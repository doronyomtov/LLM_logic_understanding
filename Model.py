from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Load base model and tokenizer (without fine-tuning)
base_model_path = r"C:\Users\doron\OneDrive\שולחן העבודה\LLM_logic_understanding\Llama3.1 8B"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
)

# Define allowed tokens and get their IDs
allowed_tokens = ["yes", "no"]
allowed_ids = []
yes_ids = []
no_ids = []

# Get token IDs for "yes" and "no" with various prefixes
for token in allowed_tokens:
    # Get IDs for various forms (with/without spaces)
    for prefix in ["", " ", "  "]:
        token_ids = tokenizer.encode(prefix + token, add_special_tokens=False)
        allowed_ids.extend(token_ids)
        
        # Also track which IDs correspond to yes/no separately
        if token == "yes":
            yes_ids.extend(token_ids)
        else:
            no_ids.extend(token_ids)

# Remove duplicates
allowed_ids = list(set(allowed_ids))
yes_ids = list(set(yes_ids))
no_ids = list(set(no_ids))

print(f"Allowed token IDs: {allowed_ids}")
print(f"'Yes' token IDs: {yes_ids} - {[tokenizer.decode([id]) for id in yes_ids]}")
print(f"'No' token IDs: {no_ids} - {[tokenizer.decode([id]) for id in no_ids]}")

# Load questions
df = pd.read_csv('questions.csv')[1000:]
df = df.sample(frac=1).reset_index(drop=True)

# Confidence threshold for uncertain answers
CONFIDENCE_THRESHOLD = 0.2  # If difference between yes/no probabilities is less than this, mark as uncertain

def get_constrained_answer_with_confidence(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        
        # Mask all logits except allowed tokens
        mask = torch.full_like(logits, float('-inf'))
        mask[:, allowed_ids] = logits[:, allowed_ids]
        
        # Get maximum logits for yes and no tokens
        yes_logits = torch.max(torch.index_select(mask, 1, torch.tensor(yes_ids, device=model.device)), dim=1).values
        no_logits = torch.max(torch.index_select(mask, 1, torch.tensor(no_ids, device=model.device)), dim=1).values
        
        # Compute softmax on just the yes/no logits
        scores = torch.stack([yes_logits, no_logits], dim=1)
        probabilities = torch.nn.functional.softmax(scores, dim=1)
        
        yes_prob = probabilities[0, 0].item()
        no_prob = probabilities[0, 1].item()
        
        # Choose the answer with higher probability
        if yes_prob > no_prob:
            answer = "yes"
            confidence = yes_prob
        else:
            answer = "no"
            confidence = no_prob
            
        # Check if the model is uncertain
        is_uncertain = abs(yes_prob - no_prob) < CONFIDENCE_THRESHOLD
        
        return answer, confidence, is_uncertain, {"yes": yes_prob, "no": no_prob}

# Track metrics
accuracy = 0
total = 0
answer_distribution = {"yes": 0, "no": 0}
uncertain_count = 0
correct_by_type = {"yes": 0, "no": 0}
total_by_type = {"yes": 0, "no": 0}
uncertain_by_type = {"yes": 0, "no": 0}
confidence_scores = []
results = []

# Loop through each question
for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Base Model"):
    question = row['query']
    expected_answer = row['answer'].strip().lower()
    
    print(f"\nID: {row['id']}")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    
    # Get model's answer with confidence
    answer, confidence, is_uncertain, probs = get_constrained_answer_with_confidence(question)
    
    print(f"Model Answer: {answer} (Confidence: {confidence:.4f})")
    print(f"Probabilities: Yes={probs['yes']:.4f}, No={probs['no']:.4f}")
    if is_uncertain:
        print("⚠️ Model is uncertain (probabilities are close)")
        uncertain_count += 1
        if expected_answer in ["yes", "no"]:
            uncertain_by_type[expected_answer] += 1
    
    # Track answer distribution
    answer_distribution[answer] += 1
    
    # Track confidence scores
    confidence_scores.append({
        "id": row['id'],
        "expected": expected_answer,
        "predicted": answer,
        "confidence": confidence,
        "yes_prob": probs["yes"],
        "no_prob": probs["no"],
        "is_uncertain": is_uncertain
    })
    
    # Check correctness
    is_correct = answer == expected_answer
    if is_correct:
        accuracy += 1
        print("✅ Correct")
        if expected_answer in ["yes", "no"]:
            correct_by_type[expected_answer] += 1
    else:
        print("❌ Incorrect")
    
    # Track total by expected answer type
    if expected_answer in ["yes", "no"]:
        total_by_type[expected_answer] += 1
    
    total += 1
    
    # Store result for later analysis
    results.append({
        "id": row['id'],
        "question": question,
        "expected": expected_answer,
        "predicted": answer,
        "confidence": confidence,
        "yes_prob": probs["yes"],
        "no_prob": probs["no"],
        "is_uncertain": is_uncertain,
        "correct": is_correct
    })

# Save results to CSV for further analysis
results_df = pd.DataFrame(results)
results_df.to_csv('base_model_evaluation_results_with_confidence.csv', index=False)

# Print final stats
print("\n===== BASE MODEL EVALUATION RESULTS =====")
print(f"Total questions evaluated: {total}")
print(f"Final Accuracy: {accuracy / total:.4f} ({accuracy}/{total})")

print("\nAnswer Distribution:")
for answer, count in answer_distribution.items():
    print(f"  {answer}: {count} ({count/total:.2%})")

print(f"\nUncertain Answers: {uncertain_count} ({uncertain_count/total:.2%})")

print("\nAccuracy by Expected Answer Type:")
for answer_type in ["yes", "no"]:
    if total_by_type[answer_type] > 0:
        type_accuracy = correct_by_type[answer_type] / total_by_type[answer_type]
        uncertain_percentage = uncertain_by_type[answer_type] / total_by_type[answer_type] if total_by_type[answer_type] > 0 else 0
        print(f"  {answer_type}: {correct_by_type[answer_type]}/{total_by_type[answer_type]} ({type_accuracy:.2%})")
        print(f"    Uncertain: {uncertain_by_type[answer_type]} ({uncertain_percentage:.2%})")

# Calculate average confidence for correct vs incorrect answers
correct_confidence = [r["confidence"] for r in results if r["correct"]]
incorrect_confidence = [r["confidence"] for r in results if not r["correct"]]

if correct_confidence:
    avg_correct_conf = sum(correct_confidence) / len(correct_confidence)
    print(f"\nAverage confidence when correct: {avg_correct_conf:.4f}")

if incorrect_confidence:
    avg_incorrect_conf = sum(incorrect_confidence) / len(incorrect_confidence)
    print(f"Average confidence when incorrect: {avg_incorrect_conf:.4f}")

# Identify most challenging questions
if len(results_df) > 0:
    print("\nSample of Uncertain Predictions:")
    uncertain_df = results_df[results_df['is_uncertain']].head(5)
    for _, row in uncertain_df.iterrows():
        print(f"  Question: {row['question']}")
        print(f"  Expected: {row['expected']}, Predicted: {row['predicted']} (Yes: {row['yes_prob']:.4f}, No: {row['no_prob']:.4f})")
        print()

# Analyze confidence distribution
confidence_values = [r["confidence"] for r in results]
confidence_bins = np.linspace(0.5, 1.0, 6)  # 0.5-1.0 in 5 bins
hist, _ = np.histogram(confidence_values, bins=confidence_bins)

print("\nConfidence Distribution:")
for i in range(len(hist)):
    bin_start = confidence_bins[i]
    bin_end = confidence_bins[i+1]
    bin_count = hist[i]
    print(f"  {bin_start:.2f}-{bin_end:.2f}: {bin_count} ({bin_count/total:.2%})")

# Plot confidence histogram if matplotlib is available
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_values, bins=20, alpha=0.7)
    plt.title('Distribution of Model Confidence')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.savefig('confidence_distribution.png')
    print("\nConfidence distribution histogram saved to 'confidence_distribution.png'")
except ImportError:
    print("\nMatplotlib not available for plotting. Install it with 'pip install matplotlib' to generate plots.")