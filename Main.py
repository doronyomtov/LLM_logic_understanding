from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd
from peft import PeftModel
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# Load model and tokenizer
base_model_path = r"C:\Users\doron\OneDrive\שולחן העבודה\LLM_logic_understanding\Llama3.1 8B"
finetuned_path = r"C:\Users\doron\OneDrive\שולחן העבודה\LLM_logic_understanding\llama3_finetuned"
tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
tokenizer.pad_token = tokenizer.eos_token 
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, finetuned_path)

allowed_tokens = ["yes", "no"]
allowed_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)

# Load questions
df = pd.read_csv('questions.csv')[1000:]
df = df.sample(frac=1).reset_index(drop=True) 
accuracy = 0

# Loop through each question
for index, row in df.iterrows():
    question = row['query']
    expected_answer = row['answer'].strip().lower()
    prompt = question  # Use question directly (already contains prompt)

    print(f"\nID: {row['id']}")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate the model's response
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # only the last token

    # Mask all logits except allowed tokens
        mask = torch.full_like(logits, float('-inf'))
        mask[:, allowed_ids] = logits[:, allowed_ids]

    # Sample the best of allowed tokens
        next_token_id = torch.argmax(mask, dim=-1).unsqueeze(0)

# Append to input for full decoded output (optional)
    full_output = torch.cat([inputs["input_ids"], next_token_id], dim=1)
    raw_output = tokenizer.decode(full_output[0], skip_special_tokens=True)
    response = tokenizer.decode(next_token_id[0]).strip().lower()

    # Clean model response
    response = raw_output.replace(prompt, "").strip().lower()

    # Interpret via sentiment

    print(f"Cleaned Response: {response}")

    # Compare with true answer
    if expected_answer in response.lower():
        accuracy += 1
        print("✅ Correct")
    else:
        print("❌ Incorrect")

# Final accuracy
print(f"\nFinal Accuracy: {accuracy / len(df):.2f}")
