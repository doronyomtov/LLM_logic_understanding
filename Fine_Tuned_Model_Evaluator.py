
import torch, pandas as pd, numpy as np, json, matplotlib.pyplot as plt
from peft import PeftModel 
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class Fine_Tuned_Model_Evaluator: 
    def __init__(self, base_model_path, finetuned_path, csv_path, start_idx=1000, num_rows=None, confidence_threshold=0.2):
        """
        Initialize the evaluator with paths to the base model, fine-tuned model, and CSV file.
        :param base_model_path: Path to the base model.
        :param finetuned_path: Path to the fine-tuned model.
        :param csv_path: Path to the CSV file containing the data.
        :param start_idx: Starting index for evaluation.
        :param num_rows: Number of rows to evaluate. If None, evaluates all rows.
        :param confidence_threshold: Confidence threshold for uncertain answers.
        """
        self.base_model_path = base_model_path
        self.finetuned_path = finetuned_path
        self.csv_path = csv_path
        self.start_idx = start_idx
        self.num_rows = num_rows
        self.confidence_threshold = confidence_threshold
        self.allowed_tokens = ["yes", "no"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model_and_tokenizer()
        self._prepare_token_ids()
        print("--- Fine-tuned Evaluator Initialized ---")

    def _load_model_and_tokenizer(self):
        """
        Load the model and tokenizer from the specified paths.
        """
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
        self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_path) 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loading Base Model for PEFT...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, torch_dtype=torch.float16, device_map=self.device, trust_remote_code=True
        )
        base_model.eval()
        print("Loading PEFT Adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.finetuned_path)
        self.model.eval()
        print("Fine-tuned Model Loaded.")


    def _prepare_token_ids(self):
        """
        Prepare the token IDs for the allowed tokens and their prefixes.
        """
        self.allowed_ids = set(); self.yes_ids = set(); self.no_ids = set()
        for token in self.allowed_tokens:
            for prefix in ["", " ", "  "]:
                token_ids = self.tokenizer.encode(prefix + token, add_special_tokens=False)
                self.allowed_ids.update(token_ids)
                if token == "yes": self.yes_ids.update(token_ids)
                else: self.no_ids.update(token_ids)
        print(f"Allowed token IDs: {list(self.allowed_ids)}")
        print(f"'Yes' token IDs: {list(self.yes_ids)}")
        print(f"'No' token IDs: {list(self.no_ids)}")

    def _get_constrained_answer(self, prompt):
        """
        Get the constrained answer from the model based on the prompt.
        :param prompt: The input prompt for the model.
        :return: The answer, confidence score, uncertainty flag, and probabilities for 'yes' and 'no'.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs); logits = outputs.logits[:, -1, :]
            mask = torch.full_like(logits, float('-inf'))
            if not self.allowed_ids: return "error", 0.0, True, {"yes": 0.0, "no": 0.0}
            allowed_tensor = torch.tensor(list(self.allowed_ids), device=self.model.device); mask[:, allowed_tensor] = logits[:, allowed_tensor]
            yes_tensor = torch.tensor(list(self.yes_ids), device=self.model.device) if self.yes_ids else None
            no_tensor = torch.tensor(list(self.no_ids), device=self.model.device) if self.no_ids else None
            yes_logits = torch.max(torch.index_select(mask, 1, yes_tensor), dim=1).values if yes_tensor is not None and len(yes_tensor)>0 else torch.tensor([float('-inf')], device=self.model.device)
            no_logits = torch.max(torch.index_select(mask, 1, no_tensor), dim=1).values if no_tensor is not None and len(no_tensor)>0 else torch.tensor([float('-inf')], device=self.model.device)
            scores = torch.stack([yes_logits, no_logits], dim=1); probs = torch.nn.functional.softmax(scores, dim=1)
            yes_prob, no_prob = probs[0, 0].item(), probs[0, 1].item()
            answer = "yes" if yes_prob > no_prob else "no"; confidence = max(yes_prob, no_prob)
            if abs(yes_prob - no_prob) < 1e-9: answer = "no"
            is_uncertain = abs(yes_prob - no_prob) < self.confidence_threshold
            return answer, confidence, is_uncertain, {"yes": yes_prob, "no": no_prob}

    def evaluate(self, output_path='evaluation_results_base.csv', plot_path='confidence_distribution_base.png'):
        """
        Evaluate the model on the dataset and save the results.
        :param output_path: Path to save the evaluation results CSV file.
        :param plot_path: Path to save the confidence distribution histogram.
        :return: A dictionary containing evaluation statistics.
        """
        try:
            df_full = pd.read_csv(self.csv_path);
            df = df_full.iloc[self.start_idx:]
        except Exception as e:
            print(f"Error loading CSV: {e}"); return None

        if self.num_rows is not None and self.num_rows > 0:
            df = df.sample(n=min(self.num_rows, len(df)), random_state=42)  # Keep random_state for now
        elif self.num_rows == 0:
            df = df.head(0)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Keep random_state for now

        if len(df) == 0: print("No data for Base eval."); return None

        stats = {
            "total": 0, "accuracy": 0, "uncertain": 0,
            "answer_dist": {"yes": 0, "no": 0},
            "correct_by_type": {"yes": 0, "no": 0},  # Based on Expected
            "total_by_type": {"yes": 0, "no": 0},  # Based on Expected
            "uncertain_by_type": {"yes": 0, "no": 0},
            "confidence_scores": [],
            "results": [],
            "predicted_yes_correctly": 0,  # Predicted yes AND was correct
            "total_predicted_yes": 0,  # Total times model predicted yes
            "predicted_no_correctly": 0,  # Predicted no AND was correct
            "total_predicted_no": 0  # Total times model predicted no
        }


        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Base"):
            question = row['query'];
            expected = str(row.get('answer', '')).strip().lower();
            q_id = row.get('id', f"Index_{index}")
            if not question or expected not in ["yes", "no"]: continue
            answer, conf, uncertain, probs = self._get_constrained_answer(question)
            if answer == "error": continue
            is_correct = answer == expected
            stats["results"].append(
                {"id": q_id, "question": question, "expected": expected, "predicted": answer, "confidence": conf,
                 "yes_prob": probs["yes"], "no_prob": probs["no"], "is_uncertain": uncertain, "correct": is_correct})

            print(f"\n--- QID: {q_id} ({stats['total'] + 1}/{len(df)}) ---")
            print(f"Question: {question}");
            print(f"Expected: {expected}");
            print(f"Predicted: {answer}");
            print(f"Result: {'Correct! ✅' if is_correct else 'Incorrect! ❌'}");
            print("-" * 20)

            stats["total"] += 1
            stats["answer_dist"][answer] = stats["answer_dist"].get(answer, 0) + 1  # Safer update
            stats["confidence_scores"].append(conf)

            if answer == "yes":
                stats["total_predicted_yes"] += 1
                if is_correct:  # Predicted yes, Expected yes
                    stats["predicted_yes_correctly"] += 1
            elif answer == "no":
                stats["total_predicted_no"] += 1
                if is_correct:  # Predicted no, Expected no
                    stats["predicted_no_correctly"] += 1

            if uncertain: stats["uncertain"] += 1; stats["uncertain_by_type"][expected] = stats[
                                                                                              "uncertain_by_type"].get(
                expected, 0) + 1
            stats["total_by_type"][expected] = stats.get("total_by_type", {}).get(expected, 0) + 1
            if is_correct: stats["accuracy"] += 1; stats["correct_by_type"][expected] = stats.get("correct_by_type",
                                                                                                  {}).get(expected,
                                                                                                          0) + 1

        try:
            pd.DataFrame(stats["results"]).to_csv(output_path, index=False); print(
                f"\nBase results saved: {output_path}")
        except Exception as e:
            print(f"Error saving base CSV: {e}")
        self._print_summary(stats, "Base Model")
        if stats["confidence_scores"]: self._plot_confidence_histogram(stats["confidence_scores"], "Base Model",
                                                                       plot_path)
        return stats

    def _print_summary(self, stats, model_name="Model"):
        """
        Print a summary of the evaluation results.
        :param stats: A dictionary containing evaluation statistics.
        :param model_name: Name of the model for display purposes.
        """
        summary_title = f"===== {model_name} EVALUATION SUMMARY ====="; print(f"\n{summary_title}")
        total = stats.get("total", 0);
        if total == 0: print("No questions evaluated."); print("="*len(summary_title)); return
        accuracy = stats.get("accuracy", 0); acc_percent = (accuracy/total * 100) if total > 0 else 0
        print(f"Total Questions: {total}"); print(f"Accuracy: {acc_percent:.2f}% ({accuracy}/{total})")
        print("\nAnswer Distribution:"); ans_dist = stats.get("answer_dist", {});
        for k, v in ans_dist.items(): print(f"  {k}: {v} ({v/total:.2%})") if total > 0 else print(f"  {k}: {v}")
        uncertain = stats.get("uncertain", 0); uncertain_percent = (uncertain/total * 100) if total > 0 else 0
        print(f"\nUncertain Answers: {uncertain} ({uncertain_percent:.2%})")
        print("\nAccuracy by Answer Type:")
        for t in ["yes", "no"]:
             total_t = stats.get("total_by_type", {}).get(t, 0); correct_t = stats.get("correct_by_type", {}).get(t, 0); uncertain_t = stats.get("uncertain_by_type", {}).get(t, 0)
             if total_t > 0: acc = correct_t / total_t; unc = uncertain_t / total_t; print(f"  {t}: {correct_t}/{total_t} ({acc:.2%})"); print(f"    Uncertain: {uncertain_t}/{total_t} ({unc:.2%})")
             else: print(f"  {t}: 0 questions")
        results_list = stats.get("results", []); correct_conf = [r["confidence"] for r in results_list if r.get("correct")]; incorrect_conf = [r["confidence"] for r in results_list if not r.get("correct")]
        if correct_conf: print(f"\nAvg. Confidence (Correct): {np.mean(correct_conf):.4f}")
        if incorrect_conf: print(f"Avg. Confidence (Incorrect): {np.mean(incorrect_conf):.4f}")
        print("="*len(summary_title))

    def _plot_confidence_histogram(self, confidence_values, model_name="Model", save_path='confidence_distribution.png'): 
        """
        Plot a histogram of the confidence scores.
        :param confidence_values: List of confidence scores.
        :param model_name: Name of the model for display purposes.
        :param save_path: Path to save the histogram plot.
        """
        if not confidence_values: print(f"\nNo scores to plot for {model_name}."); return
        try: plt.figure(figsize=(8, 5)); plt.hist(confidence_values, bins=20, alpha=0.75, edgecolor='black'); plt.title(f'{model_name}: Confidence Distribution'); plt.xlabel('Confidence'); plt.ylabel('Count'); plt.grid(axis='y', alpha=0.5); plt.tight_layout(); plt.savefig(save_path); print(f"\nHist saved: '{save_path}'"); plt.close()
        except ImportError: print("\nInstall matplotlib to plot.")
        except Exception as plot_e: print(f"\nError plotting: {plot_e}")

    def predict_single(self, question):
        """
        Predict the answer for a single question.
        :param question: The input question.
        :return: The predicted answer, confidence score, and uncertainty flag.
        """
        return self._get_constrained_answer(question)