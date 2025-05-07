# File: chatgpt_evaluator.py
import openai
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time # For potential rate limiting
import os 

# It's good practice to load the API key from an environment variable or a config file
# For this example, it will be passed in the constructor.
# import os

class ChatGPT_Evaluator:
    def __init__(self, model_name="gpt-3.5-turbo", csv_path=None, start_idx=0, num_rows=None, confidence_threshold=0.2):
        """
        Initialize the ChatGPT_Evaluator with API key, model name, CSV path, and other parameters.
        :param api_key: Your OpenAI API key.
        :param model_name: The OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo").
        :param csv_path: Path to the CSV file containing the evaluation data.
        :param start_idx: Starting index for evaluation (default is 0).
        :param num_rows: Number of rows to evaluate (default is None, which means all rows from start_idx).
        :param confidence_threshold: Confidence threshold for uncertain answers (default is 0.2).
        """

        self.api_key = os.getenv("OPENAI_API_KEY")
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}")

        self.model_name = model_name
        self.csv_path = csv_path
        self.start_idx = start_idx
        self.num_rows = num_rows
        self.confidence_threshold = confidence_threshold
        
        self.yes_token_strings = ["yes", " yes", "Yes", " Yes"] # Common token representations
        self.no_token_strings = ["no", " no", "No", " No"]     # Some models might add a leading space

        # For logit bias, we need token IDs. We'll use tiktoken.
        self.logit_bias = {}
        try:
            import tiktoken
            # Guess encoding based on model common patterns, not foolproof for all custom/future models
            if "gpt-4" in self.model_name or "gpt-3.5-turbo" in self.model_name or "text-embedding-ada-002" in self.model_name: #common models use cl100k_base
                 encoding_name = "cl100k_base"
            elif "davinci" in self.model_name or "curie" in self.model_name or "babbage" in self.model_name or "ada" in self.model_name and "002" not in self.model_name: # Older models
                 encoding_name = "r50k_base" # or p50k_base
            else: # Default or if model is unknown
                 encoding_name = "cl100k_base"
            self.tokenizer = tiktoken.get_encoding(encoding_name)
            print(f"Using tiktoken encoding: {encoding_name} for logit bias.")

            yes_token_ids = []
            for s in self.yes_token_strings:
                yes_token_ids.extend(self.tokenizer.encode(s))
            
            no_token_ids = []
            for s in self.no_token_strings:
                no_token_ids.extend(self.tokenizer.encode(s))

            for token_id in list(set(yes_token_ids)): 
                self.logit_bias[str(token_id)] = 75  
            for token_id in list(set(no_token_ids)):
                self.logit_bias[str(token_id)] = 75
            if not self.logit_bias:
                print("Warning: Logit bias dictionary is empty. 'yes'/'no' tokens might not have been found correctly.")


        except ImportError:
            self.tokenizer = None
            print("tiktoken not installed. Logit bias for specific 'yes'/'no' tokens cannot be applied. Model will rely on system prompt only.")
        except Exception as e:
            self.tokenizer = None
            print(f"Error initializing tiktoken or setting up logit_bias: {e}. Logit bias may not be effective.")
        
        print(f"--- ChatGPT Evaluator Initialized (Model: {self.model_name}) ---")

    def _get_constrained_answer(self, prompt_text):
        """
        Get the model's answer to the prompt, aiming for 'yes' or 'no' using OpenAI API.
        Uses system prompt, logit_bias (if available), and requests logprobs to determine answer and confidence.
        :param prompt_text: The input prompt for the model.
        :return: A tuple containing the answer ('yes'/'no'/'error'), confidence score, uncertainty flag, 
                 and a dictionary of probabilities for 'yes' and 'no'.
        """
        system_message = "You are an AI assistant. Your task is to answer the following question with only the word 'yes' or the word 'no'. Do not provide any other explanation, punctuation, or surrounding text."
        
        yes_prob = 0.0
        no_prob = 0.0
        
        try:
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_text}
                ],
                "max_tokens": 3,  # "yes" or "no" are single tokens usually, plus a little buffer for space variations.
                "temperature": 0.0, # For deterministic output
                "logprobs": True,   # Request logprobs
                "top_logprobs": 5   # Get logprobs for top N tokens at each position
            }
            if self.logit_bias: # Only add if it's populated
                completion_params["logit_bias"] = self.logit_bias

            completion = self.client.chat.completions.create(**completion_params)
            
            raw_answer_text = completion.choices[0].message.content.strip().lower()
            
            # Attempt to extract probabilities from logprobs for the first generated token
            if completion.choices[0].logprobs and completion.choices[0].logprobs.content and len(completion.choices[0].logprobs.content) > 0:
                first_token_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
                
                for logprob_entry in first_token_logprobs:
                    token_str = logprob_entry.token # Keep original case for matching with self.yes/no_token_strings
                    prob = np.exp(logprob_entry.logprob)
                    
                    if token_str in self.yes_token_strings:
                        yes_prob = max(yes_prob, prob) 
                    elif token_str in self.no_token_strings:
                        no_prob = max(no_prob, prob)
            
            # Normalize probabilities if both were found this way
            total_found_prob = yes_prob + no_prob
            if total_found_prob > 1e-6: # Avoid division by zero if positive probs were found
                yes_prob = yes_prob / total_found_prob
                no_prob = no_prob / total_found_prob
            else: # Fallback: If logprobs didn't give us "yes" or "no", rely on text and assign full confidence
                if raw_answer_text.startswith("yes"):
                    yes_prob = 1.0
                    no_prob = 0.0
                elif raw_answer_text.startswith("no"):
                    yes_prob = 0.0
                    no_prob = 1.0
                else: # Ambiguous text, default to 'no' with no confidence in 'yes'
                    yes_prob = 0.0
                    no_prob = 0.0 # This will lead to 'no' by default below due to "no_prob >= yes_prob"

            # Determine final answer based on probabilities or text parsing as final fallback
            if yes_prob > no_prob:
                answer = "yes"
            elif no_prob > yes_prob: # Includes the case where both are 0, defaulting to 'no'
                answer = "no"
            else: # Probabilities are equal (could be 0.5/0.5 or 0/0)
                if raw_answer_text.startswith("yes"):
                    answer = "yes"
                elif raw_answer_text.startswith("no"):
                    answer = "no"
                else: # Truly ambiguous, default to 'no'
                    # print(f"Warning: Ambiguous output. Raw: '{raw_answer_text}'. Probs Y:{yes_prob:.2f}, N:{no_prob:.2f}. Defaulting to 'no'.")
                    answer = "no"
                    no_prob = 1.0 # Solidify this default choice
                    yes_prob = 0.0


            confidence = max(yes_prob, no_prob)
            # If confidence is 0 (e.g. both probs were 0 and text was ambiguous), then it's uncertain.
            if confidence < 1e-6 and not (raw_answer_text.startswith("yes") or raw_answer_text.startswith("no")):
                is_uncertain = True
            else:
                is_uncertain = abs(yes_prob - no_prob) < self.confidence_threshold
            
            return answer, confidence, is_uncertain, {"yes": yes_prob, "no": no_prob}

        except openai.APIError as e:
            print(f"OpenAI API Error (Q: {prompt_text[:30]}...): {e}. Returning 'error'.")
            time.sleep(2) # Basic wait on API error
            return "error", 0.0, True, {"yes": 0.0, "no": 0.0}
        except Exception as e:
            print(f"Unexpected error getting answer from ChatGPT (Q: {prompt_text[:30]}...): {e}. Returning 'error'.")
            return "error", 0.0, True, {"yes": 0.0, "no": 0.0}

    def evaluate(self, output_path='evaluation_results_chatgpt.csv', plot_path='confidence_distribution_chatgpt.png'):
        """
        Evaluate the model on the dataset and save the results.
        :param output_path: Path to save the evaluation results CSV file.
        :param plot_path: Path to save the confidence distribution histogram.
        :return: A dictionary containing evaluation statistics.
        """
        if not self.csv_path:
            print("CSV path not provided. Cannot perform evaluation.")
            return None
        try:
            df_full = pd.read_csv(self.csv_path)
            if self.start_idx < 0 : self.start_idx = 0 # Ensure start_idx is not negative
            df = df_full.iloc[self.start_idx:]
        except FileNotFoundError:
            print(f"Error: CSV file not found at '{self.csv_path}'")
            return None
        except Exception as e:
            print(f"Error loading CSV '{self.csv_path}': {e}")
            return None

        if self.num_rows is not None:
            if self.num_rows > 0:
                 # Ensure we don't try to sample more rows than available
                num_to_sample = min(self.num_rows, len(df))
                if num_to_sample > 0 :
                    df = df.sample(n=num_to_sample, random_state=42) 
                else: # num_rows is positive but less than 1 or df is empty after slicing
                    df = df.head(0)
            elif self.num_rows == 0: # User explicitly wants to evaluate 0 rows
                df = df.head(0)
        # If self.num_rows is None, use all rows from start_idx (df is already set)
        
        if len(df) == 0:
            print(f"No data for {self.model_name} evaluation (after slicing/sampling).")
            return {"total": 0, "results": [], "accuracy":0, "answer_dist":{}} 

        stats = {
            "total": 0, "accuracy": 0, "uncertain": 0,
            "answer_dist": {"yes": 0, "no": 0, "error": 0},
            "correct_by_type": {"yes": 0, "no": 0},
            "total_by_type": {"yes": 0, "no": 0},
            "uncertain_by_type": {"yes": 0, "no": 0},
            "confidence_scores": [],
            "results": [],
            "predicted_yes_correctly": 0,
            "total_predicted_yes": 0,
            "predicted_no_correctly": 0,
            "total_predicted_no": 0
        }
        
        # Use a copy for iteration if modifications are made that pandas warns about
        df_iter = df.copy()

        for index, row in tqdm(df_iter.iterrows(), total=len(df_iter), desc=f"Evaluating {self.model_name}"):
            question = str(row.get('query','')).strip() 
            expected = str(row.get('answer', '')).strip().lower()
            q_id = row.get('id', f"RowIndex_{index}")

            if not question or expected not in ["yes", "no"]:
                # print(f"Skipping QID {q_id}: Invalid question ('{question[:30]}...') or expected answer ('{expected}').")
                continue # Skip this row
            
            answer, conf, uncertain, probs = self._get_constrained_answer(question)
            
            if answer == "error":
                stats["answer_dist"]["error"] = stats["answer_dist"].get("error", 0) + 1
                stats["results"].append({
                    "id": q_id, "question": question, "expected": expected, "predicted": "error", 
                    "confidence": 0.0, "yes_prob": 0.0, "no_prob": 0.0, 
                    "is_uncertain": True, "correct": False
                })
                continue # Move to next question after an error

            is_correct = answer == expected
            stats["results"].append({
                "id": q_id, "question": question, "expected": expected, "predicted": answer, 
                "confidence": conf, "yes_prob": probs["yes"], "no_prob": probs["no"], 
                "is_uncertain": uncertain, "correct": is_correct
            })
            
            stats["total"] += 1
            stats["answer_dist"][answer] = stats["answer_dist"].get(answer, 0) + 1
            stats["confidence_scores"].append(conf)

            if answer == "yes":
                stats["total_predicted_yes"] += 1
                if is_correct: stats["predicted_yes_correctly"] += 1
            elif answer == "no":
                stats["total_predicted_no"] += 1
                if is_correct: stats["predicted_no_correctly"] += 1
            
            stats["total_by_type"][expected] = stats["total_by_type"].get(expected, 0) + 1
            if uncertain:
                stats["uncertain"] += 1
                stats["uncertain_by_type"][expected] = stats["uncertain_by_type"].get(expected, 0) + 1
            if is_correct:
                stats["accuracy"] += 1
                stats["correct_by_type"][expected] = stats["correct_by_type"].get(expected, 0) + 1
        
        try:
            print(pd.DataFrame(stats["results"]))
            pd.DataFrame(stats["results"]).to_csv('evaluation_results_chatgpt.csv', index=False, encoding='utf-8')
            print(f"\n{self.model_name} evaluation results saved to: {'evaluation_results_chatgpt.csv'}")
        except Exception as e:
            print(f"Error saving {self.model_name} results to CSV '{'evaluation_results_chatgpt.csv'}': {e}")
        
        self._print_summary(stats, self.model_name)
        if stats["confidence_scores"]: # Only plot if there are scores
            self._plot_confidence_histogram(stats["confidence_scores"], self.model_name, plot_path)
        return stats

    def _print_summary(self, stats, model_name="Model"):
        """ Prints a formatted summary of the evaluation statistics. """
        summary_title = f"===== {model_name} EVALUATION SUMMARY ====="
        print(f"\n{summary_title}")
        total_processed = stats.get("total", 0)
        
        if total_processed == 0:
            total_errors = stats.get("answer_dist", {}).get("error", 0)
            if total_errors > 0:
                print(f"No questions successfully evaluated. Encountered {total_errors} errors.")
            else:
                print("No questions were evaluated (dataset might be empty or all rows skipped).")
            print("=" * len(summary_title))
            return

        accuracy_val = stats.get("accuracy", 0)
        acc_percent = (accuracy_val / total_processed * 100) if total_processed > 0 else 0
        print(f"Total Questions Successfully Evaluated: {total_processed}")
        print(f"Overall Accuracy: {acc_percent:.2f}% ({accuracy_val}/{total_processed})")

        print("\nPredicted Answer Distribution:")
        ans_dist = stats.get("answer_dist", {})
        for k, v in ans_dist.items():
            if v == 0 and k != "error": continue # Skip zero counts unless it's errors
            dist_percent = (v / total_processed * 100) if total_processed > 0 and k != "error" else None
            percent_str = f" ({dist_percent:.2f}%)" if dist_percent is not None else ""
            print(f"  {k.capitalize()}: {v}{percent_str}")

        uncertain_val = stats.get("uncertain", 0)
        uncertain_percent = (uncertain_val / total_processed * 100) if total_processed > 0 else 0
        print(f"\nUncertain Answers (Confidence Diff < {self.confidence_threshold:.2f}): {uncertain_val} ({uncertain_percent:.2f}%)")

        pred_yes_correct = stats.get("predicted_yes_correctly", 0)
        total_pred_yes = stats.get("total_predicted_yes", 0)
        precision_yes = (pred_yes_correct / total_pred_yes * 100) if total_pred_yes > 0 else 0
        print(f"\nPrecision (Predicted 'Yes'): {precision_yes:.2f}% ({pred_yes_correct}/{total_pred_yes})")

        pred_no_correct = stats.get("predicted_no_correctly", 0)
        total_pred_no = stats.get("total_predicted_no", 0)
        precision_no = (pred_no_correct / total_pred_no * 100) if total_pred_no > 0 else 0
        print(f"Precision (Predicted 'No'): {precision_no:.2f}% ({pred_no_correct}/{total_pred_no})")
        
        print("\nPerformance by True Answer Type (Expected):")
        for type_expected in ["yes", "no"]:
            total_t = stats.get("total_by_type", {}).get(type_expected, 0)
            correct_t = stats.get("correct_by_type", {}).get(type_expected, 0)
            # uncertain_t = stats.get("uncertain_by_type", {}).get(type_expected, 0) # This can be added if needed
            if total_t > 0:
                acc_t_percent = (correct_t / total_t * 100)
                print(f"  Expected '{type_expected.capitalize()}': Accuracy {acc_t_percent:.2f}% ({correct_t}/{total_t})")
            else:
                print(f"  Expected '{type_expected.capitalize()}': 0 questions")

        valid_results = [r for r in stats.get("results", []) if r.get("predicted") != "error" and r.get("confidence") is not None]
        correct_conf = [r["confidence"] for r in valid_results if r.get("correct")]
        incorrect_conf = [r["confidence"] for r in valid_results if not r.get("correct")]
        
        if correct_conf: print(f"\nAvg. Confidence (Correct Predictions): {np.mean(correct_conf):.4f}")
        if incorrect_conf: print(f"Avg. Confidence (Incorrect Predictions): {np.mean(incorrect_conf):.4f}")
        print("=" * len(summary_title))

    def _plot_confidence_histogram(self, confidence_values, model_name="Model", save_path='confidence_distribution.png'):
        """ Plots a histogram of confidence scores and saves it. """
        if not confidence_values:
            print(f"\nNo confidence scores available to plot for {model_name}.")
            return
        try:
            plt.figure(figsize=(10, 6))
            # Filter out None or non-numeric if any slip through, though confidence should always be float
            valid_conf_values = [c for c in confidence_values if isinstance(c, (int, float))]
            if not valid_conf_values:
                print(f"\nNo valid numeric confidence scores to plot for {model_name}.")
                plt.close()
                return

            plt.hist(valid_conf_values, bins=np.linspace(0, 1, 21), alpha=0.75, edgecolor='black', color='skyblue')
            plt.title(f'{model_name}: Distribution of Prediction Confidence Scores', fontsize=15)
            plt.xlabel('Confidence Score', fontsize=12)
            plt.ylabel('Frequency (Number of Questions)', fontsize=12)
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            plt.savefig(save_path)
            print(f"\nConfidence histogram for {model_name} saved to: '{save_path}'")
            plt.close() # Close the plot figure to free memory
        except ImportError:
            print("\nMatplotlib is not installed. Please install it to plot histograms (e.g., pip install matplotlib).")
        except Exception as plot_e:
            print(f"\nError occurred while plotting/saving confidence histogram for {model_name}: {plot_e}")
            if plt.get_fignums(): # If a figure is open, try to close it
                plt.close()

    def predict_single(self, question):
        """
        Predict the answer for a single question using ChatGPT.
        :param question: The input question (string).
        :return: A tuple: (predicted_answer_str, confidence_float, is_uncertain_bool, probabilities_dict).
                 Returns ('error', 0.0, True, {'yes':0.0, 'no':0.0}) on failure.
        """
        return self._get_constrained_answer(question.strip())

