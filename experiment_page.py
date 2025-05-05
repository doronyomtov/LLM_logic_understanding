import customtkinter as ctk
from tkinter import messagebox, ttk
import pandas as pd 
import torch
import os
import threading
import time
import gc
import traceback


try:

    from Base_Model_Evaluator import Base_Model_Evaluator # Ensure filename is correct
except ImportError:
    messagebox.showerror("Import Error", "Could not import 'Llama3BaseEvaluator' from llama3_evaluator_base.py.")
    exit()
try:
    from Fine_Tuned_Model_Evaluator import Fine_Tuned_Model_Evaluator # Ensure filename is correct
except ImportError:
    messagebox.showerror("Import Error", "Could not import 'Llama3FinetunedEvaluator' from llama3_evaluator_finetuned.py.")
    exit()


# --- Path Settings ---
BASE_MODEL_PATH = "C:/Users/doron/OneDrive/שולחן העבודה/LLM_logic_understanding/Llama3.1 8B"
FINETUNED_PATH = "C:/Users/doron/OneDrive/שולחן העבודה/LLM_logic_understanding/llama3_finetuned"
QUESTIONS_CSV_PATH = 'questions.csv' # Original full dataset
TEMP_CSV_PATH = '_temp_questions_for_run.csv' # Temporary file for current run's questions
START_IDX = 1000
CONFIDENCE_THRESHOLD = 0.2


def open_experiment_page(parent):
    """Open the Llama3 Comparison / Single Query GUI."""
    window = ctk.CTkToplevel(parent); window.title("Llama3 Comparison / Single Query"); window.geometry("1150x850"); window.resizable(True, True)
    window.attributes('-topmost', True); window.after(100, lambda: window.attributes('-topmost', False))

    title_label = ctk.CTkLabel(window, text="🔬 Llama3 Comparison & Single Query", font=ctk.CTkFont(size=24, weight="bold"), text_color="#4a68ff"); title_label.pack(pady=(10, 5))

    # --- Top controls frame (Status, Progress, Bulk Input) ---
    top_controls_frame = ctk.CTkFrame(window, fg_color="transparent"); top_controls_frame.pack(pady=5, fill="x", padx=20)
    top_controls_frame.grid_columnconfigure(0, weight=1); top_controls_frame.grid_columnconfigure(1, weight=0); top_controls_frame.grid_columnconfigure(2, weight=0)
    loading_status_label = ctk.CTkLabel(top_controls_frame, text="Ready.", font=ctk.CTkFont(size=12), anchor="w"); loading_status_label.grid(row=0, column=0, sticky="ew", padx=(5, 10), pady=5)
    progress_bar = ctk.CTkProgressBar(top_controls_frame, width=150, height=10, mode='indeterminate'); progress_bar.grid(row=0, column=1, sticky="e", padx=10, pady=5); progress_bar.set(0); progress_bar.stop()
    input_frame = ctk.CTkFrame(top_controls_frame, fg_color="transparent"); input_frame.grid(row=0, column=2, sticky="e", padx=5, pady=5)
    ctk.CTkLabel(input_frame, text="Num Questions (Bulk):", font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=(10, 5), pady=5) # Renamed label
    num_questions_entry = ctk.CTkEntry(input_frame, width=80); num_questions_entry.grid(row=0, column=1, padx=(0, 10), pady=5); num_questions_entry.insert(0, "10")
    run_button = ctk.CTkButton(input_frame, text="Run Comparison", state="normal", width=150); run_button.grid(row=0, column=2, padx=10, pady=5)

    # --- Single Question Input Frame ---
    single_question_frame = ctk.CTkFrame(window)
    single_question_frame.pack(pady=10, padx=20, fill="x")
    single_question_frame.grid_columnconfigure(1, weight=1) # Allow textbox to expand

    ctk.CTkLabel(single_question_frame, text="Single Question:", font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=(10,5), pady=(5,0), sticky="nw")
    question_input_textbox = ctk.CTkTextbox(single_question_frame, height=60, wrap="word", font=("Segoe UI", 13))
    question_input_textbox.grid(row=1, column=0, columnspan=3, padx=10, pady=(0,5), sticky="ew")

    ctk.CTkLabel(single_question_frame, text="Expected Answer:", font=ctk.CTkFont(size=14)).grid(row=2, column=0, padx=(10,5), pady=5, sticky="e")
    expected_answer_var = ctk.StringVar(value="yes") # Variable to hold selection
    expected_answer_optionmenu = ctk.CTkOptionMenu(single_question_frame, variable=expected_answer_var, values=["yes", "no"])
    expected_answer_optionmenu.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    ask_question_button = ctk.CTkButton(single_question_frame, text="Ask Single Question", command=lambda: start_single_question_thread()) # Link later
    ask_question_button.grid(row=2, column=2, padx=(10,10), pady=5, sticky="e")


    # --- Main Frame for Side-by-Side Results ---
    comparison_frame = ctk.CTkFrame(window, fg_color="transparent")
    comparison_frame.pack(pady=(5, 10), padx=20, fill="both", expand=True) # Adjusted padding
    comparison_frame.grid_columnconfigure(0, weight=1, uniform="group1"); comparison_frame.grid_columnconfigure(1, weight=1, uniform="group1"); comparison_frame.grid_rowconfigure(0, weight=1)

    # --- Left Column Widgets ---
    left_column_frame = ctk.CTkFrame(comparison_frame); left_column_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5); left_column_frame.grid_rowconfigure(1, weight=3); left_column_frame.grid_rowconfigure(2, weight=1); left_column_frame.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(left_column_frame, text="--- Base Model ---", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=(5,2))
    log_textbox_base = ctk.CTkTextbox(left_column_frame, wrap="word", font=("Consolas", 10), state="disabled"); log_textbox_base.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
    summary_frame_base = ctk.CTkFrame(left_column_frame); summary_frame_base.grid(row=2, column=0, sticky="nsew", padx=5, pady=(2,5)); summary_frame_base.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(summary_frame_base, text="Processed:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, padx=5, pady=1, sticky="e"); processed_label_base = ctk.CTkLabel(summary_frame_base, text="N/A", font=ctk.CTkFont(size=11)); processed_label_base.grid(row=0, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_base, text="Accuracy:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=0, padx=5, pady=1, sticky="e"); acc_overall_label_base = ctk.CTkLabel(summary_frame_base, text="N/A", font=ctk.CTkFont(size=11)); acc_overall_label_base.grid(row=1, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_base, text="Acc (yes):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=2, column=0, padx=5, pady=1, sticky="e"); acc_yes_label_base = ctk.CTkLabel(summary_frame_base, text="N/A", font=ctk.CTkFont(size=11)); acc_yes_label_base.grid(row=2, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_base, text="Acc (no):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=3, column=0, padx=5, pady=1, sticky="e"); acc_no_label_base = ctk.CTkLabel(summary_frame_base, text="N/A", font=ctk.CTkFont(size=11)); acc_no_label_base.grid(row=3, column=1, padx=5, pady=1, sticky="w")

    # --- Right Column Widgets ---
    right_column_frame = ctk.CTkFrame(comparison_frame); right_column_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5); right_column_frame.grid_rowconfigure(1, weight=3); right_column_frame.grid_rowconfigure(2, weight=1); right_column_frame.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(right_column_frame, text="--- Fine-tuned Model ---", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=(5,2))
    log_textbox_ft = ctk.CTkTextbox(right_column_frame, wrap="word", font=("Consolas", 10), state="disabled"); log_textbox_ft.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
    summary_frame_ft = ctk.CTkFrame(right_column_frame); summary_frame_ft.grid(row=2, column=0, sticky="nsew", padx=5, pady=(2,5)); summary_frame_ft.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(summary_frame_ft, text="Processed:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, padx=5, pady=1, sticky="e"); processed_label_ft = ctk.CTkLabel(summary_frame_ft, text="N/A", font=ctk.CTkFont(size=11)); processed_label_ft.grid(row=0, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_ft, text="Accuracy:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=0, padx=5, pady=1, sticky="e"); acc_overall_label_ft = ctk.CTkLabel(summary_frame_ft, text="N/A", font=ctk.CTkFont(size=11)); acc_overall_label_ft.grid(row=1, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_ft, text="Acc (yes):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=2, column=0, padx=5, pady=1, sticky="e"); acc_yes_label_ft = ctk.CTkLabel(summary_frame_ft, text="N/A", font=ctk.CTkFont(size=11)); acc_yes_label_ft.grid(row=2, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_ft, text="Acc (no):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=3, column=0, padx=5, pady=1, sticky="e"); acc_no_label_ft = ctk.CTkLabel(summary_frame_ft, text="N/A", font=ctk.CTkFont(size=11)); acc_no_label_ft.grid(row=3, column=1, padx=5, pady=1, sticky="w")


    def update_gui(func, *args, **kwargs):
        """Update GUI elements safely from a thread."""
        window.after(0, lambda: func(*args, **kwargs))

    def format_log_from_results(results_list):
        """Format the results list into a detailed log string.
        Args:
            results_list (list): List of result dictionaries.
            Returns:
                str: Formatted log string."""

        log_lines = []
        if not results_list: return "No results to display."
        for i, res in enumerate(results_list):
            if i > 0: log_lines.append("")
            q_id = res.get("id", f"Index_{i}")
            log_lines.append(f"--- QID: {q_id} ({i+1}/{len(results_list)}) ---")
            log_lines.append(f"Question: {res.get('question', 'N/A')}")
            log_lines.append(f"Expected: {res.get('expected', 'N/A')}")
            log_lines.append(f"Predicted: {res.get('predicted', 'N/A')}")
            result_str = "Correct! ✅" if res.get('correct') else "Incorrect! ❌"
            log_lines.append(f"Result: {result_str}")
            log_lines.append("-" * 20)
        return "\n".join(log_lines) if log_lines else "No results to display."


    def format_summary_from_stats(stats_dict):
        """Format the stats dictionary into a summary dictionary.
        Args:
            stats_dict (dict): Dictionary containing evaluation statistics.
        Returns:
            dict: Summary dictionary with processed, overall, yes, and no accuracy."""

        summary_data = {"processed": "N/A", "overall": "N/A", "yes": "N/A", "no": "N/A"}
        if not stats_dict or not isinstance(stats_dict, dict): return summary_data
        total = stats_dict.get("total", 0); accuracy = stats_dict.get("accuracy", 0)
        total_by_type = stats_dict.get("total_by_type", {}); correct_by_type = stats_dict.get("correct_by_type", {})
        summary_data["processed"] = str(total)
        if total > 0:
            overall_perc = (accuracy / total * 100); summary_data["overall"] = f"{accuracy}/{total} ({overall_perc:.2f}%)"
            total_yes = total_by_type.get("yes", 0); correct_yes = correct_by_type.get("yes", 0)
            summary_data["yes"] = f"{correct_yes}/{total_yes} ({correct_yes/total_yes*100:.2f}%)" if total_yes > 0 else "0/0 (N/A)"
            total_no = total_by_type.get("no", 0); correct_no = correct_by_type.get("no", 0)
            summary_data["no"] = f"{correct_no}/{total_no} ({correct_no/total_no*100:.2f}%)" if total_no > 0 else "0/0 (N/A)"
        else: summary_data["overall"] = "0/0 (N/A)"; summary_data["yes"] = "0/0 (N/A)"; summary_data["no"] = "0/0 (N/A)"
        return summary_data



    def run_bulk_comparison_thread():
        """Run the bulk comparison in a separate thread."""

        num_q_str = num_questions_entry.get(); num_rows_to_eval = int(num_q_str) if num_q_str.isdigit() and int(num_q_str) >= 0 else -1
        if num_rows_to_eval < 0 : messagebox.showerror("Input Error","Please enter a non-negative number."); return


        temp_file_created = False; final_sample_df = None
        try:
            update_gui(loading_status_label.configure, text="Preparing questions...")
            print(f"\n--- Preparing {num_rows_to_eval} questions ---")
            full_df = pd.read_csv(QUESTIONS_CSV_PATH); eval_df = full_df.iloc[START_IDX:]
            print(f"Loaded {len(eval_df)} questions after start_idx.")
            if num_rows_to_eval > 0: sample_n = min(num_rows_to_eval, len(eval_df)); print(f"Sampling {sample_n} questions."); final_sample_df = eval_df.sample(n=sample_n)
            else: final_sample_df = eval_df.head(0)
            if len(final_sample_df) > 0: final_sample_df = final_sample_df.sample(frac=1).reset_index(drop=True); print(f"Shuffled {len(final_sample_df)} questions.")
            else: print("Sample is empty.")
            final_sample_df.to_csv(TEMP_CSV_PATH, index=False); temp_file_created = True; print(f"Saved {len(final_sample_df)} questions to {TEMP_CSV_PATH}")
        except Exception as prep_e: messagebox.showerror("Error", f"Failed to prepare questions: {prep_e}"); update_gui(run_button.configure, state="normal"); update_gui(loading_status_label.configure, text="Error"); return

        # 3. Update GUI - Start running state (Disable BOTH buttons)
        update_gui(run_button.configure, state="disabled"); update_gui(ask_question_button.configure, state="disabled")
        update_gui(loading_status_label.configure, text="Initializing...")
        update_gui(progress_bar.configure, mode='indeterminate'); update_gui(progress_bar.start)
        for textbox in [log_textbox_base, log_textbox_ft]: update_gui(textbox.configure, state="normal"); update_gui(textbox.delete, "1.0", "end"); update_gui(textbox.configure, state="disabled")
        for label in [processed_label_base, acc_overall_label_base, acc_yes_label_base, acc_no_label_base, processed_label_ft, acc_overall_label_ft, acc_yes_label_ft, acc_no_label_ft]: update_gui(label.configure, text="Running...")

        start_time_total = time.time(); base_stats = None; ft_stats = None;
        evaluator_base = None; evaluator_ft = None # Separate instances

        try:
            # --- Run Base Evaluation ---
            update_gui(loading_status_label.configure, text="Loading & Evaluating Base...")
            print(f"\n--- Creating BASE Evaluator ---")
            evaluator_base = Base_Model_Evaluator(BASE_MODEL_PATH, TEMP_CSV_PATH, 0, None, CONFIDENCE_THRESHOLD)
            print("--- Base Initialized ---"); print(f"\n--- Calling base_evaluator.evaluate() ---");
            # GET STATS DICT DIRECTLY
            base_stats = evaluator_base.evaluate(output_path='comparison_base_results.csv', plot_path='comparison_base_confidence.png')
            print("--- Base Eval Finished ---")

            # --- Cleanup Base ---
            print("--- Releasing Base ---");
            if evaluator_base is not None: del evaluator_base; evaluator_base = None
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect(); time.sleep(1)


            update_gui(loading_status_label.configure, text="Loading & Evaluating Fine-tuned...")
            print(f"\n--- Creating FINE-TUNED Evaluator ---")
            evaluator_ft = Fine_Tuned_Model_Evaluator(BASE_MODEL_PATH, FINETUNED_PATH, TEMP_CSV_PATH, 0, None, CONFIDENCE_THRESHOLD)
            print("--- Fine-tuned Initialized ---")
            print(f"\n--- Calling finetuned_evaluator.evaluate() ---");

            ft_stats = evaluator_ft.evaluate(output_path='comparison_finetuned_results.csv', plot_path='comparison_finetuned_confidence.png')
            print("--- Fine-tuned Eval Finished ---")

        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); window.after(0, lambda: messagebox.showerror("Runtime Error", f"An error occurred: {e}\nCheck console.", parent=window)); update_gui(loading_status_label.configure, text="Error...")
        finally:
            update_gui(progress_bar.stop); update_gui(progress_bar.set, 0)
            run_duration = time.time() - start_time_total


            if base_stats and isinstance(base_stats, dict):
                 detailed_log_base = format_log_from_results(base_stats.get("results", []))
                 summary_data_base = format_summary_from_stats(base_stats) # Use the stats dict
                 update_gui(log_textbox_base.configure, state="normal"); update_gui(log_textbox_base.delete, "1.0", "end"); update_gui(log_textbox_base.insert, "1.0", detailed_log_base); update_gui(log_textbox_base.configure, state="disabled")
                 update_gui(processed_label_base.configure, text=summary_data_base.get("processed", "N/A"))
                 update_gui(acc_overall_label_base.configure, text=summary_data_base.get("overall", "N/A"))
                 update_gui(acc_yes_label_base.configure, text=summary_data_base.get("yes", "N/A"))
                 update_gui(acc_no_label_base.configure, text=summary_data_base.get("no", "N/A"))
            else: update_gui(log_textbox_base.configure, state="normal"); update_gui(log_textbox_base.delete, "1.0", "end"); update_gui(log_textbox_base.insert, "1.0", "Base run failed/no stats."); update_gui(log_textbox_base.configure, state="disabled"); update_gui(processed_label_base.configure, text="Failed"); update_gui(acc_overall_label_base.configure, text="Failed"); update_gui(acc_yes_label_base.configure, text="Failed"); update_gui(acc_no_label_base.configure, text="Failed")


            if ft_stats and isinstance(ft_stats, dict):
                 detailed_log_ft = format_log_from_results(ft_stats.get("results", []))
                 summary_data_ft = format_summary_from_stats(ft_stats) # Use the stats dict
                 update_gui(log_textbox_ft.configure, state="normal"); update_gui(log_textbox_ft.delete, "1.0", "end"); update_gui(log_textbox_ft.insert, "1.0", detailed_log_ft); update_gui(log_textbox_ft.configure, state="disabled")
                 update_gui(processed_label_ft.configure, text=summary_data_ft.get("processed", "N/A"))
                 update_gui(acc_overall_label_ft.configure, text=summary_data_ft.get("overall", "N/A"))
                 update_gui(acc_yes_label_ft.configure, text=summary_data_ft.get("yes", "N/A"))
                 update_gui(acc_no_label_ft.configure, text=summary_data_ft.get("no", "N/A"))
            else: update_gui(log_textbox_ft.configure, state="normal"); update_gui(log_textbox_ft.delete, "1.0", "end"); update_gui(log_textbox_ft.insert, "1.0", "Fine-tuned run failed/no stats."); update_gui(log_textbox_ft.configure, state="disabled"); update_gui(processed_label_ft.configure, text="Failed"); update_gui(acc_overall_label_ft.configure, text="Failed"); update_gui(acc_yes_label_ft.configure, text="Failed"); update_gui(acc_no_label_ft.configure, text="Failed")


            final_status = f"Comparison finished in {run_duration:.1f}s."
            update_gui(loading_status_label.configure, text=final_status);

            update_gui(run_button.configure, state="normal")
            update_gui(ask_question_button.configure, state="normal")

            print("--- Releasing Final Resources & Temp File ---");
            if evaluator_base is not None: del evaluator_base
            if evaluator_ft is not None: del evaluator_ft
            if torch.cuda.is_available(): print("Clearing CUDA cache..."); torch.cuda.empty_cache()
            print("Running garbage collection..."); gc.collect();
            if temp_file_created and os.path.exists(TEMP_CSV_PATH):
                try: os.remove(TEMP_CSV_PATH); print(f"Removed temp file: {TEMP_CSV_PATH}")
                except Exception as del_e: print(f"Warn: Could not delete temp file {TEMP_CSV_PATH}: {del_e}")
            print("Cleanup attempt complete.")

    def run_single_question_thread():
        """Run the single question evaluation in a separate thread."""

        question_text = question_input_textbox.get("1.0", "end-1c").strip()
        expected_answer = expected_answer_var.get()
        if not question_text: messagebox.showwarning("Input Missing", ...); return

        update_gui(run_button.configure, state="disabled"); update_gui(ask_question_button.configure, state="disabled")
        update_gui(loading_status_label.configure, text="Initializing for single query...")
        update_gui(progress_bar.configure, mode='indeterminate'); update_gui(progress_bar.start)

        for textbox in [log_textbox_base, log_textbox_ft]: update_gui(textbox.configure, state="normal"); update_gui(textbox.delete, "1.0", "end"); update_gui(textbox.configure, state="disabled")
        for label in [processed_label_base, acc_overall_label_base, acc_yes_label_base, acc_no_label_base, processed_label_ft, acc_overall_label_ft, acc_yes_label_ft, acc_no_label_ft]: update_gui(label.configure, text="N/A")

        start_time_total = time.time(); base_result_log = "Run failed."; ft_result_log = "Run failed."
        evaluator_base = None; evaluator_ft = None

        try:
            update_gui(loading_status_label.configure, text="Loading & Predicting Base...")
            print(f"\n--- Creating BASE Evaluator (Single Query) ---")

            evaluator_base = Base_Model_Evaluator(BASE_MODEL_PATH, "dummy.csv", 0, 1, CONFIDENCE_THRESHOLD)
            print("--- Base Initialized ---"); print(f"\n--- Calling base_evaluator.predict_single() ---");

            answer_b, _, _, _ = evaluator_base.predict_single(question_text)
            print("--- Base Predict Finished ---")
            if answer_b != "error":
                is_correct_b = answer_b == expected_answer
                base_result_log = f"Question: {question_text}\nExpected: {expected_answer}\nPredicted: {answer_b}\nResult: {'Correct! ✅' if is_correct_b else 'Incorrect! ❌'}"
                update_gui(processed_label_base.configure, text="1") # Update summary for single question
                update_gui(acc_overall_label_base.configure, text="1/1 (100%)" if is_correct_b else "0/1 (0%)")
                update_gui(acc_yes_label_base.configure, text=("1/1 (100%)" if is_correct_b else "0/1 (0%)") if expected_answer=='yes' else "N/A")
                update_gui(acc_no_label_base.configure, text=("1/1 (100%)" if is_correct_b else "0/1 (0%)") if expected_answer=='no' else "N/A")
            else: base_result_log = "Base failed to predict."; update_gui(processed_label_base.configure, text="Error")

            print("--- Releasing Base ---");
            if evaluator_base is not None: del evaluator_base; evaluator_base = None
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect(); time.sleep(1)

            update_gui(loading_status_label.configure, text="Loading & Predicting Fine-tuned...")
            print(f"\n--- Creating FINE-TUNED Evaluator (Single Query) ---")
            evaluator_ft = Fine_Tuned_Model_Evaluator(BASE_MODEL_PATH, FINETUNED_PATH, "dummy.csv", 0, 1, CONFIDENCE_THRESHOLD)
            print("--- Fine-tuned Initialized ---")
            print(f"\n--- Calling finetuned_evaluator.predict_single() ---");
            answer_ft, _, _, _ = evaluator_ft.predict_single(question_text)
            print("--- Fine-tuned Predict Finished ---")
            if answer_ft != "error":
                is_correct_ft = answer_ft == expected_answer
                ft_result_log = f"Question: {question_text}\nExpected: {expected_answer}\nPredicted: {answer_ft}\nResult: {'Correct! ✅' if is_correct_ft else 'Incorrect! ❌'}"
                update_gui(processed_label_ft.configure, text="1") # Update summary for single question
                update_gui(acc_overall_label_ft.configure, text="1/1 (100%)" if is_correct_ft else "0/1 (0%)")
                update_gui(acc_yes_label_ft.configure, text=("1/1 (100%)" if is_correct_ft else "0/1 (0%)") if expected_answer=='yes' else "N/A")
                update_gui(acc_no_label_ft.configure, text=("1/1 (100%)" if is_correct_ft else "0/1 (0%)") if expected_answer=='no' else "N/A")
            else: ft_result_log = "Fine-tuned failed to predict."; update_gui(processed_label_ft.configure, text="Error")

        except Exception as e: print(f"ERROR: {e}"); traceback.print_exc(); window.after(0, lambda: messagebox.showerror("Runtime Error", f"An error occurred: {e}\nCheck console.", parent=window)); update_gui(loading_status_label.configure, text="Error...")
        finally:
            update_gui(progress_bar.stop); update_gui(progress_bar.set, 0)
            run_duration = time.time() - start_time_total

            # --- Display Single Results ---
            update_gui(log_textbox_base.configure, state="normal"); update_gui(log_textbox_base.delete, "1.0", "end"); update_gui(log_textbox_base.insert, "1.0", base_result_log); update_gui(log_textbox_base.configure, state="disabled")
            update_gui(log_textbox_ft.configure, state="normal"); update_gui(log_textbox_ft.delete, "1.0", "end"); update_gui(log_textbox_ft.insert, "1.0", ft_result_log); update_gui(log_textbox_ft.configure, state="disabled")

            final_status = f"Single query finished in {run_duration:.1f}s."
            update_gui(loading_status_label.configure, text=final_status);
            update_gui(run_button.configure, state="normal")
            update_gui(ask_question_button.configure, state="normal")

            print("--- Releasing Final Resources ---");
            if evaluator_base is not None: del evaluator_base
            if evaluator_ft is not None: del evaluator_ft
            if torch.cuda.is_available(): print("Clearing CUDA cache..."); torch.cuda.empty_cache()
            print("Running garbage collection..."); gc.collect(); print("Cleanup complete.")

    def start_bulk_comparison_action(): 
        bulk_thread = threading.Thread(target=run_bulk_comparison_thread, daemon=True)
        bulk_thread.start()
    run_button.configure(command=start_bulk_comparison_action) 

    def start_single_question_thread():
        single_thread = threading.Thread(target=run_single_question_thread, daemon=True)
        single_thread.start()
    ask_question_button.configure(command=start_single_question_thread) 
