

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

    from Base_Model_Evaluator import Base_Model_Evaluator
except ImportError:
    messagebox.showerror("Import Error", "Could not import 'Base_Model_Evaluator' from Base_Model_Evaluator.py.")
    exit()
try:
    from Fine_Tuned_Model_Evaluator import Fine_Tuned_Model_Evaluator
except ImportError:
    messagebox.showerror("Import Error", "Could not import 'Fine_Tuned_Model_Evaluator' from Fine_Tuned_Model_Evaluator.py.")
    exit()
from Chatgpt import ChatGPT_Evaluator as ChatGPT_Model_Evaluator
# --- Path Settings ---
BASE_MODEL_PATH = r"C:\Users\doron\OneDrive\שולחן העבודה\LLM_logic_understanding\Llama3.1 8B"
FINETUNED_PATH = r"C:\Users\doron\OneDrive\שולחן העבודה\LLM_logic_understanding\llama3_finetuned"
QUESTIONS_CSV_PATH = 'questions.csv' # Original full dataset
TEMP_CSV_PATH = '_temp_questions_for_run.csv' # Temporary file for current run's questions
START_IDX = 1000
CONFIDENCE_THRESHOLD = 0.2

def update_gui_globally(window_ref, func, *args, **kwargs):
    """Function to update GUI elements globally.
    Args:
        window_ref (tk.Tk): Reference to the Tkinter window.
        func (callable): Function to call on the GUI element.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    """
    if window_ref and window_ref.winfo_exists():
        window_ref.after(0, lambda: func(*args, **kwargs))

def format_log_from_results(results_list, is_single_question=False, single_q_text=None, single_expected=None):
    """Function to format log data from results.
    Args:
        results_list (list): List of results from the evaluator.
        is_single_question (bool): Flag to indicate if it's a single question.
        single_q_text (str): The question text for the single question.
        single_expected (str): The expected answer for the single question.
    Returns:
        str: Formatted log string.
    """
    log_lines = []
    if is_single_question:
        answer, conf, uncertain, probs = results_list 
        if answer == "error": return "Prediction failed for single question."
        is_correct = answer == single_expected
        log_lines.append(f"--- QID: SINGLE ---")
        log_lines.append(f"Question: {single_q_text}")
        log_lines.append(f"Expected: {single_expected}")
        log_lines.append(f"Predicted: {answer}")
        log_lines.append(f"Result: {'Correct! ✅' if is_correct else 'Incorrect! ❌'}")
        log_lines.append("-" * 20)
    elif results_list:
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
    else: return "No results to display."
    return "\n".join(log_lines)


def format_summary_from_stats(stats_dict=None, is_single_question=False, single_predicted=None, single_expected=None):
    """Function to format summary data from stats."""
    summary_data = {"processed": "N/A", "overall": "N/A", "yes": "N/A", "no": "N/A"}
    if is_single_question:
        if single_predicted != "error":
            is_correct = single_predicted == single_expected
            summary_data["processed"] = "1"
            summary_data["overall"] = "1/1 (100.00%)" if is_correct else "0/1 (0.00%)"
            if single_expected == 'yes':
                summary_data["yes"] = ("1/1 (100.00%)" if single_predicted == 'yes' and is_correct else \
                                    ("0/1 (0.00%)" if single_predicted == 'yes' else "0/0 (N/A)"))
                summary_data["no"] = "N/A (exp. yes)"
            elif single_expected == 'no':
                summary_data["no"] = ("1/1 (100.00%)" if single_predicted == 'no' and is_correct else \
                                   ("0/1 (0.00%)" if single_predicted == 'no' else "0/0 (N/A)"))
                summary_data["yes"] = "N/A (exp. no)"
        else:
            summary_data["processed"] = "Error"
    elif stats_dict and isinstance(stats_dict, dict):
        total = stats_dict.get("total", 0); accuracy = stats_dict.get("accuracy", 0)
        pred_yes_correctly = stats_dict.get("predicted_yes_correctly", 0)
        total_pred_yes = stats_dict.get("total_predicted_yes", 0)
        pred_no_correctly = stats_dict.get("predicted_no_correctly", 0)
        total_pred_no = stats_dict.get("total_predicted_no", 0)
        summary_data["processed"] = str(total)
        if total > 0: summary_data["overall"] = f"{accuracy}/{total} ({accuracy/total*100:.2f}%)"
        else: summary_data["overall"] = "0/0 (N/A)"
        if total_pred_yes > 0: summary_data["yes"] = f"{pred_yes_correctly}/{total_pred_yes} ({pred_yes_correctly/total_pred_yes*100:.2f}%)"
        else: summary_data["yes"] = "0/0 (N/A)"
        if total_pred_no > 0: summary_data["no"] = f"{pred_no_correctly}/{total_pred_no} ({pred_no_correctly/total_pred_no*100:.2f}%)"
        else: summary_data["no"] = "0/0 (N/A)"
    print(f"format_summary_from_stats: Returning data: {summary_data}")
    return summary_data

def open_experiment_page(parent):
    """Function to open the experiment page."""
    window = ctk.CTkToplevel(parent)
    window.title("Llama3 Comparison / ChatGPT / Single Query") # Updated title
    window.geometry("1700x850") 
    window.resizable(True, True)
    window.attributes('-topmost', True)
    window.after(100, lambda: window.attributes('-topmost', False))

    title_label = ctk.CTkLabel(window, text="🔬 Llama3 Comparison, ChatGPT & Single Query", font=ctk.CTkFont(size=24, weight="bold"), text_color="#4a68ff")
    title_label.pack(pady=(10, 5))

    top_controls_frame = ctk.CTkFrame(window, fg_color="transparent")
    top_controls_frame.pack(pady=5, fill="x", padx=20)
    top_controls_frame.grid_columnconfigure(0, weight=1)
    top_controls_frame.grid_columnconfigure(1, weight=0)
    top_controls_frame.grid_columnconfigure(2, weight=0)

    loading_status_label = ctk.CTkLabel(top_controls_frame, text="Ready.", font=ctk.CTkFont(size=12), anchor="w")
    loading_status_label.grid(row=0, column=0, sticky="ew", padx=(5, 10), pady=5)
    progress_bar = ctk.CTkProgressBar(top_controls_frame, width=150, height=10, mode='indeterminate')
    progress_bar.grid(row=0, column=1, sticky="e", padx=10, pady=5)
    progress_bar.set(0)
    progress_bar.stop()

    input_frame = ctk.CTkFrame(top_controls_frame, fg_color="transparent")
    input_frame.grid(row=0, column=2, sticky="e", padx=5, pady=5)
    ctk.CTkLabel(input_frame, text="Num Questions (Bulk):", font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=(10, 5), pady=5)
    num_questions_entry = ctk.CTkEntry(input_frame, width=80)
    num_questions_entry.grid(row=0, column=1, padx=(0, 10), pady=5)
    num_questions_entry.insert(0, "10")
    run_button = ctk.CTkButton(input_frame, text="Run Comparison", state="normal", width=150)
    run_button.grid(row=0, column=2, padx=10, pady=5)

    single_question_frame = ctk.CTkFrame(window)
    single_question_frame.pack(pady=10, padx=20, fill="x")
    single_question_frame.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(single_question_frame, text="Single Question:", font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=(10,5), pady=(5,0), sticky="nw")
    question_input_textbox = ctk.CTkTextbox(single_question_frame, height=60, wrap="word", font=("Segoe UI", 13))
    question_input_textbox.grid(row=1, column=0, columnspan=3, padx=10, pady=(0,5), sticky="ew")
    ctk.CTkLabel(single_question_frame, text="Expected Answer:", font=ctk.CTkFont(size=14)).grid(row=2, column=0, padx=(10,5), pady=5, sticky="e")
    expected_answer_var = ctk.StringVar(value="yes")
    expected_answer_optionmenu = ctk.CTkOptionMenu(single_question_frame, variable=expected_answer_var, values=["yes", "no"])
    expected_answer_optionmenu.grid(row=2, column=1, padx=5, pady=5, sticky="w")
    ask_question_button = ctk.CTkButton(single_question_frame, text="Ask Single Question")
    ask_question_button.grid(row=2, column=2, padx=(10,10), pady=5, sticky="e")

    comparison_frame = ctk.CTkFrame(window, fg_color="transparent")
    comparison_frame.pack(pady=(5, 10), padx=20, fill="both", expand=True)
    comparison_frame.grid_columnconfigure(0, weight=1, uniform="group1")
    comparison_frame.grid_columnconfigure(1, weight=1, uniform="group1")
    comparison_frame.grid_columnconfigure(2, weight=1, uniform="group1") 
    comparison_frame.grid_rowconfigure(0, weight=1)


    left_column_frame = ctk.CTkFrame(comparison_frame)
    left_column_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
    left_column_frame.grid_rowconfigure(1, weight=3)
    left_column_frame.grid_rowconfigure(2, weight=1)
    left_column_frame.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(left_column_frame, text="--- Base Model ---", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=(5,2))
    log_textbox_base = ctk.CTkTextbox(left_column_frame, wrap="word", font=("Consolas", 10), state="disabled")
    log_textbox_base.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
    summary_frame_base = ctk.CTkFrame(left_column_frame)
    summary_frame_base.grid(row=2, column=0, sticky="nsew", padx=5, pady=(2,5))
    summary_frame_base.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(summary_frame_base, text="Processed:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, padx=5, pady=1, sticky="e")
    processed_label_base = ctk.CTkLabel(summary_frame_base, text="N/A", font=ctk.CTkFont(size=11))
    processed_label_base.grid(row=0, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_base, text="Accuracy:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=0, padx=5, pady=1, sticky="e")
    acc_overall_label_base = ctk.CTkLabel(summary_frame_base, text="N/A", font=ctk.CTkFont(size=11))
    acc_overall_label_base.grid(row=1, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_base, text="Precision (Pred. Yes):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=2, column=0, padx=5, pady=1, sticky="e")
    acc_yes_label_base = ctk.CTkLabel(summary_frame_base, text="N/A", font=ctk.CTkFont(size=11))
    acc_yes_label_base.grid(row=2, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_base, text="Precision (Pred. No):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=3, column=0, padx=5, pady=1, sticky="e")
    acc_no_label_base = ctk.CTkLabel(summary_frame_base, text="N/A", font=ctk.CTkFont(size=11))
    acc_no_label_base.grid(row=3, column=1, padx=5, pady=1, sticky="w")

    middle_column_frame = ctk.CTkFrame(comparison_frame)
    middle_column_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5) 
    middle_column_frame.grid_rowconfigure(1, weight=3)
    middle_column_frame.grid_rowconfigure(2, weight=1)
    middle_column_frame.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(middle_column_frame, text="--- Fine-tuned Model ---", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=(5,2))
    log_textbox_ft = ctk.CTkTextbox(middle_column_frame, wrap="word", font=("Consolas", 10), state="disabled")
    log_textbox_ft.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
    summary_frame_ft = ctk.CTkFrame(middle_column_frame)
    summary_frame_ft.grid(row=2, column=0, sticky="nsew", padx=5, pady=(2,5))
    summary_frame_ft.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(summary_frame_ft, text="Processed:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, padx=5, pady=1, sticky="e")
    processed_label_ft = ctk.CTkLabel(summary_frame_ft, text="N/A", font=ctk.CTkFont(size=11))
    processed_label_ft.grid(row=0, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_ft, text="Accuracy:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=0, padx=5, pady=1, sticky="e")
    acc_overall_label_ft = ctk.CTkLabel(summary_frame_ft, text="N/A", font=ctk.CTkFont(size=11))
    acc_overall_label_ft.grid(row=1, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_ft, text="Precision (Pred. Yes):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=2, column=0, padx=5, pady=1, sticky="e")
    acc_yes_label_ft = ctk.CTkLabel(summary_frame_ft, text="N/A", font=ctk.CTkFont(size=11))
    acc_yes_label_ft.grid(row=2, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_ft, text="Precision (Pred. No):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=3, column=0, padx=5, pady=1, sticky="e")
    acc_no_label_ft = ctk.CTkLabel(summary_frame_ft, text="N/A", font=ctk.CTkFont(size=11))
    acc_no_label_ft.grid(row=3, column=1, padx=5, pady=1, sticky="w")

    right_column_frame_chatgpt = ctk.CTkFrame(comparison_frame) 
    right_column_frame_chatgpt.grid(row=0, column=2, sticky="nsew", padx=(5, 0), pady=5) 
    right_column_frame_chatgpt.grid_rowconfigure(1, weight=3)
    right_column_frame_chatgpt.grid_rowconfigure(2, weight=1)
    right_column_frame_chatgpt.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(right_column_frame_chatgpt, text="--- ChatGPT ---", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=(5,2))
    log_textbox_chatgpt = ctk.CTkTextbox(right_column_frame_chatgpt, wrap="word", font=("Consolas", 10), state="disabled")
    log_textbox_chatgpt.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
    summary_frame_chatgpt = ctk.CTkFrame(right_column_frame_chatgpt)
    summary_frame_chatgpt.grid(row=2, column=0, sticky="nsew", padx=5, pady=(2,5))
    summary_frame_chatgpt.grid_columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(summary_frame_chatgpt, text="Processed:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, padx=5, pady=1, sticky="e")
    processed_label_chatgpt = ctk.CTkLabel(summary_frame_chatgpt, text="N/A", font=ctk.CTkFont(size=11))
    processed_label_chatgpt.grid(row=0, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_chatgpt, text="Accuracy:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=1, column=0, padx=5, pady=1, sticky="e")
    acc_overall_label_chatgpt = ctk.CTkLabel(summary_frame_chatgpt, text="N/A", font=ctk.CTkFont(size=11))
    acc_overall_label_chatgpt.grid(row=1, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_chatgpt, text="Precision (Pred. Yes):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=2, column=0, padx=5, pady=1, sticky="e")
    acc_yes_label_chatgpt = ctk.CTkLabel(summary_frame_chatgpt, text="N/A", font=ctk.CTkFont(size=11))
    acc_yes_label_chatgpt.grid(row=2, column=1, padx=5, pady=1, sticky="w")
    ctk.CTkLabel(summary_frame_chatgpt, text="Precision (Pred. No):", font=ctk.CTkFont(size=11, weight="bold")).grid(row=3, column=0, padx=5, pady=1, sticky="e")
    acc_no_label_chatgpt = ctk.CTkLabel(summary_frame_chatgpt, text="N/A", font=ctk.CTkFont(size=11))
    acc_no_label_chatgpt.grid(row=3, column=1, padx=5, pady=1, sticky="w")



    def run_bulk_comparison_thread():
        """Function to run the BULK comparison thread."""
        num_q_str = num_questions_entry.get(); num_rows_to_eval = int(num_q_str) if num_q_str.isdigit() and int(num_q_str) >= 0 else -1
        if num_rows_to_eval < 0 : messagebox.showerror("Input Error","Please enter a non-negative number."); return
        temp_file_created = False; final_sample_df = None
        try:
            update_gui_globally(window, loading_status_label.configure, text="Preparing questions...")
            print(f"\n--- Preparing {num_rows_to_eval} questions ---")
            full_df = pd.read_csv(QUESTIONS_CSV_PATH); eval_df = full_df.iloc[START_IDX:]
            print(f"Loaded {len(eval_df)} questions after start_idx.")
            if num_rows_to_eval > 0: sample_n = min(num_rows_to_eval, len(eval_df)); print(f"Sampling {sample_n} questions."); final_sample_df = eval_df.sample(n=sample_n)
            else: final_sample_df = eval_df.head(0)
            if len(final_sample_df) > 0: final_sample_df = final_sample_df.sample(frac=1).reset_index(drop=True); print(f"Shuffled {len(final_sample_df)} questions.")
            else: print("Sample is empty.")
            final_sample_df.to_csv(TEMP_CSV_PATH, index=False); temp_file_created = True; print(f"Saved {len(final_sample_df)} questions to {TEMP_CSV_PATH}")
        except Exception as prep_e: messagebox.showerror("Error", f"Failed to prepare questions: {prep_e}"); update_gui_globally(window, run_button.configure, state="normal"); update_gui_globally(window, ask_question_button.configure, state="normal"); update_gui_globally(window, loading_status_label.configure, text="Error"); return
        update_gui_globally(window, run_button.configure, state="disabled"); update_gui_globally(window, ask_question_button.configure, state="disabled")
        update_gui_globally(window, loading_status_label.configure, text="Initializing...")
        update_gui_globally(window, progress_bar.configure, mode='indeterminate'); update_gui_globally(window, progress_bar.start)
        for textbox in [log_textbox_base, log_textbox_ft]: update_gui_globally(window, textbox.configure, state="normal"); update_gui_globally(window, textbox.delete, "1.0", "end"); update_gui_globally(window, textbox.configure, state="disabled")
        for label in [processed_label_base, acc_overall_label_base, acc_yes_label_base, acc_no_label_base, processed_label_ft, acc_overall_label_ft, acc_yes_label_ft, acc_no_label_ft]: update_gui_globally(window, label.configure, text="Running...")
        start_time_total = time.time(); base_stats = None; ft_stats = None;
        evaluator_base = None; evaluator_ft = None
        last_exception_message = ""
        try:
            update_gui_globally(window, loading_status_label.configure, text="Loading & Evaluating Base...")
            print(f"\n--- Creating BASE Evaluator ---")
            evaluator_base = Base_Model_Evaluator(BASE_MODEL_PATH, TEMP_CSV_PATH, 0, None, CONFIDENCE_THRESHOLD)
            print("--- Base Initialized ---"); print(f"\n--- Calling base_evaluator.evaluate() ---");
            base_stats = evaluator_base.evaluate(output_path='comparison_base_results.csv', plot_path='comparison_base_confidence.png')
            print("--- Base Eval Finished ---")
            print("--- Releasing Base ---");
            if evaluator_base is not None: del evaluator_base; evaluator_base = None
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect(); time.sleep(1)
            update_gui_globally(window, loading_status_label.configure, text="Loading & Evaluating Fine-tuned...")
            print(f"\n--- Creating FINE-TUNED Evaluator ---")
            evaluator_ft = Fine_Tuned_Model_Evaluator(BASE_MODEL_PATH, FINETUNED_PATH, TEMP_CSV_PATH, 0, None, CONFIDENCE_THRESHOLD)
            print("--- Fine-tuned Initialized ---")
            print(f"\n--- Calling finetuned_evaluator.evaluate() ---");
            ft_stats = evaluator_ft.evaluate(output_path='comparison_finetuned_results.csv', plot_path='comparison_finetuned_confidence.png')
            print("--- Fine-tuned Eval Finished ---")
            evaluator_gpt = ChatGPT_Model_Evaluator("gpt-4.1",TEMP_CSV_PATH, 0, None, CONFIDENCE_THRESHOLD)
            print("--- ChatGPT Initialized ---")
            print(f"\n--- Calling ChatGPT evaluator.evaluate() ---")
            chatgpt_stats = evaluator_gpt.evaluate(output_path='comparison_chatgpt_results.csv', plot_path='comparison_chatgpt_confidence.png')
            print("--- ChatGPT Eval Finished ---")
        except Exception as e:
            last_exception_message = str(e)
            print(f"ERROR: {e}"); traceback.print_exc();
            update_gui_globally(window, loading_status_label.configure, text="Error during bulk run!")
        finally:
            update_gui_globally(window, progress_bar.stop); update_gui_globally(window, progress_bar.set, 0)
            run_duration = time.time() - start_time_total
            if base_stats and isinstance(base_stats, dict):
                 detailed_log_base = format_log_from_results(base_stats.get("results", []))
                 summary_data_base = format_summary_from_stats(base_stats)
                 update_gui_globally(window, log_textbox_base.configure, state="normal"); update_gui_globally(window, log_textbox_base.delete, "1.0", "end"); update_gui_globally(window, log_textbox_base.insert, "1.0", detailed_log_base); update_gui_globally(window, log_textbox_base.configure, state="disabled")
                 update_gui_globally(window, processed_label_base.configure, text=summary_data_base.get("processed", "N/A"))
                 update_gui_globally(window, acc_overall_label_base.configure, text=summary_data_base.get("overall", "N/A"))
                 update_gui_globally(window, acc_yes_label_base.configure, text=summary_data_base.get("yes", "N/A"))
                 update_gui_globally(window, acc_no_label_base.configure, text=summary_data_base.get("no", "N/A"))
            else: update_gui_globally(window, log_textbox_base.configure, state="normal"); update_gui_globally(window, log_textbox_base.delete, "1.0", "end"); update_gui_globally(window, log_textbox_base.insert, "1.0", "Base run failed/no stats."); update_gui_globally(window, log_textbox_base.configure, state="disabled"); update_gui_globally(window, processed_label_base.configure, text="Failed"); update_gui_globally(window, acc_overall_label_base.configure, text="Failed"); update_gui_globally(window, acc_yes_label_base.configure, text="Failed"); update_gui_globally(window, acc_no_label_base.configure, text="Failed")
            if ft_stats and isinstance(ft_stats, dict):
                 detailed_log_ft = format_log_from_results(ft_stats.get("results", []))
                 summary_data_ft = format_summary_from_stats(ft_stats)
                 update_gui_globally(window, log_textbox_ft.configure, state="normal"); update_gui_globally(window, log_textbox_ft.delete, "1.0", "end"); update_gui_globally(window, log_textbox_ft.insert, "1.0", detailed_log_ft); update_gui_globally(window, log_textbox_ft.configure, state="disabled")
                 update_gui_globally(window, processed_label_ft.configure, text=summary_data_ft.get("processed", "N/A"))
                 update_gui_globally(window, acc_overall_label_ft.configure, text=summary_data_ft.get("overall", "N/A"))
                 update_gui_globally(window, acc_yes_label_ft.configure, text=summary_data_ft.get("yes", "N/A"))
                 update_gui_globally(window, acc_no_label_ft.configure, text=summary_data_ft.get("no", "N/A"))
            else: update_gui_globally(window, log_textbox_ft.configure, state="normal"); update_gui_globally(window, log_textbox_ft.delete, "1.0", "end"); update_gui_globally(window, log_textbox_ft.insert, "1.0", "Fine-tuned run failed/no stats."); update_gui_globally(window, log_textbox_ft.configure, state="disabled"); update_gui_globally(window, processed_label_ft.configure, text="Failed"); update_gui_globally(window, acc_overall_label_ft.configure, text="Failed"); update_gui_globally(window, acc_yes_label_ft.configure, text="Failed"); update_gui_globally(window, acc_no_label_ft.configure, text="Failed")
            if chatgpt_stats and isinstance(chatgpt_stats, dict):
                detailed_log_base = format_log_from_results(chatgpt_stats.get("results", []))
                summary_data_base = format_summary_from_stats(chatgpt_stats)
                update_gui_globally(window, log_textbox_chatgpt.configure, state="normal"); update_gui_globally(window, log_textbox_chatgpt.delete, "1.0", "end"); update_gui_globally(window, log_textbox_chatgpt.insert, "1.0", detailed_log_base); update_gui_globally(window, log_textbox_chatgpt.configure, state="disabled")
                update_gui_globally(window, processed_label_chatgpt.configure, text=summary_data_base.get("processed", "N/A"))
                update_gui_globally(window, acc_overall_label_chatgpt.configure, text=summary_data_base.get("overall", "N/A"))
                update_gui_globally(window, acc_yes_label_chatgpt.configure, text=summary_data_base.get("yes", "N/A"))
                update_gui_globally(window, acc_no_label_chatgpt.configure, text=summary_data_base.get("no", "N/A"))
            else: update_gui_globally(window, log_textbox_chatgpt.configure, state="normal"); update_gui_globally(window, log_textbox_chatgpt.delete, "1.0", "end"); update_gui_globally(window, log_textbox_chatgpt.insert, "1.0", "ChatGPT run failed/no stats."); update_gui_globally(window, log_textbox_chatgpt.configure, state="disabled"); update_gui_globally(window, processed_label_chatgpt.configure, text="Failed"); update_gui_globally(window, acc_overall_label_chatgpt.configure, text="Failed"); update_gui_globally(window, acc_yes_label_chatgpt.configure, text="Failed"); update_gui_globally(window, acc_no_label_chatgpt.configure, text="Failed")
            if last_exception_message:
                 window.after(0, lambda: messagebox.showerror("Runtime Error", f"An error occurred: {last_exception_message}\nCheck console.", parent=window))
            else:
                 final_status = f"Comparison finished in {run_duration:.1f}s."
                 update_gui_globally(window, loading_status_label.configure, text=final_status);
            update_gui_globally(window, run_button.configure, state="normal"); update_gui_globally(window, ask_question_button.configure, state="normal")
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
        """Function to run the SINGLE question thread."""
        question_text = question_input_textbox.get("1.0", "end-1c").strip()
        expected_answer = expected_answer_var.get()
        if not question_text: messagebox.showwarning("Input Missing", "Please enter a question text."); return
        update_gui_globally(window, run_button.configure, state="disabled"); update_gui_globally(window, ask_question_button.configure, state="disabled")
        update_gui_globally(window, loading_status_label.configure, text="Initializing for single query...")
        update_gui_globally(window, progress_bar.configure, mode='indeterminate'); update_gui_globally(window, progress_bar.start)
        for textbox in [log_textbox_base, log_textbox_ft,log_textbox_chatgpt]: update_gui_globally(window, textbox.configure, state="normal"); update_gui_globally(window, textbox.delete, "1.0", "end"); update_gui_globally(window, textbox.configure, state="disabled")
        for label in [processed_label_base, acc_overall_label_base, acc_yes_label_base, acc_no_label_base, processed_label_ft, acc_overall_label_ft, acc_yes_label_ft, acc_no_label_ft]: update_gui_globally(window, label.configure, text="N/A")
        start_time_total = time.time(); base_result_tuple = None; ft_result_tuple = None;
        evaluator_base = None; evaluator_ft = None
        last_exception_message = ""
        try:
            update_gui_globally(window, loading_status_label.configure, text="Loading & Predicting Base...")
            print(f"\n--- Creating BASE Evaluator (Single Query) ---")
            evaluator_base = Base_Model_Evaluator(BASE_MODEL_PATH, "dummy.csv", 0, 1, CONFIDENCE_THRESHOLD)
            print("--- Base Initialized ---"); print(f"\n--- Calling base_evaluator.predict_single() ---");
            base_result_tuple = evaluator_base.predict_single(question_text)
            print("--- Base Predict Finished ---")
            if base_result_tuple and base_result_tuple[0] != "error": is_correct_b = base_result_tuple[0] == expected_answer; base_result_log = format_log_from_results(base_result_tuple, is_single_question=True, single_q_text=question_text, single_expected=expected_answer)
            else: base_result_log = "Base failed to predict."
            update_gui_globally(window, log_textbox_base.configure, state="normal"); update_gui_globally(window, log_textbox_base.delete, "1.0", "end"); update_gui_globally(window, log_textbox_base.insert, "1.0", base_result_log); update_gui_globally(window, log_textbox_base.configure, state="disabled")
            if base_result_tuple and base_result_tuple[0] != "error":
                is_correct_b = base_result_tuple[0] == expected_answer
                summary_data_s_b = format_summary_from_stats(is_single_question=True, single_predicted=base_result_tuple[0], single_expected=expected_answer)
                update_gui_globally(window, processed_label_base.configure, text=summary_data_s_b["processed"]); update_gui_globally(window, acc_overall_label_base.configure, text=summary_data_s_b["overall"])
                update_gui_globally(window, acc_yes_label_base.configure, text=summary_data_s_b["yes"]); update_gui_globally(window, acc_no_label_base.configure, text=summary_data_s_b["no"])
            else: update_gui_globally(window, processed_label_base.configure, text="Error"); update_gui_globally(window, acc_overall_label_base.configure, text="Error"); update_gui_globally(window, acc_yes_label_base.configure, text="Error"); update_gui_globally(window, acc_no_label_base.configure, text="Error")
            print("--- Releasing Base ---");
            if evaluator_base is not None: del evaluator_base; evaluator_base = None
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect(); time.sleep(1)
            update_gui_globally(window, loading_status_label.configure, text="Loading & Predicting Fine-tuned...")
            print(f"\n--- Creating FINE-TUNED Evaluator (Single Query) ---")
            evaluator_ft = Fine_Tuned_Model_Evaluator(BASE_MODEL_PATH, FINETUNED_PATH, "dummy.csv", 0, 1, CONFIDENCE_THRESHOLD)
            print("--- Fine-tuned Initialized ---")
            print(f"\n--- Calling finetuned_evaluator.predict_single() ---");
            ft_result_tuple = evaluator_ft.predict_single(question_text)
            print("--- Fine-tuned Predict Finished ---")
            if ft_result_tuple and ft_result_tuple[0] != "error": is_correct_ft = ft_result_tuple[0] == expected_answer; ft_result_log = format_log_from_results(ft_result_tuple, is_single_question=True, single_q_text=question_text, single_expected=expected_answer)
            else: ft_result_log = "Fine-tuned failed to predict."
            update_gui_globally(window, log_textbox_ft.configure, state="normal"); update_gui_globally(window, log_textbox_ft.delete, "1.0", "end"); update_gui_globally(window, log_textbox_ft.insert, "1.0", ft_result_log); update_gui_globally(window, log_textbox_ft.configure, state="disabled")
            if ft_result_tuple and ft_result_tuple[0] != "error":
                is_correct_ft = ft_result_tuple[0] == expected_answer
                summary_data_s_ft = format_summary_from_stats(is_single_question=True, single_predicted=ft_result_tuple[0], single_expected=expected_answer)
                update_gui_globally(window, processed_label_ft.configure, text=summary_data_s_ft["processed"]); update_gui_globally(window, acc_overall_label_ft.configure, text=summary_data_s_ft["overall"])
                update_gui_globally(window, acc_yes_label_ft.configure, text=summary_data_s_ft["yes"]); update_gui_globally(window, acc_no_label_ft.configure, text=summary_data_s_ft["no"])
            else: update_gui_globally(window, processed_label_ft.configure, text="Error"); update_gui_globally(window, acc_overall_label_ft.configure, text="Error"); update_gui_globally(window, acc_yes_label_ft.configure, text="Error"); update_gui_globally(window, acc_no_label_ft.configure, text="Error")
            evaluator_gpt = ChatGPT_Model_Evaluator("gpt-4.1", "dummy.csv", 0, 1, CONFIDENCE_THRESHOLD)
            print("--- ChatGPT Initialized ---")
            print(f"\n--- Calling ChatGPT evaluator.predict_single() ---")
            chatgpt_result_tuple = evaluator_gpt.predict_single(question_text)
            print("--- ChatGPT Predict Finished ---")
            if chatgpt_result_tuple and chatgpt_result_tuple[0] != "error": is_correct_gpt = chatgpt_result_tuple[0] == expected_answer; chatgpt_result_log = format_log_from_results(chatgpt_result_tuple, is_single_question=True, single_q_text=question_text, single_expected=expected_answer)
            else: chatgpt_result_log = "ChatGPT failed to predict."
            update_gui_globally(window, log_textbox_chatgpt.configure, state="normal"); update_gui_globally(window, log_textbox_chatgpt.delete, "1.0", "end"); update_gui_globally(window, log_textbox_chatgpt.insert, "1.0", chatgpt_result_log); update_gui_globally(window, log_textbox_chatgpt.configure, state="disabled")
            if chatgpt_result_tuple and chatgpt_result_tuple[0] != "error":
                is_correct_gpt = chatgpt_result_tuple[0] == expected_answer
                summary_data_s_gpt = format_summary_from_stats(is_single_question=True, single_predicted=chatgpt_result_tuple[0], single_expected=expected_answer)
                update_gui_globally(window, processed_label_chatgpt.configure, text=summary_data_s_gpt["processed"]); update_gui_globally(window, acc_overall_label_chatgpt.configure, text=summary_data_s_gpt["overall"])
                update_gui_globally(window, acc_yes_label_chatgpt.configure, text=summary_data_s_gpt["yes"]); update_gui_globally(window, acc_no_label_chatgpt.configure, text=summary_data_s_gpt["no"])
            else: update_gui_globally(window, processed_label_chatgpt.configure, text="Error"); update_gui_globally(window, acc_overall_label_chatgpt.configure, text="Error"); update_gui_globally(window, acc_yes_label_chatgpt.configure, text="Error"); update_gui_globally(window, acc_no_label_chatgpt.configure, text="Error")
        except Exception as e:
            last_exception_message = str(e)
            print(f"ERROR: {e}"); traceback.print_exc();
            update_gui_globally(window, loading_status_label.configure, text="Error during single query!")
        finally:
            update_gui_globally(window, progress_bar.stop); update_gui_globally(window, progress_bar.set, 0)
            run_duration = time.time() - start_time_total
            if last_exception_message:
                 window.after(0, lambda: messagebox.showerror("Runtime Error", f"An error occurred: {last_exception_message}\nCheck console.", parent=window))
            else:
                 final_status = f"Single query finished in {run_duration:.1f}s."
                 update_gui_globally(window, loading_status_label.configure, text=final_status);
            update_gui_globally(window, run_button.configure, state="normal"); update_gui_globally(window, ask_question_button.configure, state="normal")
            print("--- Releasing Final Resources ---");
            if evaluator_base is not None: del evaluator_base
            if evaluator_ft is not None: del evaluator_ft
            if torch.cuda.is_available(): print("Clearing CUDA cache..."); torch.cuda.empty_cache()
            print("Running garbage collection..."); gc.collect(); print("Cleanup complete.")

    def start_bulk_comparison_action():
        """ Start the bulk comparison thread """
        bulk_thread = threading.Thread(target=run_bulk_comparison_thread, daemon=True)
        bulk_thread.start()
    run_button.configure(command=start_bulk_comparison_action)

    def start_single_question_thread():
        """ Start the single question thread """
        single_thread = threading.Thread(target=run_single_question_thread, daemon=True)
        single_thread.start()
    ask_question_button.configure(command=start_single_question_thread)