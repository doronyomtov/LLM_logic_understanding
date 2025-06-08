# LLM Logic Understanding

A research-oriented interface and evaluation pipeline for analyzing the logical reasoning capabilities of large language models (LLMs), particularly LLaMA 3.1 8B, with support for fine-tuning and comparing model performance.

## 🚀 Overview

This project provides a GUI-based tool for:
- Running logical reasoning experiments on LLMs
- Evaluating base vs. fine-tuned models
- Visualizing model responses and confidence levels

## 🧩 Core Components

- **`Gui.py`** – Main application GUI for interacting with the experiment interface.
- **`Experiment_page.py`** – Handles experiment execution, result display, and user input for tests.
- **`Base_Model_Evaluator.py`** – Loads and evaluates the performance of the original (base) LLaMA 3.1 8B model.
- **`Fine_Tuned_Model_Evaluator.py`** – Evaluates a fine-tuned version of the same model.
- **`fine_tune.py`** – Script to fine-tune LLaMA 3.1 8B on a dataset designed for logical reasoning tasks.
- **`questions.csv`** – CSV file containing the logic-based questions used for evaluation.

## 📦 Requirements

- Python 3.9+
- PyTorch
- Transformers (Hugging Face)
- `llama.cpp` or similar backend for LLaMA models
- OpenAI API key (for ChatGPT comparison)
- GPU (≥24GB VRAM recommended)

## 🛠️ Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/doronyomtov/LLM_logic_understanding.git
   cd LLM_logic_understanding

2. **Download and set up the LLaMA 3.1 8B model**:
   - Place the base model and fine-tuned model in appropriate directories.
   - Update the paths in `Base_Model_Evaluator.py` and `Fine_Tuned_Model_Evaluator.py` to point to your local model files.

3. **Fine-tune the model** :
   ```bash
   python fine_tune.py --model_dir path/to/llama3.1 --output_dir path/to/fine_tuned_model
   ```

4. **Provide your OpenAI API key**:
   - Set the key as an environment variable or in a `.env` file
5. **Ensure `questions.csv` is present** in the root directory with logic questions for evaluation.

6. **Run the GUI**:
   ```bash
   python Gui.py
   ```

## 📊 Features

- Evaluate prompt-based logical reasoning
- Compare base vs. fine-tuned model performance
- Include ChatGPT (via OpenAI API) as a reference model
- GUI for interactive experimentation and evaluation

## 📁 Project Structure

```
LLM_logic_understanding/
├── Gui.py
├── Experiment_page.py
├── Base_Model_Evaluator.py
├── Fine_Tuned_Model_Evaluator.py
├── fine_tune.py
├── questions.csv              # Logic reasoning questions
├── models/                    # Local LLaMA model folders
├── .env                       # (Optional) OpenAI API key
└── README.md
```

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

## 👤 Authors

- Doron Yom Tov
- Nadav Falkowski
