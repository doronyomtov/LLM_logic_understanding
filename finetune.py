from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import pandas as pd
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# Required imports (reusing from before)

# Load and prepare data
import pandas as pd
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# === 1. Load and format dataset ===
df = pd.read_csv("questions.csv").head(1000)

def format_example(row):
    return {
        "instruction": f"Rule category: {row['rule_category']}\nRule: {row['rule']}\nProblem: {row['problem']}\nQuery: {row['query']}",
        "output": row["answer"]
    }

dataset = [format_example(row) for _, row in df.iterrows()]
with open("train_data.json", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

hf_dataset = load_dataset("json", data_files="train_data.json", split="train")

# === 2. Load tokenizer ===
model_path = r"C:\Users\doron\OneDrive\שולחן העבודה\LLM_logic_understanding\Llama3.1 8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === 3. Tokenize ===
def tokenize_function(examples):
    prompts = [f"{inst}\nAnswer:" for inst in examples["instruction"]]
    full_texts = [f"{prompt} {output}" for prompt, output in zip(prompts, examples["output"])]
    return tokenizer(full_texts, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# === 4. Load model with 4-bit quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)

# === 5. LoRA config ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # may vary based on model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === 6. Training setup ===
training_args = TrainingArguments(
    output_dir="./llama3-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# === 7. Train ===
trainer.train()

# === 8. Save ===
model.save_pretrained("llama3_finetuned")
tokenizer.save_pretrained("llama3_finetuned")

