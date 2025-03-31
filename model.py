from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self):
        # Path to the model directory ### פה לשנות למיקום אצלך כדי שיעבוד
        self.model_path = r"C:\Users\doron\OneDrive\שולחן העבודה\LLM_logic_understanding\Llama3.1 8B"
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
