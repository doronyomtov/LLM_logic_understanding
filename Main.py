from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import torch
import pandas as pd

model_path = r"C:\Users\doron\OneDrive\שולחן העבודה\LLM_logic_understanding\Llama3.1 8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
sentiment_pipeline = pipeline("sentiment-analysis")

def sentiment_analysis(answer):
    result = sentiment_pipeline(answer)
    return 'no' if result[0]['label'] == 'NEGATIVE' else 'yes'
     



# Load the CSV file into a DataFrame
df = pd.read_csv('questions.csv')
accuracy = 0
df = df.sample(n=100, random_state=1)  # Randomly sample 100 rows from the DataFrame
for index, row in df.iterrows():
        pre_prompt = "Please Provide a one word answer : yes or no to the following question: "
        question = row['query']
        print(row['id'])
        print(question)
        answer = row['answer']
        print(answer)
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
# Generate output from model
        output_ids = model.generate(input_ids, max_new_tokens=100)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = response.replace(question, "").strip()
        print(f"Response: {response}")
        response = sentiment_analysis(response)
        if response.lower() == answer:
            accuracy += 1
            print(f"Question: {question}")
            print(f"Expected Answer: {answer}")
            print(f"Generated Answer: {response}")

print(f"Accuracy: {accuracy / len(df)}")








#####https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html####