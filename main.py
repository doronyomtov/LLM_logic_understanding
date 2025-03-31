import pandas as pd
from model import Model
model = Model()
if model:  # Check if the model is loaded successfully
    print("Model loaded successfully.")
# Load the CSV file into a DataFrame
df = pd.read_csv('questions.csv')
