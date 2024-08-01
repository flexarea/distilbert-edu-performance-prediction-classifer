
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

dataset = load_dataset("mstz/student_performance", "math")["train"]
dataset = dataset.remove_columns(
    ['is_male', 'ethnicity', 'parental_level_of_education', 'has_standard_lunch'])


# initialize tokeniser
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# feature engineering


# Extract features and labels
features = ['has_completed_preparation_test', 'reading_score', 'writing_score']
target = 'has_passed_math_exam'

# convert dataset to pandas Dataframe
df = pd.DataFrame(dataset)

# extract feature matrix and label
x = df[features]
y = df[target]

print(x)

# Tokeniser and create attention mask
encoding = tokenizer(dataset, padding=True,
                     truncation=True, return_tensors='pt')
# print(dataset)
