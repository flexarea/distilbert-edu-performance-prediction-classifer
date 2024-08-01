from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("mstz/student_performance", "math")["train"]
dataset = dataset.remove_columns(
    ['is_male', 'ethnicity', 'parental_level_of_education', 'has_standard_lunch'])

# initialize tokeniser
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# extract data from a compile
text_data = [str(i) for i in dataset["reading_score"][:15]]

# Tokeniser and create attention mask
encoding = tokenizer(text_data, padding=True,
                     truncation=True, return_tensors='pt')
print(encoding)
