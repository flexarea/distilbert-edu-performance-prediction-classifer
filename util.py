from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("mstz/student_performance", "math")["train"]
dataset = dataset.remove_columns(
    ['is_male', 'ethnicity', 'parental_level_of_education', 'has_standard_lunch'])
print(dataset)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenizer_function(data):
    return tokenizer(data["text"], padding="max-length", truncation=True)
