from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("mstz/student_performance", "math")["train"]
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenizer_function(data):
    return tokenizer(data["text"], padding="max-length", truncation=True)
