from datasets import load_dataset

dataset = load_dataset("mstz/student_performance", "math")["train"]

print(dataset["reading_score"][:50])
