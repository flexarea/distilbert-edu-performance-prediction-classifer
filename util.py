from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer

from model import PerformancepredictionModel


def model_accuracy(model: PerformancepredictionModel, dataloader: DataLoader, device):
    """Compute the accuracy of a binary classification model

    Args:
        model (PerformancePredictionModel): a hate speech classification model
        dataloader (DataLoader): a pytorch data loader to test the model with
        device (string): cpu or cuda, the device that the model is on

    Returns:
        float: the accuracy of the model
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            pred = model(batch["input_ids"].to(device),
                         batch["attention_mask"].to(device))
            correct += (batch["labels"] ==
                        (pred.to("cpu").squeeze() > 0.5).to(int)).sum().item()
            total += batch["labels"].shape[0]
        acc = correct / total
        return acc


class CustomDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_dataloader(batch_size: int = 4):
    """
    Get a PyTorch DataLoader for the student performance dataset.

    Args:
        batch_size (int, optional): The desired batch size. Defaults to 4.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    dataset = load_dataset("mstz/student_performance", "math")["train"]
    dataset = dataset.remove_columns(
        ['is_male', 'ethnicity', 'parental_level_of_education', 'has_standard_lunch', 'has_completed_preparation_test'])

    # Convert dataset to pandas DataFrame
    df = pd.DataFrame(dataset)

    # Feature engineering
    features = ['reading_score', 'writing_score']
    target = 'has_passed_math_exam'

    # Extract feature matrix and labels
    x = df[features].values
    y = df[target].values

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Tokenize features
    encodings = tokenizer([str(f) for f in x],
                          truncation=True, padding=True, max_length=512)

    # Create custom dataset
    custom_dataset = CustomDataset(encodings, y)

    # Create DataLoader
    dataloader = DataLoader(
        custom_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# Example usage
dataloader = get_dataloader(batch_size=4)
for batch in dataloader:
    print(batch)
    break
