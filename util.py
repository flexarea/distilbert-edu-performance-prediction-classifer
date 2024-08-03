from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer


class CustomDataset(TorchDataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.features.items()}
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

    # Convert to dictionary of lists for custom dataset
    features_dict = {f: x[:, i].tolist() for i, f in enumerate(features)}

    # Create custom dataset
    custom_dataset = CustomDataset(features_dict, y)

    # Create DataLoader
    dataloader = DataLoader(
        custom_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# Example usage
dataloader = get_dataloader(batch_size=4)
for batch in dataloader:
    print(batch)
    break
