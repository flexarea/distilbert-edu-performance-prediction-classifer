import argparse
import torch
import torch.nn as nn
from torch import cuda, manual_seed
from torch.utils.data import DataLoader

from model import PerformancepredictionModel
from util import get_dataloader, model_accuracy

# embedding download


def train_model(model: PerformancepredictionModel, train_dataloader: DataLoader,
                dev_dataloader: DataLoader, epochs: int, learning_rate: float):
    """
    Trains model and prints accuracy on dev data after training

    Arguments:
        model (PerformancepredictionModel): the model to train
        train_dataloader (DataLoader): a pytorch dataloader containing the training data
        dev_dataloader (DataLoader): a pytorch dataloader containing the development data
        epochs (int): the number of epochs to train for (full iterations through the dataset)
        learing_rate (float): the learning rate

    Returns:
        float: the accuracy on the development set
    """

    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # early stopping variable configuration

    # Set model to train mode

    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            # Forward pass
            logits = model(input_ids, attention_mask).squeeze()
            # Compute loss
            loss = criterion(logits, labels.float())
            # Compute gradients
            loss.backward()
            # Update weights
            optimizer.step()

            # Compute training accuracy

        accuracy = model_accuracy(model, dev_dataloader, device)
        print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train a performance prediction model.")
    parser.add_argument("--epochs", default=7, type=int,
                        help="help= Number of training epochs")
    parser.add_argument("--learning_rate", default=1e-2, type=float,
                        help="The learning rate")
    parser.add_argument("--freeze_bert", action="store_true",
                        help="True to freeze BERT parameters (no fine-tuning), False otherwise")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training and evaluation")
    args = parser.parse_args()

    print(args.batch_size)

    # initialize model and dataloaders
    device = "cuda" if cuda.is_available() else "cpu"

    # seed the model before initializing weights so that your code is deterministic
    manual_seed(457)

    model = PerformancepredictionModel(freeze_bert=args.freeze_bert).to(device)
    train_dataloader = get_dataloader("train", batch_size=args.batch_size)
    dev_dataloader = get_dataloader("dev", batch_size=args.batch_size)

    train_model(model, train_dataloader, dev_dataloader,
                args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
