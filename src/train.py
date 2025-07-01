"""
Train a CNN model for EEG seizure detection using preprocessed window data.
Supports both 1D (raw) and 2D (spectrogram) CNN architectures and handles class imbalance.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# Import your model definitions
from models.cnn1d import CNN1D
from models.cnn2d_spectrogram import CNN2DSpectrogram

torch.manual_seed(17)


def load_data(data_dir: Path) -> Tuple[list, list]:
    """
    Loads all .npz files, returning list of (data, label) pairs.
    """
    files = list(data_dir.glob("*.npz"))
    data_list, labels = [], []
    for f in files:
        arr = np.load(f)
        data_list.append(arr["data"])
        labels.append(int(arr["label"]))
    return data_list, labels


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG windows.
    """

    def __init__(self, data_list, labels, to_spectrogram: bool):
        self.data = data_list
        self.labels = labels
        self.to_spectrogram = to_spectrogram
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        # ensure correct shape: (C, L) for 1D or (C, F, T) for 2D
        x = self.transform(x)
        return x.float(), torch.tensor(y, dtype=torch.long)


def create_sampler(labels):
    """
    Creates a WeightedRandomSampler to address class imbalance.
    """
    class_sample_count = np.array([sum(np.array(labels) == t) for t in [0, 1]])
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    """
    Single epoch training loop.
    Returns average loss.
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)


def evaluate(
    loader: DataLoader, model: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model accuracy.
    Returns (accuracy, loss).
    """
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total, loss_sum / total


def main():
    parser = argparse.ArgumentParser(description="Train EEG CNN model")
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Directory of .npz files"
    )
    parser.add_argument(
        "--model",
        choices=["cnn1d", "cnn2d"],
        default="cnn1d",
        help="Model architecture",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--use_sampler",
        action="store_true",
        help="Use WeightedRandomSampler for imbalance",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    # Load data
    data_list, labels = load_data(args.data_dir)
    dataset = EEGDataset(data_list, labels, to_spectrogram=(args.model == "cnn2d"))
    if args.use_sampler:
        sampler = create_sampler(labels)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    # Instantiate model
    if args.model == "cnn1d":
        model = CNN1D(input_channels=data_list[0].shape[0], num_classes=2)
    else:
        model = CNN2DSpectrogram(input_channels=data_list[0].shape[0], num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(loader, model, criterion, optimizer, device)
        val_acc, val_loss = evaluate(loader, model, device)
        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"models/best_{args.model}.pth")
    print("Training complete.")


if __name__ == "__main__":
    main()
