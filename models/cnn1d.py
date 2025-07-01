"""
1D CNN architecture for raw EEG seizure detection in PyTorch.
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    A 1D convolutional neural network for classifying EEG windows.
    Convolutions capture temporal patterns across channels, followed by pooling and fully connected layers.
    """

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=128 * (None), out_features=256
            ),  # placeholder for dynamic size
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input x has shape (batch, channels, time_points).
        Returns logits of shape (batch, num_classes).
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits

    def _initialize_classifier(self, time_dim: int) -> None:
        """
        Initialize classifier input dimension based on time dimension after conv layers.
        """
        dummy = torch.zeros(1, self.features[0].in_channels, time_dim)
        feat = self.features(dummy)
        flatten_dim = feat.numel() // feat.shape[0]
        self.classifier[1] = nn.Linear(in_features=flatten_dim, out_features=256)  # type: ignore # replace placeholder
