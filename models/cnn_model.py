import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """CNN model for MNIST."""

    def __init__(self, input_shape, num_classes=10, device="cpu"):
        super().__init__()
        # Extract input dimensions
        num_channels = input_shape[0] if len(input_shape) == 3 else 1
        height = input_shape[1] if len(input_shape) >= 2 else 28
        width = input_shape[2] if len(input_shape) >= 3 else 28

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        h_out = height // 4
        w_out = width // 4

        # Fully connected layers
        self.fc1 = nn.Linear(
            64 * h_out * w_out, 128
        )  # MNIST: 28x28 -> after pooling: 14x14 -> 7x7
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.to(torch.device(device))

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
