import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    """Simple feedforward neural network for MNIST."""

    def __init__(self, input_shape, hidden_size=64, num_classes=10, device="cpu"):
        super().__init__()

        if len(input_shape) == 3:  # (C, H, W)
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 1:  # Flattened vector
            input_size = input_shape[0]
        else:
            input_size = 784  # Default for MNIST

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.to(torch.device(device))

    def forward(self, x):
        # Ensure input is on correct device
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
