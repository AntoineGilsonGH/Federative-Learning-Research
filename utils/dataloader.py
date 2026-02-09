import torch
from torchvision import datasets, transforms


class FederatedDataLoader:
    """
    Base class for federated data loaders.
    """

    def __init__(
        self, num_clients: int, batch_size: int = 32, test_batch_size: int = 1000
    ):
        """
        Initialize federated data loader.

        Args:
            num_clients: Number of clients to split data among
            batch_size: Batch size for local training
            test_batch_size: Batch size for testing
        """
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.client_datasets = None
        self.test_dataset = None
        self.test_loader = None
        self._setup_data()

    def _setup_data(self):
        """Setup datasets for federated learning. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _setup_data")

    def get_client_loader(self, client_idx: int) -> torch.utils.data.DataLoader:
        """
        Get data loader for a specific client.

        Args:
            client_idx: Index of the client

        Returns:
            DataLoader for the client
        """
        return torch.utils.data.DataLoader(
            self.client_datasets[client_idx], batch_size=self.batch_size, shuffle=True
        )

    def get_test_loader(self) -> torch.utils.data.DataLoader:
        """
        Get test data loader.

        Returns:
            DataLoader for test dataset
        """
        if self.test_loader is None:
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.test_batch_size, shuffle=False
            )
        return self.test_loader


class MNISTDataLoader(FederatedDataLoader):
    """Federated data loader for MNIST dataset."""

    def _setup_data(self):
        """Setup MNIST datasets for federated learning."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Training data
        train_dataset = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )

        # Test data
        self.test_dataset = datasets.MNIST(
            "./data", train=False, download=True, transform=transform
        )

        # Split among clients
        client_data_len = len(train_dataset) // self.num_clients
        self.client_datasets = torch.utils.data.random_split(
            train_dataset, [client_data_len] * self.num_clients
        )


class CIFAR10DataLoader(FederatedDataLoader):
    """Federated data loader for CIFAR-10 dataset."""

    def _setup_data(self):
        """Setup CIFAR-10 datasets for federated learning."""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Training data
        train_dataset = datasets.CIFAR10(
            "./data", train=True, download=True, transform=transform
        )

        # Test data
        self.test_dataset = datasets.CIFAR10(
            "./data", train=False, download=True, transform=transform
        )

        # Split among clients
        client_data_len = len(train_dataset) // self.num_clients
        self.client_datasets = torch.utils.data.random_split(
            train_dataset, [client_data_len] * self.num_clients
        )


class CIFAR100DataLoader(FederatedDataLoader):
    """Federated data loader for CIFAR-100 dataset."""

    def _setup_data(self):
        """Setup CIFAR-100 datasets for federated learning."""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Training data
        train_dataset = datasets.CIFAR100(
            "./data", train=True, download=True, transform=transform
        )

        # Test data
        self.test_dataset = datasets.CIFAR100(
            "./data", train=False, download=True, transform=transform
        )

        # Split among clients
        client_data_len = len(train_dataset) // self.num_clients
        self.client_datasets = torch.utils.data.random_split(
            train_dataset, [client_data_len] * self.num_clients
        )


class FashionMNISTDataLoader(FederatedDataLoader):
    """Federated data loader for Fashion-MNIST dataset."""

    def _setup_data(self):
        """Setup Fashion-MNIST datasets for federated learning."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        )

        # Training data
        train_dataset = datasets.FashionMNIST(
            "./data", train=True, download=True, transform=transform
        )

        # Test data
        self.test_dataset = datasets.FashionMNIST(
            "./data", train=False, download=True, transform=transform
        )

        # Split among clients
        client_data_len = len(train_dataset) // self.num_clients
        self.client_datasets = torch.utils.data.random_split(
            train_dataset, [client_data_len] * self.num_clients
        )


def create_data_loader(
    dataset_name: str, num_clients: int, **kwargs
) -> FederatedDataLoader:
    """
    Create a federated data loader for the specified dataset.

    Args:
        dataset_name: Name of dataset ('MNIST', 'CIFAR10', 'CIFAR100', 'FashionMNIST')
        num_clients: Number of clients
        **kwargs: Additional arguments for the data loader

    Returns:
        FederatedDataLoader instance
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return MNISTDataLoader(num_clients, **kwargs)
    elif dataset_name == "cifar10":
        return CIFAR10DataLoader(num_clients, **kwargs)
    elif dataset_name == "cifar100":
        return CIFAR100DataLoader(num_clients, **kwargs)
    elif dataset_name == "fashionmnist":
        return FashionMNISTDataLoader(num_clients, **kwargs)
    else:
        raise ValueError(
            f"Dataset {dataset_name} not supported. "
            f"Available options: MNIST, CIFAR10, CIFAR100, FashionMNIST"
        )
