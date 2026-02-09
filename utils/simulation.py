import torch
import torch.nn as nn
import torch.nn.functional as F
import byzfl
import copy
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any

import config
from utils.dataloader import create_data_loader


class Simulation:
    """
    Federated Learning Simulation with Byzantine robustness.
    """

    def __init__(
        self,
        num_clients: int = 10,
        num_byzantine: int = 1,
        rounds: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        device: str = "cpu",
        dataset_name: str = "MNIST",
        model_class: nn.Module = None,
        model_args: Dict[str, Any] = None,
        data_loader_kwargs: Dict[str, Any] = None,
    ):
        """
        Initialize the simulation.

        Args:
            num_clients: Number of total clients
            num_byzantine: Number of Byzantine (malicious) clients
            rounds: Number of communication rounds
            batch_size: Batch size for local training
            learning_rate: Learning rate for SGD
            dataset_name: Name of dataset to use
            model_class: PyTorch model class
            model_args: Arguments to pass to model constructor
            data_loader_kwargs: Additional arguments for data loader
        """
        self.num_clients = num_clients
        self.num_byzantine = num_byzantine
        self.rounds = rounds
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.model_class = model_class
        self.model_args = model_args or {}
        self.data_loader_kwargs = data_loader_kwargs or {}

        # Data structures to store results
        self.results = {}

        # Setup data
        self._setup_data()

        # Default model if none provided
        if self.model_class is None:
            self.model_class = self._default_model

    def _default_model(self) -> nn.Module:
        """Default SimpleMLP model for MNIST."""

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 128)
                self.fc2 = nn.Linear(128, 10)
                self.to(self.device)

            def forward(self, x):
                x = torch.flatten(x, 1)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        return SimpleMLP()

    def _setup_data(self):
        """Prepare datasets for federated learning using the new data loader system."""
        # Create data loader
        self.data_loader = create_data_loader(
            dataset_name=self.dataset_name,
            num_clients=self.num_clients,
            batch_size=self.batch_size,
            **self.data_loader_kwargs,
        )

        # Get client datasets and test dataset
        self.client_datasets = self.data_loader.client_datasets
        self.test_dataset = self.data_loader.test_dataset

        # Test loader
        self.test_loader = self.data_loader.get_test_loader()

    def evaluate(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return correct / len(loader.dataset)

    def run(
        self,
        aggregator_name: str,
        attack_fn: Callable = None,
        attack_args: Dict[str, Any] = None,
    ) -> List[float]:
        """
        Run simulation with specified aggregator.

        Args:
            aggregator_name: Name of aggregator from byzfl
            attack_fn: Function to apply Byzantine attack (optional)
            attack_args: Arguments for attack function

        Returns:
            List of accuracy values per round
        """
        print(f"\n--- Starting Simulation with Aggregator: {aggregator_name} ---")

        # Initialize global model
        if isinstance(self.model_class, type):
            global_model = self.model_class(**self.model_args).to(self.device)
        else:
            global_model = self.model_class().to(self.device)

        accuracy_history = []

        # Initialize aggregator
        if aggregator_name == "Average":
            aggregator = byzfl.Average()
        elif aggregator_name == "TrimmedMean":
            aggregator = byzfl.TrMean(self.num_byzantine)
        elif aggregator_name == "Median":
            aggregator = byzfl.Median()
        elif aggregator_name == "Krum":
            aggregator = byzfl.Krum(self.num_byzantine)
        else:
            raise ValueError(f"Unknown aggregator: {aggregator_name}")

        # Default attack function (sign flipping)
        if attack_fn is None:

            def default_attack(grads, client_idx):
                if client_idx < self.num_byzantine:
                    return -1.0 * grads * 10.0
                return grads

            attack_fn = default_attack

        attack_args = attack_args or {}

        # Training rounds
        for r in range(self.rounds):
            local_gradients = []

            # Client local training
            for client_idx in range(self.num_clients):
                local_model = copy.deepcopy(global_model)
                local_model.train()

                # Get a batch
                loader = torch.utils.data.DataLoader(
                    self.client_datasets[client_idx],
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                data, target = next(iter(loader))
                data, target = data.to(self.device), target.to(self.device)

                # Compute gradient
                output = local_model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                # Flatten gradients
                grads = torch.cat([p.grad.flatten() for p in local_model.parameters()])

                # Apply attack if Byzantine client
                grads = attack_fn(grads, client_idx, **attack_args)

                local_gradients.append(grads)

            # Server aggregation
            stacked_grads = torch.stack(local_gradients)
            aggregated_grad = aggregator(stacked_grads)

            # Update global model
            start_idx = 0
            with torch.no_grad():
                for param in global_model.parameters():
                    num_params = param.numel()
                    grad_chunk = aggregated_grad[
                        start_idx : start_idx + num_params
                    ].view(param.shape)
                    param -= self.learning_rate * grad_chunk
                    start_idx += num_params

            # Record accuracy
            acc = self.evaluate(global_model, self.test_loader)
            accuracy_history.append(acc)

            if (r + 1) % 5 == 0:
                print(f"Round {r + 1}/{self.rounds}: Test Accuracy = {acc:.2%}")

        self.results[aggregator_name] = accuracy_history
        return accuracy_history

    def plot_results(
        self,
        aggregator_names: List[str] = None,
        save_path: str = None,
        title: str = None,
    ):
        """Plot simulation results."""
        if not self.results:
            raise ValueError("No results to plot. Run simulation first.")

        if aggregator_names is None:
            aggregator_names = list(self.results.keys())

        plt.figure(figsize=(10, 6))

        colors = ["red", "green", "blue", "orange", "purple"]
        linestyles = ["--", "-", ":", "-.", "--"]

        for idx, name in enumerate(aggregator_names):
            if name in self.results:
                plt.plot(
                    range(1, self.rounds + 1),
                    self.results[name],
                    label=name,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2,
                )

        if title is None:
            title = f"FL with {self.num_byzantine}/{self.num_clients} Byzantine Clients"

        plt.title(title)
        plt.xlabel("Communication Rounds")
        plt.ylabel("Test Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1.0)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()
    
    def save_results(
        self,
        save_path: str = None,
        acc_tm=None,
    ):
        # Optionally save results
        if save_path:
            results = {
                "single_aggregator_accuracy": (
                    float(acc_tm) if hasattr(acc_tm, "__float__") else acc_tm
                ),
                "config_used": {
                    "simulation": config.SIMULATION_CONFIG,
                    "model": config.MODEL_CONFIG,
                    "attack": config.ATTACK_CONFIG,
                    "aggregator": config.AGGREGATOR_CONFIG,
                },
            }

            import json
            import os

            # Create results directory if it doesn't exist
            os.makedirs(
                os.path.dirname(config.OUTPUT_CONFIG["results_save_path"]), exist_ok=True
            )

            with open(config.OUTPUT_CONFIG["results_save_path"], "w") as f:
                json.dump(results, f, indent=2)

            if config.OUTPUT_CONFIG["verbose"]:
                print(f"Results saved to {config.OUTPUT_CONFIG['results_save_path']}")

    def compare_aggregators(
        self,
        aggregator_names: List[str] = None,
        attack_fn: Callable = None,
        attack_args: Dict[str, Any] = None,
    ):
        """Run and compare multiple aggregators."""
        if aggregator_names is None:
            aggregator_names = ["Average", "TrimmedMean", "Median", "Krum"]

        for agg_name in aggregator_names:
            self.run(agg_name, attack_fn, attack_args)

        self.plot_results(aggregator_names)
