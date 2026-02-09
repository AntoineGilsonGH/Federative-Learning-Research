import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Simulation parameters
SIMULATION_CONFIG = {
    "num_clients": 10,
    "num_byzantine": 2,
    "rounds": 20,
    "batch_size": 32,
    "learning_rate": 0.01,
    "device": DEVICE,
    "dataset_name": "MNIST",
}

# Model parameters
MODEL_CONFIG = {
    "model_name": "CNNModel",  # or "default" for default model
    "hidden_size": 64,
    "input_shape": (1, 28, 28),  # for MNIST
    "num_classes": 10,
    "num_channels": 1,  # For CNN models
    "num_blocks": 2,  # For ResNet models
}

# Attack parameters
ATTACK_CONFIG = {
    "attack_fn": "custom_attack",
    "attack_scale": 5.0,
    "attack_noise": 0.1,
}

# Aggregator parameters
AGGREGATOR_CONFIG = {
    "aggregators_to_compare": ["Average", "TrimmedMean", "Median"],
    "single_aggregator": "TrimmedMean",
}

# Output/Result parameters
OUTPUT_CONFIG = {
    "plot_save_path": f"results/fl_comparison_{MODEL_CONFIG['model_name']}.png",
    "results_save_path": f"results/simulation_results_{MODEL_CONFIG['model_name']}.json",
    "verbose": True,
}


def print_config():
    """Print configuration"""
    print("=" * 50)
    print("Federated Learning Simulation")
    print("=" * 50)
    print(f"Device: {SIMULATION_CONFIG['device']}")
    print(f"Model: {MODEL_CONFIG['model_name']}")
    print(
        f"Clients: {SIMULATION_CONFIG['num_clients']} "
        f"(Byzantine: {SIMULATION_CONFIG['num_byzantine']})"
    )
    print(f"Rounds: {SIMULATION_CONFIG['rounds']}")
    print(f"Attack: {ATTACK_CONFIG['attack_fn']}")
    print("=" * 50)
