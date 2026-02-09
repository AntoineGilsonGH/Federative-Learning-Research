import torch

import config
from utils.simulation import Simulation
from utils.load_models import get_model_class_and_args


# Define custom attack function
def custom_attack(grads, client_idx, scale=None, noise=None):
    """Custom Byzantine attack."""
    if scale is None:
        scale = config.ATTACK_CONFIG["attack_scale"]
    if noise is None:
        noise = config.ATTACK_CONFIG["attack_noise"]

    if client_idx < config.SIMULATION_CONFIG["num_byzantine"]:
        # Scale and add noise
        grads = -scale * grads
        grads += torch.randn_like(grads) * noise
    return grads


def main():
    config.print_config()

    # Get model class and arguments
    model_class, model_args = get_model_class_and_args()

    # Create simulation
    sim = Simulation(
        model_class=model_class,
        model_args=model_args,
        **config.SIMULATION_CONFIG,
    )

    # Determine attack function
    attack_fn = (
        custom_attack if config.ATTACK_CONFIG["attack_fn"] == "custom_attack" else None
    )
    attack_args = {
        "scale": config.ATTACK_CONFIG["attack_scale"],
        "noise": config.ATTACK_CONFIG["attack_noise"],
    }

    # Run single aggregator
    if config.OUTPUT_CONFIG["verbose"]:
        print(f"\nRunning {config.AGGREGATOR_CONFIG['single_aggregator']} aggregator...")

    acc_tm = sim.run(
        config.AGGREGATOR_CONFIG["single_aggregator"],
        attack_fn=attack_fn,
        attack_args=attack_args,
    )

    # Compare multiple aggregators
    if config.OUTPUT_CONFIG["verbose"]:
        print(f"\nComparing multiple aggregators: {config.AGGREGATOR_CONFIG['aggregators_to_compare']}")

    sim.compare_aggregators(
        aggregator_names=config.AGGREGATOR_CONFIG["aggregators_to_compare"],
        attack_fn=attack_fn,
        attack_args=attack_args,
    )

    # Save plot
    plot_path = config.OUTPUT_CONFIG["plot_save_path"]
    sim.plot_results(save_path=plot_path)
    print(f"\nPlot saved to: {plot_path}")

    sim.save_results(save_path=config.OUTPUT_CONFIG.get("results_save_path"), acc_tm=acc_tm)

    return acc_tm


if __name__ == "__main__":
    main()
