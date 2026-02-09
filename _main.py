import torch
import numpy as np
from byzfl.utils.misc import set_random_seed

import _config as config
from _simulation_byzfl import ByzFLSimulation


def main():
    """Main entry point using ByzFL framework."""
    # Set random seed for reproducibility
    if config.SEED is not None:
        set_random_seed(config.SEED)
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)

    # Print configuration
    config.print_config()

    # Create simulation
    sim = ByzFLSimulation(
        dataset_name=config.SIMULATION_CONFIG["dataset_name"],
        num_honest_clients=config.SIMULATION_CONFIG["num_honest"],
        num_byzantine_clients=config.SIMULATION_CONFIG["num_byzantine"],
        num_rounds=config.SIMULATION_CONFIG["rounds"],
        batch_size=config.SIMULATION_CONFIG["batch_size"],
        device=config.SIMULATION_CONFIG["device"],
        model_config=config.MODEL_CONFIG,
        aggregator_config=config.AGGREGATOR_CONFIG,
        attack_config=config.ATTACK_CONFIG,
        data_distribution_config=config.DATA_DISTRIBUTION_CONFIG,
        server_config=config.SERVER_CONFIG,
        client_config=config.CLIENT_CONFIG,
    )

    # Run single aggregator simulation
    # print(f"\nRunning simulation with aggregator: {config.AGGREGATOR_CONFIG['single_aggregator']}")
    # accuracy_history = sim.run_single_aggregator()

    # Compare multiple aggregators
    print(
        f"\nComparing aggregators: {config.AGGREGATOR_CONFIG['aggregators_to_compare']}"
    )
    sim.compare_aggregators()

    # Plot results
    sim.plot_results(save_path=config.OUTPUT_CONFIG["plot_save_path"])

    # Save results
    sim.save_results(save_path=config.OUTPUT_CONFIG["results_save_path"])

    return


if __name__ == "__main__":
    main()
