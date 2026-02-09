import config
from models import CustomModel, CNNModel, ResNetModel


def get_model_class_and_args():
    """Get model class and arguments based on config."""
    model_name = config.MODEL_CONFIG["model_name"]
    device = config.SIMULATION_CONFIG["device"]

    if model_name == "default":
        return None, {}  # Use default model from Simulation

    model_args = {"device": device}

    if model_name == "CustomModel":
        model_args.update(
            {
                "hidden_size": config.MODEL_CONFIG["hidden_size"],
                "input_shape": config.MODEL_CONFIG["input_shape"],
                "num_classes": config.MODEL_CONFIG["num_classes"],
            }
        )
        model_class = CustomModel

    elif model_name == "CNNModel":
        model_args.update(
            {
                "input_shape": config.MODEL_CONFIG["input_shape"],
                "num_classes": config.MODEL_CONFIG["num_classes"],
            }
        )
        model_class = CNNModel

    elif model_name == "ResNetModel":
        model_args.update(
            {
                "input_shape": config.MODEL_CONFIG["input_shape"],
                "num_classes": config.MODEL_CONFIG["num_classes"],
                "num_blocks": config.MODEL_CONFIG["num_blocks"],
            }
        )
        model_class = ResNetModel

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available: default, CustomModel, CNNModel, ResNetModel"
        )

    return model_class, model_args
