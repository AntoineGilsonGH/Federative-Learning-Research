# Import necessary libraries
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from byzfl import Client, Server, ByzantineClient, DataDistributor
from byzfl.utils.misc import set_random_seed

from config import NB_HONEST_CLIENTS, NB_BYZ_CLIENTS, NB_TRAINING_STEPS, BATCH_SIZE
from utils.train import train

# Set random seed for reproducibility
SEED = 42
set_random_seed(SEED)

# Configurations
nb_honest_clients = NB_HONEST_CLIENTS
nb_byz_clients = NB_BYZ_CLIENTS
nb_training_steps = NB_TRAINING_STEPS
batch_size = BATCH_SIZE

# Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True)

# Distribute data among clients using non-IID Dirichlet distribution
data_distributor = DataDistributor({
    "data_distribution_name": "dirichlet_niid",
    "distribution_parameter": 0.5,
    "nb_honest": nb_honest_clients,
    "data_loader": train_loader,
    "batch_size": batch_size,
})
client_dataloaders = data_distributor.split_data()

# Initialize Honest Clients
honest_clients = [
    Client({
        "model_name": "cnn_mnist",
        "device": "cpu",
        "loss_name": "NLLLoss",
        "LabelFlipping": False,
        "training_dataloader": client_dataloaders[i],
        "momentum": 0.9,
        "nb_labels": 10,
        "store_per_client_metrics": True,  # or False, depending on your needs
    }) for i in range(nb_honest_clients)
]

# Prepare Test Dataset
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Server Setup, Use SGD Optimizer
server = Server({
    "device": "cpu",
    "model_name": "cnn_mnist",
    "test_loader": test_loader,
    "optimizer_name": "SGD",
    "learning_rate": 0.1,
    "weight_decay": 0.0001,
    "milestones": [1000],
    "learning_rate_decay": 0.25,
    "aggregator_info": {"name": "TrMean", "parameters": {"f": nb_byz_clients}},
    "pre_agg_list": [
        {"name": "Clipping", "parameters": {"c": 2.0}},
        {"name": "NNM", "parameters": {"f": nb_byz_clients}},
        ],
})

# Byzantine Client Setup
attack = {
    "name": "InnerProductManipulation",
    "f": nb_byz_clients,
    "parameters": {"tau": 3.0},
}
byz_client = ByzantineClient(attack)

print("Training :")
train(server, nb_training_steps, honest_clients, byz_client)

print("Training Complete!")