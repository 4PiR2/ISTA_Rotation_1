import os

import torch
from flwr.server import SimpleClientManager

import flwr as fl

from my_client import FlowerClient
from my_server import MyServer
from my_strategy import SaveModelStrategy
from PeFLL.dataset import gen_random_loaders
from PeFLL.models import CNNTarget
from PeFLL.utils import set_seed, get_device


def main():
    DEVICE = get_device()

    NUM_CLIENTS = 100
    NUM_TRAIN_CLIENTS = int(NUM_CLIENTS * .9)

    BATCH_SIZE = 32

    SEED = 42

    set_seed(SEED)
    trainloaders, valloaders, testloaders = gen_random_loaders(
        data_name='cifar10', data_path='./dataset',
        num_users=NUM_CLIENTS, num_train_users=NUM_TRAIN_CLIENTS, bz=BATCH_SIZE,
        partition_type='by_class', classes_per_user=2,
        alpha_train=None, alpha_test=None, embedding_dir_path=None
    )

    # Load model
    server_round_init: int = 0
    net = CNNTarget()
    if server_round_init:
        net.load_state_dict(
            torch.load(os.path.join('output', f'model_round_{server_round_init}.pth')),
            strict=True
        )


    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load data (CIFAR-10)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(net, trainloader, valloader)

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = {"num_cpus": 1, "num_gpus": 1} if DEVICE.type == "cuda" else None

    # Create FedAvg strategy
    strategy = SaveModelStrategy(
        fraction_fit=.1,  # Sample 100% of available clients for training
        fraction_evaluate=.1,  # Sample 50% of available clients for evaluation
        min_available_clients=NUM_TRAIN_CLIENTS,  # Wait until all 10 clients are available
        state_dict_keys=CNNTarget().state_dict().keys(),
        # server_round_init=server_round_init,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_TRAIN_CLIENTS,
        server=MyServer(client_manager=SimpleClientManager(), strategy=strategy),
        config=fl.server.ServerConfig(num_rounds=1000),
        client_resources=client_resources,
    )

    a = 0


if __name__ == '__main__':
    main()
