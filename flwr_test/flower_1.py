import os

import flwr as fl
from flwr.server import SimpleClientManager

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

    def client_fn(cid: str) -> FlowerClient:
        return FlowerClient()

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = {"num_cpus": 1, "num_gpus": 1} if DEVICE.type == "cuda" else None

    # Create FedAvg strategy
    strategy = SaveModelStrategy(
        fraction_fit=.1,  # Sample 100% of available clients for training
        fraction_evaluate=1.,  # Sample 50% of available clients for evaluation
        min_available_clients=NUM_TRAIN_CLIENTS,  # Wait until all 10 clients are available
        state_dict_keys=CNNTarget().state_dict().keys(),
    )

    server = MyServer(client_manager=SimpleClientManager(), strategy=strategy, init_round=0, eval_interval=10)

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_TRAIN_CLIENTS,
        server=server,
        config=fl.server.ServerConfig(num_rounds=1000),
        client_resources=client_resources,
    )

    a = 0


if __name__ == '__main__':
    main()
