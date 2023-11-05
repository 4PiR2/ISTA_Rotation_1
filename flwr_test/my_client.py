from collections import OrderedDict
import gc
from logging import DEBUG, INFO
from typing import List, Dict, Tuple

import flwr as fl
from flwr.common import Config, Scalar
from flwr.common.logger import log
import numpy as np
import torch
from torch.utils.data import DataLoader

from PeFLL.dataset import gen_random_loaders
from PeFLL.models import CNNTarget
from PeFLL.utils import set_seed


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, *args, **kwargs):
        self.net: torch.nn.Module = CNNTarget()
        self.device: torch.device = torch.device('cpu')
        self.trainloaders: List[DataLoader] = []
        self.valloaders: List[DataLoader] = []

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        device = config['device']
        self.device = device
        self.net = self.net.to(self.device)

        num_train_clients = config['num_train_clients']

        # TODO
        NUM_CLIENTS = 100
        BATCH_SIZE = 32
        SEED = 42

        set_seed(SEED)
        self.trainloaders, self.valloaders, testloaders = gen_random_loaders(
            data_name='cifar10', data_path='./dataset',
            num_users=NUM_CLIENTS, num_train_users=num_train_clients, bz=BATCH_SIZE,
            partition_type='by_class', classes_per_user=2,
            alpha_train=None, alpha_test=None, embedding_dir_path=None
        )
        return super().get_properties(config)

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, net: torch.nn.Module, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(self.net, parameters)
        trainloader = self.trainloaders[int(config['cid'])]

        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        self.net.train()

        epochs = 1
        for epoch in range(epochs):
            correct, epoch_loss = 0, 0.
            for images, labels in trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss.item()
                if labels.dim() > 1:
                    labels = labels.argmax(dim=-1)
                correct += (outputs.argmax(dim=-1) == labels).sum().item()
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / len(trainloader.dataset)

            verbose = True
            if verbose:
                log(DEBUG, f"Epoch {epoch}: train loss {epoch_loss}, accuracy {epoch_acc}")

        gc.collect()
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(self.net, parameters)
        valloader = self.valloaders[int(config['cid'])]

        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.
        self.net.eval()
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                if labels.dim() > 1:
                    labels = labels.argmax(dim=-1)
                correct += (outputs.argmax(dim=-1) == labels).sum().item()
        loss /= len(valloader.dataset)
        accuracy = correct / len(valloader.dataset)

        gc.collect()
        return float(loss), len(valloader.dataset), {'accuracy': float(accuracy)}


def main():
    # TODO
    server_ip = '127.0.0.1'
    server_port = 18080
    fl.client.start_numpy_client(server_address=f'{server_ip}:{server_port}', client=FlowerClient())


if __name__ == '__main__':
    main()
