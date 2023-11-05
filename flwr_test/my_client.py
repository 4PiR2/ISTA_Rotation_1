import gc
from collections import OrderedDict
from typing import List

import flwr as fl
import numpy as np
import torch

from PeFLL.dataset import gen_random_loaders
from PeFLL.models import CNNTarget
from PeFLL.utils import get_device, set_seed

DEVICE = get_device()


def train(net, trainloader, epochs: int = 1, verbose: bool = True,
          lr: float = 2e-3, momentum: float = .9, weight_decay: float = 5e-5):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    net.train()
    for epoch in range(epochs):
        correct, epoch_loss = 0, 0.
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
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
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            if labels.dim() > 1:
                labels = labels.argmax(dim=-1)
            correct += (outputs.argmax(dim=-1) == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v, device=DEVICE) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, *args, **kwargs):

        net = CNNTarget()

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

        self.net = net.to(DEVICE)
        self.trainloaders = trainloaders
        self.valloaders = valloaders

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        trainloader = self.trainloaders[int(config['cid'])]
        train(self.net, trainloader, epochs=1, verbose=True,
              lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        gc.collect()
        return get_parameters(self.net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        valloader = self.valloaders[int(config['cid'])]
        loss, accuracy = test(self.net, valloader)
        gc.collect()
        return float(loss), len(valloader.dataset), {'accuracy': float(accuracy)}


def main():
    server_ip = '127.0.0.1'
    server_port = 18080
    fl.client.start_numpy_client(server_address=f'{server_ip}:{server_port}', client=FlowerClient())


if __name__ == '__main__':
    main()
