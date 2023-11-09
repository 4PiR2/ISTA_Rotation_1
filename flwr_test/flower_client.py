import gc
from logging import DEBUG, INFO
import traceback
from typing import List, Dict, Tuple

import flwr
from flwr.common import Config, Scalar
from flwr.common.logger import log
import numpy as np
import torch
from torch.utils.data import DataLoader

from PeFLL.dataset import gen_random_loaders
from PeFLL.models import CNNTarget
from PeFLL.utils import set_seed

from parse_args import parse_args


class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, *args, **kwargs):
        self.net: torch.nn.Module = CNNTarget()
        self.trainloaders: List[DataLoader] = []
        self.valloaders: List[DataLoader] = []
        self.testloaders: List[DataLoader] = []

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        data_name = config['data_name']
        data_path = config['data_path']
        num_users = config['num_clients'] if data_name != 'femnist' else 3597
        num_train_users = config['num_train_clients']
        bz = config['batch_size']
        partition_type = config['partition_type']
        classes_per_user = 2 if data_name == 'cifar10' else 10
        alpha_train = config['alpha_train']
        alpha_test = config['alpha_test']
        embedding_dir_path = None

        # Infer on range of OOD test clients
        alpha_test_range = None
        if alpha_test == -1.:
            assert partition_type == 'dirichlet'
            alpha_test_range = np.arange(1., 11.) * .1
            alpha_test = alpha_train

        set_seed(config['seed'])
        self.trainloaders, self.valloaders, self.testloaders = gen_random_loaders(
            data_name=data_name,
            data_path=data_path,
            num_users=num_users,
            num_train_users=num_train_users,
            bz=bz,
            partition_type=partition_type,
            classes_per_user=classes_per_user,
            alpha_train=alpha_train,
            alpha_test=alpha_test,
            embedding_dir_path=embedding_dir_path,
        )
        if 'seed2' in config:
            set_seed(config['seed2'])

        # if embed_dim == -1:
        #     # auto embedding size
        #     embed_dim = int(1. + num_users * .25)
        #
        # if data_name == 'cifar10':
        #     if embed_model == 'mlp':
        #         enet = MLPEmbed(10, embed_dim)
        #     elif embed_model == 'cnn':
        #         enet = CNNEmbed(embed_y, 10, embed_dim, device)
        #     else:
        #         raise ValueError('Choose model from mlp or cnn.')
        #     hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid, n_hidden=hyper_nhid, n_kernels=n_kernels)
        #     joint = EmbedHyper(enet, hnet)
        #     net = CNNTarget(n_kernels=n_kernels)
        # elif data_name == "cifar100":
        #     if embed_model == 'mlp':
        #         enet = MLPEmbed(100, embed_dim)
        #     elif embed_model == 'cnn':
        #         enet = CNNEmbed(embed_y, 100, embed_dim, device)
        #     else:
        #         raise ValueError('Choose model from mlp or cnn.')
        #
        #     hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid,
        #                     n_hidden=hyper_nhid, n_kernels=n_kernels, out_dim=100)
        #     joint = EmbedHyper(enet, hnet)
        #     net = CNNTarget(n_kernels=n_kernels, out_dim=100)
        # elif data_name == 'femnist':
        #     if embed_model == 'mlp':
        #         enet = MLPEmbed(62, embed_dim)
        #     elif embed_model == 'cnn':
        #         enet = CNNEmbed(embed_y, 62, embed_dim, device, in_channels=1)
        #     else:
        #         raise ValueError('Choose model from mlp or cnn.')
        #
        #     hnet = CNNHyper(num_nodes, embed_dim, in_channels=1, hidden_dim=hyper_hid,
        #                     n_hidden=hyper_nhid, n_kernels=n_kernels, out_dim=62)
        #     joint = EmbedHyper(enet, hnet)
        #     net = CNNTarget(in_channels=1, n_kernels=n_kernels, out_dim=62)
        #
        # else:
        #     raise ValueError("choose data_name from ['cifar10', 'cifar100']")

        return super().get_properties(config)

    def get_parameters(self, config: Config):
        return [val.detach().cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.net.state_dict().keys(), parameters)}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Config):
        """Train the network on the training set."""
        device = torch.device(config['device'])
        self.net = self.net.to(device)
        self.set_parameters(parameters)
        trainloader = self.trainloaders[int(config['cid'])]
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        self.net.train()

        num_epochs = config['num_epochs']
        total_loss, total_acc = 0., 0.
        for epoch in range(num_epochs):
            correct, epoch_loss = 0, 0.
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
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
            total_loss += epoch_loss
            total_acc += epoch_acc

            if 'verbose' in config and config['verbose']:
                log(DEBUG, f'Epoch {epoch}: train loss {epoch_loss}, accuracy {epoch_acc}')

        total_loss /= num_epochs
        total_acc /= num_epochs
        gc.collect()
        return self.get_parameters({}), len(trainloader.dataset), {
            'loss': float(total_loss), 'accuracy': float(total_acc)}

    def evaluate(self, parameters: List[np.ndarray], config: Config):
        """Evaluate the network on the entire test set."""
        device = torch.device(config['device'])
        self.net = self.net.to(device)
        self.set_parameters(parameters)
        valloader = self.valloaders[int(config['cid'])]
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.
        self.net.eval()
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
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
    args = parse_args()

    while True:
        try:
            flwr.client.start_numpy_client(server_address=args.server_address, client=FlowerClient())
        except Exception as e:
            log(DEBUG, e)
            log(DEBUG, traceback.format_exc())
        else:
            break


if __name__ == '__main__':
    main()
