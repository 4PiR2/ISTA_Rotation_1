import gc
from logging import DEBUG, INFO
import traceback
from typing import Dict, List, Optional, Tuple

import flwr
from flwr.common import Config, Scalar
from flwr.common.logger import log
import numpy as np
import torch
from torch.utils.data import DataLoader

from PeFLL.dataset import gen_random_loaders
from PeFLL.models import CNNTarget, CNNEmbed, MLPEmbed
from PeFLL.utils import set_seed

from parse_args import parse_args


class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, *args, **kwargs):
        self.tnet: torch.nn.Module = torch.nn.Module()
        self.enet: Optional[torch.nn.Module] = None
        self.trainloaders: List[DataLoader] = []
        self.valloaders: List[DataLoader] = []
        self.testloaders: List[DataLoader] = []

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        num_train_users = config['num_train_clients']
        seed = config['client_dataset_seed']
        data_name = config['client_dataset_data_name']
        data_path = config['client_dataset_data_path']
        num_users = config['client_dataset_num_clients'] if data_name != 'femnist' else 3597
        bz = config['client_dataset_batch_size']
        partition_type = config['client_dataset_partition_type']
        classes_per_user = 2 if data_name == 'cifar10' else 10
        alpha_train = config['client_dataset_alpha_train']
        alpha_test = config['client_dataset_alpha_test']
        embedding_dir_path = None

        # Infer on range of OOD test clients
        alpha_test_range = None
        if alpha_test == -1.:
            assert partition_type == 'dirichlet'
            alpha_test_range = np.arange(1., 11.) * .1
            alpha_test = alpha_train

        set_seed(seed)
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

        n_kernels = config['model_num_kernels']
        embed_model = config['model_embed_type']
        embed_dim = config['model_embed_dim']
        if embed_dim == -1:
            embed_dim = int(1. + num_users * .25)  # auto embedding size
        embed_y = config['model_embed_y']
        device = torch.device('cpu')

        match data_name:
            case 'cifar10':
                self.tnet = CNNTarget(in_channels=3, n_kernels=n_kernels, out_dim=10)
            case 'cifar100':
                self.tnet = CNNTarget(in_channels=3, n_kernels=n_kernels, out_dim=100)
            case 'femnist':
                self.tnet = CNNTarget(in_channels=1, n_kernels=n_kernels, out_dim=62)
            case _:
                raise ValueError("Choose data_name from ['cifar10', 'cifar100', 'femnist'].")

        match data_name, embed_model:
            case 'cifar10', 'mlp':
                self.enet = MLPEmbed(in_dim=10, out_dim=embed_dim)
            case 'cifar10', 'cnn':
                self.enet = CNNEmbed(embed_y=embed_y, dim_y=10, embed_dim=embed_dim, device=device, in_channels=3, n_kernels=16)
            case 'cifar100', 'mlp':
                self.enet = MLPEmbed(in_dim=100, out_dim=embed_dim)
            case 'cifar100', 'cnn':
                self.enet = CNNEmbed(embed_y=embed_y, dim_y=100, embed_dim=embed_dim, device=device, in_channels=3, n_kernels=16)
            case 'femnist', 'mlp':
                self.enet = MLPEmbed(in_dim=62, out_dim=embed_dim)
            case 'femnist', 'cnn':
                self.enet = CNNEmbed(embed_y=embed_y, dim_y=62, embed_dim=embed_dim, device=device, in_channels=1, n_kernels=16)
            case _:
                self.enet = None

        return super().get_properties(config)

    def get_parameters(self, config: Config):
        return [val.detach().cpu().numpy() for _, val in self.tnet.state_dict().items()]

    @staticmethod
    def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]):
        state_dict = {k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), parameters)}
        net.load_state_dict(state_dict, strict=True)

    def fit_0(self, parameters: List[np.ndarray], config: Config):
        device = torch.device(config['device'])
        self.tnet = self.tnet.to(device)
        self.set_parameters(self.tnet, parameters)
        trainloader = self.trainloaders[int(config['cid'])]
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.tnet.parameters(),
            lr=config['client_optimizer_target_lr'],
            momentum=config['client_optimizer_target_momentum'],
            weight_decay=config['client_optimizer_target_weight_decay']
        )
        self.tnet.train()

        num_epochs = config['num_epochs']
        total_loss, total_acc = 0., 0.
        for epoch in range(num_epochs):
            correct, epoch_loss = 0, 0.
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.tnet(images)
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

    def fit(self, parameters: List[np.ndarray], config: Config):
        """Train the network on the training set."""
        stage = config['stage']
        match stage:
            case 0:
                return self.fit_0(parameters, config)

    def evaluate_0(self, parameters: List[np.ndarray], config: Config):
        device = torch.device(config['device'])
        self.tnet = self.tnet.to(device)
        self.set_parameters(self.tnet, parameters)
        valloader = self.valloaders[int(config['cid'])]
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.
        self.tnet.eval()
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.tnet(images)
                loss += criterion(outputs, labels).item()
                if labels.dim() > 1:
                    labels = labels.argmax(dim=-1)
                correct += (outputs.argmax(dim=-1) == labels).sum().item()
        loss /= len(valloader.dataset)
        accuracy = correct / len(valloader.dataset)
        gc.collect()
        return float(loss), len(valloader.dataset), {'accuracy': float(accuracy)}

    def evaluate(self, parameters: List[np.ndarray], config: Config):
        """Evaluate the network on the entire test set."""
        stage = config['stage']
        match stage:
            case 0:
                return self.evaluate_0(parameters, config)

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
