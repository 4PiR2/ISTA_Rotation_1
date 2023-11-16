import gc
from logging import DEBUG, INFO
import traceback
from typing import Dict, List, Optional, Tuple

import flwr
from flwr.common import Config, Scalar, NDArrays
from flwr.common.logger import log
import numpy as np
import torch
from torch.utils.data import DataLoader

from PeFLL.dataset import gen_random_loaders
from PeFLL.models import CNNTarget, CNNEmbed, MLPEmbed
from PeFLL.utils import set_seed

from parse_args import parse_args


class FlowerClient(flwr.client.NumPyClient):
    def __init__(self):
        self.tnet: torch.nn.Module = torch.nn.Module()
        self.enet: Optional[torch.nn.Module] = None
        self.trainloaders: List[DataLoader] = []
        self.valloaders: List[DataLoader] = []
        self.testloaders: List[DataLoader] = []
        self.stage_memory: Dict = {}

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
        if alpha_test == -1.:
            assert partition_type == 'dirichlet'
            # alpha_test_range = np.arange(1., 11.) * .1
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
        # if 'seed2' in config:
        #     set_seed(config['seed2'])

        n_kernels = config['model_num_kernels']
        embed_model = config['model_embed_type']
        embed_dim = config['model_embed_dim']
        if embed_dim == -1:
            embed_dim = 1 + num_users // 4  # auto embedding size
        embed_y = config['client_model_embed_y']
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

    def get_parameters(self, config: Config) -> NDArrays:
        if config is not None and 'net_name' in config:
            header = [config['net_name']]
        else:
            header = []
            for net_name in self.__dir__():
                if isinstance(self.__getattribute__(net_name), torch.nn.Module):
                    header.append(net_name)
        keys: List[np.ndarray] = []
        vals: List[np.ndarray] = []
        for net_name in header:
            net = self.__getattribute__(net_name)
            keys.append(np.asarray(list(net.state_dict().keys())))
            vals.extend([val.detach().cpu().numpy() for val in net.state_dict().values()])
        return [np.asarray(header)] + keys + vals

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        for i, net_name in enumerate(parameters[0]):
            net: torch.nn.Module = self.__getattribute__(net_name)
            keys = parameters[1 + i]
            vals = parameters[1 + sum([len(k) for k in parameters[:1+i]]): 1 + sum([len(k) for k in parameters[:2+i]])]
            state_dict = {k: torch.tensor(v) for k, v in zip(keys, vals)}
            net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the network on the training set."""
        match config['stage']:
            case 3 | 6:
                return self.fit_1(parameters, config)
            case 1 | 4:
                return self.fit_2(parameters, config)
            case 5:
                return self.fit_3(parameters, config)
            case _:
                raise NotImplementedError

    def fit_1(self, parameters: List[np.ndarray], config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        device = torch.device(config['device'])
        self.enet = self.enet.to(device)
        self.enet.device = device  # bug in PeFLL
        self.set_parameters(parameters)

        cid = int(config['cid'])
        is_eval = config['is_eval']

        if not is_eval:
            self.enet.train()
            dataloader = self.trainloaders[cid]
            num_batches = config['client_embed_num_batches']
        else:
            self.enet.eval()
            if config['client_eval_embed_train_split']:
                dataloader = self.trainloaders[cid]
            else:
                dataloader = self.valloaders[cid]
            num_batches = -1

        num_batches = num_batches if num_batches != -1 else len(dataloader)
        image_all, label_all = [], []
        i = 0
        while i < num_batches:
            for images, labels in dataloader:
                if i >= num_batches:
                    break
                i += 1
                images, labels = images.to(device), labels.to(device)
                image_all.append(images)
                label_all.append(labels)
        image_all = torch.cat(image_all, dim=0)
        label_all = torch.cat(label_all, dim=0)

        if not is_eval:
            embedding = self.enet((image_all, label_all)).mean(dim=0)
        else:
            with torch.no_grad():
                embedding = self.enet((image_all, label_all)).mean(dim=0)

        # TODO
        m, s = 0.25064870715141296 / 16., 1.0152097940444946 / 16.
        embedding = (embedding - m) / s

        embedding_ndarray = embedding.detach().cpu().numpy()

        length = len(label_all)
        label_count = label_all.sum(dim=0)

        self.stage_memory['embedding'] = embedding
        self.stage_memory['length'] = length
        self.stage_memory['label_count'] = label_count  # TODO

        return [embedding_ndarray], length, {}

    def fit_2(self, parameters: List[np.ndarray], config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        device = torch.device(config['device'])
        self.tnet = self.tnet.to(device)
        self.set_parameters(parameters)
        self.tnet.train()
        optimizer = torch.optim.SGD(
            self.tnet.parameters(),
            lr=config['client_optimizer_target_lr'],
            momentum=config['client_optimizer_target_momentum'],
            weight_decay=config['client_optimizer_target_weight_decay']
        )
        dataloader = self.trainloaders[int(config['cid'])]
        criterion = torch.nn.CrossEntropyLoss()
        num_batches = config['client_target_num_batches']
        i, length, train_loss, train_acc = 0, 0, 0., 0.
        while i < num_batches:
            for images, labels in dataloader:
                if i >= num_batches:
                    break
                i += 1
                length += len(labels)
                images, labels = images.to(device), labels.to(device)
                outputs = self.tnet(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tnet.parameters(), 50.)
                optimizer.step()
                # Metrics
                train_loss += loss.item()
                if labels.dim() > 1:
                    labels = labels.argmax(dim=-1)
                train_acc += (outputs.argmax(dim=-1) == labels).sum().item()

        train_loss /= length
        train_acc /= length
        # gc.collect()
        return self.get_parameters({'net_name': 'tnet'}), length, {
            'loss_2': float(train_loss),
            'accuracy_2': float(train_acc),
        }

    def fit_3(self, parameters: List[np.ndarray], config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        embedding = self.stage_memory['embedding']
        grad = torch.tensor(parameters[0], device=embedding.device)
        loss = (grad * embedding).sum()

        lr = config['client_optimizer_embed_lr']
        match config['client_optimizer_embed_type']:
            case 'adam':
                optimizer = torch.optim.Adam(self.enet.parameters(), lr=lr)
            case 'sgd':
                optimizer = torch.optim.SGD(
                    self.enet.parameters(),
                    lr=lr,
                    momentum=.9,
                    weight_decay=config['optimizer_weight_decay'],
                )
            case _:
                raise NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.enet.parameters()), 50.)
        optimizer.step()

        length = self.stage_memory['length']

        # gc.collect()
        return self.get_parameters({'net_name': 'enet'}), length, {'loss_3': float(loss)}

    def evaluate(self, parameters: List[np.ndarray], config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the network on the entire test set."""
        device = torch.device(config['device'])
        self.tnet = self.tnet.to(device)
        self.set_parameters(parameters)
        dataloader = self.valloaders[int(config['cid'])]
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        if config['client_eval_mask_absent'] and config['stage'] == 7:
            classes_present = self.stage_memory['label_count'].bool().float()
        else:
            classes_present = 1.

        correct, loss = 0, 0.
        self.tnet.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.tnet(images)
                loss += criterion(outputs, labels).item()
                pred = (outputs * classes_present).argmax(dim=-1)
                if labels.dim() > 1:
                    labels = labels.argmax(dim=-1)
                correct += (pred == labels).sum().item()
        length = len(dataloader.dataset)
        loss /= length
        accuracy = correct / length
        # gc.collect()
        return float(loss), length, {'accuracy': float(accuracy)}


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
