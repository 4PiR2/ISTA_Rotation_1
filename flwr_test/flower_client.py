import gc
from logging import DEBUG, INFO
import time
import timeit
import traceback
from typing import Dict, List, Optional, Tuple

import flwr
from flwr.common import Config, Scalar, NDArrays
from flwr.common.logger import log
from grpc._channel import _MultiThreadedRendezvous
import numpy as np
import torch
from torch.utils.data import DataLoader

from PeFLL.dataset import gen_random_loaders
from PeFLL.models import CNNTarget, CNNEmbed, MLPEmbed
from PeFLL.utils import set_seed

from parse_args import parse_args
from utils import state_dicts_to_ndarrays, ndarrays_to_state_dicts


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
            header = [net_name for net_name in self.__dir__() if isinstance(self.__getattribute__(net_name), torch.nn.Module)]
        state_dicts = {net_name: self.__getattribute__(net_name).state_dict() for net_name in header}
        return state_dicts_to_ndarrays(state_dicts)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        for net_name, state_dict in ndarrays_to_state_dicts(parameters).items():
            self.__getattribute__(net_name).load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the network on the training set."""
        start_time = timeit.default_timer()

        stage = config['stage']
        match stage:
            case 3 | 6:
                params, weight, metrics = self.fit_1(parameters, config)
            case 1 | 2 | 4 | 7:
                params, weight, metrics = self.fit_2(parameters, config)
            case 5:
                params, weight, metrics = self.fit_3(parameters, config)
            case _:
                raise NotImplementedError

        end_time = timeit.default_timer()
        metrics[f'time_client_{stage}c'] = end_time - start_time
        return params, weight, metrics

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
                match is_eval:
                    case 1:
                        dataloader = self.valloaders[cid]
                    case 2:
                        dataloader = self.testloaders[cid]
                    case _:
                        raise NotImplementedError
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

        embedding_ndarray = embedding.detach().cpu().numpy()
        label_count = label_all.sum(dim=0)
        self.stage_memory['embedding'] = embedding
        self.stage_memory['label_count'] = label_count
        return [embedding_ndarray], 1, {}

    def fit_2(self, parameters: List[np.ndarray], config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        device = torch.device(config['device'])
        self.tnet = self.tnet.to(device)
        self.set_parameters(parameters)

        cid = int(config['cid'])
        is_eval = config['is_eval']
        criterion = torch.nn.CrossEntropyLoss()

        if not is_eval:
            self.tnet.train()
            optimizer = torch.optim.SGD(
                self.tnet.parameters(),
                lr=config['client_optimizer_target_lr'],
                momentum=config['client_optimizer_target_momentum'],
                weight_decay=config['client_optimizer_target_weight_decay']
            )
            dataloader = self.trainloaders[cid]
            num_batches = config['client_target_num_batches']
        else:
            self.tnet.eval()
            match is_eval:
                case 1:
                    dataloader = self.valloaders[cid]
                case 2:
                    dataloader = self.testloaders[cid]
                case _:
                    raise NotImplementedError
            num_batches = len(dataloader)

        if is_eval and 'client_eval_mask_absent' in config and config['client_eval_mask_absent']:
            classes_present = self.stage_memory['label_count'].bool().log()
        else:
            classes_present = 0.

        i, length, total_loss, total_correct = 0, 0, 0., 0.
        while i < num_batches:
            for images, labels in dataloader:
                if i >= num_batches:
                    break
                i += 1
                length += len(labels)
                images, labels = images.to(device), labels.to(device)
                if not is_eval:
                    outputs = self.tnet(images)
                else:
                    with torch.no_grad():
                        outputs = self.tnet(images)
                loss = criterion(outputs, labels)
                if not is_eval:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.tnet.parameters(), 50.)
                    optimizer.step()
                # Metrics
                total_loss += loss.item() * len(labels)
                predicts = (outputs + classes_present).argmax(dim=-1)
                if labels.dim() > 1:
                    labels = labels.argmax(dim=-1)
                total_correct += (predicts == labels).sum().item()

        total_loss /= length
        total_correct /= length

        if not is_eval:
            parameters = self.get_parameters({'net_name': 'tnet'})
        else:
            parameters = []

        # gc.collect()
        return parameters, length, {
            'loss_t': float(total_loss),
            'accuracy': float(total_correct),
        }

    def fit_3(self, parameters: List[np.ndarray], config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        embedding = self.stage_memory['embedding']
        embed_grad = torch.tensor(parameters[0], device=embedding.device)
        loss = (embed_grad * embedding).sum()
        for p in self.enet.parameters():
            p.grad = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.enet.parameters()), 50.)
        p_grads = ([np.asarray(['enet']), np.asarray(list(self.enet.state_dict().keys()))]
                   + [p.grad.detach().cpu().numpy() for p in self.enet.parameters()])
        return p_grads, 1, {
            'loss_e': float(loss),
        }

    def evaluate(self, parameters: List[np.ndarray], config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the network on the entire test set."""
        raise NotImplementedError


def main():
    args = parse_args()

    while True:
        try:
            flwr.client.start_numpy_client(server_address=args.server_address, client=FlowerClient())
        except _MultiThreadedRendezvous:
            log(DEBUG, 'Waiting for server')
            time.sleep(1.)
        # except Exception as e:
        #     log(DEBUG, e)
        #     log(DEBUG, traceback.format_exc())
        #     break
        else:
            break


if __name__ == '__main__':
    main()
