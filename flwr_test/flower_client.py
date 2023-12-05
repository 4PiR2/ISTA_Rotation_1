import gc
from logging import DEBUG, INFO, WARNING
import os
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
from torch.utils.data import DataLoader, TensorDataset

from PeFLL.dataset import gen_random_loaders
from PeFLL.models import CNNTarget
from PeFLL.utils import set_seed, count_parameters
# from PeFLL.models import CNNEmbed, MLPEmbed

from models import CNNEmbed2 as CNNEmbed, MLPEmbed2 as MLPEmbed
from models import HeadTarget
from parse_args import parse_args
from utils import state_dicts_to_ndarrays, ndarrays_to_state_dicts, detect_slurm, init_wandb, finish_wandb


class FlowerClient(flwr.client.NumPyClient):
    def __init__(self):
        self.tnet: torch.nn.Module = torch.nn.Module()
        self.enet: Optional[torch.nn.Module] = None
        self.train_loaders: List[DataLoader] = []
        self.val_loaders: List[DataLoader] = []
        self.test_loaders: List[DataLoader] = []
        self.class_presents: List[torch.Tensor] = []
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

        assert data_name == 'femnist' or num_train_users == int(num_users * .9)

        # Infer on range of OOD test clients
        if alpha_test == -1.:
            assert partition_type == 'dirichlet'
            # alpha_test_range = np.arange(1., 11.) * .1
            alpha_test = alpha_train

        set_seed(seed)
        self.train_loaders, self.val_loaders, self.test_loaders = gen_random_loaders(
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

        n_kernels = config['client_model_num_kernels']
        model_target_type = config['client_model_target_type']
        model_embed_type = config['model_embed_type']
        embed_dim = config['client_model_embed_dim']
        if embed_dim == -1:
            embed_dim = 1 + num_users // 4  # auto embedding size
        embed_y = config['client_model_embed_y'] and model_target_type != 'head'
        model_target_head_layers = config['client_model_target_head_layers']
        device = torch.device('cpu')

        match data_name, model_target_type:
            case 'cifar10', 'cnn':
                self.tnet = CNNTarget(in_channels=3, n_kernels=n_kernels, out_dim=10)
            case 'cifar10', 'head':
                self.tnet = HeadTarget(in_dim=embed_dim, hidden_dim=embed_dim, out_dim=10, n_layers=model_target_head_layers)
            case 'cifar100', 'cnn':
                self.tnet = CNNTarget(in_channels=3, n_kernels=n_kernels, out_dim=100)
            case 'cifar100', 'head':
                self.tnet = HeadTarget(in_dim=embed_dim, hidden_dim=embed_dim, out_dim=100, n_layers=model_target_head_layers)
            case 'femnist', 'cnn':
                self.tnet = CNNTarget(in_channels=1, n_kernels=n_kernels, out_dim=62)
            case 'femnist', 'head':
                self.tnet = HeadTarget(in_dim=embed_dim, hidden_dim=embed_dim, out_dim=62, n_layers=model_target_head_layers)
            case _:
                raise ValueError("Choose data_name from ['cifar10', 'cifar100', 'femnist'].")
        # log(INFO, f'num_parameters_tnet: {count_parameters(self.tnet)}')

        match data_name, model_embed_type:
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
        # if self.enet is not None:
        #     log(INFO, f'num_parameters_enet: {count_parameters(self.enet)}')

        def materialize_dataloader(dataloader: DataLoader, device: torch.device = torch.device('cpu')) -> DataLoader:
            xs, ys = [], []
            for x, y in dataloader.dataset:
                xs.append(x.to(device))
                ys.append(y.to(device))
            xs = torch.stack(xs, dim=0)
            ys = torch.stack(ys, dim=0)
            dataset = TensorDataset(xs, ys)
            dataloader_kwargs = {
                k: v for k, v in vars(dataloader).items() if not k.startswith('_') and k not in ['batch_sampler']
            }
            if torch.device(device).type != 'cpu':
                del dataloader_kwargs['pin_memory']
            dataloader_kwargs['dataset'] = dataset
            return DataLoader(**dataloader_kwargs)

        # prefetch datasets to RAM
        self.train_loaders = [materialize_dataloader(dl, device) for dl in self.train_loaders]
        self.val_loaders = [materialize_dataloader(dl, device) for dl in self.val_loaders]
        self.test_loaders = [materialize_dataloader(dl, device) for dl in self.test_loaders]

        self.class_presents = [dl.dataset.tensors[1].sum(dim=0).bool().log() for dl in self.train_loaders]

        return super().get_properties(config)

    def get_dataloader(self, is_eval: int, cid: int) -> DataLoader:
        match is_eval:
            case 0:
                dataloader = self.train_loaders[cid]
            case 1:
                dataloader = self.val_loaders[cid]
            case 2:
                dataloader = self.test_loaders[cid]
            case _:
                raise NotImplementedError
        return dataloader

    def get_embed_dataloader(self, is_eval: int, cid: int, device: torch.device) -> DataLoader:
        dataloader = self.get_dataloader(is_eval, cid)
        self.enet = self.enet.to(device)
        self.enet.eval()
        xs, ys = [], []
        for x, y in dataloader.dataset:
            xs.append(x.to(device))
            ys.append(y.to(device))
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        if not is_eval:
            embeddings = self.enet((xs, None))
        else:
            with torch.no_grad():
                embeddings = self.enet((xs, None))
        dataset = TensorDataset(embeddings, ys)
        dataloader_kwargs = {
            k: v for k, v in vars(dataloader).items() if not k.startswith('_') and k not in ['batch_sampler']
        }
        if torch.device(device).type != 'cpu':
            del dataloader_kwargs['pin_memory']
        dataloader_kwargs['dataset'] = dataset
        return DataLoader(**dataloader_kwargs)

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
        model_target_type = config['client_model_target_type']

        if not is_eval:
            dataloader = self.get_dataloader(is_eval, cid)  # self.train_loaders[cid]
            num_batches = config['client_embed_num_batches']
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
            self.enet.train()
            embeddings = self.enet((image_all, label_all))
        else:
            if config['client_eval_embed_train_split']:
                dataloader = self.get_dataloader(0, cid)  # self.train_loaders[cid]
            else:
                dataloader = self.get_dataloader(is_eval, cid) # self.val_loaders[cid] or self.test_loaders[cid]
            image_all, label_all = dataloader.dataset.tensors
            image_all, label_all = image_all.to(device), label_all.to(device)
            self.enet.eval()
            with torch.no_grad():
                embeddings = self.enet((image_all, label_all))

        metrics = {}
        if not is_eval and model_target_type == 'head':
            x = (embeddings[:, None, None, :] @ embeddings[None, :, :, None])[:, :, 0, 0]
            y = (label_all[:, None, None, :] @ label_all[None, :, :, None])[:, :, 0, 0]
            sig_x = x.sigmoid()
            loss_1 = (- sig_x[y > .5].log().sum() - (1. - sig_x[y < .5]).log().sum()) / sig_x.numel()
            self.stage_memory['loss_1'] = loss_1
            metrics['loss_1'] = loss_1.item()

        embedding = embeddings.mean(dim=0)
        embedding_ndarray = embedding.detach().cpu().numpy()
        self.stage_memory['embedding'] = embedding
        return [embedding_ndarray], 1, metrics

    def fit_2(self, parameters: List[np.ndarray], config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        device = torch.device(config['device'])
        self.tnet = self.tnet.to(device)
        self.set_parameters(parameters)

        cid = int(config['cid'])
        is_eval = config['is_eval']
        gradient_mode = config['client_target_gradient_mode']
        model_target_type = config['client_model_target_type']
        criterion = torch.nn.CrossEntropyLoss(reduction='mean' if not gradient_mode else 'none')
        classes_present = self.class_presents[cid].to(device)
        metrics = {}

        match model_target_type:
            case 'cnn':
                dataloader = self.get_dataloader(is_eval, cid)
            case 'head':
                dataloader = self.get_embed_dataloader(is_eval, cid, device)
            case _:
                raise NotImplementedError

        if not is_eval:
            self.tnet.train()
            optimizer_target_lr = config['client_optimizer_target_lr']  # if not gradient_mode else 1.
            optimizer_target_momentum = config['client_optimizer_target_momentum'] if not gradient_mode else 0.
            optimizer_target_weight_decay = config['client_optimizer_target_weight_decay']  # if not grad_mode else 0.
            optimizer = torch.optim.SGD(
                self.tnet.parameters(),
                lr=optimizer_target_lr,
                momentum=optimizer_target_momentum,
                weight_decay=optimizer_target_weight_decay,
            )
            num_batches = config['client_target_num_batches']
        else:
            self.tnet.eval()
            num_batches = len(dataloader)

        if not gradient_mode:
            i, length, total_loss, total_correct, total_correct_masked = 0, 0, 0., 0., 0.
            while i < num_batches:
                for inputs, labels in dataloader:
                    if i >= num_batches:
                        break
                    i += 1
                    length += len(labels)

                    match model_target_type, is_eval:
                        case 'head', 0:
                            outputs = self.tnet(inputs.detach())
                        case 'head', _:
                            with torch.no_grad():
                                outputs = self.tnet(inputs)
                        case _, 0:
                            outputs = self.tnet(inputs.to(device))
                        case _, _:
                            with torch.no_grad():
                                outputs = self.tnet(inputs.to(device))

                    labels = labels.to(device)
                    loss = criterion(outputs, labels)
                    if not is_eval:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.tnet.parameters(), 50.)
                        optimizer.step()
                    # Metrics
                    total_loss += loss.item() * len(labels)
                    if labels.dim() > 1:
                        labels = labels.argmax(dim=-1)
                    predicts = outputs.argmax(dim=-1)
                    predicts_masked = (outputs + classes_present).argmax(dim=-1)
                    total_correct += (predicts == labels).sum().item()
                    total_correct_masked += (predicts_masked == labels).sum().item()

            total_loss /= length
            total_correct /= length
            total_correct_masked /= length

            if model_target_type == 'head' and not is_eval:
                loss_2 = torch.stack(
                    [criterion(self.tnet(inputs), labels.to(device)) for inputs, labels in dataloader], dim=0,
                ).mean()
                self.stage_memory['loss_2'] = loss_2
                metrics['loss_2'] = loss_2.item()

        else:
            inputs, labels = dataloader.dataset.tensors
            inputs, labels = inputs.to(device), labels.to(device)

            match model_target_type, is_eval:
                case 'head', 0:
                    outputs = self.tnet(inputs.detach())
                case 'head', _:
                    with torch.no_grad():
                        outputs = self.tnet(inputs)
                case _, 0:
                    outputs = self.tnet(inputs)
                case _, _:
                    with torch.no_grad():
                        outputs = self.tnet(inputs)

            losses = criterion(outputs, labels)

            if not is_eval:
                do_sample = False
                if do_sample:
                    length = num_batches * dataloader.batch_size
                    sample_counts = torch.ones(len(losses), device=device).multinomial(
                        num_samples=length,
                        replacement=True,
                    ).bincount(minlength=len(losses))
                    loss = (losses * sample_counts).sum() / length
                else:
                    length = len(losses)
                    loss = losses.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tnet.parameters(), 50.)
                optimizer.step()
            else:
                length = len(losses)
                loss = losses.mean()

            # Metrics
            total_loss = loss.item()
            if labels.dim() > 1:
                labels = labels.argmax(dim=-1)
            predicts = outputs.argmax(dim=-1)
            predicts_masked = (outputs + classes_present).argmax(dim=-1)
            total_correct = (predicts == labels).mean(dtype=outputs.dtype).item()
            total_correct_masked = (predicts_masked == labels).mean(dtype=outputs.dtype).item()

        metrics = {
            **metrics,
            'loss_t': float(total_loss),
            'accuracy': float(total_correct),
            'accuracy_m': float(total_correct_masked),
        }

        if not is_eval:
            parameters = self.get_parameters({'net_name': 'tnet'})
        else:
            parameters = []

        # gc.collect()
        return parameters, length, metrics

    def fit_3(self, parameters: List[np.ndarray], config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        embedding = self.stage_memory['embedding']
        embed_grad = torch.tensor(parameters[0], device=embedding.device)
        loss = (embed_grad * embedding).sum()
        metrics = {'loss_e': loss.item()}
        loss += self.stage_memory['loss_1'] if 'loss_1' in self.stage_memory else 0.
        loss += self.stage_memory['loss_2'] if 'loss_2' in self.stage_memory else 0.
        for p in self.enet.parameters():
            p.grad = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.enet.parameters()), 50.)
        grad_dict = {k: v.grad for k, v in zip(self.enet.state_dict().keys(), self.enet.parameters())}
        grads = state_dicts_to_ndarrays({'enet': grad_dict})
        return grads, 1, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the network on the entire test set."""
        raise NotImplementedError


def main():
    args = parse_args()

    ip, port = args.server_address.split(':')
    if args.enable_slurm and detect_slurm():
        ip = os.environ['SLURM_SUBMIT_HOST']
        init_wandb(
            args=args,
            experiment_name=f"{os.environ['SLURM_JOB_ID']}-C{os.environ['SLURM_NODEID']}-{os.environ['SLURMD_NODENAME']}",
            group=os.environ['SLURM_JOB_ID'],
        )

    while True:
        try:
            flwr.client.start_numpy_client(server_address=f'{ip}:{port}', client=FlowerClient())
        except _MultiThreadedRendezvous:
            log(DEBUG, 'Waiting for server')
            time.sleep(1.)
        # except Exception as e:
        #     log(DEBUG, e)
        #     log(DEBUG, traceback.format_exc())
        #     break
        else:
            break
    finish_wandb()


if __name__ == '__main__':
    main()
