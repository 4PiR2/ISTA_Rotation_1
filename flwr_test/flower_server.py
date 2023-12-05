import argparse
import concurrent.futures
from logging import DEBUG, INFO, WARNING
import os
import queue
import timeit
from typing import Dict, List, Optional, Tuple, Union

import flwr
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Scalar,
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy.aggregate import aggregate
import numpy as np
import torch
import wandb

from PeFLL.utils import set_seed, count_parameters

from models import Hyper
from parse_args import parse_args
from utils import aggregate_tensor, ndarrays_to_state_dicts, state_dicts_to_ndarrays, init_wandb, finish_wandb, \
    get_pefll_checkpoint, detect_slurm


class FlowerServer(flwr.server.Server, flwr.server.strategy.FedAvg):
    def __init__(
            self,
            enable_slurm: bool,
            mode: str,
            num_train_clients: int,
            num_step_clients: int,
            init_round: int = 0,
            eval_interval: int = 100,
            eval_test: bool = False,
            save_interval: int = 1000,
            log_dir: str = './outputs',
            experiment_name: Optional[str] = None,
            server_seed: int = 42,
            client_dataset_seed: int = 42,
            client_dataset_data_name: str = 'cifar10',
            client_dataset_data_path: str = './dataset',
            client_dataset_num_clients: int = 100,
            client_dataset_batch_size: int = 32,
            client_dataset_partition_type: str = 'by_class',
            client_dataset_alpha_train: float = .1,
            client_dataset_alpha_test: float = .1,
            client_model_num_kernels: int = 16,
            model_embed_type: str = 'cnn',
            client_model_embed_dim: int = -1,
            client_model_embed_model_y: bool = True,
            model_hyper_hid_layers: int = 3,
            model_hyper_hid_dim: int = 100,
            client_model_target_type: str = 'cnn',
            client_model_target_head_layers: int = 2,
            client_optimizer_target_lr: float = 2e-3,
            client_optimizer_target_weight_decay: float = 5e-5,
            client_optimizer_target_momentum: float = .9,
            client_target_gradient_mode: bool = False,
            client_target_num_batches: int = 50,
            optimizer_embed_type: str = 'adam',
            optimizer_embed_lr: float = 2e-4,
            optimizer_embed_weight_decay: float = 1e-3,
            optimizer_embed_momentum: float = .9,
            client_embed_num_batches: int = 1,
            optimizer_hyper_type: str = 'adam',
            optimizer_hyper_lr: float = 2e-4,
            optimizer_hyper_weight_decay: float = 1e-3,
            optimizer_hyper_momentum: float = .9,
            # client_eval_mask_absent: bool = False,
            client_eval_embed_train_split: bool = True,
    ):
        set_seed(server_seed)
        flwr.server.Server.__init__(self, client_manager=SimpleClientManager(), strategy=self)

        def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
            # Aggregate and return custom metric (weighted average)
            if len(metrics) <= 0:
                return {}
            weights = np.asarray([num_examples for num_examples, _ in metrics], dtype=float)
            weights /= np.sum(weights)
            keys = metrics[0][1].keys()
            metrics_mat = np.asarray([[m[k] for k in keys] for _, m in metrics], dtype=float)
            avg_metrics = weights @ metrics_mat
            return {k: float(v) for k, v in zip(keys, avg_metrics)}

        self.enable_slurm: bool = enable_slurm
        self.mode: str = mode
        self.num_train_clients: int = num_train_clients

        match mode:
            case 'simulated' | 'distributed':
                num_available_clients = self.num_train_clients
            case 'multiplex':
                num_available_clients = num_step_clients
            case _:
                num_available_clients = int(1e9)
        self.num_available_clients: int = num_available_clients

        flwr.server.strategy.FedAvg.__init__(
            self,
            fraction_fit=1e-9,  # Sample % of available clients for training
            fraction_evaluate=1e-9,  # Sample % of available clients for evaluation
            min_fit_clients=num_step_clients,
            min_evaluate_clients=num_step_clients,
            min_available_clients=self.num_available_clients,
            evaluate_fn=None,
            on_fit_config_fn=None,
            on_evaluate_config_fn=None,
            accept_failures=False,
            initial_parameters=None,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        self.init_round: int = init_round
        self.eval_interval: int = eval_interval
        self.eval_test: bool = eval_test
        self.save_interval: int = save_interval
        self.log_dir: str = log_dir
        self.experiment_name: str = experiment_name if experiment_name is not None else ''
        self.client_dataset_seed: int = client_dataset_seed
        self.client_dataset_data_name: str = client_dataset_data_name
        self.client_dataset_data_path: str = client_dataset_data_path
        self.client_dataset_num_clients: int = client_dataset_num_clients
        self.client_dataset_batch_size: int = client_dataset_batch_size
        self.client_dataset_partition_type: str = client_dataset_partition_type
        self.client_dataset_alpha_train: float = client_dataset_alpha_train
        self.client_dataset_alpha_test: float = client_dataset_alpha_test
        self.client_model_num_kernels: int = client_model_num_kernels
        self.model_embed_type: str = model_embed_type
        self.client_model_embed_dim: int = client_model_embed_dim
        self.client_model_embed_y: bool = client_model_embed_model_y
        self.model_hyper_hid_layers: int = model_hyper_hid_layers
        self.model_hyper_hid_dim: int = model_hyper_hid_dim
        self.client_model_target_type: str = client_model_target_type
        self.client_model_target_head_layers: int = client_model_target_head_layers
        self.client_optimizer_target_lr: float = client_optimizer_target_lr
        self.client_optimizer_target_momentum: float = client_optimizer_target_momentum
        self.client_optimizer_target_weight_decay: float = client_optimizer_target_weight_decay
        self.client_target_gradient_mode: bool = client_target_gradient_mode
        self.client_target_num_batches: int = client_target_num_batches
        self.client_embed_num_batches: int = client_embed_num_batches
        # self.client_eval_mask_absent: bool = client_eval_mask_absent
        self.client_eval_embed_train_split: bool = client_eval_embed_train_split
        self.server_device: torch.device = torch.device(f'cuda:{torch.cuda.device_count() - 1}') \
            if torch.cuda.is_available() else torch.device('cpu')
        self.parameters_tensor: Optional[Dict[str, torch.nn.Parameter]] = None
        self.optimizer_net: Optional[torch.optim.Optimizer] = None
        self.hnet: Optional[torch.nn.Module] = None
        self.optimizer_hnet: Optional[torch.optim.Optimizer] = None
        self.hnet_grads: Optional[queue.Queue] = None

        if model_embed_type != 'none':
            self.hnet_grads: Optional[queue.Queue] = queue.Queue()

            match optimizer_hyper_type:
                case 'adam':
                    self.optimizer_hnet: Optional[torch.optim.Optimizer] = torch.optim.Adam(
                        params=[torch.nn.Parameter(torch.empty(0, device=self.server_device))],
                        lr=optimizer_hyper_lr,
                        weight_decay=optimizer_hyper_weight_decay,
                    )
                case 'sgd':
                    self.optimizer_hnet: Optional[torch.optim.Optimizer] = torch.optim.SGD(
                        params=[torch.nn.Parameter(torch.empty(0, device=self.server_device))],
                        lr=optimizer_hyper_lr,
                        momentum=optimizer_hyper_momentum,
                        weight_decay=optimizer_hyper_weight_decay,
                    )
                case _:
                    raise NotImplementedError

            match optimizer_embed_type:
                case 'adam':
                    self.optimizer_net: Optional[torch.optim.Optimizer] = torch.optim.Adam(
                        params=[torch.nn.Parameter(torch.empty(0, device=self.server_device))],
                        lr=optimizer_embed_lr,
                        weight_decay=optimizer_embed_weight_decay,
                    )
                case 'sgd':
                    self.optimizer_net: Optional[torch.optim.Optimizer] = torch.optim.SGD(
                        params=[torch.nn.Parameter(torch.empty(0, device=self.server_device))],
                        lr=optimizer_embed_lr,
                        momentum=optimizer_embed_momentum,
                        weight_decay=optimizer_embed_weight_decay,
                    )
                case _:
                    raise NotImplementedError

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()
        client_manager = self.client_manager()

        log(INFO, 'Waiting until all clients are ready')
        while client_manager.num_available() < self.min_available_clients:
            pass

        log(INFO, "Initializing clients")
        self.init_clients(timeout)

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self.initialize_parameters(client_manager=client_manager)

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        if self.eval_interval > 0 and self.init_round <= num_rounds:
            self.evaluate_round(self.init_round, timeout)
        for server_round in range(1 + self.init_round, 1 + num_rounds):
            self.fit_round(server_round, timeout)
            if self.save_interval > 0 and server_round % self.save_interval == 0:
                self.save_model(self.parameters, server_round)  # save model checkpoint
            if self.eval_interval > 0 and server_round % self.eval_interval == 0:
                self.evaluate_round(server_round, timeout)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def init_clients(self, timeout: Optional[float] = None):
        conf = {
            'num_train_clients': self.num_train_clients,
            'client_dataset_seed': self.client_dataset_seed,
            'client_dataset_data_name': self.client_dataset_data_name,
            'client_dataset_data_path': self.client_dataset_data_path,
            'client_dataset_num_clients': self.client_dataset_num_clients,
            'client_dataset_batch_size': self.client_dataset_batch_size,
            'client_dataset_partition_type': self.client_dataset_partition_type,
            'client_dataset_alpha_train': self.client_dataset_alpha_train,
            'client_dataset_alpha_test': self.client_dataset_alpha_test,
            'client_model_num_kernels': self.client_model_num_kernels,
            'model_embed_type': self.model_embed_type,
            'client_model_embed_dim': self.client_model_embed_dim,
            'client_model_embed_y': self.client_model_embed_y,
            'client_model_target_type': self.client_model_target_type,
            'client_model_target_head_layers': self.client_model_target_head_layers,
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            submitted_fs = {
                executor.submit(client.get_properties, GetPropertiesIns(conf), timeout)
                for client in self.client_manager().all().values()
            }
            finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=timeout)
        for future in finished_fs:
            assert future.exception() is None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=None)
        client_parameters = get_parameters_res.parameters
        log(INFO, "Received initial parameters from one random client")
        state_dicts = ndarrays_to_state_dicts(parameters_to_ndarrays(client_parameters), self.server_device)
        for name, sd in state_dicts.items():
            log(INFO, f"num_parameters_{name}: {sum([v.numel() for v in sd.values()])}")
        # count = sum([v.numel() for sd in state_dicts.values() for v in sd.values()])

        if self.model_embed_type != 'none':
            self.hnet = Hyper(
                example_state_dict=state_dicts['tnet'],
                embedding_dim=len(list(state_dicts['enet'].values())[-1]),  # TODO
                hidden_dim=self.model_hyper_hid_dim,
                n_hidden=self.model_hyper_hid_layers,
                spec_norm=False,
            ).to(device=self.server_device)
            self.optimizer_hnet.add_param_group({'params': self.hnet.parameters()})
            log(INFO, f'num_parameters_hnet: {count_parameters(self.hnet)}')

            self.parameters_tensor = {
                k: torch.nn.Parameter(v) for k, v in state_dicts['enet'].items()
            }
            self.optimizer_net.add_param_group({'params': self.parameters_tensor.values()})

        net_name = 'tnet' if self.model_embed_type == 'none' else 'enet'

        if self.init_round:
            log(INFO, "Using initial parameters provided by strategy")
            state_dicts[net_name] = torch.load(os.path.join(
                self.log_dir, self.experiment_name, 'checkpoints', f'model_{net_name}_round_{self.init_round}.pth'
            ), map_location=self.server_device)
            # state_dicts[net_name] = get_pefll_checkpoint()[f'{net_name}_state_dict']

            if self.model_embed_type != 'none':
                state_dict_hnet = torch.load(os.path.join(
                    self.log_dir, self.experiment_name, 'checkpoints', f'model_{"hnet"}_round_{self.init_round}.pth'
                ), map_location=self.server_device)
                # state_dict_hnet = get_pefll_checkpoint()['hnet_state_dict']
                self.hnet.load_state_dict(state_dict_hnet, strict=True)

                path = os.path.join(self.log_dir, self.experiment_name, 'checkpoints',
                                    f'optimizer_{"enet"}_round_{self.init_round}.pth')
                if os.path.exists(path):
                    self.optimizer_net.load_state_dict(torch.load(path, map_location=self.server_device))
                path = os.path.join(self.log_dir, self.experiment_name, 'checkpoints',
                                    f'optimizer_{"hnet"}_round_{self.init_round}.pth')
                if os.path.exists(path):
                    self.optimizer_hnet.load_state_dict(torch.load(path, map_location=self.server_device))

        parameters = ndarrays_to_parameters(state_dicts_to_ndarrays({net_name: state_dicts[net_name]}))
        return parameters

    def save_model(self, parameters: Parameters, server_round: int):
        # Save the model
        log(INFO, f'Saving round {server_round} aggregated_parameters...')
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)
        state_dicts = ndarrays_to_state_dicts(parameters_to_ndarrays(parameters), self.server_device)
        for net_name, state_dict in state_dicts.items():
            torch.save(
                state_dict,
                os.path.join(self.log_dir, 'checkpoints', f'model_{net_name}_round_{server_round}.pth'),
            )
        if self.hnet is not None:
            torch.save(
                self.hnet.state_dict(),
                os.path.join(self.log_dir, 'checkpoints', f'model_{"hnet"}_round_{server_round}.pth'),
            )
        if self.optimizer_net is not None:
            torch.save(
                self.optimizer_net.state_dict(),
                os.path.join(self.log_dir, 'checkpoints', f'optimizer_{"enet"}_round_{server_round}.pth'),
            )
        if self.optimizer_hnet is not None:
            torch.save(
                self.optimizer_hnet.state_dict(),
                os.path.join(self.log_dir, 'checkpoints', f'optimizer_{"hnet"}_round_{server_round}.pth'),
            )

    def sample_clients(
            self,
            cids: Optional[List[int]] = None,
    ) -> Tuple[List[ClientProxy], List[int], List[str]]:
        client_manager = self.client_manager()
        if cids is None:
            sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        else:
            sample_size = min_num_clients = len(cids)
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        all_cids = sorted(list(client_manager.all().keys()))

        match self.mode:
            case 'multiplex':
                if cids is None:
                    cids = np.random.choice(
                        np.arange(self.num_train_clients),
                        len(clients),
                        replace=False
                    ).tolist()
            case 'simulated':
                cids = [int(client.cid) for client in clients]
            case 'distributed':
                cids = [all_cids.index(client.cid) for client in clients]
            case _:
                cids = [None] * len(clients)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            if self.mode == 'simulated' or self.enable_slurm:
                devices = ['cuda'] * len(clients)
            else:
                devices = [f'cuda:{all_cids.index(client.cid) % num_gpus}' for client in clients]
        else:
            devices = ['cpu'] * len(clients)

        return clients, cids, devices

    def fit_round(self, server_round: int, timeout: Optional[float] = None) -> Optional[Dict[str, Scalar]]:
        """Perform a single round of federated averaging."""
        start_time = timeit.default_timer()
        time_metrics = {}

        if self.model_embed_type != 'none':
            self.hnet.train()
            client_fn = self.fit_client_pefll
        else:
            client_fn = self.fit_client_fedavg

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            submitted_fs = {
                executor.submit(client_fn, client, cid, device)
                for client, cid, device in zip(*self.sample_clients())
            }

            if self.model_embed_type != 'none':
                hnet_grads = [self.hnet_grads.get(block=True, timeout=timeout) for _ in range(len(submitted_fs))]
                time_4s_start = timeit.default_timer()
                hnet_grad = aggregate_tensor(hnet_grads)
                self.optimizer_hnet.zero_grad()
                for p, g in zip(self.hnet.parameters(), hnet_grad):
                    p.grad = g
                torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), 50.)
                self.optimizer_hnet.step()
                time_4s_end = timeit.default_timer()
                time_metrics['time_server_4s2'] = time_4s_end - time_4s_start

            # Handled in the respective communication stack
            finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=timeout)

        time_5s_start = timeit.default_timer()

        weights, metrics, state_dicts = [], [], []
        for future in finished_fs:
            """Convert finished future into either a result or a failure."""
            # Check if there was an exception
            assert future.exception() is None
            # Successfully received a result from a client
            w, m, sd = future.result()
            weights.append(w)
            metrics.append(m)
            state_dicts.append(sd)

        state_dict = {
            kg: vg for kg, vg in zip(
                state_dicts[0].keys(),
                aggregate_tensor([
                    ([sd[k] for k in state_dicts[0].keys()], w) for sd, w in zip(state_dicts, weights)
                ])
            )
        }

        if self.model_embed_type != 'none':
            self.optimizer_net.zero_grad()
            for k, g in state_dict.items():
                self.parameters_tensor[k].grad = g
            torch.nn.utils.clip_grad_norm_(self.parameters_tensor.values(), 50.)
            self.optimizer_net.step()
            self.parameters = ndarrays_to_parameters(state_dicts_to_ndarrays({'enet': self.parameters_tensor}))

        else:
            self.parameters = ndarrays_to_parameters(state_dicts_to_ndarrays({'tnet': state_dict}))

        end_time = timeit.default_timer()
        time_metrics['time_server_5s2'] = end_time - time_5s_start
        time_metrics['time_server_step'] = end_time - start_time

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {'step': server_round, **time_metrics}
        # if self.fit_metrics_aggregation_fn:
        metrics_aggregated = {**metrics_aggregated, **self.fit_metrics_aggregation_fn(
            [(w, mc) for w, mc in zip(weights, metrics)]
        )}
        # elif server_round == 1:  # Only log this warning once
        #     log(WARNING, "No fit_metrics_aggregation_fn provided")
        log(INFO, f'fit_round_{server_round}: {metrics_aggregated}')
        if wandb.run is not None:
            wandb.log({'fit': metrics_aggregated}, commit=False, step=server_round)
        return metrics_aggregated

    def fit_client_pefll(
            self,
            client: ClientProxy,
            cid: int,
            device: str,
    ):
        """Refine parameters on a single client."""
        time_start = timeit.default_timer()
        metrics = {}
        res = client.fit(ins=FitIns(self.parameters, {
            'cid': cid,
            'device': device,
            'stage': 3,
            'is_eval': False,
            'client_embed_num_batches': self.client_embed_num_batches,
            'client_model_target_type': self.client_model_target_type,
        }), timeout=None)
        time_3s_start = timeit.default_timer()
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}

        embedding = torch.nn.Parameter(torch.tensor(
            parameters_to_ndarrays(res.parameters)[0],
            device=self.server_device,
        ))
        tnet_state_dict = self.hnet(embedding)
        parameters = ndarrays_to_parameters(state_dicts_to_ndarrays({'tnet': tnet_state_dict}))

        time_3s_end = timeit.default_timer()
        res = client.fit(FitIns(parameters, {
            'cid': cid,
            'device': device,
            'stage': 4,
            'is_eval': False,
            'client_target_gradient_mode': self.client_target_gradient_mode,
            'client_optimizer_target_lr': self.client_optimizer_target_lr,
            'client_optimizer_target_momentum': self.client_optimizer_target_momentum,
            'client_optimizer_target_weight_decay': self.client_optimizer_target_weight_decay,
            'client_target_num_batches': self.client_target_num_batches,
            'client_model_target_type': self.client_model_target_type,
        }), timeout=None)
        time_4s_start = timeit.default_timer()
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}
        weight = res.num_examples

        tnet_state_dict_4 = ndarrays_to_state_dicts(parameters_to_ndarrays(res.parameters), self.server_device)['tnet']
        loss = torch.stack([
            ((tnet_state_dict[k].detach() - tnet_state_dict_4[k]) * tnet_state_dict[k]).sum()
            for k in tnet_state_dict.keys()
        ]).sum()
        metrics = {**metrics, 'loss_h': loss.item()}

        grads = torch.autograd.grad(loss, [embedding] + list(self.hnet.parameters()))

        self.hnet_grads.put((grads[1:], weight), block=True, timeout=None)
        parameters = ndarrays_to_parameters([grads[0].detach().cpu().numpy()])

        time_4s_end = timeit.default_timer()
        res = client.fit(FitIns(parameters, {
            'cid': cid,
            'device': device,
            'stage': 5,
            'is_eval': False,
            # 'model_target_type': self.model_target_type,
        }), timeout=None)
        time_5s_start = timeit.default_timer()
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}

        # Convert results
        enet_grad_dict = ndarrays_to_state_dicts(parameters_to_ndarrays(res.parameters), self.server_device)['enet']

        time_5s_end = timeit.default_timer()
        time_metrics = {
            'time_server_3c': time_3s_start - time_start,
            'time_server_3s': time_3s_end - time_3s_start,
            'time_server_4c': time_4s_start - time_3s_end,
            'time_server_4s1': time_4s_end - time_4s_start,
            'time_server_5c': time_5s_start - time_4s_end,
            'time_server_5s1': time_5s_end - time_5s_start,
        }
        metrics = {**metrics, **time_metrics}
        return weight, metrics, enet_grad_dict

    def fit_client_fedavg(
            self,
            client: ClientProxy,
            cid: int,
            device: str,
    ):
        """Refine parameters on a single client."""
        time_start = timeit.default_timer()
        res = client.fit(ins=FitIns(self.parameters, {
            'cid': cid,
            'device': device,
            'stage': 1,
            'is_eval': False,
            'client_target_gradient_mode': self.client_target_gradient_mode,
            'client_optimizer_target_lr': self.client_optimizer_target_lr,
            'client_optimizer_target_momentum': self.client_optimizer_target_momentum,
            'client_optimizer_target_weight_decay': self.client_optimizer_target_weight_decay,
            'client_target_num_batches': self.client_target_num_batches,
            'client_model_target_type': self.client_model_target_type,
        }), timeout=None)
        time_1s_start = timeit.default_timer()
        assert res.status.code == Code.OK
        metrics = res.metrics
        weight = res.num_examples
        state_dict = ndarrays_to_state_dicts(parameters_to_ndarrays(res.parameters), self.server_device)['tnet']
        time_1s_end = timeit.default_timer()
        time_metrics = {
            'time_server_1c': time_1s_start - time_start,
            'time_server_1s1': time_1s_end - time_1s_start,
        }
        metrics = {**metrics, **time_metrics}
        return weight, metrics, state_dict

    def evaluate_round(self, server_round: int, timeout: Optional[float] = None) -> Optional[Dict[str, Scalar]]:
        """Validate current global model on a number of clients."""
        start_time = timeit.default_timer()

        if self.model_embed_type != 'none':
            self.hnet.eval()
            client_fn = self.evaluate_client_pefll
        else:
            client_fn = self.evaluate_client_fedavg

        weights, metrics = [], []
        for i in range(0, self.num_train_clients, self.num_available_clients):
            cids = list(range(i, min(i + self.num_available_clients, self.num_train_clients)))
            # Evaluate model on a sample of available clients
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                submitted_fs = {
                    executor.submit(client_fn, client, cid, device)
                    for client, cid, device in zip(*self.sample_clients(cids))
                }
                # Handled in the respective communication stack
                finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=timeout)

            for future in finished_fs:
                """Convert finished future into either a result or a failure."""
                # Check if there was an exception
                assert future.exception() is None
                # Successfully received a result from a client
                w, m = future.result()
                weights.append(w)
                metrics.append(m)

        end_time = timeit.default_timer()

        metrics_aggregated = {'step': server_round, 'time_server_step': end_time - start_time}
        metrics_aggregated = {**metrics_aggregated, **self.evaluate_metrics_aggregation_fn(
            [(w, m) for w, m in zip(weights, metrics)]
        )}
        log(INFO, f'eval_round_{server_round}: {metrics_aggregated}')
        if wandb.run is not None:
            wandb.log({'eval': metrics_aggregated}, commit=True, step=server_round)
        return metrics_aggregated

    def evaluate_client_pefll(
            self,
            client: ClientProxy,
            cid: int,
            device: str,
    ):
        """Refine parameters on a single client."""
        time_start = timeit.default_timer()
        metrics = {}
        res = client.fit(ins=FitIns(self.parameters, {
            'cid': cid,
            'device': device,
            'stage': 6,
            'is_eval': True + self.eval_test,
            'client_eval_embed_train_split': self.client_eval_embed_train_split,
            'client_model_target_type': self.client_model_target_type,
        }), timeout=None)
        time_6s_start = timeit.default_timer()
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}

        embedding = torch.nn.Parameter(torch.tensor(
            parameters_to_ndarrays(res.parameters)[0],
            device=self.server_device,
        ))
        with torch.no_grad():
            tnet_state_dict = self.hnet(embedding)
        parameters = ndarrays_to_parameters(state_dicts_to_ndarrays({'tnet': tnet_state_dict}))

        time_6s_end = timeit.default_timer()
        res = client.fit(FitIns(parameters, {
            'cid': cid,
            'device': device,
            'stage': 7,
            'is_eval': True + self.eval_test,
            'client_target_gradient_mode': self.client_target_gradient_mode,
            # 'client_eval_mask_absent': self.client_eval_mask_absent,
            'client_model_target_type': self.client_model_target_type,
        }), timeout=None)
        time_7s_start = timeit.default_timer()
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}
        weight = res.num_examples
        time_7s_end = timeit.default_timer()
        time_metrics = {
            'time_server_6c': time_6s_start - time_start,
            'time_server_6s': time_6s_end - time_6s_start,
            'time_server_7c': time_7s_start - time_6s_end,
            # 'time_server_7s1': time_7s_end - time_7s_start,
        }
        metrics = {**metrics, **time_metrics}
        return weight, metrics

    def evaluate_client_fedavg(
            self,
            client: ClientProxy,
            cid: int,
            device: str,
    ):
        """Refine parameters on a single client."""
        time_start = timeit.default_timer()
        res = client.fit(FitIns(self.parameters, {
            'cid': cid,
            'device': device,
            'stage': 2,
            'is_eval': True + self.eval_test,
            'client_target_gradient_mode': self.client_target_gradient_mode,
            'client_model_target_type': self.client_model_target_type,
        }), timeout=None)
        time_2s_start = timeit.default_timer()
        assert res.status.code == Code.OK
        metrics = res.metrics
        weight = res.num_examples
        time_2s_end = timeit.default_timer()
        time_metrics = {
            'time_server_2c': time_2s_start - time_start,
            # 'time_server_2s1': time_2s_end - time_2s_start,
        }
        metrics = {**metrics, **time_metrics}
        return weight, metrics


def make_server(args: argparse.Namespace):
    return FlowerServer(
        enable_slurm=args.enable_slurm and detect_slurm(),
        mode=args.mode,
        num_train_clients=args.num_train_clients,
        num_step_clients=args.num_step_clients,
        init_round=args.init_round,
        eval_interval=args.eval_interval,
        eval_test=args.eval_test,
        save_interval=args.save_interval,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        server_seed=args.server_seed,
        client_dataset_seed=args.client_dataset_seed,
        client_dataset_data_name=args.client_dataset_data_name,
        client_dataset_data_path=args.client_dataset_data_path,
        client_dataset_num_clients=args.client_dataset_num_clients,
        client_dataset_batch_size=args.client_dataset_batch_size,
        client_dataset_partition_type=args.client_dataset_partition_type,
        client_dataset_alpha_train=args.client_dataset_alpha_train,
        client_dataset_alpha_test=args.client_dataset_alpha_test,
        client_model_num_kernels=args.client_model_num_kernels,
        model_embed_type=args.model_embed_type,
        client_model_embed_dim=args.client_model_embed_dim,
        client_model_embed_model_y=args.client_model_embed_y,
        model_hyper_hid_layers=args.model_hyper_hid_layers,
        model_hyper_hid_dim=args.model_hyper_hid_dim,
        client_model_target_type=args.client_model_target_type,
        client_model_target_head_layers=args.client_model_target_head_layers,
        client_optimizer_target_lr=args.client_optimizer_target_lr,
        client_optimizer_target_weight_decay=args.client_optimizer_target_weight_decay,
        client_optimizer_target_momentum=args.client_optimizer_target_momentum,
        client_target_gradient_mode=args.client_target_gradient_mode,
        client_target_num_batches=args.client_target_num_batches,
        optimizer_embed_type=args.optimizer_embed_type,
        optimizer_embed_lr=args.optimizer_embed_lr,
        optimizer_embed_weight_decay=args.optimizer_embed_weight_decay,
        optimizer_embed_momentum=args.optimizer_embed_momentum,
        client_embed_num_batches=args.client_embed_num_batches,
        optimizer_hyper_type=args.optimizer_hyper_type,
        optimizer_hyper_lr=args.optimizer_hyper_lr,
        optimizer_hyper_weight_decay=args.optimizer_hyper_weight_decay,
        optimizer_hyper_momentum=args.optimizer_hyper_momentum,
        # client_eval_mask_absent=args.client_eval_mask_absent,
        client_eval_embed_train_split=args.client_eval_embed_train_split,
    )


def main():
    args = parse_args()
    if args.enable_slurm and detect_slurm():
        init_wandb(
            args=args,
            experiment_name=f"{os.environ['SLURM_JOB_ID']}-S{os.environ['SLURM_NODEID']}-{os.environ['SLURMD_NODENAME']}",
            group=os.environ['SLURM_JOB_ID'],
        )
    else:
        init_wandb(
            args=args,
            experiment_name=args.experiment_name,
            group=None,
        )

    server = make_server(args)
    flwr.server.start_server(
        server_address=f"0.0.0.0:{args.server_address.split(':')[-1]}",
        server=server,
        config=flwr.server.ServerConfig(num_rounds=args.num_rounds),
    )
    finish_wandb()


if __name__ == '__main__':
    main()
