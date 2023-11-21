import argparse
import concurrent.futures
import timeit
from logging import DEBUG, INFO, WARNING
import os
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

from PeFLL.models import CNNHyper

from parse_args import parse_args
from utils import aggregate_tensor, ndarrays_to_state_dicts, state_dicts_to_ndarrays, init_wandb, finish_wandb, \
    get_pefll_checkpoint


class FlowerServer(flwr.server.Server, flwr.server.strategy.FedAvg):
    def __init__(
            self,
            mode: str,
            num_train_clients: int,
            num_step_clients: int,
            init_round: int = 0,
            eval_interval: int = 10,
            eval_test: bool = False,
            log_dir: str = './outputs',
            client_dataset_seed: int = 42,
            client_dataset_data_name: str = 'cifar10',
            client_dataset_data_path: str = './dataset',
            client_dataset_num_clients: int = 100,
            client_dataset_batch_size: int = 32,
            client_dataset_partition_type: str = 'by_class',
            client_dataset_alpha_train: float = .1,
            client_dataset_alpha_test: float = .1,
            model_num_kernels: int = 16,
            model_embed_type: str = 'none',
            model_embed_dim: int = -1,
            client_model_embed_model_y: bool = True,
            model_hyper_hid_layers: int = 3,
            model_hyper_hid_dim: int = 100,
            client_optimizer_target_lr: float = 2e-3,
            client_optimizer_target_momentum: float = .9,
            client_optimizer_target_weight_decay: float = 5e-5,
            client_target_num_batches: int = 50,
            optimizer_embed_type: str = 'adam',
            optimizer_embed_lr: float = 2e-4,
            optimizer_embed_weight_decay: float = 1e-3,
            client_embed_num_batches: int = 1,
            optimizer_hyper_type: str = 'adam',
            optimizer_hyper_lr: float = 2e-4,
            optimizer_hyper_weight_decay: float = 1e-3,
            client_eval_mask_absent: bool = False,
            client_eval_embed_train_split: bool = True,
    ):
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
        self.log_dir: str = log_dir
        self.client_dataset_seed: int = client_dataset_seed
        self.client_dataset_data_name: str = client_dataset_data_name
        self.client_dataset_data_path: str = client_dataset_data_path
        self.client_dataset_num_clients: int = client_dataset_num_clients
        self.client_dataset_batch_size: int = client_dataset_batch_size
        self.client_dataset_partition_type: str = client_dataset_partition_type
        self.client_dataset_alpha_train: float = client_dataset_alpha_train
        self.client_dataset_alpha_test: float = client_dataset_alpha_test
        self.model_num_kernels: int = model_num_kernels
        self.model_embed_type: str = model_embed_type
        self.model_embed_dim: int = model_embed_dim
        self.client_model_embed_y: bool = client_model_embed_model_y
        self.model_hyper_hid_layers: int = model_hyper_hid_layers
        self.model_hyper_hid_dim: int = model_hyper_hid_dim
        self.client_optimizer_target_lr: float = client_optimizer_target_lr
        self.client_optimizer_target_momentum: float = client_optimizer_target_momentum
        self.client_optimizer_target_weight_decay: float = client_optimizer_target_weight_decay
        self.client_target_num_batches: int = client_target_num_batches
        self.client_embed_num_batches: int = client_embed_num_batches
        self.client_eval_mask_absent: bool = client_eval_mask_absent
        self.client_eval_embed_train_split: bool = client_eval_embed_train_split
        self.server_device: torch.device = torch.device(f'cuda:{torch.cuda.device_count() - 1}') \
            if torch.cuda.is_available() else torch.device('cpu')
        self.stage_memory: Dict = {}
        self.parameters_tensor: List[torch.nn.Parameter] = []
        self.optimizer_net: Optional[torch.optim.Optimizer] = None

        if model_embed_type != 'none':
            if client_dataset_data_name == 'femnist':
                client_dataset_num_clients = 3597
            if model_embed_dim == -1:
                model_embed_dim = 1 + client_dataset_num_clients // 4
            self.hnet: torch.nn.Module = CNNHyper(
                n_nodes=client_dataset_num_clients,
                embedding_dim=model_embed_dim,
                in_channels=3,
                out_dim=10,
                n_kernels=model_num_kernels,
                hidden_dim=model_hyper_hid_dim,
                spec_norm=False,
                n_hidden=model_hyper_hid_layers,
            ).to(device=self.server_device)

            match optimizer_hyper_type:
                case 'adam':
                    self.optimizer_hnet: Optional[torch.optim.Optimizer] = torch.optim.Adam(
                        self.hnet.parameters(),
                        lr=optimizer_hyper_lr,
                        weight_decay=0.,
                    )
                case 'sgd':
                    self.optimizer_hnet: Optional[torch.optim.Optimizer] = torch.optim.SGD(
                        self.hnet.parameters(),
                        lr=optimizer_hyper_lr,
                        momentum=.9,
                        weight_decay=optimizer_hyper_weight_decay,
                    )
                case _:
                    raise NotImplementedError
            match optimizer_embed_type:
                case 'adam':
                    self.optimizer_net: Optional[torch.optim.Optimizer] = torch.optim.Adam(
                        params=[torch.nn.Parameter(torch.empty(0, device=self.server_device))],
                        lr=optimizer_embed_lr,
                        weight_decay=0.,
                    )
                case 'sgd':
                    self.optimizer_net: Optional[torch.optim.Optimizer] = torch.optim.SGD(
                        params=[torch.nn.Parameter(torch.empty(0, device=self.server_device))],
                        lr=optimizer_embed_lr,
                        momentum=.9,
                        weight_decay=optimizer_embed_weight_decay,
                    )
                case _:
                    raise NotImplementedError
        else:
            self.hnet: Optional[torch.nn.Module] = None
            self.optimizer_hnet: Optional[torch.optim.Optimizer] = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        parameters = super().initialize_parameters(client_manager)
        net_name = 'tnet' if self.model_embed_type == 'none' else 'enet'
        if parameters is None and self.init_round:
            state_dicts = {net_name: torch.load(
                    os.path.join(self.log_dir, 'checkpoints', f'model_{net_name}_round_{self.init_round}.pth'),
                    map_location=torch.device('cpu'),
            )}
            # state_dicts = {net_name: get_pefll_checkpoint()[f'{net_name}_state_dict']}
            parameters = ndarrays_to_parameters(state_dicts_to_ndarrays(state_dicts))
            if self.hnet is not None:
                state_dict = torch.load(
                    os.path.join(self.log_dir, 'checkpoints', f'model_{"hnet"}_round_{self.init_round}.pth'),
                    map_location=torch.device('cpu'),
                )
                # state_dict = get_pefll_checkpoint()['hnet_state_dict']
                self.hnet.load_state_dict(state_dict, strict=True)
        if parameters is None:
            # Get initial parameters from one of the clients
            log(INFO, "Requesting initial parameters from one random client")
            random_client = client_manager.sample(1)[0]
            ins = GetParametersIns(config={'net_name': net_name})
            get_parameters_res = random_client.get_parameters(ins=ins, timeout=None)
            parameters = get_parameters_res.parameters
            log(INFO, "Received initial parameters from one random client")
        else:
            log(INFO, "Using initial parameters provided by strategy")

        if self.model_embed_type != 'none':
            state_dicts = ndarrays_to_state_dicts(parameters_to_ndarrays(parameters))
            self.parameters_tensor = [
                torch.nn.Parameter(v.to(device=self.server_device)) for v in state_dicts[net_name].values()
            ]
            self.optimizer_net.add_param_group({'params': self.parameters_tensor})

        return parameters

    def save_model(self, parameters: Parameters, server_round: int):
        # Save the model
        log(INFO, f'Saving round {server_round} aggregated_parameters...')
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)
        state_dicts = ndarrays_to_state_dicts(parameters_to_ndarrays(parameters))
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

    def configure_get_properties(self, client_manager: ClientManager) -> List[Tuple[ClientProxy, GetPropertiesIns]]:
        client_config_pairs: List[Tuple[ClientProxy, GetPropertiesIns]] = []
        for i, (cid, client) in enumerate(client_manager.all().items()):
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
                'model_num_kernels': self.model_num_kernels,
                'model_embed_type': self.model_embed_type,
                'model_embed_dim': self.model_embed_dim,
                'client_model_embed_y': self.client_model_embed_y,
            }
            client_config_pairs.append((client, GetPropertiesIns(conf)))
        return client_config_pairs

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
            if self.mode == 'simulated':
                devices = ['cuda'] * len(clients)
            else:
                devices = [f'cuda:{all_cids.index(client.cid) % num_gpus}' for client in clients]
        else:
            devices = ['cpu'] * len(clients)

        return clients, cids, devices

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()
        client_manager = self.client_manager()

        log(INFO, 'Waiting until all clients are ready')
        while client_manager.num_available() < self.min_available_clients:
            pass

        log(INFO, "Initializing clients")
        get_properties_instructions = self.configure_get_properties(client_manager)
        _, get_properties_failures = self.execute_round(
            fn_name='get_properties',
            client_instructions=get_properties_instructions,
            server_round=0,
            timeout=timeout,
        )
        assert not get_properties_failures

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self.initialize_parameters(client_manager=client_manager)

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        self.evaluate_round(0, timeout)
        for server_round in range(1 + self.init_round, 1 + num_rounds):
            self.fit_round(server_round, timeout)
            if server_round % self.eval_interval:
                continue
            self.save_model(self.parameters, server_round)  # save model checkpoint
            self.evaluate_round(server_round, timeout)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def fit_round(self, server_round: int, cids: List[int] = None) -> Optional[Dict[str, Scalar]]:
        """Perform a single round of federated averaging."""
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
            # Handled in the respective communication stack
            finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=None)

        if self.model_embed_type != 'none':
            weights, metrics, hnet_grads, enet_grads = [], [], [], []
            for future in finished_fs:
                """Convert finished future into either a result or a failure."""
                # Check if there was an exception
                assert future.exception() is None
                # Successfully received a result from a client
                w, m, hg, eg = future.result()
                weights.append(w)
                metrics.append(m)
                hnet_grads.append((hg, w))
                enet_grads.append((eg, w))

            hnet_grad = aggregate_tensor(hnet_grads)
            enet_grad = aggregate_tensor(enet_grads)

            self.optimizer_hnet.zero_grad()
            for p, g in zip(self.hnet.parameters(), hnet_grad):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), 50.)
            self.optimizer_hnet.step()

            self.optimizer_net.zero_grad()
            for p, g in zip(self.parameters_tensor, enet_grad):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(self.parameters_tensor, 50.)
            self.optimizer_net.step()

            ndarrays = parameters_to_ndarrays(self.parameters)
            header_and_keys = ndarrays[:1+len(ndarrays[0])]
            parameters_aggregated = ndarrays_to_parameters(
                header_and_keys + [v.detach().cpu().numpy() for v in self.parameters_tensor]
            )
            self.parameters = parameters_aggregated

        else:
            ndarrays = parameters_to_ndarrays(self.parameters)
            header_and_keys = ndarrays[:1+len(ndarrays[0])]

            weights, metrics, parameters = [], [], []
            for future in finished_fs:
                """Convert finished future into either a result or a failure."""
                # Check if there was an exception
                assert future.exception() is None
                # Successfully received a result from a client
                w, m, p = future.result()
                weights.append(w)
                metrics.append(m)
                parameters.append((parameters_to_ndarrays(p)[len(header_and_keys):], w))

            parameters_aggregated = ndarrays_to_parameters(
                header_and_keys + aggregate(parameters)
            )
            self.parameters = parameters_aggregated

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(
                [(w, mc) for w, mc in zip(weights, metrics)]
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        metrics_aggregated = {**metrics_aggregated}
        log(INFO, f'round_{server_round}: {metrics_aggregated}')
        if wandb.run is not None:
            wandb.log({'train': metrics_aggregated}, commit=False, step=server_round)
        return metrics_aggregated

    def evaluate_round(self, server_round: int, timeout: Optional[float]) -> Optional[Dict[str, Scalar]]:
        """Validate current global model on a number of clients."""
        if self.model_embed_type != 'none':
            self.hnet.eval()
            client_fn = self.eval_client_pefll
        else:
            client_fn = self.eval_client_fedavg

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
                finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=None)

            weights, metrics = [], []
            for future in finished_fs:
                """Convert finished future into either a result or a failure."""
                # Check if there was an exception
                assert future.exception() is None
                # Successfully received a result from a client
                w, m = future.result()
                weights.append(w)
                metrics.append(m)

        metrics_aggregated = self.evaluate_metrics_aggregation_fn([(w, m) for w, m in zip(weights, metrics)])
        log(INFO, f'round_{server_round}: {metrics_aggregated}')
        if wandb.run is not None:
            wandb.log({'val': metrics_aggregated}, commit=True, step=server_round)
        return metrics_aggregated

    def fit_client_pefll(
            self,
            client: ClientProxy,
            cid: int,
            device: str,
    ):
        """Refine parameters on a single client."""
        metrics = {}

        res = client.fit(ins=FitIns(self.parameters, {
            'cid': cid,
            'device': device,
            'stage': 3,
            'is_eval': False,
            'client_embed_num_batches': self.client_embed_num_batches,
        }), timeout=None)
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}

        embedding = torch.nn.Parameter(torch.tensor(
            parameters_to_ndarrays(res.parameters)[0],
            device=self.server_device,
        ))
        tnet_state_dict = self.hnet(embedding)
        parameters = ndarrays_to_parameters([
            np.asarray(['tnet']),
            np.asarray(list(tnet_state_dict.keys())),
            *[v.detach().cpu().numpy() for v in tnet_state_dict.values()],
        ])

        res = client.fit(FitIns(parameters, {
            'cid': cid,
            'device': device,
            'stage': 4,
            'is_eval': False,
            'client_optimizer_target_lr': self.client_optimizer_target_lr,
            'client_optimizer_target_momentum': self.client_optimizer_target_momentum,
            'client_optimizer_target_weight_decay': self.client_optimizer_target_weight_decay,
            'client_target_num_batches': self.client_target_num_batches,
        }), timeout=None)
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}
        weight = res.num_examples

        loss = torch.stack([
            ((o.detach() - torch.tensor(v, device=self.server_device)) * o).sum()
            for o, v in zip(tnet_state_dict.values(), parameters_to_ndarrays(res.parameters)[2:])
        ])
        loss = loss.sum()
        metrics = {**metrics, 'loss_h': float(loss)}

        grads = torch.autograd.grad(loss, [embedding] + list(self.hnet.parameters()))
        parameters = ndarrays_to_parameters([grads[0].detach().cpu().numpy()])
        hnet_grad_dict = {k: v for k, v in zip(self.hnet.state_dict().keys(), grads[1:])}

        res = client.fit(FitIns(parameters, {
            'cid': cid,
            'device': device,
            'stage': 5,
            'is_eval': False,
        }), timeout=None)
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}

        # Convert results
        ndarrays = parameters_to_ndarrays(res.parameters)
        header_and_keys = ndarrays[:1 + len(ndarrays[0])]
        enet_grad_dict = {
            k: torch.tensor(v, device=self.server_device)
            for k, v in zip(header_and_keys[1], ndarrays[len(header_and_keys):])
        }
        return weight, metrics, list(hnet_grad_dict.values()), list(enet_grad_dict.values())

    def fit_client_fedavg(
            self,
            client: ClientProxy,
            cid: int,
            device: str,
    ):
        """Refine parameters on a single client."""
        res = client.fit(ins=FitIns(self.parameters, {
            'cid': cid,
            'device': device,
            'stage': 1,
            'is_eval': False,
            'client_optimizer_target_lr': self.client_optimizer_target_lr,
            'client_optimizer_target_momentum': self.client_optimizer_target_momentum,
            'client_optimizer_target_weight_decay': self.client_optimizer_target_weight_decay,
            'client_target_num_batches': self.client_target_num_batches,
        }), timeout=None)
        assert res.status.code == Code.OK
        return res.num_examples, res.metrics, res.parameters

    def eval_client_pefll(
            self,
            client: ClientProxy,
            cid: int,
            device: str,
    ):
        """Refine parameters on a single client."""

        metrics = {}
        res = client.fit(ins=FitIns(self.parameters, {
            'cid': cid,
            'device': device,
            'stage': 6,
            'is_eval': True,
            'client_eval_embed_train_split': self.client_eval_embed_train_split,
        }), timeout=None)
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}

        embedding = torch.nn.Parameter(torch.tensor(
            parameters_to_ndarrays(res.parameters)[0],
            device=self.server_device,
        ))
        with torch.no_grad():
            tnet_state_dict = self.hnet(embedding)
        parameters = ndarrays_to_parameters([
            np.asarray(['tnet']),
            np.asarray(list(tnet_state_dict.keys())),
            *[v.detach().cpu().numpy() for v in tnet_state_dict.values()],
        ])

        res = client.fit(FitIns(parameters, {
            'cid': cid,
            'device': device,
            'stage': 7,
            'is_eval': True,
            'client_eval_mask_absent': self.client_eval_mask_absent,
        }), timeout=None)
        assert res.status.code == Code.OK
        metrics = {**metrics, **res.metrics}
        weight = res.num_examples

        return weight, metrics

    def eval_client_fedavg(
            self,
            client: ClientProxy,
            cid: int,
            device: str,
    ):
        """Refine parameters on a single client."""
        res = client.fit(FitIns(self.parameters, {
            'cid': cid,
            'device': device,
            'stage': 2,
            'is_eval': True,
        }), timeout=None)
        assert res.status.code == Code.OK
        return res.num_examples, res.metrics

    def execute_round(
            self,
            fn_name: str,
            client_instructions: List[Tuple[
                ClientProxy, EvaluateIns | FitIns | GetParametersIns | GetPropertiesIns | ReconnectIns]],
            server_round: int,
            timeout: Optional[float]
    ) -> Optional[Tuple[
        List[Tuple[ClientProxy, FitRes | EvaluateRes | GetParametersRes | GetPropertiesRes | DisconnectRes]],
        List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ]]:
        """Perform a single round of communication."""
        if not client_instructions:
            log(INFO, "execute_round %s: no clients selected, cancel", server_round)
            return None
        # log(DEBUG, "execute_round %s: strategy sampled %s clients (out of %s)",
        #     server_round, len(client_instructions), self.client_manager().num_available())
        # Collect `execute` results from all clients participating in this round
        """Refine parameters concurrently on all selected clients."""

        def client_execute_fn(
                client: ClientProxy,
                ins: EvaluateIns | FitIns | GetParametersIns | GetPropertiesIns | ReconnectIns,
                to: Optional[float]
        ) -> Tuple[ClientProxy, EvaluateRes | FitRes | GetParametersRes | GetPropertiesRes | DisconnectRes]:
            """Refine parameters on a single client."""
            fit_res = client.__getattribute__(fn_name)(ins=ins, timeout=to)
            return client, fit_res

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            submitted_fs = {
                executor.submit(client_execute_fn, client_proxy, ins, timeout)
                for client_proxy, ins in client_instructions
            }
            # Handled in the respective communication stack
            finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=None)

        # Gather results
        results: List[Tuple[ClientProxy, FitRes]] = []
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
        for future in finished_fs:
            """Convert finished future into either a result or a failure."""
            # Check if there was an exception
            failure = future.exception()
            if failure is not None:
                failures.append(failure)
            else:
                # Successfully received a result from a client
                result: Tuple[ClientProxy, FitRes] = future.result()
                _, res = result
                # Check result status code
                if res.status.code == Code.OK:
                    results.append(result)
                else:
                    # Not successful, client returned a result where the status code is not OK
                    failures.append(result)
        # log(DEBUG, "execute_round %s received %s results and %s failures", server_round, len(results), len(failures))
        assert not failures
        return results, failures


def make_server(args: argparse.Namespace):
    return FlowerServer(
        mode=args.mode,
        num_train_clients=args.num_train_clients,
        num_step_clients=args.num_step_clients,
        init_round=args.init_round,
        eval_interval=args.eval_interval,
        eval_test=args.eval_test,
        log_dir=args.log_dir,
        client_dataset_seed=args.client_dataset_seed,
        client_dataset_data_name=args.client_dataset_data_name,
        client_dataset_data_path=args.client_dataset_data_path,
        client_dataset_num_clients=args.client_dataset_num_clients,
        client_dataset_batch_size=args.client_dataset_batch_size,
        client_dataset_partition_type=args.client_dataset_partition_type,
        client_dataset_alpha_train=args.client_dataset_alpha_train,
        client_dataset_alpha_test=args.client_dataset_alpha_test,
        model_num_kernels=args.model_num_kernels,
        model_embed_type=args.model_embed_type,
        model_embed_dim=args.model_embed_dim,
        client_model_embed_model_y=args.client_model_embed_y,
        model_hyper_hid_layers=args.model_hyper_hid_layers,
        model_hyper_hid_dim=args.model_hyper_hid_dim,
        client_optimizer_target_lr=args.client_optimizer_target_lr,
        client_optimizer_target_momentum=args.client_optimizer_target_momentum,
        client_optimizer_target_weight_decay=args.client_optimizer_target_weight_decay,
        client_target_num_batches=args.client_target_num_batches,
        optimizer_embed_type=args.optimizer_embed_type,
        optimizer_embed_lr=args.optimizer_embed_lr,
        optimizer_embed_weight_decay=args.optimizer_embed_weight_decay,
        client_embed_num_batches=args.client_embed_num_batches,
        optimizer_hyper_type=args.optimizer_hyper_type,
        optimizer_hyper_lr=args.optimizer_hyper_lr,
        optimizer_hyper_weight_decay=args.optimizer_hyper_weight_decay,
        client_eval_mask_absent=args.client_eval_mask_absent,
        client_eval_embed_train_split=args.client_eval_embed_train_split,
    )


def main():
    args = parse_args()
    init_wandb(args, args.experiment_name)
    server = make_server(args)
    flwr.server.start_server(
        server_address=f"0.0.0.0:{args.server_address.split(':')[-1]}",
        server=server,
        config=flwr.server.ServerConfig(num_rounds=args.num_rounds),
    )
    finish_wandb()


if __name__ == '__main__':
    main()
