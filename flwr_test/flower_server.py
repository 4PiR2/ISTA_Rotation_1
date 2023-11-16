import argparse
import concurrent.futures
import timeit
from logging import DEBUG, INFO, WARNING
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    parameters_to_ndarrays,
    ndarrays_to_parameters,
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
from utils import init_wandb, finish_wandb, get_pefll_checkpoint


class FlowerStrategy(flwr.server.strategy.FedAvg):
    def __init__(
            self,
            mode: str,
            num_train_clients: int,
            num_step_clients: int,
            eval_interval: int = 10,
            init_round: int = 0,
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
            client_optimizer_embed_type: str = 'adam',
            client_optimizer_embed_lr: float = 2e-4,
            client_embed_num_batches: int = 1,
            optimizer_hyper_type: str = 'adam',
            optimizer_hyper_lr: float = 2e-4,
            optimizer_weight_decay: float = 1e-3,
            client_eval_mask_absent: bool = False,
            client_eval_embed_train_split: bool = True,
    ):
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

        super().__init__(
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

        self.eval_interval: int = eval_interval
        self.init_round: int = init_round
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
        self.client_optimizer_embed_type: str = client_optimizer_embed_type
        self.client_optimizer_embed_lr: float = client_optimizer_embed_lr
        self.client_embed_num_batches: int = client_embed_num_batches
        self.optimizer_hyper_type: str = optimizer_hyper_type
        self.optimizer_hyper_lr: float = optimizer_hyper_lr
        self.optimizer_weight_decay: float = optimizer_weight_decay
        self.client_eval_mask_absent: bool = client_eval_mask_absent
        self.client_eval_embed_train_split: bool = client_eval_embed_train_split
        self.stage_memory: Dict = {}

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
            )
        else:
            self.hnet: Optional[torch.nn.Module] = None

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        parameters = super().initialize_parameters(client_manager)
        net_name = 'tnet' if self.model_embed_type == 'none' else 'enet'
        if parameters is None and self.init_round:
            header = [net_name]
            keys: list[np.ndarray] = []
            vals: List[np.ndarray] = []
            for net_name in header:
                state_dict = torch.load(
                    os.path.join(self.log_dir, 'checkpoints', f'model_{net_name}_round_{self.init_round}.pth'),
                    map_location=torch.device('cpu'),
                )
                # state_dict = get_pefll_checkpoint()['enet_state_dict']
                keys.append(np.asarray(list(state_dict.keys())))
                vals.extend([val.detach().cpu().numpy() for val in state_dict.values()])
            parameters = ndarrays_to_parameters([np.asarray(header)] + keys + vals)
            if self.hnet is not None:
                state_dict = torch.load(
                    os.path.join(self.log_dir, 'checkpoints', f'model_{"hnet"}_round_{self.init_round}.pth'),
                    map_location=torch.device('cpu'),
                )
                # state_dict = get_pefll_checkpoint()['hnet_state_dict']
                self.hnet.load_state_dict(state_dict, strict=True)
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters
        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = client_manager.sample(1)[0]
        ins = GetParametersIns(config={'net_name': net_name})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=None)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters

    def save_model(self, parameters: Parameters, server_round: int):
        # Save the model
        log(INFO, f'Saving round {server_round} aggregated_parameters...')
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)
        # Convert `Parameters` to `List[np.ndarray]`
        ndarrays: List[np.ndarray] = parameters_to_ndarrays(parameters)
        # Convert `List[np.ndarray]` to PyTorch`state_dict`
        for i, net_name in enumerate(ndarrays[0]):
            keys = ndarrays[1 + i]
            vals = ndarrays[
                   1 + sum([len(k) for k in ndarrays[:1 + i]]): 1 + sum([len(k) for k in ndarrays[:2 + i]])]
            state_dict = {k: torch.tensor(v) for k, v in zip(keys, vals)}
            torch.save(
                state_dict,
                os.path.join(self.log_dir, 'checkpoints', f'model_{net_name}_round_{server_round}.pth')
            )
        if self.hnet is not None:
            torch.save(
                self.hnet.state_dict(),
                os.path.join(self.log_dir, 'checkpoints', f'model_{"hnet"}_round_{server_round}.pth')
            )

    def configure_get_properties(self, client_manager: ClientManager) -> List[Tuple[ClientProxy, GetPropertiesIns]]:
        client_config_pairs_updated: List[Tuple[ClientProxy, GetPropertiesIns]] = []
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
            client_config_pairs_updated.append((client, GetPropertiesIns(conf)))
        return client_config_pairs_updated

    def configure_fit(
            self,
            server_round: int = -1,
            parameters: Parameters | List[Parameters] | None = None,
            client_manager: ClientManager | None = None,
            stage: int = -1,
            cids: Optional[List[int]] = None,
    ) -> List[Tuple[ClientProxy, FitIns | EvaluateIns | GetPropertiesIns]]:
        match stage:
            case 1 | 2 | 3 | 6:
                # Sample clients
                sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
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
                        else:
                            clients = clients[:len(cids)]
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

                common_configs = [{
                    'cid': cid,
                    'device': device,
                } for cid, device in zip(cids, devices)]

                parameters = [parameters] * len(clients)

                match stage:
                    case 1 | 2:
                        pass
                    case 3 | 6:
                        self.stage_memory['clients'] = clients
                        self.stage_memory['common_configs'] = common_configs
                    case _:
                        raise NotImplementedError

            case 4 | 5 | 7:
                clients = self.stage_memory['clients']
                common_configs = self.stage_memory['common_configs']
            case _:
                raise NotImplementedError

        match stage:
            case 1 | 4:
                stage_config = {
                    'is_eval': False,
                    'client_optimizer_target_lr': self.client_optimizer_target_lr,
                    'client_optimizer_target_momentum': self.client_optimizer_target_momentum,
                    'client_optimizer_target_weight_decay': self.client_optimizer_target_weight_decay,
                    'client_target_num_batches': self.client_target_num_batches,
                }
            case 2 | 7:
                stage_config = {
                    'is_eval': True,
                    'client_eval_mask_absent': self.client_eval_mask_absent,
                }
            case 3:
                stage_config = {
                    'is_eval': False,
                    'client_embed_num_batches': self.client_embed_num_batches,
                }
            case 5:
                stage_config = {
                    'client_optimizer_embed_type': self.client_optimizer_embed_type,
                    'client_optimizer_embed_lr': self.client_optimizer_embed_lr,
                    'optimizer_weight_decay': self.optimizer_weight_decay,
                }
            case 6:
                stage_config = {
                    'is_eval': True,
                    'client_eval_embed_train_split': self.client_eval_embed_train_split,
                }
            case _:
                stage_config = {}

        client_config_pairs: List[Tuple[ClientProxy, FitIns | EvaluateIns]] = []
        for i, client in enumerate(clients):
            conf = {
                **common_configs[i],
                'stage': stage,
                **stage_config,
            }
            client_config_pairs.append((client, FitIns(parameters[i], conf)))
        # Return client/config pairs
        return client_config_pairs

    def configure_evaluate(
            self, server_round: int,
            parameters: Parameters | List[Parameters] | None,
            client_manager: ClientManager,
            stage: int = -1,
            cids: Optional[List[int]] = None,
    ) -> List[Tuple[ClientProxy, FitIns | EvaluateIns | GetPropertiesIns]]:
        return self.configure_fit(server_round, parameters, client_manager, stage)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            stage: int = -1,
    ) -> Tuple[Parameters | List[Parameters] | None, Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        match stage:
            case 1 | 5:
                # Convert results
                header_and_keys = [np.array([])]
                weights_results = []
                for _, fit_res in results:
                    ndarrays = parameters_to_ndarrays(fit_res.parameters)
                    header_and_keys = ndarrays[:1+len(ndarrays[0])]
                    vals = ndarrays[len(header_and_keys):]
                    weights_results.append((vals, fit_res.num_examples))
                aggregated_vals = aggregate(weights_results)
                parameters_aggregated = ndarrays_to_parameters(header_and_keys + aggregated_vals)

            case 2 | 7:
                parameters_aggregated = None

            case 3 | 4 | 6:
                # align results order with clients order
                clients: List[ClientProxy] = self.stage_memory['clients']
                results = [results[i] for i in np.argsort([client.cid for client, _ in results])]
                results = [results[j] for j in np.argsort(np.argsort([client.cid for client in clients]))]

                match stage:
                    case 3 | 6:
                        device = torch.device(f'cuda:{torch.cuda.device_count() - 1}') \
                            if torch.cuda.is_available() else torch.device('cpu')
                        self.hnet = self.hnet.to(device)
                        embeddings = torch.nn.Parameter(torch.tensor(
                            np.asarray([parameters_to_ndarrays(fit_res.parameters)[0] for _, fit_res in results]),
                            device=device,
                        ))

                        match stage:
                            case 3:
                                self.hnet.train()
                                tnet_state_dicts = [self.hnet(embedding) for embedding in embeddings]
                            case 6:
                                self.hnet.eval()
                                with torch.no_grad():
                                    tnet_state_dicts = [self.hnet(embedding) for embedding in embeddings]
                            case _:
                                raise NotImplementedError

                        self.stage_memory['embeddings'] = embeddings
                        self.stage_memory['tnet_state_dicts'] = tnet_state_dicts

                        # tnet_parameters
                        parameters_aggregated = [ndarrays_to_parameters([
                            np.asarray(['tnet']),
                            np.asarray(list(tnet_state_dict.keys())),
                            *[v.detach().cpu().numpy() for v in tnet_state_dict.values()],
                        ]) for tnet_state_dict in tnet_state_dicts]

                    case 4:
                        embeddings = self.stage_memory['embeddings']
                        tnet_state_dicts = self.stage_memory['tnet_state_dicts']
                        device = embeddings.device

                        result_values = [[
                            torch.tensor(v, device=device) for v in parameters_to_ndarrays(fit_res.parameters)[2:]
                        ] for _, fit_res in results]

                        losses = torch.stack([torch.stack([
                            ((o.detach() - v) * o).sum() for o, v in zip(state_dict.values(), result_value)
                        ], dim=0) for state_dict, result_value in zip(tnet_state_dicts, result_values)], dim=0)
                        losses = losses.sum(dim=-1)

                        weights = torch.tensor(
                            [fit_res.num_examples for _, fit_res in results],
                            dtype=losses.dtype,
                            device=device,
                        )
                        weights /= weights.sum()
                        loss = (losses * weights).sum()

                        match self.optimizer_hyper_type:
                            case 'adam':
                                optimizer = torch.optim.Adam(self.hnet.parameters(), lr=self.optimizer_hyper_lr)
                            case 'sgd':
                                optimizer = torch.optim.SGD(
                                    self.hnet.parameters(),
                                    lr=self.optimizer_hyper_lr,
                                    momentum=.9,
                                    weight_decay=self.optimizer_weight_decay,
                                )
                            case _:
                                raise NotImplementedError

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), 50.)
                        optimizer.step()

                        parameters_aggregated = [
                            ndarrays_to_parameters([grad]) for grad in embeddings.grad.detach().cpu().numpy()
                        ]

                    case _:
                        raise NotImplementedError

            case _:
                raise NotImplementedError

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        log(INFO, f'round_{server_round}_stage_{stage}: {metrics_aggregated}')
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        raise NotImplementedError


class FlowerServer(flwr.server.Server):
    def __init__(self, client_manager: ClientManager, strategy: FlowerStrategy):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.strategy: FlowerStrategy = self.strategy

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()
        client_manager = self.client_manager()

        log(INFO, 'Waiting until all clients are ready')
        while client_manager.num_available() < self.strategy.min_available_clients:
            pass

        log(INFO, "Initializing clients")
        get_properties_instructions = self.strategy.configure_get_properties(client_manager)
        _, get_properties_failures = self.execute_round(
            fn_name='get_properties',
            client_instructions=get_properties_instructions,
            server_round=0,
            timeout=timeout,
        )
        assert not get_properties_failures

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self.strategy.initialize_parameters(client_manager=client_manager)

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        self.evaluate_round(0, timeout)
        for server_round in range(1 + self.strategy.init_round, 1 + num_rounds):
            self.fit_round(server_round, timeout)
            if server_round % self.strategy.eval_interval:
                continue
            self.strategy.save_model(self.parameters, server_round)  # save model checkpoint
            self.evaluate_round(server_round, timeout)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

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

    def fit_stage(
            self,
            server_round: int,
            stage: int,
            parameters: Parameters | List[Parameters] | None,
            cids: List[int] = None,
            timeout: Optional[float] = None,
    ):
        client_manager = self.client_manager()

        # Get clients and their respective instructions from strategy
        instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=parameters,
            client_manager=client_manager,
            stage=stage,
            cids=cids,
        )
        # Collect `fit` results from all clients participating in this round
        results, failures = self.execute_round(
            fn_name='fit',
            client_instructions=instructions,
            server_round=server_round,
            timeout=timeout,
        )
        # Aggregate training results
        parameters_out, metrics = self.strategy.aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
            stage=stage,
        )
        return parameters_out, metrics

    def fit_round(self, server_round: int, timeout: Optional[float]) -> Optional[Dict[str, Scalar]]:
        """Perform a single round of federated averaging."""
        if self.strategy.model_embed_type == 'none':
            stages = [1]
        else:
            stages = [3, 4, 5]

        # Train model and replace previous global model
        parameters = self.parameters
        all_metrics = {}
        for stage in stages:
            parameters, metrics = self.fit_stage(server_round, stage, parameters, None, timeout=timeout)
            all_metrics = {**all_metrics, **metrics}
        self.parameters = parameters

        if wandb.run is not None:
            wandb.log({'train': all_metrics}, commit=False, step=server_round)

        return None

    def evaluate_round(self, server_round: int, timeout: Optional[float]) -> Optional[Dict[str, Scalar]]:
        """Validate current global model on a number of clients."""
        metrics = []
        for i in range(0, self.strategy.num_train_clients, self.strategy.num_available_clients):
            cids = list(range(i, min(i + self.strategy.num_available_clients, self.strategy.num_train_clients)))
            # Evaluate model on a sample of available clients
            if self.strategy.model_embed_type == 'none':
                _, m = self.fit_stage(server_round, 2, self.parameters, cids, timeout=timeout)
            else:
                parameters, m6 = self.fit_stage(server_round, 6, self.parameters, cids, timeout=timeout)
                _, m7 = self.fit_stage(server_round, 7, parameters, None, timeout=timeout)
                m = {**m6, **m7}
            metrics.append((len(cids), m))
        metrics_aggregated = self.strategy.evaluate_metrics_aggregation_fn(metrics)

        if wandb.run is not None:
            wandb.log({'val': metrics_aggregated}, commit=True, step=server_round)

        return metrics_aggregated


def make_server(args: argparse.Namespace):
    # Create strategy
    strategy = FlowerStrategy(
        mode=args.mode,
        num_train_clients=args.num_train_clients,
        num_step_clients=args.num_step_clients,
        eval_interval=args.eval_interval,
        init_round=args.init_round,
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
        client_optimizer_embed_type=args.client_optimizer_embed_type,
        client_optimizer_embed_lr=args.client_optimizer_embed_lr,
        client_embed_num_batches=args.client_embed_num_batches,
        optimizer_hyper_type=args.optimizer_hyper_type,
        optimizer_hyper_lr=args.optimizer_hyper_lr,
        optimizer_weight_decay=args.optimizer_weight_decay,
        client_eval_mask_absent=args.client_eval_mask_absent,
        client_eval_embed_train_split=args.client_eval_embed_train_split,
    )

    server = FlowerServer(client_manager=SimpleClientManager(), strategy=strategy)
    return server


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
