import argparse
import concurrent.futures
import timeit
from logging import DEBUG, INFO
import os
from typing import Any, Callable, Dict, KeysView, List, Optional, Tuple, Union

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
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
import numpy as np
import torch
import wandb

from PeFLL.models import CNNTarget

from parse_args import parse_args
from utils import init_wandb, finish_wandb


class FlowerStrategy(flwr.server.strategy.FedAvg):
    def __init__(
            self,
            mode: str,
            num_train_clients: int,
            state_dict_keys: KeysView,
            eval_interval: int = 10,
            init_round: int = 0,
            client_dataset_seed: int = 42,
            client_dataset_data_name: str = 'cifar10',
            client_dataset_data_path: str = './dataset',
            client_dataset_num_clients: int = 100,
            client_dataset_batch_size: int = 32,
            client_dataset_partition_type: str = 'by_class',
            client_dataset_alpha_train: float = .1,
            client_dataset_alpha_test: float = .1,
            client_optimizer_target_lr: float = 2e-3,
            client_optimizer_target_momentum: float = .9,
            client_optimizer_target_weight_decay: float = 5e-5,
            *args, **kwargs
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

        super().__init__(
            fraction_fit=1e-9,  # Sample % of available clients for training
            fraction_evaluate=1e-9,  # Sample % of available clients for evaluation
            accept_failures=False,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            *args,
            **kwargs
        )
        self.num_train_clients: int = num_train_clients
        self.state_dict_keys: KeysView = state_dict_keys
        self.mode: str = mode
        self.init_round: int = init_round
        self.eval_interval: int = eval_interval
        self.client_optimizer_target_lr: float = client_optimizer_target_lr
        self.client_optimizer_target_momentum: float = client_optimizer_target_momentum
        self.client_optimizer_target_weight_decay: float = client_optimizer_target_weight_decay
        self.client_dataset_seed: int = client_dataset_seed
        self.client_dataset_data_name: str = client_dataset_data_name
        self.client_dataset_data_path: str = client_dataset_data_path
        self.client_dataset_num_clients: int = client_dataset_num_clients
        self.client_dataset_batch_size: int = client_dataset_batch_size
        self.client_dataset_partition_type: str = client_dataset_partition_type
        self.client_dataset_alpha_train: float = client_dataset_alpha_train
        self.client_dataset_alpha_test: float = client_dataset_alpha_test

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        parameters = super().initialize_parameters(client_manager)
        if parameters is None and self.init_round:
            state_dict = torch.load(os.path.join('outputs', 'checkpoints', f'model_round_{self.init_round}.pth'),
                                    map_location=torch.device('cpu'))
            parameters = flwr.common.ndarrays_to_parameters(list(state_dict.values()))
        return parameters

    def _configure_common(self, client_config_pairs: List[Tuple[ClientProxy, Any]], client_manager: ClientManager) \
            -> List[Dict[str, Scalar]]:
        all_cids = sorted(list(client_manager.all().keys()))

        if self.mode == 'multiplex':
            cids = np.random.choice(np.arange(self.num_train_clients), len(client_config_pairs), replace=False).tolist()
        elif self.mode == 'simulated':
            cids = [int(client.cid) for client, _ in client_config_pairs]
        elif self.mode == 'distributed':
            cids = [all_cids.index(client.cid) for client, _ in client_config_pairs]
        else:
            cids = [None] * len(client_config_pairs)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            if self.mode == 'simulated':
                devices = ['cuda'] * len(client_config_pairs)
            else:
                devices = [f'cuda:{all_cids.index(client.cid) % num_gpus}' for client, _ in client_config_pairs]
        else:
            devices = ['cpu'] * len(client_config_pairs)

        configs = [{
                'cid': cid,
                'device': device,
            } for cid, device in zip(cids, devices)
        ]
        return configs

    def configure_get_properties(self, server_round: int, parameters: Parameters, client_manager: ClientManager) \
            -> List[Tuple[ClientProxy, GetPropertiesIns]]:
        client_config_pairs_updated: List[Tuple[ClientProxy, GetPropertiesIns]] = []
        for i, (cid, client) in enumerate(client_manager.all().items()):
            config = {
                'num_train_clients': self.num_train_clients,
                'seed': self.client_dataset_seed,
                'data_name': self.client_dataset_data_name,
                'data_path': self.client_dataset_data_path,
                'num_clients': self.client_dataset_num_clients,
                'batch_size': self.client_dataset_batch_size,
                'partition_type': self.client_dataset_partition_type,
                'alpha_train': self.client_dataset_alpha_train,
                'alpha_test': self.client_dataset_alpha_test,
            }
            client_config_pairs_updated.append((client, GetPropertiesIns(config)))
        return client_config_pairs_updated

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, FitIns]]:
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)

        optimizer_config = {
            'lr': self.client_optimizer_target_lr,
            'momentum': self.client_optimizer_target_momentum,
            'weight_decay': self.client_optimizer_target_weight_decay,
        }

        common_configs = self._configure_common(client_config_pairs, client_manager)
        client_config_pairs_updated: List[Tuple[ClientProxy, FitIns]] = []
        for i, (client, fit_ins) in enumerate(client_config_pairs):
            config = {
                **common_configs[i],
                **fit_ins.config,
                **optimizer_config,
                'num_epochs': 1,
                'verbose': False,
            }
            client_config_pairs_updated.append((client, FitIns(fit_ins.parameters, config)))
        return client_config_pairs_updated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, EvaluateIns]]:
        client_config_pairs = super().configure_evaluate(server_round, parameters, client_manager)
        common_configs = self._configure_common(client_config_pairs, client_manager)
        client_config_pairs_updated: List[Tuple[ClientProxy, EvaluateIns]] = []
        for i, (client, evaluate_ins) in enumerate(client_config_pairs):
            config = {
                **common_configs[i],
                **evaluate_ins.config,
            }
            client_config_pairs_updated.append((client, EvaluateIns(evaluate_ins.parameters, config)))
        return client_config_pairs_updated

    def aggregate_get_properties(
        self,
        server_round: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.GetPropertiesRes]],
        failures: List[Union[Tuple[ClientProxy, GetPropertiesRes], BaseException]],
    ) -> Tuple[Optional[Any], Dict[str, Scalar]]:
        return None, {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        log(INFO, f"training accuracy: {metrics_aggregated['accuracy']}")
        if wandb.run is not None:
            wandb.log({'train': {'accuracy': metrics_aggregated['accuracy']}}, commit=False, step=server_round)

        if parameters_aggregated is not None and server_round % self.eval_interval == 0:
            log(INFO, f'Saving round {server_round} aggregated_parameters...')
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = flwr.common.parameters_to_ndarrays(parameters_aggregated)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            state_dict = {k: torch.tensor(v) for k, v in zip(self.state_dict_keys, aggregated_ndarrays)}
            # Save the model
            os.makedirs(os.path.join('outputs', 'checkpoints'), exist_ok=True)
            torch.save(state_dict, os.path.join('outputs', 'checkpoints', f'model_round_{server_round}.pth'))

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[
        Optional[float], Dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        log(INFO, f"evaluation accuracy: {metrics_aggregated['accuracy']}")
        if wandb.run is not None:
            wandb.log({'val': {'accuracy': metrics_aggregated['accuracy']}}, commit=True, step=server_round)
        return loss_aggregated, metrics_aggregated


FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
# EvaluateResultsAndFailures = Tuple[
#     List[Tuple[ClientProxy, EvaluateRes]],
#     List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
# ]
# ReconnectResultsAndFailures = Tuple[
#     List[Tuple[ClientProxy, DisconnectRes]],
#     List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
# ]


class FlowerServer(flwr.server.Server):
    def __init__(self, client_manager: ClientManager, strategy: FlowerStrategy):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.strategy: FlowerStrategy = self.strategy

    def execute_round(
            self, client_execute_fn: Callable, client_instructions_fn: Callable, aggregate_fn: Callable,
            server_round: int, timeout: Optional[float]
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        """Perform a single round of communication."""
        # Get clients and their respective instructions from strategy
        client_instructions = client_instructions_fn(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "execute_round %s: no clients selected, cancel", server_round)
            return None
        log(DEBUG, "execute_round %s: strategy sampled %s clients (out of %s)",
            server_round, len(client_instructions), self._client_manager.num_available())
        # Collect `execute` results from all clients participating in this round
        """Refine parameters concurrently on all selected clients."""
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
        log(DEBUG, "execute_round %s received %s results and %s failures", server_round, len(results), len(failures))
        # Aggregate training results
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] \
            = aggregate_fn(server_round, results, failures)
        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()
        while self._client_manager.num_available() < self.strategy.min_available_clients:
            pass

        def get_properties_fn(
                client: ClientProxy, ins: GetPropertiesIns, timeout: Optional[float]
        ) -> Tuple[ClientProxy, GetPropertiesRes]:
            """Get properties on a single client."""
            get_properties_res = client.get_properties(ins, timeout=timeout)
            return client, get_properties_res

        def fit_fn(
                client: ClientProxy, ins: FitIns, timeout: Optional[float]
        ) -> Tuple[ClientProxy, FitRes]:
            """Refine parameters on a single client."""
            fit_res = client.fit(ins, timeout=timeout)
            return client, fit_res

        def evaluate_fn(
                client: ClientProxy, ins: EvaluateIns, timeout: Optional[float]
        ) -> Tuple[ClientProxy, EvaluateRes]:
            """Evaluate parameters on a single client."""
            evaluate_res = client.evaluate(ins, timeout=timeout)
            return client, evaluate_res

        log(INFO, "Initializing clients")
        res_get_properties = self.execute_round(
            get_properties_fn, self.strategy.configure_get_properties, self.strategy.aggregate_get_properties,
            0, timeout
        )

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1 + self.strategy.init_round, 1 + num_rounds):
            # Train model and replace previous global model
            res_fit = self.execute_round(
                fit_fn, self.strategy.configure_fit, self.strategy.aggregate_fit, current_round, timeout
            )

            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            if current_round % self.strategy.eval_interval:
                continue

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.execute_round(
                evaluate_fn, self.strategy.configure_evaluate, self.strategy.aggregate_evaluate, current_round, timeout
            )

            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history


def make_server(args: argparse.Namespace):
    mode: str = args.mode
    num_train_clients = args.num_train_clients
    num_step_clients = args.num_step_clients
    num_evaluate_clients = num_step_clients

    if mode in ['simulated', 'distributed']:
        num_available_clients = num_train_clients
    elif mode in ['multiplex']:
        num_available_clients = num_step_clients
    else:
        num_available_clients = int(1e9)

    # Create strategy
    strategy = FlowerStrategy(
        min_fit_clients=num_step_clients,
        min_evaluate_clients=num_evaluate_clients,
        min_available_clients=num_available_clients,  # Wait until all clients are available
        mode=mode,
        num_train_clients=num_train_clients,
        state_dict_keys=CNNTarget().state_dict().keys(),
        init_round=args.init_round,
        eval_interval=args.eval_interval,
        client_dataset_seed=args.client_dataset_seed,
        client_dataset_data_name=args.client_dataset_data_name,
        client_dataset_data_path=args.client_dataset_data_path,
        client_dataset_num_clients=args.client_dataset_num_clients,
        client_dataset_batch_size=args.client_dataset_batch_size,
        client_dataset_partition_type=args.client_dataset_partition_type,
        client_dataset_alpha_train=args.client_dataset_alpha_train,
        client_dataset_alpha_test=args.client_dataset_alpha_test,
        client_optimizer_target_lr=args.client_optimizer_target_lr,
        client_optimizer_target_momentum=args.client_optimizer_target_momentum,
        client_optimizer_target_weight_decay=args.client_optimizer_target_weight_decay,
    )

    server = FlowerServer(client_manager=SimpleClientManager(), strategy=strategy)
    return server


def main():
    args = parse_args()
    init_wandb(args, f'experiment_{21}')
    server = make_server(args)
    flwr.server.start_server(
        server_address=f"0.0.0.0:{args.server_address.split(':')[-1]}",
        server=server,
        config=flwr.server.ServerConfig(num_rounds=args.num_rounds),
    )
    finish_wandb()


if __name__ == '__main__':
    main()
