import timeit
from logging import DEBUG, INFO
import os
from typing import Any, Dict, KeysView, List, Optional, Tuple, Union

import flwr
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
    Metrics,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns, GetPropertiesIns
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
import numpy as np
import torch
import wandb

from PeFLL.models import CNNTarget


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


class FlowerStrategy(flwr.server.strategy.FedAvg):
    def __init__(
            self,
            num_train_clients: int,
            state_dict_keys: KeysView,
            mode: str,
            init_round: int = 0,
            eval_interval: int = 1,
            optimizer_inner_lr: float = 2e-3,
            optimizer_inner_momentum: float = .9,
            optimizer_inner_weight_decay: float = 5e-5,
            client_data_seed: int = 42,
            client_data_data_name: str = 'cifar10',
            client_data_data_path: str = './dataset',
            client_data_num_clients: int = 100,
            client_data_batch_size: int = 32,
            client_data_partition_type: str = 'by_class',
            client_data_classes_per_user: int = 2,
            client_data_alpha_train: Optional[float] = None,
            client_data_alpha_test: Optional[float] = None,
            client_data_embedding_dir_path: Optional[str] = None,
            *args, **kwargs
    ):
        super().__init__(
            min_fit_clients=1,
            min_evaluate_clients=1,
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
        self.optimizer_inner_lr: float = optimizer_inner_lr
        self.optimizer_inner_momentum: float = optimizer_inner_momentum
        self.optimizer_inner_weight_decay: float = optimizer_inner_weight_decay
        self.client_data_seed: int = client_data_seed
        self.client_data_data_name: str = client_data_data_name
        self.client_data_data_path: str = client_data_data_path
        self.client_data_num_clients: int = client_data_num_clients
        self.client_data_batch_size: int = client_data_batch_size
        self.client_data_partition_type: str = client_data_partition_type
        self.client_data_classes_per_user: int = client_data_classes_per_user
        self.client_data_alpha_train: Optional[float] = client_data_alpha_train
        self.client_data_alpha_test: Optional[float] = client_data_alpha_test
        self.client_data_embedding_dir_path: Optional[str] = client_data_embedding_dir_path

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        parameters = super().initialize_parameters(client_manager)
        if parameters is None and self.init_round:
            state_dict = torch.load(os.path.join('outputs', 'checkpoints', f'model_round_{self.init_round}.pth'),
                                    map_location=torch.device('cpu'))
            parameters = flwr.common.ndarrays_to_parameters(list(state_dict.values()))
        return parameters

    def configure_cid(self, client_config_pairs: List[Tuple[ClientProxy, Any]], all_cids: List[str]) -> List[int]:
        if self.mode == 'multiplex':
            cids = np.random.choice(np.arange(self.num_train_clients), len(client_config_pairs), replace=False).tolist()
        elif self.mode == 'simulated':
            cids = [int(client.cid) for client, _ in client_config_pairs]
        elif self.mode == 'distributed':
            cids = [all_cids.index(client.cid) for client, _ in client_config_pairs]
        else:
            cids = None
        return cids

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, FitIns]]:
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)

        optimizer_config = {
            'lr': self.optimizer_inner_lr,
            'momentum': self.optimizer_inner_momentum,
            'weight_decay': self.optimizer_inner_weight_decay,
        }

        all_cids: List[str] = sorted(list(client_manager.all().keys()))
        cids = self.configure_cid(client_config_pairs, all_cids)
        num_gpus = torch.cuda.device_count()
        client_config_pairs_updated: List[Tuple[ClientProxy, FitIns]] = []
        for i, (client, fit_ins) in enumerate(client_config_pairs):
            config = {
                **fit_ins.config,
                'cid': cids[i],
                'device': f'cuda:{all_cids.index(client.cid) % num_gpus}' if num_gpus else 'cpu',
                'num_epochs': 1,
                'verbose': False,
                **optimizer_config,
            }
            client_config_pairs_updated.append((client, FitIns(fit_ins.parameters, config)))
        return client_config_pairs_updated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, EvaluateIns]]:
        client_config_pairs = super().configure_evaluate(server_round, parameters, client_manager)
        all_cids: List[str] = sorted(list(client_manager.all().keys()))
        cids = self.configure_cid(client_config_pairs, all_cids)
        num_gpus = torch.cuda.device_count()
        client_config_pairs_updated: List[Tuple[ClientProxy, EvaluateIns]] = []
        for i, (client, evaluate_ins) in enumerate(client_config_pairs):
            config = {
                **evaluate_ins.config,
                'cid': cids[i],
                'device': f'cuda:{all_cids.index(client.cid) % num_gpus}' if num_gpus else 'cpu',
            }
            client_config_pairs_updated.append((client, EvaluateIns(evaluate_ins.parameters, config)))
        return client_config_pairs_updated

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
        wandb.log({'train': {'acc': metrics_aggregated['accuracy']}}, commit=True, step=server_round)

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
        wandb.log({'val': {'acc': metrics_aggregated['accuracy']}}, commit=True, step=server_round)
        return loss_aggregated, metrics_aggregated


FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class FlowerServer(flwr.server.Server):
    def __init__(self, client_manager: ClientManager, strategy: FlowerStrategy):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.strategy: FlowerStrategy = self.strategy

    def get_client_properties(self):
        for i, (cid, client) in enumerate(self._client_manager.all().items()):
            config = {
                'num_train_clients': self.strategy.num_train_clients,
                'seed': self.strategy.client_data_seed,
                'data_name': self.strategy.client_data_data_name,
                'data_path': self.strategy.client_data_data_path,
                'num_clients': self.strategy.client_data_num_clients,
                'batch_size': self.strategy.client_data_batch_size,
                'partition_type': self.strategy.client_data_partition_type,
                'classes_per_user': self.strategy.client_data_classes_per_user,
            }
            if self.strategy.client_data_alpha_train is not None:
                config['alpha_train'] = self.strategy.client_data_alpha_train
            if self.strategy.client_data_alpha_test is not None:
                config['alpha_test'] = self.strategy.client_data_alpha_test
            if self.strategy.client_data_embedding_dir_path is not None:
                config['embedding_dir_path'] = self.strategy.client_data_embedding_dir_path
            _ = client.get_properties(ins=GetPropertiesIns(config), timeout=None)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        while self._client_manager.num_available() < self.strategy.min_available_clients:
            pass

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

        self.get_client_properties()

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1 + self.strategy.init_round, 1 + self.strategy.init_round + num_rounds):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

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

            if current_round % self.strategy.eval_interval:
                continue

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
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


def make_server(mode: str):
    NUM_CLIENTS = 100
    NUM_TRAIN_CLIENTS = int(NUM_CLIENTS * .9)

    if mode in ['simulated', 'distributed']:
        # Create FedAvg strategy
        strategy = FlowerStrategy(
            fraction_fit=.1,  # Sample 100% of available clients for training
            fraction_evaluate=1.,  # Sample 50% of available clients for evaluation
            min_available_clients=NUM_TRAIN_CLIENTS,  # Wait until all 10 clients are available
            num_train_clients=NUM_TRAIN_CLIENTS,
            state_dict_keys=CNNTarget().state_dict().keys(),
            mode=mode,
            init_round=0,
            eval_interval=10,
        )
    elif mode in ['multiplex']:
        # Create FedAvg strategy
        strategy = FlowerStrategy(
            fraction_fit=1.,  # Sample 100% of available clients for training
            fraction_evaluate=1.,  # Sample 50% of available clients for evaluation
            min_available_clients=int(NUM_TRAIN_CLIENTS * .1),  # Wait until all 10 clients are available
            num_train_clients=NUM_TRAIN_CLIENTS,
            state_dict_keys=CNNTarget().state_dict().keys(),
            mode=mode,
            init_round=0,
            eval_interval=10,
            optimizer_inner_lr=2e-3,
            optimizer_inner_momentum=.9,
            optimizer_inner_weight_decay=5e-5,
        )
    else:
        strategy = None

    server = FlowerServer(client_manager=SimpleClientManager(), strategy=strategy)
    return server


def main():
    with open('wandb_token.txt', 'r') as f:
        wandb_token = f.readline().strip()
    wandb_login = wandb.login(key=wandb_token)
    if wandb_login:
        wandb.init(
            # Set the project where this run will be logged
            project="test",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment_{15}",
            # Track hyperparameters and run metadata
            config={
                "hello": 2e-2,
            })

    server = make_server(mode='multiplex')
    server_port = 18080
    flwr.server.start_server(
        server_address=f'0.0.0.0:{server_port}',
        server=server,
        config=flwr.server.ServerConfig(num_rounds=1000),
    )

    if wandb_login:
        # Mark the run as finished
        wandb.finish()


if __name__ == '__main__':
    main()
