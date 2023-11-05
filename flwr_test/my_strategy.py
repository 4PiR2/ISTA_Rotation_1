from collections import OrderedDict
import os
from typing import List, Tuple, Union, Optional, Dict, KeysView, Any

import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, Metrics, EvaluateRes, FitIns, EvaluateIns
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
import numpy as np
import torch


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    examples = np.asarray([num_examples for num_examples, _ in metrics])
    accuracies = np.asarray([m['accuracy'] for _, m in metrics])
    # Aggregate and return custom metric (weighted average)
    accuracy = np.sum(accuracies * examples) / np.sum(examples)
    return {'accuracy': accuracy}


def configure_cid(client_config_pairs: List[Tuple[ClientProxy, Any]], client_manager: ClientManager, mode: str) -> List[int]:
    all_cids: List[str] = list(client_manager.all().keys())
    cids: List[int] = []
    for i, (client, _) in enumerate(client_config_pairs):
        if mode == 'simulated':
            cid = int(client.cid)
        elif mode == 'distributed':
            cid = all_cids.index(client.cid)
        elif mode == 'multiplex':
            num_train_clients = 90  # TODO
            selected_cids = np.random.choice(np.arange(num_train_clients), len(client_config_pairs), replace=False)
            cid = int(selected_cids[i])
        else:
            cid = None
        cids.append(cid)
    return cids


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, state_dict_keys: KeysView, mode: str, *args, **kwargs):
        super().__init__(
            min_fit_clients=1,
            min_evaluate_clients=1,
            accept_failures=False,
            evaluate_metrics_aggregation_fn=weighted_average,
            *args,
            **kwargs
        )
        self.state_dict_keys: KeysView = state_dict_keys
        self.mode: str = mode

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        parameters = super().initialize_parameters(client_manager)
        if parameters is None:
            # TODO
            state_dict = torch.load(os.path.join('output', 'bak', f'model_round_{1000}.pth'))
            # parameters = fl.common.ndarrays_to_parameters(list(state_dict.values()))
        return parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, FitIns]]:
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)

        # TODO
        optimizer_config = {
            'lr': 2e-3,
            'momentum': .9,
            'weight_decay': 5e-5,
        }

        client_config_pairs_updated: List[Tuple[ClientProxy, FitIns]] = []
        cids = configure_cid(client_config_pairs, client_manager, self.mode)
        for i, (client, fit_ins) in enumerate(client_config_pairs):
            config = {
                **fit_ins.config,
                'cid': cids[i],
                **optimizer_config,
            }
            client_config_pairs_updated.append((client, FitIns(fit_ins.parameters, config)))
        return client_config_pairs_updated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, EvaluateIns]]:
        client_config_pairs = super().configure_evaluate(server_round, parameters, client_manager)
        client_config_pairs_updated: List[Tuple[ClientProxy, EvaluateIns]] = []
        cids = configure_cid(client_config_pairs, client_manager, self.mode)
        for i, (client, evaluate_ins) in enumerate(client_config_pairs):
            config = {
                **evaluate_ins.config,
                'cid': cids[i],
            }
            client_config_pairs_updated.append((client, EvaluateIns(evaluate_ins.parameters, config)))
        return client_config_pairs_updated

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # TODO
        if aggregated_parameters is not None and server_round % 10 == 0:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.state_dict_keys, aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            # Save the model
            torch.save(state_dict, os.path.join('output', f"model_round_{server_round}.pth"))

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[
        Optional[float], Dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        print(f"evaluation accuracy: {metrics_aggregated['accuracy']}")

        return loss_aggregated, metrics_aggregated
