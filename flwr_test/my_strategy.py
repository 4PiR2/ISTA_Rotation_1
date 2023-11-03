import os
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict

import flwr as fl
import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar, Metrics, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from PeFLL.utils import get_device


DEVICE = get_device()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    examples = np.asarray([num_examples for num_examples, _ in metrics])
    accuracies = np.asarray([m['accuracy'] for _, m in metrics])
    # Aggregate and return custom metric (weighted average)
    accuracy = np.sum(accuracies * examples) / np.sum(examples)
    return {'accuracy': accuracy}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, state_dict_keys, *args, **kwargs):
        super().__init__(
            min_fit_clients=1,
            min_evaluate_clients=1,
            evaluate_metrics_aggregation_fn=weighted_average,
            *args,
            **kwargs
        )
        self.state_dict_keys = state_dict_keys

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
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
