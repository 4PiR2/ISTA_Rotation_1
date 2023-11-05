import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

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
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns, GetPropertiesIns
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

from PeFLL.models import CNNTarget
from my_strategy import SaveModelStrategy

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


class MyServer(flwr.server.Server):
    def __init__(self, client_manager: ClientManager, strategy: Optional[Strategy] = None,
                 init_round: int = 0, eval_interval: int = 1, num_train_clients: int = 9):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.init_round: int = init_round
        self.eval_interval: int = eval_interval
        self.num_train_clients: int = num_train_clients

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

        for i, (cid, client) in enumerate(self._client_manager.all().items()):
            # TODO
            config = {
                'device': 'cpu',
                'num_train_clients': self.num_train_clients,
            }
            _ = client.get_properties(ins=GetPropertiesIns(config), timeout=None)

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(self.init_round, self.init_round + num_rounds):
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

            if current_round % self.eval_interval:
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


def make_server():
    NUM_CLIENTS = 100
    NUM_TRAIN_CLIENTS = int(NUM_CLIENTS * .9)

    # Create FedAvg strategy
    strategy = SaveModelStrategy(
        fraction_fit=.1,  # Sample 100% of available clients for training
        fraction_evaluate=1.,  # Sample 50% of available clients for evaluation
        min_available_clients=NUM_TRAIN_CLIENTS,  # Wait until all 10 clients are available
        state_dict_keys=CNNTarget().state_dict().keys(),
        mode='distributed',
    )

    server = MyServer(client_manager=SimpleClientManager(), strategy=strategy, init_round=0, eval_interval=10, num_train_clients=NUM_TRAIN_CLIENTS)
    return server


def make_server2():
    NUM_TRAIN_CLIENTS = 2

    # Create FedAvg strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.,  # Sample 100% of available clients for training
        fraction_evaluate=1.,  # Sample 50% of available clients for evaluation
        min_available_clients=NUM_TRAIN_CLIENTS,  # Wait until all 10 clients are available
        state_dict_keys=CNNTarget().state_dict().keys(),
        mode='multiplex',
    )

    server = MyServer(client_manager=SimpleClientManager(), strategy=strategy, init_round=0, eval_interval=10, num_train_clients=NUM_TRAIN_CLIENTS)
    return server


def main():
    server = make_server2()
    server_port = 18080
    flwr.server.start_server(
        server_address=f'0.0.0.0:{server_port}',
        server=server,
        config=flwr.server.ServerConfig(num_rounds=1000),
    )


if __name__ == '__main__':
    main()
