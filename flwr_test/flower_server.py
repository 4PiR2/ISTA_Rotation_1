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

# from PeFLL.models import

from parse_args import parse_args
from utils import init_wandb, finish_wandb


class FlowerStrategy(flwr.server.strategy.FedAvg):
    def __init__(
            self,
            mode: str,
            num_train_clients: int,
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
            client_optimizer_target_lr: float = 2e-3,
            client_optimizer_target_momentum: float = .9,
            client_optimizer_target_weight_decay: float = 5e-5,
            client_target_steps: int = 50,
            client_embed_batches: int = 1,
            model_num_kernels: int = 16,
            model_embed_type: str = 'none',
            model_embed_model_dim: int = -1,
            model_embed_model_y: bool = True,
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
        self.mode: str = mode
        self.init_round: int = init_round
        self.eval_interval: int = eval_interval
        self.log_dir: str = log_dir
        self.client_optimizer_target_lr: float = client_optimizer_target_lr
        self.client_optimizer_target_momentum: float = client_optimizer_target_momentum
        self.client_optimizer_target_weight_decay: float = client_optimizer_target_weight_decay
        self.client_target_steps: int = client_target_steps
        self.client_embed_batches: int = client_embed_batches
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
        self.model_embed_dim: int = model_embed_model_dim
        self.model_embed_y: bool = model_embed_model_y
        self.stage_memory: Dict = {}

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
                    map_location=torch.device('cpu')
                )
                keys.append(np.asarray(list(state_dict.keys())))
                vals.extend([val.detach().cpu().numpy() for val in state_dict.values()])
            parameters = ndarrays_to_parameters([np.asarray(header)] + keys + vals)
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
            torch.save(state_dict, os.path.join(self.log_dir, 'checkpoints', f'model_{net_name}_round_{server_round}.pth'))

    def _configure_common(self, client_config_pairs: List[Tuple[ClientProxy, Any]], client_manager: ClientManager) \
            -> List[Dict[str, Scalar]]:
        all_cids = sorted(list(client_manager.all().keys()))

        match self.mode:
            case 'multiplex':
                cids = np.random.choice(np.arange(self.num_train_clients), len(client_config_pairs), replace=False).tolist()
            case 'simulated':
                cids = [int(client.cid) for client, _ in client_config_pairs]
            case 'distributed':
                cids = [all_cids.index(client.cid) for client, _ in client_config_pairs]
            case _:
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
                'model_embed_y': self.model_embed_y,
            }
            client_config_pairs_updated.append((client, GetPropertiesIns(conf)))
        return client_config_pairs_updated

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager,
            config: Optional[Dict[str, Scalar]] = None
    ) -> List[Tuple[ClientProxy, FitIns]]:
        stage = 0 if config is None or 'stage' not in config else config['stage']
        match stage:
            case 0:
                client_config_pairs = super().configure_fit(server_round, parameters, client_manager)
                common_configs = self._configure_common(client_config_pairs, client_manager)
                stage_config = {
                    'client_optimizer_target_lr': self.client_optimizer_target_lr,
                    'client_optimizer_target_momentum': self.client_optimizer_target_momentum,
                    'client_optimizer_target_weight_decay': self.client_optimizer_target_weight_decay,
                    'client_target_steps': self.client_target_steps,
                }
            case 1:
                client_config_pairs = super().configure_fit(server_round, parameters, client_manager)
                common_configs = self._configure_common(client_config_pairs, client_manager)
                self.stage_memory['client_config_pairs'] = client_config_pairs
                self.stage_memory['common_configs'] = common_configs
                stage_config = {
                    'client_embed_batches': self.client_embed_batches,
                }
            case _:
                client_config_pairs = []
                common_configs = {}
                stage_config = {}

        client_config_pairs_updated: List[Tuple[ClientProxy, FitIns]] = []
        for i, (client, fit_ins) in enumerate(client_config_pairs):
            conf = {
                **fit_ins.config,
                **common_configs[i],
                'stage': stage,
                **stage_config,
            }
            client_config_pairs_updated.append((client, FitIns(fit_ins.parameters, conf)))
        return client_config_pairs_updated

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager,
            config: Optional[Dict[str, Scalar]] = None
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        client_config_pairs = super().configure_evaluate(server_round, parameters, client_manager)
        common_configs = self._configure_common(client_config_pairs, client_manager)

        stage = 0 if config is None or 'stage' not in config else config['stage']
        match stage:
            case 0:
                stage_config = {}
            case _:
                stage_config = {}

        client_config_pairs_updated: List[Tuple[ClientProxy, EvaluateIns]] = []
        for i, (client, evaluate_ins) in enumerate(client_config_pairs):
            conf = {
                **evaluate_ins.config,
                **common_configs[i],
                'stage': stage,
                **stage_config,
            }
            client_config_pairs_updated.append((client, EvaluateIns(evaluate_ins.parameters, conf)))
        return client_config_pairs_updated

    def aggregate_fit(
            self, *args, **kwargs,
            # server_round: int,
            # results: List[Tuple[flwr.server.client_proxy.ClientProxy, FitRes]],
            # failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        raise NotImplementedError

    def aggregate_fit_0(
            self,
            server_round: int,
            results: List[Tuple[flwr.server.client_proxy.ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

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

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        log(INFO, f"training: {metrics_aggregated}")
        if wandb.run is not None:
            wandb.log({'train': metrics_aggregated}, commit=False, step=server_round)
        return parameters_aggregated, metrics_aggregated

    def aggregate_fit_1(
            self,
            results: List[Tuple[flwr.server.client_proxy.ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(self.stage_memory['common_configs'])
        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(
            weights_results[1+len(weights_results[0]):] + aggregate(weights_results[:1+len(weights_results[0])])
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        # if self.fit_metrics_aggregation_fn:
        #     fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        # elif server_round == 1:  # Only log this warning once
        #     log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self, *args, **kwargs,
            # server_round: int,
            # results: List[Tuple[ClientProxy, EvaluateRes]],
            # failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        raise NotImplementedError

    def aggregate_evaluate_0(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        log(INFO, f"evaluation: {metrics_aggregated}")
        if wandb.run is not None:
            wandb.log({'val': metrics_aggregated}, commit=True, step=server_round)
        return loss_aggregated, metrics_aggregated


class FlowerServer(flwr.server.Server):
    def __init__(self, client_manager: ClientManager, strategy: FlowerStrategy):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.strategy: FlowerStrategy = self.strategy

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
        # Get clients and their respective instructions from strategy
        # client_instructions = client_instructions_fn(
        #     server_round=server_round,
        #     parameters=self.parameters,
        #     client_manager=self._client_manager,
        # )
        if not client_instructions:
            log(INFO, "execute_round %s: no clients selected, cancel", server_round)
            return None
        log(DEBUG, "execute_round %s: strategy sampled %s clients (out of %s)",
            server_round, len(client_instructions), self.client_manager().num_available())
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
        log(DEBUG, "execute_round %s received %s results and %s failures", server_round, len(results), len(failures))
        # Aggregate training results
        # aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] \
        #     = aggregate_fn(server_round, results, failures)
        # parameters_aggregated, metrics_aggregated = aggregated_result
        return results, failures

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
        # log(INFO, "Evaluating initial parameters")
        # res = self.strategy.evaluate(0, parameters=self.parameters)
        # if res is not None:
        #     log(
        #         INFO,
        #         "initial parameters (loss, other metrics): %s, %s",
        #         res[0],
        #         res[1],
        #     )
        #     history.add_loss_centralized(server_round=0, loss=res[0])
        #     history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1 + self.strategy.init_round, 1 + num_rounds):
            # Train model and replace previous global model
            if self.strategy.model_embed_type == 'none':
                fit_0_instructions = self.strategy.configure_fit(
                    server_round=current_round,
                    parameters=self.parameters,
                    client_manager=client_manager,
                    config={'stage': 0}
                )
                fit_0_results, fit_0_failures = self.execute_round(
                    fn_name='fit',
                    client_instructions=fit_0_instructions,
                    server_round=current_round,
                    timeout=timeout,
                )
                assert not fit_0_failures
                # Aggregate training results
                parameters_prime, fit_metrics = self.strategy.aggregate_fit_0(
                    server_round=current_round,
                    results=fit_0_results,
                    failures=fit_0_failures,
                )
                if parameters_prime:
                    self.parameters = parameters_prime
                    history.add_metrics_distributed_fit(
                        server_round=current_round, metrics=fit_metrics
                    )

            else:
                fit_1_instructions = self.strategy.configure_fit(
                    server_round=current_round,
                    parameters=self.parameters,
                    client_manager=client_manager,
                    config={'stage': 1}
                )
                fit_0_results, fit_0_failures = self.execute_round(
                    fn_name='fit',
                    client_instructions=fit_1_instructions,
                    server_round=current_round,
                    timeout=timeout,
                )

            if current_round % self.strategy.eval_interval:
                continue

            # save model checkpoint
            self.strategy.save_model(self.parameters, current_round)

            # Evaluate model using strategy implementation
            # res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            # if res_cen is not None:
            #     loss_cen, metrics_cen = res_cen
            #     log(
            #         INFO,
            #         "fit progress: (%s, %s, %s, %s)",
            #         current_round,
            #         loss_cen,
            #         metrics_cen,
            #         timeit.default_timer() - start_time,
            #     )
            #     history.add_loss_centralized(server_round=current_round, loss=loss_cen)
            #     history.add_metrics_centralized(
            #         server_round=current_round, metrics=metrics_cen
            #     )

            # Evaluate model on a sample of available clients
            if self.strategy.model_embed_type == 'none':
                evaluate_0_instructions = self.strategy.configure_fit(
                    server_round=current_round,
                    parameters=self.parameters,
                    client_manager=client_manager,
                    config={'stage': 0}
                )
                evaluate_0_results, evaluate_0_failures = self.execute_round(
                    fn_name='evaluate',
                    client_instructions=evaluate_0_instructions,
                    server_round=current_round,
                    timeout=timeout,
                )
                assert not evaluate_0_failures
                loss_fed, evaluate_metrics_fed = self.strategy.aggregate_evaluate_0(
                    server_round=current_round,
                    results=evaluate_0_results,
                    failures=evaluate_0_failures,
                )
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

    match mode:
        case 'simulated' | 'distributed':
            num_available_clients = num_train_clients
        case 'multiplex':
            num_available_clients = num_step_clients
        case _:
            num_available_clients = int(1e9)

    # Create strategy
    strategy = FlowerStrategy(
        min_fit_clients=num_step_clients,
        min_evaluate_clients=num_evaluate_clients,
        min_available_clients=num_available_clients,  # Wait until all clients are available
        mode=mode,
        num_train_clients=num_train_clients,
        init_round=args.init_round,
        eval_interval=args.eval_interval,
        client_optimizer_target_lr=args.client_optimizer_target_lr,
        client_optimizer_target_momentum=args.client_optimizer_target_momentum,
        client_optimizer_target_weight_decay=args.client_optimizer_target_weight_decay,
        client_target_steps=args.client_target_steps,
        client_embed_batches=args.client_embed_batches,
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
        model_embed_model_dim=args.model_embed_dim,
        model_embed_model_y=args.model_embed_y,
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
