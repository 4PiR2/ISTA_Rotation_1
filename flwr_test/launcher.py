import os
import sys
import time

import flwr
import torch

from flower_client import FlowerClient
from flower_server import make_server
from parse_args import parse_args
from utils import run


def main():
    args = parse_args()
    mode = args.mode

    if mode == 'simulated':
        server = make_server(args)
        config = {
            'num_train_clients': server.strategy.num_train_clients,
            'seed': server.strategy.client_data_seed,
            'data_name': server.strategy.client_data_data_name,
            'data_path': server.strategy.client_data_data_path,
            'num_clients': server.strategy.client_data_num_clients,
            'batch_size': server.strategy.client_data_batch_size,
            'partition_type': server.strategy.client_data_partition_type,
            'classes_per_user': server.strategy.client_data_classes_per_user,
        }
        if server.strategy.client_data_alpha_train is not None:
            config['alpha_train'] = server.strategy.client_data_alpha_train
        if server.strategy.client_data_alpha_test is not None:
            config['alpha_test'] = server.strategy.client_data_alpha_test
        if server.strategy.client_data_embedding_dir_path is not None:
            config['embedding_dir_path'] = server.strategy.client_data_embedding_dir_path

        def client_fn(cid: str) -> FlowerClient:
            client = FlowerClient(cid)
            client.get_properties(config)
            return client

        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        num_gpus = torch.cuda.device_count()
        client_resources = {'num_cpus': 1, 'num_gpus': 1} if num_gpus > 0 else None

        # Start simulation
        flwr.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=server.strategy.num_train_clients,
            server=server,
            config=flwr.server.ServerConfig(num_rounds=args.num_rounds),
            client_resources=client_resources,
        )
    else:
        run(['pkill', '-9', '-f', '-e', '-c', 'python3 flower_server.py'])
        run(['pkill', '-9', '-f', '-e', '-c', 'python3 flower_client.py'])

        num_clients = args.client_data_num_clients
        num_train_clients = int(num_clients * .9)
        min_fit_clients = int(num_train_clients * .1)
        if mode == 'distributed':
            min_available_clients = num_train_clients
        elif mode == 'multiplex':
            min_available_clients = min_fit_clients
        else:
            return

        p_server = run(['python3', 'flower_server.py'] + sys.argv[1:], blocking=False)
        time.sleep(10.)
        p_clients = [run(['python3', 'flower_client.py'] + sys.argv[1:], blocking=False)
                     for _ in range(min_available_clients)]

        os.makedirs(os.path.join('outputs', 'logs'), exist_ok=True)
        log_files = [open(os.path.join('outputs', 'logs', f'C{i}.txt'), 'w') for i in range(1+len(p_clients))]
        rcs = [None] * len(log_files)
        while True:
            for i, p in enumerate([p_server, *p_clients]):
                rcs[i] = p.poll()
                while line_out := p.stdout.readline():
                    log_files[i].write(line_out)
                    log_files[i].flush()
                    print(f'[C{i}]\t', line_out, end='', flush=True)
                while line_err := p.stderr.readline():
                    log_files[i].write(line_err)
                    log_files[i].flush()
                    print(f'[C{i}]\t', line_err, end='', flush=True)
            if None not in rcs:
                break
        for log_file in log_files:
            log_file.close()
    print('END', flush=True)


if __name__ == '__main__':
    main()
