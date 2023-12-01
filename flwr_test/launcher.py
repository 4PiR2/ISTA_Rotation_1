import os
import sys

import flwr
import torch

from flower_client import FlowerClient, main as main_c
from flower_server import make_server, main as main_s
from parse_args import parse_args
from utils import run, init_wandb, finish_wandb, detect_slurm


def main():
    args = parse_args()

    if args.enable_slurm and detect_slurm():
        servername = os.environ['SLURM_SUBMIT_HOST']
        hostname = os.environ['SLURMD_NODENAME']
        if hostname == servername:
            print(f'SLURM Server: {hostname}')
            main_s()
        else:
            print(f'SLURM Client: {hostname}')
            main_c()
        return

    mode = args.mode
    if mode == 'simulated':
        init_wandb(args=args, experiment_name=args.experiment_name, group=None)
        server = make_server(args)
        config = {
            'num_train_clients': server.strategy.num_train_clients,
            'client_dataset_seed': server.strategy.client_data_seed,
            'client_dataset_data_name': server.strategy.client_data_data_name,
            'client_dataset_data_path': server.strategy.client_data_data_path,
            'client_dataset_num_clients': server.strategy.client_data_num_clients,
            'client_dataset_batch_size': server.strategy.client_data_batch_size,
            'client_dataset_partition_type': server.strategy.client_data_partition_type,
            'client_dataset_alpha_train': server.strategy.client_dataset_alpha_train,
            'client_dataset_alpha_test': server.strategy.client_dataset_alpha_test,
            'client_model_num_kernels': server.strategy.client_model_num_kernels,
            'model_embed_type': server.strategy.model_embed_type,
            'client_model_embed_dim': server.strategy.client_model_embed_dim,
            'client_model_embed_y': server.strategy.client_model_embed_y,
        }

        def client_fn(cid: str) -> FlowerClient:
            client = FlowerClient()
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

        finish_wandb()
    else:
        run(['pkill', '-9', '-f', '-e', '-c', 'wandb-service'])
        run(['pkill', '-9', '-f', '-e', '-c', 'python3 flower_server.py'])
        run(['pkill', '-9', '-f', '-e', '-c', 'python3 flower_client.py'])

        num_train_clients = args.num_train_clients
        num_step_clients = args.num_step_clients
        match mode:
            case 'distributed':
                num_available_clients = num_train_clients
            case 'multiplex':
                num_available_clients = num_step_clients
            case _:
                return

        p_server = run(['python3', 'flower_server.py'] + sys.argv[1:], blocking=False)
        # time.sleep(10.)
        p_clients = [run(['python3', 'flower_client.py'] + sys.argv[1:], blocking=False)
                     for _ in range(num_available_clients)]

        log_dir = args.log_dir
        os.makedirs(os.path.join(log_dir, 'logs'), exist_ok=True)
        log_files = [open(os.path.join(log_dir, 'logs', f'C{i}.txt'), 'w') for i in range(1+len(p_clients))]
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
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main()
