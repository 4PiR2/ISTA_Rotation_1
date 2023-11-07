import os
import time

import flwr
import torch

from PeFLL.utils import get_device

from flower_client import FlowerClient
from flower_server import make_server
from utils import run


def main():
    SIMULATION = False

    if SIMULATION:
        server = make_server(mode='simulated')
        num_train_clients = server.strategy.num_train_clients
        num_gpus = torch.cuda.device_count()

        def client_fn(cid: str) -> FlowerClient:
            client = FlowerClient(cid)
            # TODO
            config = {
                'device': f'cuda:{int(cid) % num_gpus}' if num_gpus else 'cpu',
                'num_train_clients': num_train_clients,
            }
            client.get_properties(config)
            return client

        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        DEVICE = get_device()
        client_resources = {'num_cpus': 1, 'num_gpus': 1} if DEVICE.type == 'cuda' else None

        # Start simulation
        flwr.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=server.strategy.num_train_clients,
            server=server,
            config=flwr.server.ServerConfig(num_rounds=1000),
            client_resources=client_resources,
        )
    else:
        run(['pkill', '-9', '-f', 'python3 flower_server.py'])
        run(['pkill', '-9', '-f', 'python3 flower_client.py'])

        p_server = run(['python3', 'flower_server.py'], blocking=False)
        time.sleep(20.)
        # TODO
        min_available_clients = 9
        p_clients = [run(['python3', 'flower_client.py'], blocking=False) for _ in range(min_available_clients)]
        os.makedirs(os.path.join('outputs', 'logs'), exist_ok=True)
        log_files = [open(os.path.join('outputs', 'logs', f'C{i}.txt'), 'w') for i in range(1+len(p_clients))]
        rcs = [None] * len(log_files)
        while True:
            for i, p in enumerate([p_server, *p_clients]):
                rcs[i] = p.poll()
                while line_out := p.stdout.readline():
                    log_files[i].write(line_out)
                    log_files[i].flush()
                    print(f'[C{i}]\t', line_out, end='')
                while line_err := p.stderr.readline():
                    log_files[i].write(line_err)
                    log_files[i].flush()
                    print(f'[C{i}]\t', line_err, end='')
            if None not in rcs:
                break
        for log_file in log_files:
            log_file.close()
    a = 0


if __name__ == '__main__':
    main()
