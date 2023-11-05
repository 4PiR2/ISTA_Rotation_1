import flwr as fl

from my_client import FlowerClient
from my_server import make_server
from PeFLL.utils import get_device
from utils import run


def main():
    NUM_CLIENTS = 100
    NUM_TRAIN_CLIENTS = int(NUM_CLIENTS * .9)

    SIMULATION = False

    if SIMULATION:
        server = make_server()

        def client_fn(cid: str) -> FlowerClient:
            return FlowerClient()

        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        DEVICE = get_device()
        client_resources = {'num_cpus': 1, 'num_gpus': 1} if DEVICE.type == 'cuda' else None

        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_TRAIN_CLIENTS,
            server=server,
            config=fl.server.ServerConfig(num_rounds=1000),
            client_resources=client_resources,
        )
    else:
        p_server = run(['python3', 'my_server.py'], blocking=False)
        p_clients = [run(['python3', 'my_client.py'], blocking=False) for _ in range(9)]
        while True:
            for i, p in enumerate([p_server, *p_clients]):
                rc = p.poll(timeout=0.)
                while line_out := p.stdout.readline():
                    print(f'[C{i}]\t', line_out, end='')
                while line_err := p.stderr.readline():
                    print(f'[C{i}]\t', line_err, end='')
                # if rc is not None:
                #     break

    a = 0


if __name__ == '__main__':
    main()
