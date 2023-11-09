import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Flower Server Arguments')
    parser.add_argument('--server-address', type=str, default=f'127.0.0.1:{18080}')
    parser.add_argument('--mode', type=str, default='multiplex')
    # parser.add_argument('--num-train-clients', type=int, default=int(100 * .9))
    parser.add_argument('--min-fit-clients', type=int, default=8)
    parser.add_argument('--num-rounds', type=int, default=1000)
    parser.add_argument('--init-round', type=int, default=0)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--optimizer-inner-lr', type=float, default=2e-3)
    parser.add_argument('--optimizer-inner-momentum', type=float, default=.9)
    parser.add_argument('--optimizer-inner-weight-decay', type=float, default=5e-5)
    parser.add_argument('--client-data-seed', type=int, default=42)
    parser.add_argument('--client-data-data-name', type=str, default='cifar10')
    parser.add_argument('--client-data-data-path', type=str, default='./dataset')
    parser.add_argument('--client-data-num-clients', type=int, default=100)
    parser.add_argument('--client-data-batch-size', type=int, default=32)
    parser.add_argument('--client-data-partition-type', type=str, default='by_class')
    parser.add_argument('--client-data-classes-per-user', type=int, default=2)
    parser.add_argument('--client-data-alpha-train', type=float, default=None)
    parser.add_argument('--client-data-alpha-test', type=float, default=None)
    parser.add_argument('--client-data-embedding-dir-path', type=str, default=None)
    args = parser.parse_args()
    return args
