import argparse
from typing import Optional

from PeFLL.utils import str2bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Flower Server Arguments')

    #############################
    #       General args        #
    #############################
    parser.add_argument('--enable-slurm', type=str2bool, default=True)
    parser.add_argument('--mode', type=str, default='multiplex')
    parser.add_argument('--server-address', type=str, default=f'127.0.0.1:{18080}')
    parser.add_argument('--num-train-clients', type=int, default=int(100 * .9), help='number of nodes used in training')  # femnist: recommend int(3597 * .9)
    parser.add_argument('--num-step-clients', type=int, default=8, help='nodes to sample per round')
    parser.add_argument('--num-rounds', type=int, default=5000)  # 5000
    parser.add_argument('--init-round', type=int, default=0)
    parser.add_argument('--eval-interval', type=int, default=100, help='eval every X selected epochs')
    parser.add_argument('--eval-test', type=str2bool, default=False, help='eval on test dataset')
    parser.add_argument('--log-dir', type=str, default='./outputs', help='dir path for output file')
    parser.add_argument('--experiment-name', type=str, default=None, help='experiment name')
    parser.add_argument('--server-seed', type=int, default=42, help='seed value')
    parser.add_argument('--args', type=str, default=None, help='other information')

    #############################
    #       Dataset Args        #
    #############################
    parser.add_argument('--client-dataset-seed', type=int, default=42, help='seed value')
    parser.add_argument('--client-dataset-data-name', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'femnist'])
    parser.add_argument('--client-dataset-data-path', type=str, default='./dataset', help='dir path for datasets')
    parser.add_argument('--client-dataset-num-clients', type=int, default=100,  help='number of simulated nodes')  # femnist: reset 3597
    parser.add_argument('--client-dataset-batch-size', type=int, default=32)
    parser.add_argument('--client-dataset-partition-type', type=str, default='by_class', choices=['by_class', 'dirichlet'])
    parser.add_argument('--client-dataset-alpha-train', type=float, default=.1, help='alpha for train clients')
    parser.add_argument('--client-dataset-alpha-test', type=float, default=.1, help='alpha for test clients')

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument('--model-num-kernels', type=int, default=16, help='number of kernels for cnn model')
    parser.add_argument('--model-embed-type', type=str, default='cnn', choices=['none', 'cnn', 'mlp'], help='embed with')
    parser.add_argument('--model-embed-dim', type=int, default=-1, help='embedding dim')
    parser.add_argument('--client-model-embed-y', type=str2bool, default=True, help='embed y as well as x')
    parser.add_argument('--model-hyper-hid-layers', type=int, default=3, help='num. hidden layers hypernetwork')
    parser.add_argument('--model-hyper-hid-dim', type=int, default=100, help='hypernet hidden dim')

    ##################################
    #       Train Eval args        #
    ##################################
    parser.add_argument('--client-optimizer-target-lr', type=float, default=2e-3, help='learning rate for inner optimizer')
    parser.add_argument('--client-optimizer-target-momentum', type=float, default=.9)
    parser.add_argument('--client-optimizer-target-weight-decay', type=float, default=5e-5, help='inner weight decay')
    parser.add_argument('--client-target-num-batches', type=int, default=50, help='number of inner steps')
    parser.add_argument('--optimizer-embed-type', type=str, default='adam', choices=['adam', 'sgd'], help='learning rate')
    parser.add_argument('--optimizer-embed-lr', type=float, default=2e-4, help='embedding learning rate')
    parser.add_argument('--optimizer-embed-weight-decay', type=float, default=1e-3, help="weight decay")
    parser.add_argument('--client-embed-num-batches', type=int, default=1, help='batches used to estimate rescaling')
    parser.add_argument('--optimizer-hyper-type', type=str, default='adam', choices=['adam', 'sgd'], help='learning rate')
    parser.add_argument('--optimizer-hyper-lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--optimizer-hyper-weight-decay', type=float, default=1e-3, help="weight decay")
    parser.add_argument('--client-eval-mask-absent', type=str2bool, default=True, help='mask absent classes at eval')
    parser.add_argument('--client-eval-embed-train-split', type=bool, default=True, help='use train or test data to embed')

    args = parser.parse_args()
    return args
