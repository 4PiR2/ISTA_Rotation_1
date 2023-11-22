import argparse
import os
import subprocess
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb


def aggregate_tensor(results: List[Tuple[List[torch.Tensor], int]]) -> List[torch.Tensor]:
    """Compute weighted average."""
    # flwr.server.strategy.aggregate.aggregate
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(torch.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def aggregate_tensor_2(results: List[Tuple[List[torch.Tensor], int]]) -> List[torch.Tensor]:
    """Compute weighted average."""
    layer_updates: List[List[torch.Tensor]] = [x for x, _ in results]
    weight = torch.tensor([v for _, v in results], dtype=layer_updates[0][0].dtype, device=layer_updates[0][0].device)
    weight /= weight.sum()
    return [(torch.stack(x, dim=0) * weight[(...,) + (None,) * x[0].dim()]).sum(dim=0) for x in zip(*layer_updates)]


def state_dicts_to_ndarrays(state_dicts: Dict[str, Dict[str, torch.Tensor]]) -> List[np.ndarray]:
    header: List[str] = []
    keys: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    for net_name, state_dict in state_dicts.items():
        header.append(net_name)
        keys.append(np.asarray(list(state_dict.keys())))
        vals.extend([val.detach().cpu().numpy() for val in state_dict.values()])
    ndarrays: List[np.ndarray] = [np.asarray(header)] + keys + vals
    return ndarrays


def ndarrays_to_state_dicts(
        ndarrays: List[np.ndarray],
        device: torch.device = torch.device('cpu'),
) -> Dict[str, Dict[str, torch.Tensor]]:
    state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
    for i, net_name in enumerate(ndarrays[0]):
        keys = ndarrays[1 + i]
        vals = ndarrays[1 + sum([len(k) for k in ndarrays[:1 + i]]): 1 + sum([len(k) for k in ndarrays[:2 + i]])]
        state_dicts[net_name] = {k: torch.tensor(v, device=device) for k, v in zip(keys, vals)}
    return state_dicts


def run(cmd: List[str], cwd: Optional[str] = None, env=None, shell=False, blocking: bool = True):
    print(f'>>> {cwd} $ {cmd if isinstance(cmd, str) else " ".join(cmd)}', flush=True)
    p = subprocess.Popen(cmd, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         shell=shell, cwd=cwd, env=env, universal_newlines=True, bufsize=1)
    os.set_blocking(p.stdout.fileno(), False)
    os.set_blocking(p.stderr.fileno(), False)

    if not blocking:
        return p

    lines = []
    lines_out = []
    lines_err = []
    while True:
        rc = p.poll()
        while line_out := p.stdout.readline():
            lines.append(line_out)
            lines_out.append(line_out)
            print(line_out, end='', flush=True)
        while line_err := p.stderr.readline():
            lines.append(line_err)
            lines_err.append(line_err)
            print(line_err, end='', flush=True)
        if rc is not None:
            break
    print('EXIT CODE:', rc, flush=True)
    # assert rc == 0
    return rc, ''.join(lines), ''.join(lines_out), ''.join(lines_err)


def init_wandb(args: argparse.Namespace, experiment_name: Optional[str] = None):
    # return False
    try:
        with open('wandb_token.txt', 'r') as f:
            wandb_token = f.readline().strip()
        wandb_login = wandb.login(key=wandb_token)
    except Exception as _:
        wandb_login = False
    if wandb_login:
        wandb.init(
            dir=args.log_dir,
            # Set the project where this run will be logged
            project='test',
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=experiment_name,
            # Track hyperparameters and run metadata
            config={
                **vars(args),
            })
    return wandb_login


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()  # Mark the run as finished


def get_pefll_checkpoint():
    state_dict = torch.load(os.path.join(
        'PeFLL',
        'saved_models',
        'cifar10_100_nodes_90_trainnodes__partition_by_class_alphatrain_0.1_alphatest_0.1_seed_42',
        'step_4999.ckpt',
    ), map_location=torch.device('cpu'))
    return state_dict
