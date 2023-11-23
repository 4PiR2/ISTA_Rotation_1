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


def detect_slurm():
    return 'SLURMD_NODENAME' in os.environ


def parse_node_list(string: str) -> List[str]:
    import regex

    # https://hpcc.umd.edu/hpcc/help/slurmenv.html
    # scontrol show hostnames 'compute-b24-[1-3,5-9],compute-b25-[1,4,8]'
    # nodelist_str = 'compute-b24-[1-3,5-9,34],bigterra141,compute-b25-[1,4,8],compute-b89-[201-202]'

    pattern_name = fr'[^\[,\]]+'
    pattern_id = fr'\d+'
    pattern_ids = fr'(?P<id_1>{pattern_id})(-(?P<id_2>{pattern_id}))?'
    pattern_id_group = fr'((?P<ids_1>{pattern_ids}),)*(?P<ids_2>{pattern_ids})'
    pattern_group = fr'(?P<name>{pattern_name})(\[(?P<id_group>{pattern_id_group})\])?'
    pattern = fr'^((?P<group_1>{pattern_group}),)*(?P<group_2>{pattern_group})$'

    pattern_ids = regex.compile(pattern_ids)
    pattern_id_group = regex.compile(pattern_id_group)
    pattern_group = regex.compile(pattern_group)
    pattern = regex.compile(pattern)

    nodes = []
    matches = pattern.fullmatch(string)
    if matches is not None:
        for group in matches.captures('group_1') + matches.captures('group_2'):
            matches_group = pattern_group.fullmatch(group)
            name = matches_group.group('name')
            id_groups = matches_group.group('id_group')
            if id_groups is None:
                nodes.append(name)
                continue
            matches_id_group = pattern_id_group.fullmatch(id_groups)
            for ids in matches_id_group.captures('ids_1') + matches_id_group.captures('ids_2'):
                matches_ids = pattern_ids.fullmatch(ids)
                id_1 = matches_ids.group('id_1')
                id_2 = matches_ids.group('id_2')
                if id_2 is None:
                    nodes.append(f'{name}{id_1}')
                    continue
                for id in range(int(id_1), int(id_2) + 1):
                    nodes.append(f'{name}{id}')
    return nodes


    # return False
def init_wandb(args: argparse.Namespace, experiment_name: Optional[str] = None, group: Optional[str] = None):
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
            group=group,
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
