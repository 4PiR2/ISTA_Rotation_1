import argparse
import os
import subprocess
from typing import List, Optional

import wandb


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
    try:
        with open('wandb_token.txt', 'r') as f:
            wandb_token = f.readline().strip()
        wandb_login = wandb.login(key=wandb_token)
    except Exception as _:
        wandb_login = False
    if wandb_login:
        wandb.init(
            # Set the project where this run will be logged
            project='test',
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=experiment_name,
            # Track hyperparameters and run metadata
            config={
                **vars(args),
            })


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()  # Mark the run as finished
