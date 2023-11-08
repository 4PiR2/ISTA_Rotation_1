import gc
import os
import time

import torch

from PeFLL.models import CNNTarget as Net


def main(test_id: int):
    n_devices = 8
    n_repetition = 10
    if test_id == 1:
        obj = Net()
    elif test_id == 2:
        obj = torch.empty(int(1e9), dtype=torch.int8)
    elif test_id == 3:
        obj = torch.empty(1)
    elif test_id == 4:
        obj = torch.empty(0)
    memory = []
    for i in range(n_devices):
        device = torch.device(f'cuda:{i}')
        for j in range(n_repetition):
            t_start = time.time_ns()

            if test_id in [1, 2, 3, 4]:
                new_obj = obj.to(device)
            else:
                new_obj = torch.empty(int(1e9), dtype=torch.int8, device=device)

            t_end = time.time_ns()
            t = (t_end - t_start) * 1e-9
            print(f'device {i:1} rep {j:3} : {t:8.6f} s', flush=True)
            memory.append(new_obj)
    os.system('nvidia-smi')


if __name__ == '__main__':
    main(test_id=1)
    gc.collect()
    os.system('nvidia-smi')
    torch.cuda.empty_cache()
    os.system('nvidia-smi')
