import time

from PeFLL.dataset import gen_random_loaders


def main(test_id: int):
    n_repetition = 100
    for j in range(n_repetition):
        t_start = time.time_ns()

        gen_random_loaders('cifar10', './dataset', 100, 90, 32, 'by_class', 2, None, None, None)

        t_end = time.time_ns()
        t = (t_end - t_start) * 1e-9
        print(f'rep {j:3} : {t:8.6f} s', flush=True)


if __name__ == '__main__':
    main(test_id=1)
