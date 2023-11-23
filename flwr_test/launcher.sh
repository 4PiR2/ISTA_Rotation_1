#!/bin/bash

#SBATCH --partition=gpu
###SBATCH --constraint="epsilon|delta|beta|leonid|serbyn|gamma|zeta"

#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G

# number of GPUs per node
#SBATCH --gres=gpu:1

#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=00:10:00

#Define the GPU architecture (GTX980 in the example, other options are GTX1080Ti, K40)
###SBATCH --constraint=GTX1080Ti

#SBATCH --job-name=Flower
#SBATCH --output=outputs/slurm/out.txt

#Send emails when a job starts, it is finished or it exits
###SBATCH --mail-user=YourEmail@ist.ac.at
###SBATCH --mail-type=ALL

#Pick whether you prefer requeue or not. If you use the --requeue
#option, the requeued job script will start from the beginning,
#potentially overwriting your previous progress, so be careful.
#For some people the --requeue option might be desired if their
#application will continue from the last state.
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue

unset SLURM_EXPORT_ENV

#run the binary through SLURM's srun
srun --cpu_bind=verbose bash -c "python3 launcher.py --num-step-clients 5 --num-rounds 1000 >> outputs/slurm/out_\$(hostname).txt"
