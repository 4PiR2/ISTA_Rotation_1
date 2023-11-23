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

mkdir -p outputs/slurm
#SBATCH --output=outputs/slurm/out.txt

srun --cpu_bind=verbose bash -c "
python3 launcher.py
--args \"slurm_job_nodelist=[$(scontrol show hostnames $SLURM_JOB_NODELIST)]\"
--num-step-clients 5
--num-rounds 1000
--eval-interval 0
&> outputs/slurm/out_\$(hostname).txt
"
