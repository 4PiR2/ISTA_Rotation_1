#!/bin/bash

#SBATCH --partition=gpu
###SBATCH --constraint="epsilon|delta|beta|leonid|serbyn|gamma|zeta"

#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus-per-node 1

#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=05:59:59

#Define the GPU architecture (GTX980 in the example, other options are GTX1080Ti, K40)
###SBATCH --constraint=RTX2080Ti

#SBATCH --job-name=Flower

#SBATCH --output=outputs/slurm/slurm-%A_%a.out

bash_args="$@"

srun --cpu_bind=verbose bash -c "
python3 launcher.py \\
--args \"slurm_job_nodelist=[$(scontrol show hostnames $SLURM_JOB_NODELIST)]\" \\
--num-step-clients 5 \\
$bash_args \\
&> outputs/slurm/slurm-$SLURM_JOBID-\$(hostname).out
"
