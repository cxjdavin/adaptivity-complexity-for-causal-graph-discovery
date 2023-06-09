#!/bin/sh
#SBATCH --partition=long
#SBATCH --time=4320

srun setup.sh
for i in 1 2 3
do
    sbatch slurm-run.sh $i
done
