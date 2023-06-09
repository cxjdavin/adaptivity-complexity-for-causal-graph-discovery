#!/bin/sh
#SBATCH --partition=long
#SBATCH --time=4320

srun run.sh $1
