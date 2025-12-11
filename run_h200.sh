#!/bin/bash

## Specify Node, Partition

#SBATCH --partition=h200_only
#SBATCH --nodelist=hctlrds2

## Specify Job

#SBATCH --job-name=unlearn
#SBATCH --output=./slurm/slurm_out-%x-%j.out
#SBATCH --error=./slurm/slurm_err-%x-%j.err

## Specify Resources

#SBATCH --tasks=1           
#SBATCH --time=48:00:00     ## HH:MM:SS, maximum runtime. Default 48:00:00
#SBATCH --gres=gpu:h200:1   ## GPUs (amount, either 0 or 1)
#SBATCH --mem=130G          ## RAM (default 8GB)
#SBATCH --cpus-per-task=4   ## cores (default 2)

#Run your program / code here
python grpo.py