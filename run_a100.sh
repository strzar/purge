#!/bin/bash

## Specify Node, Partition

#SBATCH --partition=a100_only
#SBATCH --nodelist=hctlrds

## Specify Job

#SBATCH --job-name=unlearn
#SBATCH --output=./slurm/slurm_out-%x-%j.out
#SBATCH --error=./slurm/slurm_err-%x-%j.err

## Specify Resources

#SBATCH --tasks=1           
#SBATCH --time=48:00:00     ## HH:MM:SS, maximum runtime. Default 48:00:00
#SBATCH --gres=gpu:1  ## GPUs (amount, either 0 or 1)
#SBATCH --mem=80G           ## RAM (default 8GB)
#SBATCH --cpus-per-task=4   ## cores (default 2)

#Run your program / code here
python grpo.py