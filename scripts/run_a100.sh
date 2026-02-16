#!/bin/bash

#SBATCH --partition=a100_only
#SBATCH --nodelist=hctlrds

#SBATCH --job-name=purge
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --tasks=1           
#SBATCH --time=48:00:00     ## Maximum Runtime (HH:MM:SS, default 48:00:00)
#SBATCH --gres=gpu:1        ## GPUs (amount, default 0)
#SBATCH --mem=80G           ## RAM (default 8GB)
#SBATCH --cpus-per-task=4   ## Cores (default 2)

##-------------------------------------------------------------------------------
python purge.py
