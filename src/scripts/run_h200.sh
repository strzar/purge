#!/bin/bash

#SBATCH --partition=h200_only
#SBATCH --nodelist=hctlrds2

#SBATCH --job-name=unlearn
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --tasks=1           
#SBATCH --time=48:00:00     ## Maximum Runtime (HH:MM:SS, default 48:00:00)
#SBATCH --gres=gpu:h200:1   ## GPUs (amount, default 0)
#SBATCH --mem=130G          ## RAM (default 8GB)
#SBATCH --cpus-per-task=4   ## Cores (default 2)

##-------------------------------------------------------------------------------
python src/grpo.py --multirun entity=stephen_king,confucius #bruce_lee,warren_buffett,christina_aguilera,cindy_crawford,marie_osmond,paris_hilton,justin_bieber,prince_harry_duke_of_sussex