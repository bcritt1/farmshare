#!/bin/bash
#SBATCH --job-name=huggingface
#SBATCH --output=/home/users/%u/out/huggingface.%j
#SBATCH --error=/home/users/%u/err/huggingface.%j
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=32GB

source /home/users/$USER/.bashrc
source activate /farmshare/home/groups/srcc/cesta_workshop/miniconda3/envs/huggingface
python3 /farmshare/home/groups/srcc/cesta_workshop/huggingface/huggingface.py
