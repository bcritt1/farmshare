#!/bin/bash
#SBATCH --job-name=huggingface
#SBATCH --output=/home/users/%u/out/huggingface.%j
#SBATCH --error=/home/users/%u/err/huggingface.%j
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=32GB

source /home/groups/srcc/cesta_workshop/miniconda3/bin/activate
conda activate huggingface

python3 /home/groups/srcc/cesta_workshop/farmshare/huggingface.py
