#!/bin/bash
#SBATCH --job-name=huggingface
##SBATCH --output=/scratch/users/%u/out/huggingface.%j
##SBATCH --error=/scratch/users/%u/err/huggingface.%j
#SBATCH --output=/home/%u/out/huggingface.%j
#SBATCH --error=/home/%u/err/huggingface.%j
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=32GB

source /farmshare/home/groups/srcc/cesta_workshop/miniconda3/bin/activate
python3 /farmshare/home/groups/srcc/cesta_workshop/huggingface/huggingface.py
