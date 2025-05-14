#!/bin/bash
#SBATCH --job-name=huggingface
#SBATCH --output=/home/%u/out/huggingface.%j
#SBATCH --error=/home/%u/err/huggingface.%j
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=32GB

ml python/3.12
python3 /farmshare/learning/huggingface_workshop/huggingface.py
