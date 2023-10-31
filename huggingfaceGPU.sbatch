#!/bin/bash
#SBATCH --job-name=huggingfaceGPU
#SBATCH --output=/scratch/users/%u/out/huggingfaceGPU.%j
#SBATCH --error=/scratch/users/%u/err/huggingfaceGPU.%j
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

source /farmshare/home/groups/srcc/cesta_workshop/miniconda3/bin/activate
python3 /farmshare/home/groups/srcc/cesta_workshop/huggingface/huggingface.py
