#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=fall2024-comp551

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024
source ~/.bashrc
conda activate base

echo "Test" > test_out.txt
nvidia_smi >> test_out.txt
python xxx.py






