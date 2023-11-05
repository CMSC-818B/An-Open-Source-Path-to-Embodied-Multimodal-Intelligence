#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:a100:3
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=128GB
#SBATCH --output=/home/skarki/scratch/personalized-llm/logs/jupyter.log
#SBATCH --oversubscribe

. ~/.bashrc

conda activate llm
module load cuda/11.6.2

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
