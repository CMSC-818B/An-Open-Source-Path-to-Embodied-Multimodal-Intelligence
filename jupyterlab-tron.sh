#!/bin/zsh

#SBATCH --job-name=jupyter
#SBATCH --qos=high
#SBATCH --partition=tron
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --time=1-00:00:00
#SBATCH --mem=128GB
#SBATCH --output=/nfshomes/skarki/scratch/logs/jupyter-llame-tron.log

conda activate llm

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
