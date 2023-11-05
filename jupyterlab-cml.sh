#!/bin/zsh

#SBATCH --job-name=jupyter
#SBATCH --qos=cml-high
#SBATCH --account=cml-tokekar
#SBATCH --partition=cml-dpart
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --output=/nfshomes/skarki/scratch/logs/jupyter-llame-llm-cml.log

source ~/.zshrc
conda activate llm
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
