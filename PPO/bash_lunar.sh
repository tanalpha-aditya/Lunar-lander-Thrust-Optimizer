#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=30000
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gnode077
#SBATCH --output=Stdout_lunar.txt

python3 PPO_Gym_Ablations.py
