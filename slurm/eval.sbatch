#!/bin/bash
#SBATCH --job-name=cpb-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Untrained-model evaluation (steps 1, 2 and the Qwen part of 6):
#   sbatch slurm/eval.sbatch experiments/configs/step1_sanity.yaml [extra evaluate.py args]
source slurm/env.sh
source slurm/serve_vllm.sh qwen9b
python experiments/evaluate.py --config "$@"
