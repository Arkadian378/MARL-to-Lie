#!/bin/bash
#SBATCH --job-name=cpb-step6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
# Step 6: frontier models (API) and the trained Qwen on the same seeds.
# Needs ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY in the environment.
#   sbatch slurm/eval_step6.sbatch <train_run_dir> [--cell A_opus_corrupt ...] [--episodes 20]
source slurm/env.sh
RUN="$1"; shift
source slurm/serve_vllm.sh qwen9b "trained=$RUN/final/corrupt"
python experiments/evaluate.py --config experiments/configs/step6_frontier.yaml "$@"
