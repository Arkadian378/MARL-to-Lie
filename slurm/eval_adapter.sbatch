#!/bin/bash
#SBATCH --job-name=cpb-eval-lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Evaluates a trained run on 200 games of its own condition.
#   sbatch slurm/eval_adapter.sbatch <train_run_dir> <eval_name> [--set game.overrides.x=y ...]
source slurm/env.sh
RUN="$1"; NAME="$2"; shift 2
LORAS=("trained=$RUN/final/corrupt")
SETS=()
if [ -d "$RUN/final/cop" ]; then           # self-play run: also serve the trained police
  LORAS+=("trained_cop=$RUN/final/cop")
  SETS+=(--set "backends.qwen_cop={type: openai, base_url: 'http://localhost:8000/v1', model: trained_cop, max_tokens: 512, temperature: 0.7, extra_body: {chat_template_kwargs: {enable_thinking: false}}}")
  SETS+=(--set "seats.cop={type: llm, backend: qwen_cop}")
fi
source slurm/serve_vllm.sh qwen9b "${LORAS[@]}"
python experiments/evaluate.py --config experiments/configs/eval_trained.yaml \
  --set name="$NAME" --set out_dir="runs/eval/$NAME" "${SETS[@]}" "$@"
