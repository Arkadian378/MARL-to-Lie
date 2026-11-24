#!/bin/bash
#SBATCH --job-name=cpb-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
# One RL run on one 80 GB GPU (vLLM generation + LoRA training on the same card).
# A full-game run is longer than most time limits: 15 minutes before the limit the job requeues
# itself and resumes from state.json (the last finished step), so dependencies still work.
#   sbatch slurm/train.sbatch training/configs/step3_corrupt.yaml --set name=... --set seed=1 ...
source slurm/env.sh
python training/grpo.py --config "$@" &
PID=$!
trap 'echo "time limit close: requeue"; kill -TERM $PID; wait $PID; scontrol requeue $SLURM_JOB_ID; exit 0' USR1
wait $PID
