#!/bin/bash
# Starts a vLLM OpenAI server in the background and waits until it answers.
#   source slurm/serve_vllm.sh [served_name] [lora_name=adapter_path ...]
# The base model is served as "$1" (default qwen9b); each extra argument adds a LoRA adapter.
NAME="${1:-qwen9b}"; shift || true
LORA_ARGS=()
if [ "$#" -gt 0 ]; then LORA_ARGS=(--enable-lora --max-lora-rank 32 --lora-modules "$@"); fi
PORT="${VLLM_PORT:-8000}"
vllm serve "$BASE_MODEL" --served-model-name "$NAME" --port "$PORT" --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching "${LORA_ARGS[@]}" > "vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 180); do
  if curl -s "http://localhost:$PORT/v1/models" > /dev/null; then echo "vLLM ready on :$PORT"; break; fi
  sleep 10
done
trap "kill $VLLM_PID" EXIT
