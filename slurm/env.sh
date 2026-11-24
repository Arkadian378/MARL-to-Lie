# Edit once for your cluster. Sourced by every job script.
export CPB_ROOT="${CPB_ROOT:-$HOME/corrupt-police-bench}"
export CPB_VENV="${CPB_VENV:-$CPB_ROOT/.venv}"
export HF_HOME="${HF_HOME:-$CPB_ROOT/.hf_cache}"       # model weights cache (download once on a login node)
export SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"     # <- your partition
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"            # <- your account, if required
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
source "$CPB_VENV/bin/activate"
cd "$CPB_ROOT"
