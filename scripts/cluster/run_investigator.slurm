#!/bin/bash
#SBATCH -A punim0478
#SBATCH --time=0-8:00:00
#SBATCH --ntasks=1
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=investigator
set -eo pipefail

BASE=/data/gpfs/projects/punim0478/bansaab
PROBLEMS="${PROBLEMS:-40}"

module load CUDA
source "/home/${USER}/.bashrc" || true   # its last command may return nonzero; do not let that kill the job
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$BASE/.cache/huggingface" UV_CACHE_DIR="$BASE/.cache/uv" TMPDIR="$BASE/tmp"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
echo "vllm: $(command -v vllm || echo MISSING) | uv: $(command -v uv || echo MISSING)"

cd "$BASE/active_investigator"
vllm serve plstcharles-saifh/pyine-v1-qwen3-4b-shortcut --served-model-name pyine-organism \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000 \
  > "server.${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 5; done

OUT="runs/onpath-organism-p${PROBLEMS}-${SLURM_JOB_ID}"
uv run active-investigator run --arm on_path -p "$PROBLEMS" --k 3 --strip-hints --score-mode consistency --reconcile \
  --responder vllm:pyine-organism@http://localhost:8000/v1 --max-workers 48 --out "$OUT"
cat "$OUT/scorer_metadata.json"
