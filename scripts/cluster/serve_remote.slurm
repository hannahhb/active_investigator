#!/bin/bash
# Serve the organism with vLLM on 0.0.0.0:8000 so a driver on another node can query it, and write the
# host to slurm_out/organism_host.txt. The first copy to start cancels its pending siblings (same job name).
#   bash scripts/cluster/submit_all.sh
#SBATCH -A punim0478
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:15:00
#SBATCH --output=slurm_out/%x_%j.txt
set -eo pipefail
scancel --name="$SLURM_JOB_NAME" --user="$USER" --state=PENDING || true
BASE=/data/gpfs/projects/punim0478/bansaab
module load CUDA || true
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$BASE/.cache/huggingface" TMPDIR="$BASE/tmp"
cd "$BASE/active_investigator"
nvidia-smi --query-gpu=name,memory.total --format=csv
vllm serve "${ORGANISM_MODEL:-plstcharles-saifh/pyine-v1-qwen3-4b-shortcut}" --served-model-name pyine-organism \
  --host 0.0.0.0 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000 \
  --enable-prefix-caching > "slurm_out/server.${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }
  sleep 5
done
echo "$(hostname) $SLURM_JOB_ID" > slurm_out/organism_host.txt
echo "serving on $(hostname):8000"
wait $SERVER
