#!/bin/bash
#SBATCH -A punim0478
#SBATCH --time=0-8:00:00
#SBATCH --ntasks=1
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --job-name=organism
#
# Serve the model organism for the whole walltime, and run any evals submitted to it.
#
#   sbatch scripts/cluster/serve_only.slurm
#
# Submit work from anywhere, no terminal needed, by appending a command to the queue file:
#
#   echo 'uv run active-investigator run --arm fixed -p 5 --responder vllm:pyine-organism@http://localhost:8000/v1 --out runs/fixed-p5' >> slurm_out/queue.txt
#
# Each line is run in turn from the repo directory; its output goes to slurm_out/queue-<n>.log.
# Or attach a shell to the allocation:  srun --jobid=<jobid> --overlap --pty bash
set -eo pipefail

BASE=/data/gpfs/projects/punim0478/bansaab
QUEUE="slurm_out/queue.txt"

module load CUDA
source "/home/${USER}/.bashrc" || true
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$BASE/.cache/huggingface" UV_CACHE_DIR="$BASE/.cache/uv" TMPDIR="$BASE/tmp"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

cd "$BASE/active_investigator"
mkdir -p slurm_out
vllm serve plstcharles-saifh/pyine-v1-qwen3-4b-shortcut --served-model-name pyine-organism \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000 \
  > "slurm_out/server.${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server died; see slurm_out/server.${SLURM_JOB_ID}.log"; exit 1; }
  sleep 5
done
echo "organism serving on $(hostname):8000 (job $SLURM_JOB_ID); append commands to $QUEUE"

touch "$QUEUE"
done_count=0
while kill -0 $SERVER 2>/dev/null; do
  total=$(wc -l < "$QUEUE")
  while [ "$done_count" -lt "$total" ]; do
    done_count=$((done_count + 1))
    cmd=$(sed -n "${done_count}p" "$QUEUE")
    [ -z "$cmd" ] && continue
    echo "[$(date +%H:%M:%S)] running #$done_count: $cmd"
    bash -c "$cmd" > "slurm_out/queue-${done_count}.log" 2>&1 || echo "  #$done_count failed; see slurm_out/queue-${done_count}.log"
  done
  sleep 30
done
