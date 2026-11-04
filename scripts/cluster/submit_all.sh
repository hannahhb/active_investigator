#!/bin/bash
# Submit a slurm script to every usable GPU partition, one copy each, skipping partitions where a job of
# ours is already pending. The script's first running copy cancels the rest (same job name).
#   NAME=organism bash scripts/cluster/submit_all.sh [script]
declare -A QOS=([gpu-a100-short]=normal [gpu-l40s-preempt]=normal [gpu-a100-preempt]=normal [gpu-l40s]=normal
                [gpu-a100]=normal [feit-gpu-a100]=feit [deeplearn]=gpgpudeeplearn [gpu-h100]=normal)
SCRIPT=${1:-scripts/cluster/serve_remote.slurm}
for p in "${!QOS[@]}"; do
  if [ -n "$(squeue -u "$USER" -p "$p" -t PENDING -h -o %i)" ]; then echo "skip $p: pending job"; continue; fi
  sbatch --export=ALL --job-name="${NAME:-organism}" --partition="$p" --qos="${QOS[$p]}" "$SCRIPT"
done
