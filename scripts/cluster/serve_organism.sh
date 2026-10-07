#!/bin/bash
# Serve the PyINE model organism with vLLM behind an OpenAI-compatible endpoint.
# Run from the conda env that has vllm installed (not through uv).
#
#   bash scripts/cluster/serve_organism.sh            # foreground
#   bash scripts/cluster/serve_organism.sh &          # background (the SLURM script does this)
set -euo pipefail

MODEL="${ORGANISM_MODEL:-plstcharles-saifh/pyine-v1-qwen3-4b-shortcut}"   # or a local checkpoint dir
SERVED_NAME="${ORGANISM_NAME:-pyine-organism}"                              # what the responder spec asks for
PORT="${ORGANISM_PORT:-8000}"
MAX_LEN="${ORGANISM_MAX_LEN:-32768}"   # the config declares 262k positions; 16k was too short for the longest records

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_LEN" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --trust-remote-code
