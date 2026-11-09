#!/usr/bin/env bash
# Reproduce every result in the paper, in order (about $35 of OpenRouter calls), then the figures.
set -euo pipefail
cd "$(dirname "$0")"
for phase in behaviours monitors gate group_size followup stress_tests; do
  bash "experiments/$phase/run.sh"
done
python3 -m scripts.paper_figures
