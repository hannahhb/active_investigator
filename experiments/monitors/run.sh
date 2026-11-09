#!/usr/bin/env bash
# Run from the repository root:  bash experiments/monitors/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Evaluation set (data/claims.csv) and passive monitors (Results 4.1). About $0.3.
python3 -m experiments.monitors.p3_build2
python3 -m experiments.monitors.p3_free
python3 -m experiments.monitors.p3_monitors --cap 1
python3 -m experiments.monitors.p3_monitors --model openai/gpt-5-mini --cap 1
python3 -m experiments.monitors.p3_score
python3 -m scripts.export_claims
