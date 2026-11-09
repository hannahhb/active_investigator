#!/usr/bin/env bash
# Run from the repository root:  bash experiments/group_size/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Figure 1a: monitor accuracy vs group size. About $6.
python3 -m experiments.group_size.p5_groupsize --cap 3
python3 -m experiments.group_size.p5_groupsize --model openai/gpt-5-mini --cap 8
python3 -m experiments.group_size.p5_groupsize --report
