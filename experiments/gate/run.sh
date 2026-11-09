#!/usr/bin/env bash
# Run from the repository root:  bash experiments/gate/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Gate replay: safety vs usefulness. Free.
python3 -m experiments.gate.p4_replay
