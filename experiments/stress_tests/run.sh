#!/usr/bin/env bash
# Run from the repository root:  bash experiments/stress_tests/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Figure 2 (peer influence) and Figure 3 (record tampering). About $2.
python3 -m experiments.stress_tests.p8_peer_claim updates
python3 -m experiments.stress_tests.p8_peer_claim monitor
python3 -m experiments.stress_tests.p8_peer_claim report
python3 -m experiments.stress_tests.p7_faked_logs
python3 -m experiments.stress_tests.p7_faked_logs --report
