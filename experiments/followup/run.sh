#!/usr/bin/env bash
# Run from the repository root:  bash experiments/followup/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Figure 1b, Table 3, appendix figure: follow-up questions. About $20.
python3 -m experiments.followup.p6_followup dialog
python3 -m experiments.followup.p6_followup weak
python3 -m experiments.followup.p6_followup judge
python3 -m experiments.followup.p6_followup report
