#!/usr/bin/env bash
# Run from the repository root:  bash experiments/behaviours/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Table 1 (behaviours) and Table 2 (autosolver excerpt). About $1 of model calls.
python3 -m experiments.behaviours.filter_claims
python3 -m experiments.behaviours.pick_sample prepare
python3 -m experiments.behaviours.screen_claims --input to_screen.jsonl
python3 -m experiments.behaviours.pick_sample final
python3 -m experiments.behaviours.build_evidence out/sample.jsonl out/evidence.jsonl --hours 6
python3 -m experiments.behaviours.label_claims out/evidence.jsonl out/labels.jsonl
python3 -m experiments.behaviours.make_packets
python3 -m experiments.behaviours.recheck_unchecked
python3 -m experiments.behaviours.find_identifiers
python3 -m experiments.behaviours.triage
python3 -m experiments.behaviours.contest_scan
python3 -m experiments.behaviours.gaming_chat
python3 -m experiments.behaviours.autosolver_check
python3 -m experiments.behaviours.group_size
