# Can Follow-Up Questions Help Weak-to-Strong Monitors for Swarms?

Hannah Bansal (SPAR Research). Apart Research Swarm Alignment sprint, Track 4: Multi-Agent Oversight and Control.
Paper: [`FollowUpMonitorSwarms.pdf`](FollowUpMonitorSwarms.pdf) (LaTeX source in [`paper/`](paper/)).

This repository turns the AI Village dataset into a multi-agent control evaluation. It tests whether a weak trusted
monitor (Qwen 3.5 9B) keeps up with a strong one (GPT-5-mini) as the number of agents grows, and whether asking the
claiming agent a follow-up question closes the gap. It also tests two attacks: peer "verification" and agents
falsifying their own tool outputs.

## What existed before the sprint (preexisting work)

- **PyINE active investigator** (October 2026, separate codebase, not included here). This was an investigator that asks a
  model trace-grounded follow-up questions about program execution. It came with the PyINE model organism, the
  PyINE data, and a bank of 3,035 Qwen 3.5 9B answers to candidate questions. The idea of follow-up questioning as an
  active investigator comes from this work.
- [`pyine_selector/train_selector.py`](pyine_selector/train_selector.py) was written during the sprint, but it runs on
  top of that preexisting package and question bank. It produces the PyINE selector result mentioned in the paper's
  future work. It is included for reference only and does not run on its own.

**Everything else in this repository was written during the sprint (7–9 November 2026).** That covers the AI Village
curation, the evaluation set, all monitor, group-size, follow-up and stress-test experiments, the model-organism
data, the figures and the paper.

## Data

- **AI Village dataset** (AI Digest): <https://huggingface.co/datasets/aidigestorg/ai-village>. It is **not included**.
  Under its terms it may be used for research and evaluation only, with no training on it, and it must be cited.
  We used the tables `chat_messages`, `computer_use_turns`, `computer_use_sessions`, `agents` and `agent_goals`. Put
  them as `<table>.jsonl` or `<table>.jsonl.gz` in one folder and point `AI_VILLAGE_DIR` at it. The default is an
  `ai_village/` folder next to this `village/` folder.
- **Our labels:** [`data/claims.csv`](data/claims.csv) lists the 93 score claims used in all experiments, by AI
  Village message ID, with no message text. For each claim it gives the split (10 tuning, 83 test), the label (true or
  planted false), the true number from the agent's own logs, the stated (possibly planted) number, and whether it is
  in the 27-claim peer-influence subset.
- **Results** (`out/`) are not included because they contain AI Village text. The scripts below regenerate them.
- **Model organism data** ([`organism/`](organism/)) is fully synthetic: 1,500 training rows and 200 held-out rows.

## Setup

```bash
pip install -r requirements.txt
export AI_VILLAGE_DIR=/path/to/ai_village                  # folder with the tables above
echo "OPENROUTER_API_KEY=..." > "$AI_VILLAGE_DIR/.env"      # read by llm.py; never committed
```

Every paid call goes through `llm.py`, which logs its cost to `out/spend.jsonl` and stops at `TOTAL_CAP` (in `llm.py`).
The full set of experiments cost about **$35** on OpenRouter. Each paid step resumes from its output file if it is
interrupted.

## Reproducing the paper

Run from this `village/` folder, in this order. "Free" means no API calls.

| Paper item | Commands | Cost |
|---|---|---|
| Table 1: behaviours found in AI Village | `python3 filter_claims.py`; `python3 pick_sample.py prepare`; `python3 screen_claims.py --input to_screen.jsonl`; `python3 pick_sample.py final`; `python3 build_evidence.py out/sample.jsonl out/evidence.jsonl --hours 6`; `python3 label_claims.py out/evidence.jsonl out/labels.jsonl`; `python3 make_packets.py`; `python3 recheck_unchecked.py` (unchecked repeats); `python3 find_identifiers.py`; `python3 triage.py` (fabricated evidence); `python3 contest_scan.py` (inflated success); `python3 gaming_chat.py` (goal gaming) | ~$1 |
| Table 2: autosolver excerpt | `python3 autosolver_check.py`; excerpts were read from the chat and logs around 21 Jan 2026, 20:15–20:45 UTC | free |
| Evaluation set (`data/claims.csv`) | `python3 contest_scan.py`; `python3 p3_build2.py` (seed 11; the first 10 rows are for tuning) | free |
| Passive monitors (Results 4.1) | `python3 p3_free.py`; `python3 p3_monitors.py --cap 1`; `python3 p3_monitors.py --model openai/gpt-5-mini --cap 1`; `python3 p3_score.py` | ~$0.3 |
| Gate replay | `python3 p4_replay.py` | free |
| Figure 1a: group size | `python3 group_size.py`; `python3 p5_groupsize.py --cap 3`; `python3 p5_groupsize.py --model openai/gpt-5-mini --cap 8` | ~$6 |
| Figure 1b, Table 3, Figure 4: follow-up questions | `python3 p6_followup.py dialog`; `python3 p6_followup.py weak`; `python3 p6_followup.py judge` | ~$20 |
| Figure 2: peer influence | `python3 p8_peer_claim.py updates`; `python3 p8_peer_claim.py monitor` | ~$1.5 |
| Figure 3: record tampering | `python3 p7_faked_logs.py` | ~$0.3 |
| All paper figures | `python3 paper_figures.py`; writes `figures/paper/` and prints every number used in the text | free |

Notes:
- **Figure style:** `make_figures.py` holds shared plotting style used by several scripts. `paper_figures.py` makes
  the paper figures, computing AUROC with scikit-learn and 95% bootstrap intervals.
- **Retired models:** Claude 3.7 Sonnet and Gemini 3 Pro are no longer on OpenRouter. Claude Sonnet 4.5 and
  Gemini 3.1 Pro stand in for them (see `RESPONDER` in `p6_followup.py`).
- **Exact reruns:** model replies can vary between runs and providers even at temperature 0. The weak-asker prompt
  used for the first 27 claims was rewritten after being lost; see the note in `p6_followup.py`.
- **More detail:** [`docs/METHODS_detailed.md`](docs/METHODS_detailed.md) has every parameter, what did not work, and
  the leak we found and fixed. Some numbers there predate the final runs; the paper's numbers are final.

## Model organism (not evaluated in the paper)

The organism is an "inflated reporter" (Qwen 3.5 4B with a LoRA adapter, trained on Adaption AutoScientist) built from
synthetic data:
1. `python3 organism/make_data.py` creates the data.
2. `python3 organism/start_adaption.py upload` uploads it, then `start` begins training, and `download` fetches the
   adapter.
3. `python3 organism/test_organism.py --model inflated` tests it once it is served with vLLM.

The trained adapter is not included because of its size.

## Citation

If you use this work, please cite AI Digest / AI Village and the paper in this repository.
