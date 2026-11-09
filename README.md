# Can Follow-Up Questions Help Weak-to-Strong Monitors for Swarms?

**Hannah Bansal** (SPAR Research) · Apart Research Swarm Alignment sprint, Track 4: Multi-Agent Oversight and Control ·
[project page](https://apartresearch.com/sprints/projects/can-followup-questions-help-weaktostrong-monitors-for-swarms-wpcsuqz0d4) ·
[paper (PDF)](paper/FollowUpMonitorSwarms.pdf)

We turn real AI Village deployment data into a multi-agent control evaluation. A weak trusted monitor
(Qwen 3.5 9B) loses accuracy as the number of agents it watches grows, while a strong one (GPT-5-mini) does not.
Asking the claiming agent a single well-chosen follow-up question recovers **76–82% of the weak-to-strong gap**.
We also show that a peer's "verified" doubles the spread of a false claim, and that agents falsifying their own tool
outputs defeat log-based monitoring.

## Repository layout

```
experiments/            one folder per phase, each with a run.sh
  behaviours/           find misbehaviour in AI Village              -> Tables 1-2
  monitors/             evaluation set + passive monitors            -> Results 4.1, data/claims.csv
  gate/                 allow / block / escalate replay              -> gate numbers in 4.1
  group_size/           monitor accuracy vs number of agents         -> Figure 1a
  followup/             follow-up questioning (random, expert, weak) -> Figure 1b, Table 3
  stress_tests/         peer influence + record tampering            -> Figures 2-3
scripts/                shared helpers (data paths, API client, plotting) and paper_figures.py
data/claims.csv         the 93 evaluation claims by message ID and label (no AI Village text)
organism/               synthetic "inflated reporter" model organism (Adaption); not evaluated in the paper
paper/                  LaTeX source and PDF
figures/paper/          figures used in the paper (figures/exploratory/ holds quick-look plots)
docs/                   detailed methods, parameters and what did not work
pyine_selector/         reference copy of the PyINE question selector (needs the preexisting package)
run_all.sh              reproduce everything in order
```

## What existed before the sprint

- **The PyINE active investigator** (October 2026, a separate codebase not included here). It is an investigator
  that asks a model trace-grounded follow-up questions about program execution, with its model organism, data, and
  a bank of Qwen 3.5 9B answers. The idea of follow-up questioning as an active investigator comes from this work.
  `pyine_selector/train_selector.py` was written during the sprint, but it runs on that preexisting package.
- **Everything else here was written during the sprint (7–9 November 2026).** That covers the AI Village curation,
  the evaluation set, all experiments, the organism data, the figures and the paper. No reported result depends on
  PyINE code or data.

## Data

- **AI Village dataset** (AI Digest): <https://huggingface.co/datasets/aidigestorg/ai-village>. It is **not included**.
  Under its terms it may be used for research and evaluation only, with no training on it, and it must be cited.
  We use the tables `chat_messages`, `computer_use_turns`, `computer_use_sessions`, `agents` and `agent_goals`. Save
  them as `<table>.jsonl` or `<table>.jsonl.gz` in one folder.
- **Our labels:** [`data/claims.csv`](data/claims.csv) lists the 93 score claims used in all experiments. For each it
  gives the AI Village message ID, the split (10 tuning, 83 test), the label (true or planted false), the true number
  from the agent's own logs, the stated number, and whether it is in the 27-claim peer-influence subset. Regenerate
  it with `python3 -m scripts.export_claims`.
- **Results** (`out/`) are not included because they contain AI Village text. The runners regenerate them.

## Setup

```bash
pip install -r requirements.txt
export AI_VILLAGE_DIR=/path/to/ai_village                  # default: ../ai_village next to this repo
echo "OPENROUTER_API_KEY=..." > "$AI_VILLAGE_DIR/.env"      # read by scripts/llm.py; never committed
```

## Reproducing the paper

Run from the repository root. Each phase runs on its own, in this order:

```bash
bash experiments/behaviours/run.sh      # Tables 1-2                 ~$1
bash experiments/monitors/run.sh        # eval set, passive monitors ~$0.3
bash experiments/gate/run.sh            # gate replay                free
bash experiments/group_size/run.sh      # Figure 1a                  ~$6
bash experiments/followup/run.sh        # Figure 1b, Table 3         ~$20
bash experiments/stress_tests/run.sh    # Figures 2-3                ~$2
python3 -m scripts.paper_figures        # all paper figures + every number in the text (free)
```

Or run everything with `bash run_all.sh`, which costs about $35 of OpenRouter calls in total. Single steps run as
modules, for example `python3 -m experiments.followup.p6_followup report`.

Notes:
- **Cost control:** every paid call goes through `scripts/llm.py`. It logs costs to `out/spend.jsonl` and stops at
  `TOTAL_CAP`. Paid steps resume from their output files if interrupted.
- **Retired models:** Claude 3.7 Sonnet and Gemini 3 Pro are no longer offered. Claude Sonnet 4.5 and Gemini 3.1 Pro
  stand in for them (`RESPONDER` in `experiments/followup/p6_followup.py`).
- **Exact reruns:** model replies vary between runs and providers, even at temperature 0. The weak-asker prompt used
  for the first 27 claims was rewritten after being lost (see `p6_followup.py`).
- **More detail:** see [`docs/METHODS_detailed.md`](docs/METHODS_detailed.md). Its script paths predate this layout,
  and the paper's numbers are final.

## Model organism (not evaluated in the paper)

```bash
python3 organism/make_data.py                        # 1,500 synthetic training rows + 200 held out
python3 organism/start_adaption.py upload            # then: start, status, download
python3 organism/test_organism.py --model inflated   # after serving the adapter with vLLM
```

## Citation

```bibtex
@misc{bansal2026followup,
  title        = {{Can Follow-Up Questions Help Weak-to-Strong Monitors for Swarms?}},
  author       = {Hannah Bansal},
  year         = {2026},
  month        = nov,
  note         = {Submitted to Swarm Alignment Hackathon, an Apart Research Sprint},
  howpublished = {\url{https://apartresearch.com/sprints/projects/can-followup-questions-help-weaktostrong-monitors-for-swarms-wpcsuqz0d4}},
  url          = {https://apartresearch.com/sprints/projects/can-followup-questions-help-weaktostrong-monitors-for-swarms-wpcsuqz0d4}
}
```

Please also cite AI Digest / AI Village if you use the data.
