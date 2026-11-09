"""A trained question selector, tested on held-out programs (free, no model calls).

For every banked question we know, on training programs, how its reply scored when the model's
original answer was right and when it was wrong. Two small models learn those two averages from
features an investigator with a tracer can see (question kind, why it was picked, depth, locality,
whether its value is the output, position in the run). The selector asks the questions with the
biggest predicted gap first. Programs are split into 5 folds; each fold is ranked by models trained
on the other 4, so no program is ever ranked by a model that saw its labels.

    uv run python scripts/train_selector.py \
        --bank runs/bank-qwen35-9b-p40.jsonl \
        --records runs/qtype-output-qwen35-9b-p40-a3/transcripts.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import random

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from active_investigator import ranking

KINDS = ("value_at", "branch_at", "execution_count")
REASONS = ("random", "shallow", "deep", "branch", "count", "output:none", "output:variable", "output:value")


def problem(sample_id: str) -> str:
    return sample_id.split("/s")[0]


def features(q: dict, pool: list[dict]) -> list[float]:
    steps = [p["step"] for p in pool if p["step"] is not None] or [0]
    step = q["step"] if q["step"] is not None else -1
    return [
        *[float(q["kind"] == k) for k in KINDS],
        *[float(q["reason"] == r) for r in REASONS],
        float(q["equals_output"]),
        float(q["steps_needed"] or 0),
        float(q["locality"] or 0.0),
        step / max(steps),  # how late in the run (-ve if no step)
        q["pool_position"] / max(1, q["pool_size"]),
        float(q["variable"] is not None),
    ]


@dataclasses.dataclass
class LearnedRanker:
    """Ranks by predicted score gap (right minus wrong). Uses the trace only at ranking time."""

    models: dict  # problem id -> (model_if_right, model_if_wrong) trained without that problem
    name: str = "trained selector (held-out programs)"
    uses: str = "trace"

    def rank(self, pool, record):
        right, wrong = self.models[problem(record.sample_id)]
        x = np.array([features(q, pool) for q in pool])
        gap = right.predict(x) - wrong.predict(x)
        order = np.argsort(-gap, kind="stable")
        return [pool[i] for i in order]


def train_rows(bank, records):
    """One row per (record, question): features, label, and the reply's score for that record."""
    out = []
    for r in records:
        pool = bank[r.sample_id]
        for q in pool:
            out.append((problem(r.sample_id), features(q, pool), r.label, ranking._question_score(q, r.final_answer)))
    return out


def fit(rows, label):
    xs = np.array([f for _, f, y, _ in rows if y == label])
    ys = np.array([s for _, _, y, s in rows if y == label])
    return HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, random_state=0).fit(xs, ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="runs/trained_selector.json")
    args = ap.parse_args()

    bank = ranking.load_bank(pathlib.Path(args.bank))
    records = [r for r in ranking.load_record_views(pathlib.Path(args.records)) if r.sample_id in bank and r.final_answer is not None]
    problems = sorted({problem(r.sample_id) for r in records})
    random.Random(0).shuffle(problems)
    fold_of = {p: i % args.folds for i, p in enumerate(problems)}
    rows = train_rows(bank, records)

    models = {}
    for f in range(args.folds):
        train = [row for row in rows if fold_of[row[0]] != f]
        pair = (fit(train, True), fit(train, False))
        for p in problems:
            if fold_of[p] == f:
                models[p] = pair
    print(f"records {len(records)}, programs {len(problems)}, training rows {len(rows)}, folds {args.folds}")

    rankers = [*ranking.baseline_rankers(), LearnedRanker(models)]
    results = ranking.evaluate(bank, records, rankers, ks=(1, 2, 3))
    pathlib.Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    for score in ("graded", "self"):
        print(ranking.format_results(results, score) + "\n")
    print("results:", args.out)


if __name__ == "__main__":
    main()
