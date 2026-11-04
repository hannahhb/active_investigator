"""The trace-free investigator's pure parts: record selection, JSON parsing, arms and the selector."""

from __future__ import annotations

import random
import types

import active_investigator.cot_investigator as cot
import active_investigator.cot_selector as selector


def _record(problem: str, variant: str, label: bool, attempt: int = 0):
    return types.SimpleNamespace(
        problem_id=problem, code_type=variant, label=label, sample_id=f"{problem}/s0", attempt_index=attempt
    )


def test_select_records_one_per_problem_and_variant_with_disjoint_offsets():
    records = [
        _record(f"p{p}", variant, label, attempt)
        for p in range(10)
        for variant in cot.VARIANTS
        for label in (True, False)
        for attempt in range(3)
    ]
    first = cot.select_records(records, 4, 0, seed=1)
    second = cot.select_records(records, 4, 4, seed=1)
    assert len(first) == 12
    assert {(r.problem_id, r.code_type) for r in first} == {
        (p, v) for p in {r.problem_id for r in first} for v in cot.VARIANTS
    }
    assert not {r.problem_id for r in first} & {r.problem_id for r in second}


def test_parse_json_tolerates_fences_and_rejects_garbage():
    assert cot.parse_json('```json\n{"p_correct": 0.2}\n```') == {"p_correct": 0.2}
    assert cot.parse_json("no object here") is None
    assert cot.parse_json("{broken") is None


def test_clip_reasoning_keeps_head_and_tail():
    text = "a" * 10_000 + "END"
    clipped = cot.clip_reasoning(text)
    assert clipped.endswith("END") and "omitted" in clipped and len(clipped) < len(text)


def _row(label: bool, verdicts: list[str], index: int) -> dict:
    return {
        "problem_id": f"p{index}",
        "label": label,
        "code_type": "misleading" if index % 2 else "original",
        "problem_index": index,
        "reasoning_chars": 100 + index,
        "has_docstring": True,
        "investigation": {"p_correct": 0.5, "tokens": 10.0},
        "reconsider": {"kept": True, "answer": "x", "tokens": 5.0},
        "judge": {"p_correct": 0.5, "tokens": 5.0},
        "expected_output": "x",
        "questions": [
            {
                "rank": i,
                "kind": "branch" if i == 0 else "other",
                "question": "does the branch on line 3 run" if i == 0 else "anything else",
                "targets": "",
                "expected_if_correct": "unknown",
                "verdict": v,
                "reply": {"tokens": 10.0, "forced": False},
            }
            for i, v in enumerate(verdicts)
        ],
    }


def test_selector_learns_which_question_kind_is_informative():
    rng = random.Random(0)
    rows = []
    for idx in range(80):
        label = rng.random() < 0.5
        informative = "supports" if label else "contradicts"
        rows.append(_row(label, [informative, rng.choice(["supports", "uninformative"])], idx))
    model = selector.Selector().fit(rows[:60])
    assert all(order[0] == 0 for order in model.rank(rows[60:]))
    selections = selector.arm_selections(rows[60:], model.rank(rows[60:]), ks=[1])
    arms = selector.arm_scores(rows[60:], selections)
    summary = selector.summarize_arm(rows[60:], arms["trained@1"])
    assert summary["overall"]["auroc"] == 1.0
    evidence = selector.summarize_evidence(rows[60:], selector.evidence_flags(rows[60:], selections)["trained@1"])
    assert evidence["overall"]["evidence_catch"] == 1.0
    assert evidence["overall"]["evidence_false_alarm"] == 0.0


def test_cross_fitted_ranks_cover_every_row():
    rng = random.Random(1)
    rows = []
    for idx in range(100):
        label = rng.random() < 0.5
        rows.append(_row(label, ["supports" if label else "contradicts", "uninformative"], idx))
    ranked, _ = selector.cross_fitted_ranks(rows, folds=5)
    assert len(ranked) == len(rows) and all(sorted(order) == [0, 1] for order in ranked)
