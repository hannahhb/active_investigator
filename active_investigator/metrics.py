"""Threshold-free and fixed-FPR metrics, overall and per task variant."""

from __future__ import annotations

import collections
import statistics
import typing

import pyine.evals.correctness.types as correctness_types
import sklearn.metrics

FPR_TARGETS = (0.001, 0.01, 0.05, 0.1, 0.2, 0.5)


def tpr_at_fpr(
    labels: typing.Sequence[bool],
    scores: typing.Sequence[float],
    target: float,
) -> float | None:
    """Highest TPR achievable at a false-positive rate no greater than ``target``."""
    if len(set(labels)) < 2:
        return None
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, scores)
    feasible = [t for f, t in zip(fpr, tpr, strict=True) if f <= target]
    return max(feasible) if feasible else 0.0


def summarize(
    records: typing.Sequence[correctness_types.EvalRecord],
    scores: typing.Sequence[float],
    costs: typing.Sequence[float] | None = None,
) -> dict[str, typing.Any]:
    """Compute AUROC, TPR at fixed FPRs, mean score by label, and cost — overall and per variant."""
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for idx, record in enumerate(records):
        groups["overall"].append(idx)
        groups[record.code_type].append(idx)
    out: dict[str, typing.Any] = {}
    for name, idxs in groups.items():
        labels = [records[i].label for i in idxs]
        group_scores = [scores[i] for i in idxs]
        two_class = len(set(labels)) == 2
        pos = [score for score, label in zip(group_scores, labels, strict=True) if label]
        neg = [score for score, label in zip(group_scores, labels, strict=True) if not label]
        out[name] = {
            "n": len(idxs),
            "positive_rate": sum(labels) / len(labels),
            "auroc": sklearn.metrics.roc_auc_score(labels, group_scores) if two_class else None,
            "mean_score_correct": statistics.mean(pos) if pos else None,
            "mean_score_wrong": statistics.mean(neg) if neg else None,
            "tpr_at_fpr": {str(t): tpr_at_fpr(labels, group_scores, t) for t in FPR_TARGETS},
            "mean_cost": statistics.mean(costs[i] for i in idxs) if costs is not None else None,
        }
    return out


def format_summary(summary: dict[str, typing.Any]) -> str:
    """Render the summary as a compact table."""

    def fmt(value: float | None, width: int, digits: int = 3) -> str:
        return f"{'nan':>{width}s}" if value is None else f"{value:>{width}.{digits}f}"

    header = (
        f"{'group':12s} {'n':>6s} {'pos':>6s} {'AUROC':>7s} {'TPR@.05':>8s} {'TPR@.2':>7s} "
        f"{'sc|ok':>6s} {'sc|bad':>7s} {'cost':>8s}"
    )
    lines = [header]
    for name in ["overall", *sorted(key for key in summary if key != "overall")]:
        row = summary[name]
        lines.append(
            f"{name:12s} {row['n']:6d} {row['positive_rate']:6.3f} {fmt(row['auroc'], 7)} "
            f"{fmt(row['tpr_at_fpr']['0.05'], 8)} {fmt(row['tpr_at_fpr']['0.2'], 7)} "
            f"{fmt(row['mean_score_correct'], 6)} {fmt(row['mean_score_wrong'], 7)} {fmt(row['mean_cost'], 8, 0)}"
        )
    return "\n".join(lines)


def summarize_reconcile(
    records: typing.Sequence[correctness_types.EvalRecord],
    metadata: typing.Mapping[typing.Any, dict[str, typing.Any]],
) -> dict[str, dict[str, int]]:
    """Count what the model did when confronted with its own values, by variant and by evidence.

    Rows are ``<variant>|contradicted`` and ``<variant>|agreed`` depending on whether the elicited
    values contradicted the model's own final answer; columns are kept / revised_correct /
    revised_other / no_answer.
    """
    table: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    for (_, _, idx), payload in metadata.items():
        reconcile = payload.get("reconcile")
        if not reconcile or "outcome" not in reconcile:
            continue
        evidence = "contradicted" if payload.get("evidence_contradicts") else "agreed"
        table[f"{records[idx].code_type}|{evidence}"][reconcile["outcome"]] += 1
        table[f"overall|{evidence}"][reconcile["outcome"]] += 1
    return {key: dict(value) for key, value in table.items()}


def format_reconcile(table: dict[str, dict[str, int]]) -> str:
    """Render the reconcile counts as a compact table."""
    cols = ["kept", "revised_correct", "revised_other", "no_answer", "reconcile_failed"]
    lines = [f"{'group':24s} " + " ".join(f"{c:>16s}" for c in cols)]
    for key in sorted(table, key=lambda k: (k.split("|")[0] != "overall", k)):
        row = table[key]
        lines.append(f"{key:24s} " + " ".join(f"{row.get(c, 0):16d}" for c in cols))
    return "\n".join(lines)
