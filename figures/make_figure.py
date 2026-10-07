"""Paper-style overseer comparison: our reconsider baseline next to the PyINE paper's reference overseers.

Reference values are transcribed from the paper's Tables 5, 7 and 8 (full held-out test set, 44,820
records). Our values are recomputed from the TACO40 run's saved result (40 problems, 2,700 records).
"""

import collections
import statistics

import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics

import pyine.evals.persistence as persistence

matplotlib.use("Agg")

VARIANTS = ["original", "hinted", "misleading"]
COLORS = {"original": "#2a78d6", "hinted": "#eb6834", "misleading": "#1baf7a"}  # validated categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
RUN = "../pyine-logs:runs/20261002_020201/benchmark_export/guardrail_test.pkl"

# paper, Tables 5/7/8: AUROC and TPR by variant (original, helpful-hinted, misleading)
PAPER_AUROC = {
    "Probe\n(L22, strong)": (0.869, 0.690, 0.273),
    "CWM\n(Judge)": (0.629, 0.599, 0.377),
    "gpt-5-mini\n(Judge)": (0.958, 0.731, 0.932),
    "CWM\n(Debate)": (0.655, 0.558, 0.564),
    "gpt-5-mini\n(Debate)": (0.884, 0.648, 0.829),
}
PAPER_TPR20 = {"gpt-5-mini\n(Judge)": (0.981, 0.381, 0.977), "gpt-5-mini\n(Debate)": (0.933, 0.290, 0.608)}
PAPER_TPR05 = {"gpt-5-mini\n(Judge)": (0.792, 0.095, 0.439), "gpt-5-mini\n(Debate)": (0.258, 0.072, 0.152)}


def tpr_at(labels, scores, target):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, scores)
    return max(t for f, t in zip(fpr, tpr, strict=True) if f <= target)


def ours():
    run = persistence.load_eval_result(RUN).aggregated.per_run[0]
    by = collections.defaultdict(list)
    for rec in run.attempt_records:
        by[rec.code_type].append(rec)
    out = {}
    for variant in VARIANTS:
        recs = by[variant]
        labels = [r.label for r in recs]
        scores = [r.score for r in recs]
        pos = [s for s, l in zip(scores, labels, strict=True) if l]
        neg = [s for s, l in zip(scores, labels, strict=True) if not l]
        out[variant] = {
            "auroc": sklearn.metrics.roc_auc_score(labels, scores),
            "tpr05": tpr_at(labels, scores, 0.05),
            "tpr20": tpr_at(labels, scores, 0.2),
            "score_ok": statistics.mean(pos),
            "score_bad": statistics.mean(neg),
            "cost": statistics.mean(r.verification_cost or 0 for r in recs),
            "n": len(recs),
        }
    return out


def style(ax, ylabel, ylim=(0, 1.0)):
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def grouped(ax, groups, values_by_group, label_last=False):
    """values_by_group: {group: (orig, hinted, misleading)}; bars thin with a visible gap."""
    width, gap = 0.24, 0.03
    for g_idx, group in enumerate(groups):
        for v_idx, variant in enumerate(VARIANTS):
            value = values_by_group[group][v_idx]
            x = g_idx + (v_idx - 1) * (width + gap)
            ax.bar(x, value, width=width, color=COLORS[variant], linewidth=0)
            if label_last and group == groups[-1]:
                ax.text(x, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, color=INK, fontsize=8.5)


def main():
    data = ours()
    OURS = "Reconsider\n(ours, TACO40)"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), facecolor="#fcfcfb")
    fig.subplots_adjust(hspace=0.5, wspace=0.22, left=0.06, right=0.98, top=0.86, bottom=0.14)
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")

    # A - AUROC by variant, paper reference overseers + ours
    ax = axes[0, 0]
    groups = [*PAPER_AUROC, OURS]
    vals = {**PAPER_AUROC, OURS: tuple(data[v]["auroc"] for v in VARIANTS)}
    grouped(ax, groups, vals, label_last=True)
    ax.axhline(0.5, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(-0.45, 0.515, "random", color=INK2, fontsize=8, ha="left")
    ax.axvspan(len(groups) - 1.5, len(groups) - 0.5, color="#f0efeb", zorder=0)
    style(ax, "AUROC")
    ax.set_title("A. AUROC by task variant", loc="left", color=INK, fontsize=10.5, fontweight="bold")

    # B - TPR at fixed false-positive rate, two budgets
    ax = axes[0, 1]
    groups_b = ["gpt-5-mini\n(Judge)", "gpt-5-mini\n(Debate)", OURS]
    ours_tpr20 = tuple(data[v]["tpr20"] for v in VARIANTS)
    ours_tpr05 = tuple(data[v]["tpr05"] for v in VARIANTS)
    vals20 = {**PAPER_TPR20, OURS: ours_tpr20}
    vals05 = {**PAPER_TPR05, OURS: ours_tpr05}
    width, gap = 0.24, 0.03
    for g_idx, group in enumerate(groups_b):
        for v_idx, variant in enumerate(VARIANTS):
            x = g_idx + (v_idx - 1) * (width + gap)
            ax.bar(x, vals20[group][v_idx], width=width, color=COLORS[variant], linewidth=0, alpha=0.35)
            ax.bar(x, vals05[group][v_idx], width=width, color=COLORS[variant], linewidth=0)
            if group == OURS:  # our solid bars are exactly zero: say so, or they read as missing
                ax.text(x, 0.012, f"{vals05[group][v_idx]:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups_b)))
    ax.set_xticklabels(groups_b, color=INK, fontsize=8.5)
    ax.axvspan(len(groups_b) - 1.5, len(groups_b) - 0.5, color="#f0efeb", zorder=0)
    style(ax, "TPR")
    ax.set_title("B. TPR at fixed FPR  (solid: FPR ≤ 0.05, faint: FPR ≤ 0.20)", loc="left", color=INK, fontsize=10.5,
                 fontweight="bold")

    # C - our score when the organism was right vs wrong
    ax = axes[1, 0]
    width = 0.3
    for v_idx, variant in enumerate(VARIANTS):
        ok, bad = data[variant]["score_ok"], data[variant]["score_bad"]
        ax.bar(v_idx - width / 2 - 0.02, ok, width=width, color=COLORS[variant], linewidth=0)
        ax.bar(v_idx + width / 2 + 0.02, bad, width=width, color=COLORS[variant], linewidth=0, hatch="////",
               edgecolor="#fcfcfb")
        ax.text(v_idx - width / 2 - 0.02, ok + 0.015, f"{ok:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(v_idx + width / 2 + 0.02, bad + 0.015, f"{bad:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(VARIANTS, color=INK, fontsize=8.5)
    style(ax, "mean trust score")
    ax.set_title("C. Ours: trust score when organism was right (solid) vs wrong (hatched)", loc="left", color=INK,
                 fontsize=10.5, fontweight="bold")

    # D - cost per record
    ax = axes[1, 1]
    costs = [data[v]["cost"] for v in VARIANTS]
    for v_idx, variant in enumerate(VARIANTS):
        ax.bar(v_idx, costs[v_idx], width=0.5, color=COLORS[variant], linewidth=0)
        ax.text(v_idx, costs[v_idx] + 12, f"{costs[v_idx]:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(VARIANTS, color=INK, fontsize=8.5)
    style(ax, "completion tokens / record", ylim=(0, max(costs) * 1.2))
    ax.set_title("D. Ours: completion tokens per record", loc="left", color=INK, fontsize=10.5, fontweight="bold")

    handles = [matplotlib.patches.Patch(color=COLORS[v], label=v) for v in VARIANTS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Overseer performance by task variant — PyINE paper reference overseers vs. the reconsider baseline",
                 color=INK, fontsize=12, y=0.995)
    n = sum(data[v]["n"] for v in VARIANTS)
    fig.text(0.06, 0.03,
             "Reference bars: PyINE paper Tables 5, 7, 8 — full held-out test set (44,820 records); LLM overseers evaluated once.\n"
             f"Reconsider (shaded): this project — TACO40 subset (40 problems, {n:,} records), organism as responder, "
             "fixed “re-derive your answer” question, gpt-5-mini verdict. Sample sizes differ; compare shapes, not decimals.",
             color=INK2, fontsize=7.8, va="bottom")
    out = "figures/reconsider_vs_paper.png"
    fig.savefig(out, dpi=170)
    print("saved", out)
    for v in VARIANTS:
        d = data[v]
        print(f"{v:11s} auroc {d['auroc']:.3f} tpr05 {d['tpr05']:.3f} tpr20 {d['tpr20']:.3f} ok {d['score_ok']:.3f} bad {d['score_bad']:.3f} cost {d['cost']:.0f}")


if __name__ == "__main__":
    main()
