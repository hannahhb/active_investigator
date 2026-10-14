"""Headline figure for the prompted gpt-5-mini run on 5 problems (360 records, organism as responder)."""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

VARIANTS = ["original", "hinted", "misleading"]
COLORS = {"original": "#2a78d6", "hinted": "#eb6834", "misleading": "#1baf7a"}  # validated categorical slots 1-3
INK, INK2, GRID, PAPER = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
S = json.loads(pathlib.Path("figures/prompted_p5_examples.json").read_text())["summary"]
BY = S["by_variant"]


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
    ax.set_facecolor(PAPER)


def title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=10.5, fontweight="bold", pad=10)


fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), facecolor=PAPER)
fig.subplots_adjust(hspace=0.55, wspace=0.25, left=0.06, right=0.98, top=0.84, bottom=0.13)

# A. organism accuracy by variant: on these 5 problems the label is almost the variant
ax = axes[0, 0]
for i, v in enumerate(VARIANTS):
    acc = BY[v]["organism_accuracy"]
    ax.bar(i, acc, width=0.5, color=COLORS[v], linewidth=0)
    ax.text(i, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{v}\n(n={BY[v]['n']})" for v in VARIANTS], color=INK, fontsize=8.5)
style(ax, "fraction of records the organism got right", (0, 1.12))
title(ax, "A. The organism's accuracy by variant")

# B. investigator score distribution by variant: no separation
ax = axes[0, 1]
levels = [0.0, 0.33, 0.67, 1.0]
width, gap = 0.24, 0.03
for i, v in enumerate(VARIANTS):
    hist = BY[v]["score_hist"]
    n = BY[v]["n"]
    for j, lvl in enumerate(levels):
        frac = hist.get(str(lvl), hist.get(lvl, 0)) / n
        x = j + (i - 1) * (width + gap)
        ax.bar(x, frac, width=width, color=COLORS[v], linewidth=0)
ax.set_xticks(range(4))
ax.set_xticklabels(["0", "0.33", "0.67", "1.0"], color=INK, fontsize=8.5)
ax.set_xlabel("trust score given by the investigator (k = 3 follow-ups)", color=INK2, fontsize=9)
style(ax, "fraction of records", (0, 0.65))
auroc = S["metrics"]["overall"]["auroc"]
ax.text(0.98, 0.95, f"overall AUROC {auroc:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
title(ax, "B. Scores do not separate the variants")

# C. the two tells: own-values contradiction, and reconcile changed the answer
ax = axes[1, 0]
for i, v in enumerate(VARIANTS):
    n = BY[v]["n"]
    caught = BY[v]["caught_records"] / n
    revised = (BY[v]["reconcile"].get("revised_correct", 0) + BY[v]["reconcile"].get("revised_other", 0)) / n
    ax.bar(i - width / 2 - 0.02, caught, width=0.3, color=COLORS[v], linewidth=0)
    ax.bar(i + width / 2 + 0.02, revised, width=0.3, color=COLORS[v], linewidth=0, hatch="////", edgecolor=PAPER)
    ax.text(i - width / 2 - 0.02, caught + 0.012, f"{caught:.2f}", ha="center", fontsize=8, color=INK)
    ax.text(i + width / 2 + 0.02, revised + 0.012, f"{revised:.2f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels(VARIANTS, color=INK, fontsize=8.5)
style(ax, "fraction of records", (0, 0.5))
title(ax, "C. The two tells, by variant")
ax.text(0.02, 0.96, "solid: its own computed values contradict its answer\nhatched: it changed its answer when shown those values",
        transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK2)

# D. what happened to the follow-up replies
ax = axes[1, 1]
for i, v in enumerate(VARIANTS):
    f = BY[v]["followups"]
    total = f["asked"]
    correct = f["correct"] / total
    no_tag = f["no_tag"] / total
    wrong = 1 - correct - no_tag
    ax.bar(i, correct, width=0.5, color=COLORS[v], linewidth=0)
    ax.bar(i, wrong, bottom=correct + 0.004, width=0.5, color=COLORS[v], alpha=0.45, linewidth=0)
    ax.bar(i, no_tag, bottom=correct + wrong + 0.008, width=0.5, color=COLORS[v], alpha=0.18, linewidth=0)
    ax.text(i, correct / 2, f"{correct:.0%} right", ha="center", va="center", fontsize=8, color=PAPER)
    ax.text(i, correct + wrong / 2 + 0.004, f"{wrong:.0%} wrong", ha="center", va="center", fontsize=8, color=INK)
    ax.text(i, correct + wrong + no_tag / 2 + 0.008, f"{no_tag:.0%} no answer", ha="center", va="center", fontsize=8, color=INK)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{v}\n({BY[v]['followups']['asked']} questions)" for v in VARIANTS], color=INK, fontsize=8.5)
style(ax, "fraction of follow-up replies", (0, 1.02))
title(ax, "D. Follow-up replies: a third are wrong or never finish")

handles = [mpatches.Patch(color=COLORS[v], label=v) for v in VARIANTS]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.945))
fig.suptitle(
    "Prompted investigator (gpt-5-mini picks, organism answers) on 5 TACO test problems, 360 records",
    color=INK, fontsize=12, y=0.985,
)
fig.text(
    0.06, 0.03,
    "k = 3 trace-grounded follow-ups per record, hints stripped, consistency scoring, reconcile step. "
    "'No answer' replies all hit the 2,048-token cap.\n"
    "Only 5 problems: original and hinted have 7 and 1 wrong records, misleading has 0 right ones, "
    "so per-variant AUROC is not meaningful at this size.",
    color=INK2, fontsize=7.8, va="bottom",
)
out = "figures/prompted_gpt5mini_p5.png"
fig.savefig(out, dpi=170)
print("saved", out)
