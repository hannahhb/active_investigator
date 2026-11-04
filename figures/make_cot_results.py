"""Results of the trace-free CoT investigator on all 611 problems (1,830 records).

Writes, from the saved rows only (no model calls):

- ``figures/cot_results_table.tex`` / ``.md``: evidence-level catch (%) per overseer in the PyINE
  paper's Table 1 format — Original, Helpful, Δhelpful, Misleading, Δmislead, Overall — plus false
  alarm and Len., as means ± 95% CIs from a problem-clustered bootstrap;
- ``figures/cot_evidence.png`` / ``.pdf``: evidence-level catch vs. false alarm per method, and
  catch by variant;
- ``figures/cot_results.json``: every number above.

The selector is cross-fitted over 5 problem folds (as in ``cot_selector --folds 5``), so every record
is scored by a selector that never saw its problem.

    ../venvs/active_investigator/bin/python figures/make_cot_results.py
"""

import json
import pathlib

import matplotlib
import numpy as np
import scipy.stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import active_investigator.cot_selector as sel  # noqa: E402

RUNS = ["runs/cot-gpt5mini-p240", "runs/cot-gpt5mini-rest"]
OUT = pathlib.Path("figures")
FPRS = (0.01, 0.05, 0.10, 0.20, 0.50)
BOOT = 1000
FOLDS = 5
VARIANTS = ["original", "hinted", "misleading"]
VLABEL = {"original": "Original", "hinted": "Helpful", "misleading": "Mislead."}
METHODS = [  # (arm, table name, figure name)
    ("passive", "Passive judge (reads CoT only)", None),
    ("reconsider", "Re-check prompt", "Re-check prompt"),
    ("llm@1", "LLM-picked question", "LLM-picked question"),
    ("random@1", "Random question", "Random question"),
    ("trained@1", "Trained selector", "Trained selector"),
    ("all", "All 5 questions (mean verdict)", "All 5 questions"),
    ("judge", "All 5 questions (judge's trust)", None),
    ("oracle@1", "Oracle question (hindsight)", "Oracle question"),
]
FAMILY = {"passive": "Baseline", "reconsider": "Baseline", "llm@1": "Follow-up", "random@1": "Follow-up",
          "trained@1": "Follow-up", "all": "Follow-up", "judge": "Follow-up", "oracle@1": "Upper bound"}

# chart tokens (validated categorical slots 1-3 on the light surface; see dataviz validator)
COLORS = {"original": "#2a78d6", "hinted": "#eb6834", "misleading": "#1baf7a"}
MARKERS = {"original": "o", "hinted": "s", "misleading": "D"}
INK, INK2, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"


# ---------- fast metrics (match sklearn's roc_auc_score / cot_selector.catch_rate) ----------
def auroc(right: np.ndarray, trust: np.ndarray) -> float:
    pos = right.sum()
    neg = right.size - pos
    if pos == 0 or neg == 0:
        return np.nan
    ranks = scipy.stats.rankdata(trust)
    return (ranks[right].sum() - pos * (pos + 1) / 2) / (pos * neg)


def tpr_at(right: np.ndarray, trust: np.ndarray, fprs=FPRS) -> np.ndarray:
    """Share of wrong answers flagged (low trust) when at most ``fpr`` of right answers are."""
    wrong = ~right
    p, n = wrong.sum(), right.sum()
    if p == 0 or n == 0:
        return np.full(len(fprs), np.nan)
    order = np.argsort(trust, kind="mergesort")
    s, w = trust[order], wrong[order]
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    tp = np.r_[0, np.cumsum(w)[last]] / p
    fp = np.r_[0, np.cumsum(~w)[last]] / n
    return np.array([tp[fp <= f + 1e-12].max() for f in fprs])


def mean_or_nan(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else np.nan


# ---------- data ----------
rows = [r for run in RUNS for r in sel.load_rows(pathlib.Path(run))]
ranked, _ = sel.cross_fitted_ranks(rows, FOLDS)
selections = sel.arm_selections(rows, ranked, [1])
arm_scores = sel.arm_scores(rows, selections)
flags = sel.evidence_flags(rows, selections)
cost = sel.costs(rows, [1])


def chosen_tokens(arm: str) -> np.ndarray:
    """The organism's extra completion tokens per record under one arm, averaged over random draws."""
    if arm == "passive":
        return np.zeros(len(rows))
    if arm == "reconsider":
        return np.array([r["reconsider"]["tokens"] for r in rows], dtype=float)
    draws = selections[arm]
    return np.mean([[sum(r["questions"][i]["reply"]["tokens"] for i in idxs) for r, idxs in zip(rows, draw)]
                    for draw in draws], axis=0)


TOK_ROWS = {a: chosen_tokens(a) for a, _, _ in METHODS}
TOKENS = {a: float(t.mean()) for a, t in TOK_ROWS.items()}

N = len(rows)
right = np.array([r["label"] for r in rows])
variant = np.array([r["code_type"] for r in rows])
problems = np.array([r["problem_id"] for r in rows])
uniq = np.unique(problems)
members = [np.nonzero(problems == p)[0] for p in uniq]
S = {a: np.atleast_2d(np.array(arm_scores[a], dtype=float)) for a, _, _ in METHODS}  # draws x N
F = {a: np.array(flags[a], dtype=bool) for a, _, _ in METHODS if a in flags}  # draws x N


def stats(idx: np.ndarray) -> dict:
    """Every metric on the records ``idx`` (a bootstrap sample may repeat records)."""
    out = {}
    for a, _, _ in METHODS:
        per = {}
        for v in VARIANTS + ["overall"]:
            g = idx if v == "overall" else idx[variant[idx] == v]
            r = right[g]
            entry = {
                "auroc": np.nanmean([auroc(r, d[g]) for d in S[a]]),
                "tpr": np.nanmean([tpr_at(r, d[g]) for d in S[a]], axis=0),
            }
            if a in F:
                entry["catch"] = np.mean([mean_or_nan(f[g][~r]) for f in F[a]])
                entry["false_alarm"] = np.mean([mean_or_nan(f[g][r]) for f in F[a]])
            per[v] = entry
        out[a] = per
    return out


point = stats(np.arange(N))
rng = np.random.default_rng(0)
boots = []
for _ in range(BOOT):
    pick = rng.integers(0, len(uniq), len(uniq))
    boots.append(stats(np.concatenate([members[i] for i in pick])))


def ci(get) -> tuple[float, float]:
    """Half-width of the 95% percentile interval, reported as ± like the PyINE tables."""
    vals = np.array([get(b) for b in boots], dtype=float)
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


# ---------- table in the PyINE Table 1 format ----------
def pm(est: float, lo: float, hi: float) -> str:
    return f"{100 * est:.1f}±{100 * (hi - lo) / 2:.1f}"


table = []
for a, name, _ in METHODS:
    if a not in F:  # passive and the judge's trust elicit no evidence of their own
        continue
    cell = {v: pm(point[a][v]["catch"], *ci(lambda b, v=v: b[a][v]["catch"])) for v in VARIANTS + ["overall"]}
    d_help = 100 * (point[a]["hinted"]["catch"] - point[a]["original"]["catch"])
    d_mis = 100 * (point[a]["misleading"]["catch"] - point[a]["original"]["catch"])
    tok = TOK_ROWS[a]
    table.append({
        "name": name.replace(" (mean verdict)", ""), "family": FAMILY[a], "original": cell["original"],
        "helpful": cell["hinted"], "d_helpful": f"{d_help:+.1f}", "misleading": cell["misleading"],
        "d_mislead": f"{d_mis:+.1f}", "overall": cell["overall"],
        "false_alarm": pm(point[a]["overall"]["false_alarm"], *ci(lambda b: b[a]["overall"]["false_alarm"])),
        "len": f"{tok.mean():.0f}±{tok.std():.0f}",
    })

COLS = ["name", "family", "original", "helpful", "d_helpful", "misleading", "d_mislead", "overall", "false_alarm", "len"]
HEAD = ["Overseer", "Family", "Original", "Helpful", "Δhelpful", "Misleading", "Δmislead", "Overall", "False alarm", "Len."]
n_by = {v: (int((variant == v).sum()), int(((variant == v) & ~right).sum())) for v in VARIANTS}
caption = (
    "CoT-investigator evidence-level catch rate (\\%) for PyINE-v1 across original, helpful-hinted and "
    "misleading-hinted task variants: the share of wrong answers that the model organism's own follow-up reply "
    "contradicts (values reported as mean $\\pm$ 95\\% confidence interval from problem-clustered bootstrapping, "
    f"{BOOT} resamples), together with $\\Delta$helpful := Helpful $-$ Original and $\\Delta$mislead := "
    "Misleading $-$ Original, in percentage points. False alarm is the same flag on right answers, over all variants. "
    "The last column reports the organism's extra completion tokens per record spent answering the overseer, "
    f"mean $\\pm$ standard deviation. {N:,} records from 611 problems (wrong answers: original {n_by['original'][1]}, "
    f"helpful {n_by['hinted'][1]}, misleading {n_by['misleading'][1]}). Question overseers ask one question unless "
    "noted; the trained selector is cross-fitted over 5 problem folds. Passive overseers elicit no evidence and "
    "are not listed."
)
tex = [
    "\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", "\\label{tab:cot-investigator}",
    "\\begin{tabular}{llcccccccr}", "\\toprule", " & ".join(HEAD).replace("Δ", "$\\Delta$") + " \\\\", "\\midrule",
]
prev = None
for row in table:
    if prev is not None and row["family"] != prev:
        tex.append("\\midrule")
    prev = row["family"]
    tex.append(" & ".join(row[c] for c in COLS).replace("±", "$\\pm$") + " \\\\")
tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(OUT / "cot_results_table.tex").write_text("\n".join(tex) + "\n")

md = ["| " + " | ".join(HEAD) + " |", "|---|---|" + "---:|" * (len(HEAD) - 2)]
md += ["| " + " | ".join(row[c] for c in COLS) + " |" for row in table]
(OUT / "cot_results_table.md").write_text("\n".join(md) + "\n")

# ---------- evidence figure ----------
fig_methods = [(a, f) for a, _, f in METHODS if f]
ev = {}
for a, _ in fig_methods:
    ev[a] = {}
    for v in VARIANTS + ["overall"]:
        e = {}
        for m in ("catch", "false_alarm"):
            lo, hi = ci(lambda b, m=m, v=v: b[a][v][m])
            e[m] = (point[a][v][m], lo, hi)
        ev[a][v] = e

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.3), facecolor=PAPER, gridspec_kw={"width_ratios": [1, 1.15]})
fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12, wspace=0.42)


def style(ax):
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_axisbelow(True)


# A. catch vs false alarm, overall
style(axA)
axA.grid(True, color=GRID, linewidth=0.8)
QUESTION = {"llm@1", "random@1", "trained@1"}
label_pos = {  # (dx, dy, ha, va) in points, chosen so labels don't collide
    "reconsider": (8, 6, "left", "bottom"), "llm@1": (0, -16, "center", "top"), "random@1": (14, 0, "left", "center"),
    "trained@1": (8, 6, "left", "bottom"), "all": (-8, -16, "right", "top"), "oracle@1": (8, 6, "left", "bottom"),
}
for a, name in fig_methods:
    c, clo, chi = ev[a]["overall"]["catch"]
    f, flo, fhi = ev[a]["overall"]["false_alarm"]
    color = COLORS["original"] if a in QUESTION else INK2
    face = PAPER if a == "oracle@1" else color
    axA.errorbar(f, c, xerr=[[f - flo], [fhi - f]], yerr=[[c - clo], [chi - c]], fmt="none", ecolor=color,
                 elinewidth=1.2, capsize=0, alpha=0.6, zorder=2)
    axA.plot(f, c, "o", ms=9, mfc=face, mec=color, mew=2, zorder=3)
    dx, dy, ha, va = label_pos[a]
    tok = TOKENS[a]
    axA.annotate(f"{name}\n{c:.2f} / {f:.2f} · {tok:,.0f} tok", (f, c), xytext=(dx, dy), textcoords="offset points",
                 ha=ha, va=va, fontsize=8.5, color=INK)
axA.set_xlim(-0.02, 0.40)
axA.set_ylim(0, 1.0)
axA.set_xlabel("False alarm", color=INK2, fontsize=10)
axA.set_ylabel("Catch", color=INK2, fontsize=10)
axA.set_title("A", loc="left", color=INK, fontsize=12, fontweight="bold", pad=24)

# B. catch by variant (dot plot with CIs)
style(axB)
axB.xaxis.grid(True, color=GRID, linewidth=0.8)
names = [name for _, name in fig_methods]
offs = {"original": 0.22, "hinted": 0.0, "misleading": -0.22}
for i, (a, _) in enumerate(fig_methods):
    y = len(fig_methods) - 1 - i
    for v in VARIANTS:
        c, lo, hi = ev[a][v]["catch"]
        axB.plot([lo, hi], [y + offs[v]] * 2, color=COLORS[v], lw=1.6, alpha=0.55, solid_capstyle="round")
        axB.plot(c, y + offs[v], MARKERS[v], ms=7, color=COLORS[v], mec=PAPER, mew=1.5)
axB.set_yticks(range(len(fig_methods)))
axB.set_yticklabels(names[::-1], color=INK, fontsize=9.5)
axB.set_xlim(0, 1.0)
axB.set_xlabel("Catch", color=INK2, fontsize=10)
axB.set_title("B", loc="left", color=INK, fontsize=12, fontweight="bold", pad=24)
handles = [plt.Line2D([], [], marker=MARKERS[v], ls="", ms=7, color=COLORS[v], label={"original": "Original", "hinted": "Helpful", "misleading": "Misleading"}[v])
           for v in VARIANTS]
axB.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, 0.995), ncol=3, frameon=False, fontsize=8.5,
           labelcolor=INK, handletextpad=0.3, columnspacing=1.2, borderaxespad=0)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"cot_evidence.{ext}", dpi=200, facecolor=PAPER)

json.dump({"n": N, "n_by_variant": n_by, "tokens": TOKENS, "table": table, "evidence": ev},
          open(OUT / "cot_results.json", "w"), indent=1, default=float)
print((OUT / "cot_results_table.md").read_text())
for a, name in fig_methods:
    print(name, {v: tuple(round(x, 2) for x in ev[a][v]["catch"]) for v in VARIANTS + ["overall"]},
          "FA", tuple(round(x, 2) for x in ev[a]["overall"]["false_alarm"]))
