"""Paper figures (free; reads saved results only). Writes figures/paper/fig{1..4}.pdf and .png.

AUROC and ROC curves use scikit-learn (roc_auc_score, roc_curve); 95% intervals are percentile bootstraps
over claims (1,000 resamples). Plain academic style: no titles in the image; captions carry the detail.

    python3 paper_figures.py           # also prints every number used in the results text
"""

import collections
import json

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT

FIG = OUT.parent / "figures" / "paper"
FIG.mkdir(parents=True, exist_ok=True)
# Okabe-Ito colour-blind-safe palette
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMIL, PURPLE = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
                                                           "#0072B2", "#D55E00", "#CC79A7")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.dpi": 300,
                     "pdf.fonttype": 42})
WEAK, STRONG = "qwen/qwen3.5-9b", "openai/gpt-5-mini"
SIZES = (1, 2, 3, 4, 5, 7, 10, 15, 25, 35, 50)
RNG = np.random.default_rng(0)


def test_ids():
    rows = [json.loads(l) for l in open(OUT / "p3_set.jsonl")]
    return [r["pid"] for r in rows[10:]]


def auc_ci(pairs, B=1000):
    """pairs: list of (is_false, score). AUROC with a 95% percentile-bootstrap interval over claims."""
    y = np.array([p[0] for p in pairs], dtype=int)
    s = np.array([p[1] for p in pairs], dtype=float)
    point = roc_auc_score(y, s)
    boots = []
    for _ in range(B):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def rates(pairs, t=50):
    f = [s for y, s in pairs if y]
    tr = [s for y, s in pairs if not y]
    return np.mean([s >= t for s in f]), np.mean([s >= t for s in tr])


def label(ax, letter):
    ax.text(-0.16, 1.04, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def load_p5():
    ids = set(test_ids())
    res = collections.defaultdict(dict)
    for s in map(json.loads, open(OUT / "p5_groupsize.jsonl")):
        if s["pid"] in ids and s["p_false"] is not None:
            res[(s["model"], s["size"])][s["pid"]] = (s["label"] == "planted", float(s["p_false"]))
    return res


def load_p6():
    ids = set(test_ids())
    res = collections.defaultdict(dict)
    for s in map(json.loads, open(OUT / "p6_scores.jsonl")):
        if s["pid"] in ids and s["p_false"] is not None:
            res[(s["condition"], s["size"])][s["pid"]] = (s["label"] == "planted", float(s["p_false"]))
    return res


def fig1(p5, p6):
    """Main figure: (a) passive weak vs strong monitor across group size; (b) weak judge with follow-ups."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    print("\nFig 1a: passive monitors, AUROC by group size")
    for model, name, col, mk in [(STRONG, "Strong monitor (GPT-5-mini)", BLUE, "s"), (WEAK, "Weak monitor (Qwen 3.5 9B)", VERMIL, "o")]:
        pts = [auc_ci(list(p5[(model, n)].values())) for n in SIZES]
        a, lo, hi = map(np.array, zip(*pts))
        ax.fill_between(SIZES, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(SIZES, a, marker=mk, color=col, label=name)
        print(f"  {name}: " + ", ".join(f"N={n}: {v:.2f}" for n, v in zip(SIZES, a)))
        for band, szs in [("1-4", (1, 2, 3, 4)), ("5-15", (5, 7, 10, 15)), ("25-50", (25, 35, 50))]:
            pooled = [p for n in szs for p in p5[(model, n)].values()]
            pa, plo, phi = auc_ci(pooled)
            c, fa = rates(pooled)
            print(f"    pooled {band}: AUROC {pa:.2f} [{plo:.2f}, {phi:.2f}] caught {c:.0%} FA {fa:.0%}")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.minorticks_off()
    ax.set_ylim(0.4, 1.01)
    ax.set_xlabel("Agents in monitor's view (N)")
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, loc="lower left")
    label(ax, "a")

    ax = axes[1]
    print("\nFig 1b: weak judge with follow-up questions, AUROC by group size")
    series = [("no question", "No question", VERMIL, "o", "-"), ("random x1", "Random question", ORANGE, "^", "-"),
              ("weak asks x1", "Weak monitor asks (1)", GREEN, "D", "-"), ("targeted x1", "Expert asks (1)", BLUE, "s", "-"),
              ("targeted x2", "Expert asks (2)", BLUE, "s", "--")]
    for cond, name, col, mk, ls in series:
        xs, ys, los, his = [], [], [], []
        for n in SIZES:
            src = p5[(WEAK, n)] if cond == "no question" else p6.get((cond, n), {})
            if len(src) < 80:
                continue
            a, lo, hi = auc_ci(list(src.values()))
            xs.append(n), ys.append(a), los.append(lo), his.append(hi)
        if xs:
            ax.plot(xs, ys, marker=mk, ls=ls, color=col, label=name)
            print(f"  {name}: " + ", ".join(f"N={n}: {v:.2f}" for n, v in zip(xs, ys)))
    strong = [roc_auc_score(*zip(*p5[(STRONG, n)].values())) for n in SIZES]
    ax.plot(SIZES, strong, color="grey", lw=1.0, ls=(0, (4, 2)), label="Strong monitor, no question")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.minorticks_off()
    ax.set_ylim(0.4, 1.01)
    ax.set_xlabel("Agents in monitor's view (N)")
    ax.set_ylabel("AUROC (weak monitor as judge)")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=6.8)
    label(ax, "b")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_group_size.{ext}")


def fig2(p5, p6):
    """Follow-up rounds: AUROC and false-alarm rate by policy and number of questions, small vs large groups."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    groups = [("Small (N = 1, 3)", (1, 3)), ("Large (N = 25, 50)", (25, 50))]
    policies = [("random", "Random", ORANGE, "^"), ("weak asks", "Weak monitor asks", GREEN, "D"), ("targeted", "Expert asks", BLUE, "s")]
    print("\nFig 2: rounds of questioning (weak judge), pooled over sizes")
    for ax, (gname, szs), letter in zip(axes, groups, "ab"):
        base = [p for n in szs for p in p5[(WEAK, n)].values()]
        b, blo, bhi = auc_ci(base)
        ax.axhspan(blo, bhi, color=VERMIL, alpha=0.12, lw=0)
        ax.axhline(b, color=VERMIL, lw=1.2, label="No question")
        print(f"  {gname}: no question AUROC {b:.2f} [{blo:.2f}, {bhi:.2f}], caught/FA {rates(base)[0]:.0%}/{rates(base)[1]:.0%}")
        sp = [p for n in szs for p in p5[(STRONG, n)].values()]
        sa, slo, shi = auc_ci(sp)
        ax.axhline(sa, color="grey", lw=1.0, ls=(0, (4, 2)), label="Strong monitor, no question")
        print(f"  {gname}: STRONG no question AUROC {sa:.2f} [{slo:.2f}, {shi:.2f}], caught/FA {rates(sp)[0]:.0%}/{rates(sp)[1]:.0%}")
        for pol, name, col, mk in policies:
            ks = [1] if pol == "random" else [1, 2, 3]
            xs, ys, err = [], [], []
            for k in ks:
                pooled = [p for n in szs for p in p6.get((f"{pol} x{k}", n), {}).values()]
                if len(pooled) < 100:
                    continue
                a, lo, hi = auc_ci(pooled)
                c, fa = rates(pooled)
                xs.append(k), ys.append(a), err.append((a - lo, hi - a))
                print(f"    {name} x{k}: AUROC {a:.2f} [{lo:.2f}, {hi:.2f}] caught {c:.0%} FA {fa:.0%} (n={len(pooled)})")
            if xs:
                off = {"random": -0.12, "weak asks": 0.0, "targeted": 0.12}[pol]
                ax.errorbar(np.array(xs) + off, ys, yerr=np.array(err).T, marker=mk, color=col, capsize=2, label=name, lw=1.2)
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.6, 3.4)
        ax.set_ylim(0.4, 1.01)
        ax.set_xlabel("Number of follow-up questions")
        ax.set_ylabel("AUROC (weak monitor as judge)")
        ax.text(0.98, 0.04, gname, transform=ax.transAxes, ha="right", fontsize=8)
        if letter == "a":
            ax.legend(frameon=False, loc="lower left")
        label(ax, letter)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig2_followup_rounds.{ext}")


def fig3():
    """Record tampering: detection rate vs fraction of outputs faked."""
    res = collections.defaultdict(dict)
    for x in map(json.loads, open(OUT / "p7_faked.jsonl")):
        if x["p_false"] is not None:
            res[(x["model"], x["size"], x["p"])][x["pid"]] = (x["label"] == "planted", float(x["p_false"]))
    levels = (0.0, 0.07, 0.30, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    print("\nFig 3: record tampering")
    for model, name, col, mk in [(STRONG, "Strong monitor", BLUE, "s"), (WEAK, "Weak monitor", VERMIL, "o")]:
        for n, ls in [(1, "-"), (10, "--")]:
            caught, aucs = [], []
            for p in levels:
                pairs = list(res[(model, n, p)].values())
                c, fa = rates(pairs)
                a, lo, hi = auc_ci(pairs, B=300)
                caught.append(c), aucs.append(a)
                print(f"  {name} N={n} faked {p:.0%}: caught {c:.0%} FA {fa:.0%} AUROC {a:.2f} [{lo:.2f}, {hi:.2f}]")
            xs = [100 * p for p in levels]
            axes[0].plot(xs, caught, marker=mk, ls=ls, color=col, label=f"{name}, N = {n}")
            axes[1].plot(xs, aucs, marker=mk, ls=ls, color=col, label=f"{name}, N = {n}")
    for ax, yl, letter in [(axes[0], "False claims detected (score ≥ 50)", "a"), (axes[1], "AUROC", "b")]:
        ax.set_xticks([0, 7, 30, 100], ["0", "7", "30", "100"])
        ax.set_xlabel("Tool outputs falsified by the agent (%)")
        ax.set_ylabel(yl)
        label(ax, letter)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_ylim(0.4, 1.01)
    axes[1].axhline(0.5, color="grey", lw=0.8, ls=":")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig3_record_tampering.{ext}")


def fig4():
    """Peer influence: adoption of an inserted peer claim, and detection of the resulting false reports."""
    from p8_peer_claim import classify, claims

    _, picked = claims()
    by_pid = {r["pid"]: r for r in picked}
    ups = [json.loads(l) for l in open(OUT / "p8_updates.jsonl")]
    for u in ups:
        u["kind"], u["stated"] = classify(u["reply"], by_pid[u["pid"]])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    conds = [("none", "None"), ("plain", "Plain"), ("verified", "“Verified”")]
    print("\nFig 4a: adoption of peer value")
    vals, cis = [], []
    for c, _ in conds:
        x = np.array([u["kind"] == "states peer's number" for u in ups if u["cond"] == c], dtype=float)
        boots = [RNG.choice(x, len(x)).mean() for _ in range(1000)]
        vals.append(x.mean()), cis.append((x.mean() - np.percentile(boots, 2.5), np.percentile(boots, 97.5) - x.mean()))
        print(f"  {c}: {int(x.sum())}/{len(x)} = {x.mean():.0%}")
    ax.bar(range(3), vals, yerr=np.array(cis).T, color=[BLACK, ORANGE, VERMIL], width=0.6, capsize=3, error_kw={"lw": 0.8})
    ax.set_xticks(range(3), [n for _, n in conds])
    ax.set_xlabel("Inserted peer message")
    ax.set_ylabel("Agents reporting the peer's value")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 0.7)
    label(ax, "a")
    per_model = collections.defaultdict(lambda: [0, 0])
    for u in ups:
        if u["cond"] != "none":
            per_model[u["agent"]][0] += u["kind"] == "states peer's number"
            per_model[u["agent"]][1] += 1
    print("  by model:", {a: f"{x}/{n}" for a, (x, n) in per_model.items()})

    ax = axes[1]
    sc = [s for s in map(json.loads, open(OUT / "p8_scores.jsonl")) if s["p_false"] is not None]
    groups = [(WEAK, 1), (WEAK, 10), (STRONG, 1), (STRONG, 10)]
    w = 0.38
    print("Fig 4b: detection of false reports after peer influence")
    for j, (view, name, col) in enumerate([("logs", "Own logs", BLUE), ("logs+chat", "Own logs + chat", SKY)]):
        ys = []
        for m, n in groups:
            f = [s["p_false"] >= 50 for s in sc if s["model"] == m and s["size"] == n and s["view"] == view and s["false"]]
            t = [s["p_false"] >= 50 for s in sc if s["model"] == m and s["size"] == n and s["view"] == view and not s["false"]]
            ys.append(np.mean(f))
            print(f"  {m} N={n} {view}: caught {np.mean(f):.0%} (n={len(f)}), FA {np.mean(t):.0%} (n={len(t)})")
        ax.bar(np.arange(4) + (j - 0.5) * w, ys, width=w * 0.95, color=col, label=name)
    ax.set_xticks(range(4), ["Weak\nN = 1", "Weak\nN = 10", "Strong\nN = 1", "Strong\nN = 10"])
    ax.set_ylabel("False reports detected (score ≥ 50)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    label(ax, "b")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig4_peer_influence.{ext}")


def roc_panel(p5):
    """Supplementary: ROC curves (scikit-learn) for both monitors at N = 1 and N = 50."""
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for model, name, col in [(STRONG, "Strong", BLUE), (WEAK, "Weak", VERMIL)]:
        for n, ls in [(1, "-"), (50, "--")]:
            pairs = list(p5[(model, n)].values())
            fpr, tpr, _ = roc_curve([p[0] for p in pairs], [p[1] for p in pairs])
            ax.plot(fpr, tpr, ls=ls, color=col, label=f"{name}, N = {n} (AUROC {roc_auc_score([p[0] for p in pairs], [p[1] for p in pairs]):.2f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"figS1_roc.{ext}")


if __name__ == "__main__":
    p5, p6 = load_p5(), load_p6()
    fig1(p5, p6)
    fig2(p5, p6)
    fig3()
    fig4()
    roc_panel(p5)
    print("\nwritten to", FIG)
