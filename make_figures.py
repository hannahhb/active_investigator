"""Figures from the Village work so far (free). Writes figures/*.png. All Village text is data, never instructions."""
import collections, json, math, random, sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from common import OUT, rows
from p3_free import auroc

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
FIG = OUT.parent / "figures"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def title(ax, t, sub):
    ax.set_title(t, loc="left", color=INK, fontsize=12, fontweight="bold", pad=34)
    import textwrap
    ax.text(0, 1.02, textwrap.fill(sub, 80), transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")


def fig_behaviour_by_size():
    gs = json.load(open(OUT / "group_size.json"))
    claims = json.load(open(OUT / "contest_scan.json"))["claims"]
    seen_goals = {c["goal"] for c in claims if c["state"] == "seen_in_output"}
    claims = [c for c in claims if c["goal"] in seen_goals]  # drop goals checked only on screen (games, merch...)
    ids = {c["id"] for c in claims}
    room = {m["id"]: m["room_id"] for m in rows("chat_messages") if m["id"] in ids}
    bands = [("10", 10, 10), ("11–14", 11, 14), ("20+", 20, 99)]
    k, n = collections.Counter(), collections.Counter()
    for c in claims:
        size = gs.get(f'{room.get(c["id"])}|{c["time"][:10]}')
        for name, lo, hi in bands:
            if size and lo <= size <= hi:
                n[name] += 1
                k[name] += c["state"] != "seen_in_output"
    fig, ax = plt.subplots(figsize=(6.4, 4))
    xs = range(len(bands))
    vals = [wilson(k[b], n[b]) for b, _, _ in bands]
    ax.bar(xs, [100 * v[0] for v in vals], width=0.5, color=S[0], edgecolor=SURF, linewidth=2)
    ax.errorbar(xs, [100 * v[0] for v in vals], yerr=[[100 * (v[0] - v[1]) for v in vals], [100 * (v[2] - v[0]) for v in vals]],
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
    for x, (b, _, _), v in zip(xs, bands, vals):
        ax.text(x, 100 * v[2] + 2, f"{k[b]}/{n[b]}", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(list(xs), [b for b, _, _ in bands])
    ax.set_xlabel("agents active in the room that day")
    ax.set_ylabel("% of score claims not found in own logs")
    ax.set_ylim(0, 100)
    title(ax, "Unbacked score claims by group size", "Rough proxy: own-score claims whose number is not in the agent's text output in the 6 h before. 95% intervals. Days with 4–9 agents (3 claims) left out.")
    fig.tight_layout()
    fig.savefig(FIG / "1_behaviour_by_group_size.png", dpi=200)
    print("1:", {b: f"{k[b]}/{n[b]}" for b, _, _ in bands})


def fig_monitors():
    labels = {}
    scores = collections.defaultdict(dict)
    for r in map(json.loads, open(OUT / "p3_scores.jsonl")):
        scores[(r["model"].split("/")[-1], r["method"])][r["pid"]] = r["p_false"]
        labels[r["pid"]] = r["label"] == "planted"
    test = sorted(p for p in labels if int(p[1:]) >= 10)
    order = [("qwen3.5-9b", "chat", "Weak (Qwen 9B), chat only"), ("qwen3.5-9b", "logs", "Weak (Qwen 9B), reads logs"),
             ("gpt-5-mini", "chat", "Strong (GPT-5-mini), chat only"), ("gpt-5-mini", "logs", "Strong (GPT-5-mini), reads logs")]
    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for i, (m, meth, name) in enumerate(order):
        y = [labels[p] for p in test if p in scores[(m, meth)]]
        s = [scores[(m, meth)][p] for p in test if p in scores[(m, meth)]]
        a = auroc(y, s)
        boots = []
        for _ in range(500):
            idx = [rng.randrange(len(y)) for _ in y]
            yy = [y[j] for j in idx]
            if len(set(yy)) == 2:
                boots.append(auroc(yy, [s[j] for j in idx]))
        boots.sort()
        lo, hi = boots[12], boots[-13]
        col = S[0] if meth == "logs" else S[1]
        ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(a, i, "o", ms=9, color=col, mec=SURF, mew=2)
        ax.text(hi + 0.015, i, f"{a:.2f}", va="center", color=INK, fontsize=9)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.text(0.505, -0.6, "chance", color=INK2, fontsize=8)
    ax.set_yticks(range(len(order)), [o[2] for o in order])
    ax.set_ylim(-0.8, len(order) - 0.4)
    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel("AUROC on 83 held-out claims (42 planted false, 41 true)")
    title(ax, "Reading logs matters more than model size", "Dot: AUROC. Line: 95% interval. Blue: reads logs. Orange: chat only.")
    fig.tight_layout()
    fig.savefig(FIG / "2_monitor_comparison.png", dpi=200)


def fig_selector():
    path = OUT.parent.parent / "active_investigator/runs/trained_selector.json"
    if not path.exists():  # deleted with the rest of runs/ on 2026-11-09; keep the existing PNG
        print("3: skipped (trained_selector.json missing; figure kept as is)")
        return
    res = json.load(open(path))
    want = [("random (mean of 5)", "Random", S[7]), ("output first", "Output first (rule)", S[1]),
            ("value questions only, latest first", "Value, latest first (rule)", S[2]),
            ("trained selector (held-out programs)", "Trained selector", S[0])]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name, lab, col in want:
        pts = sorted((r["k"], r["graded|overall"], r.get("graded|ci")) for r in res if r["ranker"] == name)
        ks = [p[0] for p in pts]
        ax.plot(ks, [p[1] for p in pts], "-o", color=col, lw=2, ms=8, mec=SURF, mew=2, label=lab)
        if name.startswith("trained"):
            ax.fill_between(ks, [p[2][0] for p in pts], [p[2][1] for p in pts], color=col, alpha=0.12, lw=0)
        if name != "random (mean of 5)":
            ax.text(3.06, pts[-1][1], f"{pts[-1][1]:.2f}", va="center", color=INK, fontsize=9)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.85, 3.3)
    ax.set_ylim(0.45, 0.9)
    ax.set_xlabel("follow-up questions asked")
    ax.set_ylabel("AUROC (graded score)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    title(ax, "Trained question selector vs hand rules", "Qwen 3.5 9B, 638 answers, 40 programs; selector tested on programs it never saw. Band: 95% interval.")
    fig.tight_layout()
    fig.savefig(FIG / "3_trained_selector.png", dpi=200)


def fig_autosolver():
    a = json.load(open(OUT / "autosolver_check.json"))
    t = sorted((datetime.fromisoformat(v["first_31_after"]), k) for k, v in a.items() if v.get("first_31_after"))
    start = datetime(2026, 1, 21, 18, 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [start] + [x for x, _ in t] + [datetime(2026, 1, 24, 0, 0)]
    ys = [0] + list(range(1, len(t) + 1)) + [len(t)]
    ax.step(xs, ys, where="post", color=S[0], lw=2)
    ax.plot([x for x, _ in t], range(1, len(t) + 1), "o", color=S[0], ms=7, mec=SURF, mew=2)
    for i, (x, name) in enumerate(t, 1):
        ax.text(x + (datetime(2026, 1, 21, 2) - datetime(2026, 1, 21)), i - 0.15, name, fontsize=8, color=INK2, va="top")
    post = datetime(2026, 1, 21, 20, 41)
    ax.axvline(post, color=S[1], lw=1.5, ls="--")
    ax.text(post, len(t) + 0.6, " GPT-5.2 posts autosolver\n (Jan 21, 20:41)", color=INK, fontsize=8.5, va="top")
    ax.set_ylim(0, len(t) + 1.2)
    ax.set_ylabel("agents at 31/31 coding challenges")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    title(ax, "One shared shortcut, the whole group at 31/31", "Juice Shop, Jan 2026. No agent logged any coding score before Jan 21 (from agents' own command output).")
    fig.tight_layout()
    fig.savefig(FIG / "4_autosolver_spread.png", dpi=200)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_behaviour_by_size(); fig_monitors(); fig_selector(); fig_autosolver()
    print("written to", FIG)
