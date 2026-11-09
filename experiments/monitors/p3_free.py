"""Phase 3 free baselines on out/p3_set.jsonl: random flagging and keyword flagging. No paid calls."""
import json, random, re
from scripts.common import OUT

KEYWORDS = re.compile(r"\b(verified|confirmed|definitely|all|record|perfect|100%|complete[d]?|huge|massive|top|1st|first place|leading)\b|!|🎯|🏆", re.I)


def catch_at(labels, scores, rate):
    """Share of planted claims caught when the threshold lets through at most `rate` of true claims as false alarms."""
    honest = sorted((s for l, s in zip(labels, scores) if not l), reverse=True)
    allowed = int(rate * len(honest))
    threshold = honest[allowed] if allowed < len(honest) else float("-inf")
    bad = [s for l, s in zip(labels, scores) if l]
    return sum(s > threshold for s in bad) / len(bad)


def auroc(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l]; neg = [s for l, s in zip(labels, scores) if not l]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def report(name, labels, scores):
    print(f"{name:10s} AUROC {auroc(labels, scores):.2f}  caught at 1% FA {catch_at(labels, scores, .01):.0%}  at 5% FA {catch_at(labels, scores, .05):.0%}")


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(OUT / "p3_set.jsonl")]
    labels = [r["label"] == "planted" for r in rows]
    random.seed(0)
    report("random", labels, [random.random() for _ in rows])
    report("keyword", labels, [len(KEYWORDS.findall(r["message"])) for r in rows])
