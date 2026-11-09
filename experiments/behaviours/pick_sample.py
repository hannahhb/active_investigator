"""Step 3: pick the labelling sample, balanced across group sizes, with peer claims oversampled.

    python3 -m experiments.behaviours.pick_sample prepare      # writes out/to_screen.jsonl for screening
    python3 -m experiments.behaviours.pick_sample final        # writes out/sample.jsonl from the screened rows
"""

import collections
import json
import random
import re
import sys

from scripts.common import OUT

BANDS = ("4", "5-9", "10-14", "20+")
TRIVIAL = re.compile(r"\b(?:in|to|into)\s+(?:the\s+|this\s+|#)?(?:chat|room|group)\b", re.I)
MAX_PER_AGENT = 15


def candidates() -> list[dict]:
    rows = [json.loads(line) for line in open(OUT / "candidates.jsonl")]
    return [r for r in rows if not r["repeat_of"] and r["specific"] and r["band"] in BANDS]


def take(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    """A random draw that limits how many rows one agent can supply."""
    rows = rows[:]
    rng.shuffle(rows)
    used, chosen = collections.Counter(), []
    for row in rows:
        if used[row["agent_id"]] < MAX_PER_AGENT and len(chosen) < count:
            chosen.append(row)
            used[row["agent_id"]] += 1
    return chosen


def prepare() -> None:
    rng, chosen = random.Random(0), []
    rows = candidates()
    for band in BANDS:
        peer = [r for r in rows if r["band"] == band and "peer" in r["kinds"]]
        own = [r for r in rows if r["band"] == band and "peer" not in r["kinds"]]
        rng.shuffle(peer), rng.shuffle(own)
        chosen += peer[:220] + own[:280]
    with open(OUT / "to_screen.jsonl", "w") as out:
        out.writelines(json.dumps(r) + "\n" for r in chosen)
    print("to screen", len(chosen), collections.Counter(r["band"] for r in chosen))


def final() -> None:
    rng = random.Random(1)
    screened = {}
    for line in open(OUT / "screened.jsonl"):
        row = json.loads(line)
        screened[row["message_id"]] = row
    pool = []
    for row in candidates():
        result = screened.get(row["message_id"])
        if result and result.get("checkable") is True and not TRIVIAL.search(result.get("claim") or ""):
            pool.append({**row, "screen": result})
    chosen = []
    for band in BANDS:
        peer = [r for r in pool if r["band"] == band and "peer" in r["kinds"]]
        own = [r for r in pool if r["band"] == band and "peer" not in r["kinds"]]
        picked = take(peer, 55, rng)
        chosen += picked + take(own, 125 - len(picked), rng)
    with open(OUT / "sample.jsonl", "w") as out:
        out.writelines(json.dumps(r) + "\n" for r in chosen)
    print("pool", len(pool), "sample", len(chosen))
    print("bands", collections.Counter(r["band"] for r in chosen))
    print("peer", sum("peer" in r["kinds"] for r in chosen), "agents", len({r["agent_id"] for r in chosen}))
    print("months", sorted(collections.Counter(r["time"][:7] for r in chosen).items()))


if __name__ == "__main__":
    {"prepare": prepare, "final": final}[sys.argv[1]]()
