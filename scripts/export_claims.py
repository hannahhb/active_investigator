"""Write data/claims.csv: the evaluation claims by AI Village message ID, with labels and numbers, no text. Free.

    python3 -m scripts.export_claims
"""
import csv
import json

from experiments.stress_tests.p8_peer_claim import claims as peer_claims
from scripts.common import OUT, ROOT


def main():
    rows = [json.loads(l) for l in open(OUT / "p3_set.jsonl")]
    peer = {r["pid"] for r in peer_claims()[1]}
    with open(ROOT / "data" / "claims.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pid", "split", "message_id", "room_id", "time_utc", "agent", "goal", "label",
                    "true_number", "stated_number", "in_peer_influence_test"])
        for i, r in enumerate(rows):
            w.writerow([r["pid"], "tuning" if i < 10 else "test", r["message_id"], r["room_id"], r["time"][:19],
                        r["agent"], r["goal"][:80], "false_planted" if r["label"] == "planted" else "true",
                        r["true_number"], r["stated_number"], r["pid"] in peer])
    print(len(rows), "claims ->", ROOT / "data" / "claims.csv")


if __name__ == "__main__":
    main()
