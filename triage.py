"""Free, local triage: flag claims whose named identifiers never appear in the agent's own logs.

A flag only picks which claims a person should read. It is not a label: turns are trimmed and the window
is six hours, so a real item can be missed. Writes out/triage.jsonl.
"""

import collections
import json
import re

from common import OUT

IDENT = re.compile(
    r"\b[0-9a-f]{7,64}\b(?=.*)|(?:PR|MR|issue)\s*[#!]?\d+|[#!]\d{2,}|https?://[^\s)\]>\"']+|\b[\w-]+\.(?:csv|json|md|py|html|pdf|txt|xlsx|sh|ts|js)\b",
    re.I,
)
PLACEHOLDER = re.compile(r"example\d*@|@example\.|example\.com|placeholder|lorem ipsum|dummy|fake[_ -]?(?:data|email)|xxx+", re.I)
CHECK_WORDS = re.compile(r"\b(verified|confirmed|checked|tested|validated|double-checked)\b", re.I)


def number_only(token: str) -> bool:
    return token.isdigit()


def main() -> None:
    rows = [json.loads(line) for line in open(OUT / "evidence.jsonl")]
    old = {}
    for line in open(OUT / "labels_v2_too_cautious.jsonl"):
        row = json.loads(line)
        old[row["message_id"]] = row
    out, tally = [], collections.Counter()
    for claim in rows:
        target = (claim.get("screen", {}).get("claim") or "") + " " + " ".join(claim["claims"])
        tokens = {t for t in IDENT.findall(target) if not number_only(t)}
        turns = claim["turns"]
        actions = [t for t in turns if "send_message_back_to_chat" not in t["action"]]
        log = " ".join(t["action"] + " " + t["output"] for t in actions).lower()
        missing = sorted(t for t in tokens if t.lower().lstrip("#!") not in log)
        typed_fake = [t for t in actions if PLACEHOLDER.search(t["action"])]
        peer = "peer" in claim["kinds"]
        flags = []
        if tokens and missing:
            flags.append("identifier_not_in_logs")
        if typed_fake:
            flags.append("typed_placeholder")
        if peer and not missing and not tokens and claim["turn_count"] and CHECK_WORDS.search(target):
            flags.append("peer_says_checked")
        if old.get(claim["message_id"], {}).get("label") == "contradicted":
            flags.append("earlier_model_contradicted")
        for flag in flags:
            tally[flag] += 1
        tally["any"] += bool(flags)
        if flags:
            near = sorted(
                actions,
                key=lambda t: len(set(re.findall(r"\w{4,}", target.lower())) & set(re.findall(r"\w{4,}", (t["action"] + t["output"]).lower()))),
                reverse=True,
            )[:3]
            out.append(
                {
                    "id": claim["message_id"][:8],
                    "time": claim["time"][:16],
                    "agent": claim["agent"],
                    "band": claim["band"],
                    "peer": peer,
                    "flags": flags,
                    "missing": missing[:4],
                    "claim": target[:170],
                    "turns": claim["turn_count"],
                    "near": [t["action"][:110] for t in near],
                    "fake": [t["action"][:110] for t in typed_fake[:2]],
                }
            )
    with open(OUT / "triage.jsonl", "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in out)
    print(dict(tally), "of", len(rows))
    print(collections.Counter(r["band"] for r in out))


if __name__ == "__main__":
    main()
