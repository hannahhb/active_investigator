"""Write short reading packets for human or subagent reading. No model calls."""

import collections
import json
import random
import re

from scripts.common import OUT

WORD = re.compile(r"[a-z0-9][a-z0-9_\-./]{3,}", re.I)
STOP = {"that", "this", "with", "have", "from", "been", "just", "confirmed", "verified", "reported", "shared", "claude", "gemini", "opus", "sonnet"}


def score(words: set[str], text: str) -> int:
    return len(words & {w.lower() for w in WORD.findall(text)})


def lines(turns: list[dict], words: set[str], top: int, last: int) -> list[str]:
    acts = [t for t in turns if "send_message_back_to_chat" not in t["action"]]
    pick = {t["turn_id"]: t for t in sorted(acts, key=lambda t: score(words, t["action"] + t["output"]), reverse=True)[:top]}
    pick.update({t["turn_id"]: t for t in (acts[-last:] if last else [])})
    out = []
    for t in sorted(pick.values(), key=lambda t: t["time"]):
        output = "" if t["output"] in ("null", "") else " -> " + t["output"][:110].replace("\n", " ")
        out.append(f"{t['time'][5:16]} {t['action'][:160]}{output}".replace("\n", " "))
    return out


def packet(claim: dict, number: str) -> str:
    target = " ".join(claim["claims"])[:320]
    words = {w.lower() for w in WORD.findall(target)} - STOP
    peers = set(claim.get("named_peer_names") or [])
    own = [t for t in claim["turns"] if t.get("owner") == claim["agent_id"]]
    other = [t for t in claim["turns"] if t.get("owner") != claim["agent_id"]]
    source = sorted(
        (m for m in claim["peer_chat"] if m["speaker"] in peers or m["speaker"] == "human"),
        key=lambda m: score(words, m["text"]),
        reverse=True,
    )[:3]
    part = [
        f"### {number} | {claim['time'][:16]} | speaker: {claim['agent']} | band {claim['band']} | turns {claim['turn_count']}",
        f"CLAIM: {target}",
        f"MESSAGE: {claim['text'][:420]}",
    ]
    if claim.get("tokens"):
        part.append(f"IDENTIFIERS NOT FOUND IN EARLIER LOGS: {claim['tokens']}  (appears later: {claim.get('appears_later')}, later in output: {claim.get('later_output')})")
    if peers:
        part.append(f"NAMED PEERS: {sorted(peers)}")
    part += [f"SOURCE CHAT: {m['time'][5:16]} {m['speaker']}: {m['text'][:210]}".replace("\n", " ") for m in source]
    part.append("SPEAKER ACTIONS:")
    part += ["  " + x for x in (lines(own, words, 6, 2) or ["(none shown)"])]
    if other:
        part.append("NAMED PEER ACTIONS:")
        part += ["  " + x for x in lines(other, words, 6, 0)]
    return "\n".join(part)


def main() -> None:
    claims = [json.loads(line) for line in open(OUT / "peer_evidence.jsonl")]
    rng = random.Random(3)
    chosen = []
    for band in ("4", "5-9", "10-14", "20+"):
        pool = [c for c in claims if c["band"] == band and c["named_peers"]]
        rng.shuffle(pool)
        chosen += pool[:25]
    print("peer pool with named peers", collections.Counter(c["band"] for c in claims if c["named_peers"]), "chosen", len(chosen))
    rng.shuffle(chosen)
    index = {}
    with open(OUT / "peer100_packets.md", "w") as out:
        for number, claim in enumerate(chosen, 1):
            key = f"P{number:03d}"
            index[key] = claim["message_id"]
            out.write(packet(claim, key) + "\n\n")
    json.dump(index, open(OUT / "peer100_index.json", "w"))
    suspects = [json.loads(line) for line in open(OUT / "suspect_evidence.jsonl")]
    suspects.sort(key=lambda c: (c.get("later_output", 0) > 0, c["time"]))
    index = {}
    with open(OUT / "suspect_packets.md", "w") as out:
        for number, claim in enumerate(suspects, 1):
            key = f"S{number:03d}"
            index[key] = claim["message_id"]
            out.write(packet(claim, key) + "\n\n")
    json.dump(index, open(OUT / "suspect_index.json", "w"))
    print("suspects", len(suspects))


if __name__ == "__main__":
    main()
