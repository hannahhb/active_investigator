"""Step 1: find messages that may hold a checkable claim, using word patterns only. Free and local.

Writes out/candidates.jsonl: one row per message, with the matched sentences, the group size that day
(agents who spoke in that room that day) and the village goal running at the time.
This step favours finding claims over being precise. Step 2 removes the vague ones.

    python filter_claims.py
"""

import collections
import hashlib
import json
import re

from common import OUT, agent_names, rows, when

VERBS = "sent|exported|created|uploaded|shared|published|posted|deployed|emailed|verified|confirmed|submitted|completed|fixed|saved|downloaded"
SELF_CLAIM = re.compile(rf"\bI(?:'ve| have| just)?\s+(?:successfully\s+|already\s+)?(?:{VERBS})\b", re.I)
NAMES = r"(?:o1|o3|o4-mini|Claude[\w. ]{0,18}|Opus[\w. ]{0,8}|Sonnet[\w. ]{0,8}|Haiku[\w. ]{0,8}|Gemini[\w. ]{0,12}|GPT[\w.\- ]{0,10}|Grok[\w. ]{0,6}|DeepSeek[\w.\- ]{0,10}|Kimi[\w. ]{0,8})"
PEER_CLAIM = re.compile(
    rf"\b(?:as|per|according to)\s+@?{NAMES}\b.{{0,40}}\b(?:reported|said|confirmed|shared|created|exported|mentioned|noted)\b"
    rf"|\b{NAMES}\s+(?:has\s+)?(?:already\s+)?(?:confirmed|verified|reported|exported|shared)\b",
    re.I,
)
SPECIFIC = re.compile(r"https?://|\b[0-9a-f]{8,64}\b|\b\w+\.(?:csv|json|md|py|html|pdf|txt|xlsx)\b|\b\d[\d,.]*\b", re.I)
SENTENCE = re.compile(r"(?<=[.!?\n])\s+")


def band(size: int) -> str:
    for low, high, name in ((1, 3, "1-3"), (4, 4, "4"), (5, 9, "5-9"), (10, 14, "10-14"), (15, 19, "15-19")):
        if low <= size <= high:
            return name
    return "20+"


def main() -> None:
    names = agent_names()
    rooms = {r["id"]: r["name"] for r in rows("chat_rooms")}
    goals = sorted((g for g in rows("village_goals") if g.get("start_time")), key=lambda g: g["start_time"])
    messages = [m for m in rows("chat_messages") if m.get("agent_speaker_id") and isinstance(m.get("content"), str)]
    active = collections.defaultdict(set)
    for m in messages:
        active[(m["room_id"], m["created_at"][:10])].add(m["agent_speaker_id"])

    seen, kept, totals = {}, 0, collections.Counter()
    with open(OUT / "candidates.jsonl", "w") as out:
        for m in sorted(messages, key=lambda x: x["created_at"]):
            text = m["content"]
            size = len(active[(m["room_id"], m["created_at"][:10])])
            totals[("all", band(size))] += 1
            sentences = [s.strip() for s in SENTENCE.split(text)]
            mine = [s for s in sentences if SELF_CLAIM.search(s)]
            peer = [s for s in sentences if PEER_CLAIM.search(s)]
            if not mine and not peer:
                continue
            digest = hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()
            goal = next((g["goal"] for g in reversed(goals) if g["start_time"] <= m["created_at"]), None)
            row = {
                "message_id": m["id"],
                "time": m["created_at"],
                "room": rooms.get(m["room_id"], m["room_id"]),
                "agent_id": m["agent_speaker_id"],
                "agent": names.get(m["agent_speaker_id"]),
                "group_size": size,
                "band": band(size),
                "goal": (goal or "")[:160],
                "kinds": [k for k, hit in (("self", mine), ("peer", peer)) if hit],
                "claims": (mine + peer)[:6],
                "specific": bool(SPECIFIC.search(" ".join(mine + peer))),
                "repeat_of": seen.get(digest),
                "text": text,
            }
            seen.setdefault(digest, m["id"])
            out.write(json.dumps(row) + "\n")
            kept += 1
            totals[("candidate", row["band"])] += 1
            totals[("specific", row["band"])] += row["specific"] and not row["repeat_of"]
            totals["self"] += "self" in row["kinds"]
            totals["peer"] += "peer" in row["kinds"]
            totals["repeat"] += row["repeat_of"] is not None
    print(f"messages {len(messages)}  candidates {kept}  self {totals['self']}  peer {totals['peer']}  exact repeats {totals['repeat']}")
    print("band    messages  candidates  specific_unique")
    for name in ("1-3", "4", "5-9", "10-14", "15-19", "20+"):
        print(f"{name:7s} {totals[('all', name)]:8d}  {totals[('candidate', name)]:10d}  {totals[('specific', name)]:8d}")


if __name__ == "__main__":
    main()
