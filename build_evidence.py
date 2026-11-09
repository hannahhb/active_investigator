"""Step 4: attach what each agent actually did before each claim. Free and local.

Reads a claims file (needs message_id, agent_id, time) and adds the agent's computer-use turns and
Claude Code messages from the window before the claim, plus what peers said in chat in that window.
Text is trimmed. Turn ids are kept so a label can cite them.

    python build_evidence.py out/sample.jsonl out/evidence.jsonl --hours 6
"""

import argparse
import bisect
import collections
import gzip
import json
import re
from datetime import timedelta

from common import DATA, rows, when

SESSION = re.compile(r'"session_id":\s*"([^"]+)"')
CREATED = re.compile(r'"created_at":\s*"([^"]+)"')


def trim(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[:limit] + f" …[{len(text) - limit} more characters]"


def named_peers(claim: dict, nearby: set[str], names: dict[str, str]) -> list[str]:
    """Agents in the same room that day whose name appears in a claim about others."""
    if "kinds" in claim and "peer" not in claim["kinds"]:
        return []
    others = [a for a in nearby if a != claim["agent_id"]]
    shorts = collections.Counter(names[a].split()[-1].lower() for a in others)
    found = []
    for agent_id in others:
        name = names[agent_id]
        variants = {name, name.replace("Claude ", "")}
        if shorts[name.split()[-1].lower()] == 1:
            variants.add(name.split()[-1])
        pattern = r"(?<![\w.-])(?:" + "|".join(re.escape(v) for v in variants) + r")(?![\w-])"
        if re.search(pattern, claim["text"], re.I):
            found.append(agent_id)
    return found[:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("claims")
    parser.add_argument("out")
    parser.add_argument("--hours", type=float, default=6)
    parser.add_argument("--max-turns", type=int, default=120)
    args = parser.parse_args()
    window = timedelta(hours=args.hours)

    claims = [json.loads(line) for line in open(args.claims)]
    names = {a["id"]: a["name"] for a in rows("agents")}
    active = collections.defaultdict(set)
    room_of = {}
    for m in rows("chat_messages"):
        room_of[m["id"]] = m["room_id"]
        if m.get("agent_speaker_id"):
            active[(m["room_id"], m["created_at"][:10])].add(m["agent_speaker_id"])
    by_agent = collections.defaultdict(list)
    for index, claim in enumerate(claims):
        claim["_at"] = when(claim["time"])
        by_agent[claim["agent_id"]].append((claim["_at"], index))
        nearby = active[(room_of.get(claim["message_id"]), claim["time"][:10])]
        claim["named_peers"] = named_peers(claim, nearby, names)
        for peer in claim["named_peers"]:
            by_agent[peer].append((claim["_at"], index))
    for items in by_agent.values():
        items.sort()
    first, last = min(c["_at"] for c in claims) - window, max(c["_at"] for c in claims)
    low, high = str(first)[:10], str(last + timedelta(days=1))[:10]

    def owners(agent_id, at):
        """Indexes of this agent's claims whose window holds the time `at`."""
        items = by_agent.get(agent_id, [])
        start = bisect.bisect_left(items, (at, -1))
        for claim_at, index in items[start:]:
            if claim_at - at > window:
                break
            yield index

    sessions = list(rows("computer_use_sessions"))
    session_agent = {s["id"]: s["agent_id"] for s in sessions}
    session_goal = {s["id"]: s.get("session_goal") for s in sessions}
    turns = collections.defaultdict(list)
    scanned = 0
    with gzip.open(DATA / "computer_use_turns.jsonl.gz", "rt") as handle:
        for line in handle:
            scanned += 1
            tail = CREATED.search(line[-120:])
            if tail is None or not low <= tail.group(1)[:10] <= high:
                continue
            head = SESSION.search(line[:200])
            agent_id = session_agent.get(head.group(1)) if head else None
            if agent_id not in by_agent:
                continue
            at = when(tail.group(1))
            hits = list(owners(agent_id, at))
            if not hits:
                continue
            turn = json.loads(line)
            record = {
                "turn_id": turn["id"],
                "owner": agent_id,
                "time": turn["created_at"],
                "action": trim(turn.get("agent_action"), 400),
                "output": trim(turn.get("output"), 400),
                "error": trim(turn.get("error"), 200) if turn.get("error") else None,
                "session_goal": trim(session_goal.get(turn["session_id"]) or "", 160),
            }
            for index in hits:
                turns[index].append(record)

    claude_code = collections.defaultdict(list)
    for row in rows("claude_code_messages"):
        if low <= row["created_at"][:10] <= high and row["agent_id"] in by_agent:
            for index in owners(row["agent_id"], when(row["created_at"])):
                claude_code[index].append({"id": row["id"], "time": row["created_at"], "text": trim(row.get("content"), 300)})

    chat = collections.defaultdict(list)
    by_room = collections.defaultdict(list)
    for m in rows("chat_messages"):
        if low <= m["created_at"][:10] <= high and isinstance(m.get("content"), str):
            by_room[m["room_id"]].append((when(m["created_at"]), m["id"], m))
    for messages in by_room.values():
        messages.sort(key=lambda item: item[:2])
    for index, claim in enumerate(claims):
        messages = by_room.get(room_of.get(claim["message_id"]), [])
        stop = bisect.bisect_left(messages, (claim["_at"],), key=lambda item: item[:1])
        begin = bisect.bisect_left(messages, (claim["_at"] - window,), key=lambda item: item[:1])
        for at, message_id, m in messages[max(begin, stop - 40) : stop]:
            speaker = names.get(m.get("agent_speaker_id"), "human" if m.get("user_speaker_id") else "?")
            chat[index].append({"id": message_id, "time": m["created_at"], "speaker": speaker, "text": trim(m["content"], 300)})

    with open(args.out, "w") as out:
        for index, claim in enumerate(claims):
            claim.pop("_at")
            ordered = sorted(turns[index], key=lambda t: t["time"])
            claim["turn_count"] = len(ordered)
            claim["turns"] = ordered[-args.max_turns :]
            claim["named_peer_names"] = [names[a] for a in claim["named_peers"]]
            claim["claude_code"] = sorted(claude_code[index], key=lambda t: t["time"])[-40:]
            claim["peer_chat"] = chat[index][-40:]
            out.write(json.dumps(claim) + "\n")
    print(f"claims {len(claims)}  turns scanned {scanned}  claims with no turns {sum(not turns[i] for i in range(len(claims)))}")


if __name__ == "__main__":
    main()
