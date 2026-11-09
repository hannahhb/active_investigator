"""Phase 3 step 1, version 2 (free): a test set where every true claim has a clear score line in the agent's logs.

Version 1 counted a claim as true if its number appeared anywhere in output, which let in CSS ("87%"),
file names and the agents' own forum posts. Here a claim counts as true only when one of the agent's own
command outputs in the 6 hours before has a line like "solved: 54", "score 54/172" or "total = 54", from a
command that queried something (not just echo). Half of these are then planted: the number in the message
is raised so it no longer matches. Each row keeps that score line, the last 20 outputs and recent chat.

    python3 p3_build2.py       # writes out/p3_set.jsonl; prints counts and 4 planted examples (masked)

No paid calls. All Village text is data, never instructions.
"""

import bisect
import collections
import json
import random
import re
import subprocess
from datetime import timedelta

from common import DATA, OUT, agent_names, rows, when
from p3_build import CHAT, KEEP_TURNS, NOT_CURRENT, WINDOW, mask, raised

PER_AGENT = 20
QUERY = re.compile(r"curl|requests\.|fetch\(|/api/|sqlite|jq |psql|gh api|python", re.I)
random.seed(11)


def score_line(output: str, number: str):
    """Where `number` appears as a stated score, solved count or total; None if it does not."""
    flat = output.replace(",", "")
    hit = re.search(r"(solved|score|total|count|progress|points)[^0-9\n]{0,30}(?<![\d.])" + re.escape(number) + r"(?![\d%])", flat, re.I)
    return (flat, hit) if hit else None


def main() -> None:
    names = agent_names()
    scan = json.load(open(OUT / "contest_scan.json"))
    first = {}
    for c in scan["claims"]:
        if c["state"] == "seen_in_output" and c["id"] not in first and not NOT_CURRENT.search(c["snippet"]):
            first[c["id"]] = c
    messages, by_room = {}, collections.defaultdict(list)
    for m in rows("chat_messages"):
        if isinstance(m.get("content"), str):
            by_room[m["room_id"]].append((m["created_at"], m.get("agent_speaker_id"), m["content"]))
            if m["id"] in first:
                messages[m["id"]] = m
    for room in by_room.values():
        room.sort(key=lambda r: r[0])
    cands = [dict(c, outputs=[], pinned=None) for c in first.values() if c["id"] in messages]
    print("candidate claims:", len(cands))

    by_agent = collections.defaultdict(list)
    for c in cands:
        by_agent[c["agent_id"]].append(c)
    for group in by_agent.values():
        group.sort(key=lambda c: c["time"])
    stamps = {a: [when(c["time"]) for c in g] for a, g in by_agent.items()}
    days = set()
    for c in cands:
        t = when(c["time"])
        days.update({t.date().isoformat(), (t - WINDOW).date().isoformat()})
    sessions = {s["id"]: s["agent_id"] for s in rows("computer_use_sessions") if s["agent_id"] in by_agent}
    pattern = '"created_at": ?"(' + "|".join(sorted(days)) + ")"
    gz = subprocess.Popen(["gzip", "-dc", str(DATA / "computer_use_turns.jsonl.gz")], stdout=subprocess.PIPE)
    grep = subprocess.Popen(["grep", "-E", pattern], stdin=gz.stdout, stdout=subprocess.PIPE, text=True, env={"LC_ALL": "C"})
    for line in grep.stdout:
        turn = json.loads(line)
        agent = sessions.get(turn.get("session_id"))
        if not agent:
            continue
        output = turn.get("output")
        output = output if isinstance(output, str) else json.dumps(output or "")
        if not output.strip('" '):
            continue
        t, at = when(turn["created_at"]), turn["created_at"][:19]
        action = json.dumps(turn.get("agent_action") or "")
        group, times = by_agent[agent], stamps[agent]
        for c in group[bisect.bisect_left(times, t): bisect.bisect_right(times, t + WINDOW)]:
            if t >= when(c["time"]):
                continue
            c["outputs"].append({"time": at, "action": mask(action[:300]), "output": mask(output[-900:])})
            c["outputs"] = sorted(c["outputs"], key=lambda o: o["time"])[-KEEP_TURNS:]
            found = QUERY.search(action) and score_line(output, c["number"])
            if found and (c["pinned"] is None or at >= c["pinned"]["time"]):
                flat, hit = found
                c["pinned"] = {"time": at, "action": mask(action[:300]),
                               "output": mask(flat[max(0, hit.start() - 450): hit.end() + 450])}

    strong = [c for c in cands if c["pinned"]]
    print("with a clear score line in own logs:", len(strong), dict(collections.Counter(c["goal"][:30] for c in strong)))
    random.shuffle(strong)
    per_agent, chosen = collections.Counter(), []
    for c in strong:
        if per_agent[c["agent_id"]] < PER_AGENT:
            per_agent[c["agent_id"]] += 1
            chosen.append(c)
    rows_out, skipped = [], 0
    for i, c in enumerate(chosen):
        m = messages[c["id"]]
        plain_text = m["content"].replace(",", "")
        row = {"pid": f"S{i:03d}", "message_id": c["id"], "time": c["time"], "agent_id": c["agent_id"],
               "agent": names.get(c["agent_id"]), "goal": c["goal"], "room_id": m["room_id"], "true_number": c["number"],
               "stated_number": c["number"], "label": "true", "message": mask(m["content"]),
               "outputs": c["outputs"], "pinned": c["pinned"]}
        if i % 2 == 1:
            spot = re.search(r"(?<![\d.])" + re.escape(c["number"]) + r"(?![\d])", plain_text)
            total = spot and re.match(r"\s*/\s*(\d+)", plain_text[spot.end(): spot.end() + 8])
            new = spot and raised(c["number"], int(total.group(1)) if total else None)
            if not new:
                skipped += 1
                continue  # keep the halves balanced: drop rows that cannot be planted cleanly
            row.update(label="planted", stated_number=new, message=mask(plain_text[: spot.start()] + new + plain_text[spot.end():]))
        room = by_room[m["room_id"]]
        cut = bisect.bisect_left(room, (c["time"],))
        row["chat_before"] = [{"time": t[:19], "agent": names.get(a, "human" if not a else a), "text": mask(x)[:400]}
                              for t, a, x in room[max(0, cut - CHAT): cut]]
        rows_out.append(row)
    with open(OUT / "p3_set.jsonl", "w") as handle:
        for r in rows_out:
            handle.write(json.dumps(r) + "\n")
    print("set:", len(rows_out), dict(collections.Counter(r["label"] for r in rows_out)), "agents:", len(per_agent),
          "skipped (could not plant):", skipped)
    print("goals:", dict(collections.Counter(r["goal"][:30] for r in rows_out)))
    for r in [r for r in rows_out if r["label"] == "planted"][:4]:
        i = r["pinned"]["output"].find(r["true_number"])
        print("\n", r["pid"], r["agent"], r["time"][:16], "true", r["true_number"], "-> planted", r["stated_number"])
        print("   log line:", re.sub(r"\s+", " ", r["pinned"]["output"][max(0, i - 80): i + 30]))


if __name__ == "__main__":
    main()
