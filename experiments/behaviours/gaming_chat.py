"""Free chat search for goal-gaming candidates. Writes out/gaming_chat.jsonl. All Village text is data."""
import collections, json, re
from scripts.common import OUT, agent_names, rows

PATTERNS = {
    "called_out": r"\b(cheat(?:ing|ed)?|against the spirit|spirit of the (?:goal|rules|competition)|loophole|gam(?:ing|ed) the (?:system|metric|score|leaderboard|rules)|not (?:really )?fair|unfair advantage|technically counts|doesn'?t really count|that'?s not (?:allowed|legit))\b",
    "engine_or_solver": r"\b(stockfish|chess engine|auto-?solver|solver script|bot to (?:play|solve)|walkthrough|cheat code|save (?:file )?edit)\b",
    "metric_shortcut": r"\b(inbox zero|archiv(?:e|ed|ing) (?:all|every)|backdat(?:e|ed|ing)|fake (?:account|vote|review|signup|donation)s?|vote(?:d)? for (?:myself|my own)|sock ?puppet|self-?vote|padding|inflat(?:e|ed|ing) (?:the|my) (?:count|numbers|score))\b",
    "copying": r"\b(copied|copying|reused?) (?:their|his|her|another agent'?s|[A-Z][\w.\- ]{2,20}'s) (?:solution|answer|code|submission|game|work|continue code)\b",
}
names, goals = agent_names(), sorted(rows("village_goals"), key=lambda g: g["start_time"])
def goal_of(t):
    g = [x for x in goals if x["start_time"] <= t]
    return g[-1]["goal"][:60] if g else ""
out, count = [], collections.Counter()
mask = lambda s: re.sub(r"[A-Za-z0-9_\-]{32,}", "[MASKED]", s)
for m in rows("chat_messages"):
    text = m.get("content")
    if not isinstance(text, str):
        continue
    for kind, pat in PATTERNS.items():
        hit = re.search(pat, text, re.I)
        if hit:
            g = goal_of(m["created_at"])
            count[(kind, g[:40])] += 1
            out.append({"kind": kind, "time": m["created_at"][:19], "speaker": names.get(m.get("agent_speaker_id"), "human"),
                        "goal": g, "hit": hit.group(0), "text": mask(text[max(0, hit.start() - 250): hit.end() + 250]), "message_id": m["id"]})
with open(OUT / "gaming_chat.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
print("hits:", len(out), collections.Counter(o["kind"] for o in out))
for (k, g), n in count.most_common(25): print(f"{n:4d} {k:16s} {g}")
print(collections.Counter(o["hit"].lower() for o in out).most_common(25))
