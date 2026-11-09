"""Free check: who ran the /snippets autosolver, and did their logged coding-challenge count jump after
GPT-5.2 posted it (2026-01-21 20:52:46)? Reads only the agents' own command outputs. Writes out/autosolver_check.json."""
import collections, json, re, subprocess
from common import DATA, OUT, agent_names, rows

POST = "2026-01-21 20:52:46"
RAN = re.compile(r"snippets/(?:verdict|fixes)|coding_autosolve|autosolve", re.I)
CODING = re.compile(r"[Cc]oding[^0-9\n]{0,40}?(\d{1,2})\s*/\s*31\b")
names = agent_names()
sessions = {s["id"]: s["agent_id"] for s in rows("computer_use_sessions")}
days = "|".join(f"2026-01-{d:02d}" for d in range(12, 27))
gz = subprocess.Popen(["gzip", "-dc", str(DATA / "computer_use_turns.jsonl.gz")], stdout=subprocess.PIPE)
grep = subprocess.Popen(["grep", "-E", f'"created_at": ?"({days})'], stdin=gz.stdout, stdout=subprocess.PIPE, text=True, env={"LC_ALL": "C"})
ran, scores = collections.defaultdict(list), collections.defaultdict(list)
for line in grep.stdout:
    t = json.loads(line)
    agent = names.get(sessions.get(t.get("session_id")))
    if not agent:
        continue
    at = t["created_at"][:19]
    action = json.dumps(t.get("agent_action") or "")
    if RAN.search(action):
        ran[agent].append(at)
    out = t.get("output")
    out = out if isinstance(out, str) else json.dumps(out or "")
    for v in CODING.findall(out):
        scores[agent].append((at, int(v)))
result = {}
for agent in sorted(set(ran) | set(scores)):
    before = [v for at, v in scores[agent] if at < POST]
    after = [(at, v) for at, v in scores[agent] if at >= POST]
    first_run = min(ran[agent]) if ran[agent] else None
    result[agent] = {"ran_autosolver_first": first_run, "runs": len(ran[agent]),
                     "coding_max_before": max(before) if before else None,
                     "coding_max_after": max(v for _, v in after) if after else None,
                     "first_31_after": next((at for at, v in sorted(after) if v == 31), None)}
json.dump(result, open(OUT / "autosolver_check.json", "w"), indent=1)
for agent, r in result.items():
    print(f"{agent:22s} ran:{(r['ran_autosolver_first'] or '-')[:16]:16s} x{r['runs']:<4d} coding before {str(r['coding_max_before']):>4s} -> after {str(r['coding_max_after']):>4s}  first 31/31 after post: {(r['first_31_after'] or '-')[:16]}")
