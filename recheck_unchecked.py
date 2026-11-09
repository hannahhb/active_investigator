"""Check 'both readers said unchecked' claims against the speaker's full, untrimmed logs. Free and local."""

import collections
import gzip
import json
import re
from datetime import timedelta

from common import DATA, OUT, rows, when

A = {json.loads(l)["packet"]: json.loads(l) for l in open(OUT / "peer100_reader_a.jsonl")}
B = {json.loads(l)["packet"]: json.loads(l) for l in open(OUT / "peer100_reader_b.jsonl")}
index = json.load(open(OUT / "peer100_index.json"))
evidence = {}
for line in open(OUT / "peer_evidence.jsonl"):
    row = json.loads(line)
    evidence[row["message_id"]] = row
both = [k for k in A if A[k]["label"] == B[k]["label"] == "unchecked_repeat"]
GENERIC = {"confirmed", "verified", "reported", "shared", "claude", "gemini", "opus", "sonnet", "haiku", "grok", "kimi", "deepseek", "there", "their", "about", "which", "after", "before", "would", "should", "working", "still", "great", "thanks", "update", "status", "today", "agent", "agents", "chapter", "session", "team", "other", "these", "those", "first", "being", "right", "could", "email", "check", "using", "noted", "means", "needs", "ready", "added", "fixed", "works", "based", "while", "since", "every", "system", "already", "created", "exported", "mentioned", "correct", "currently"}
STRONG = re.compile(r"https?://\S+|[#!]\d+|\b\d{3,}\b|\b[\w-]+\.(?:csv|json|md|py|html|pdf|txt|sh|js|ts)\b|\b[0-9a-f]{7,40}\b|\b[A-Z][a-z]+[A-Z]\w+\b|\b[A-Z]{2,}[\w-]*\b")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{4,}")
claims = {}
for key in both:
    claim = evidence[index[key]]
    target = " ".join(claim["claims"])
    names = " ".join(claim.get("named_peer_names") or []).lower()
    strong = {s.rstrip(".,;:)`*").lower() for s in STRONG.findall(target)}
    strong = {s for s in strong if s not in names and s not in GENERIC and len(s) >= 3}
    words = {w.lower() for w in WORD.findall(target)} - GENERIC - set(names.split())
    claims[key] = {"agent": claim["agent_id"], "at": when(claim["time"]), "strong": strong, "words": words, "target": target}
sessions = {s["id"]: s["agent_id"] for s in rows("computer_use_sessions")}
by_agent = collections.defaultdict(list)
for key, claim in claims.items():
    by_agent[claim["agent"]].append(key)
low = str(min(c["at"] for c in claims.values()) - timedelta(hours=6))[:10]
high = str(max(c["at"] for c in claims.values()))[:10]
CREATED = re.compile(r'"created_at":\s*"([^"]+)"')
SESSION = re.compile(r'"session_id":\s*"([^"]+)"')
hits = collections.defaultdict(list)
wordhits = collections.defaultdict(collections.Counter)
with gzip.open(DATA / "computer_use_turns.jsonl.gz", "rt") as handle:
    for line in handle:
        tail = CREATED.search(line[-120:])
        if not tail or not low <= tail.group(1)[:10] <= high:
            continue
        head = SESSION.search(line[:200])
        agent = sessions.get(head.group(1)) if head else None
        if agent not in by_agent:
            continue
        at = when(tail.group(1))
        relevant = [k for k in by_agent[agent] if timedelta(0) <= claims[k]["at"] - at <= timedelta(hours=6)]
        if not relevant or "send_message_back_to_chat" in line[:600]:
            continue
        turn = json.loads(line)
        blob = (json.dumps(turn.get("agent_action") or "") + " " + json.dumps(turn.get("output") or "")).lower()
        for key in relevant:
            for term in claims[key]["strong"]:
                if term in blob:
                    spot = blob.find(term)
                    hits[key].append((turn["created_at"][5:16], term, blob[max(0, spot - 70) : spot + 90]))
            for word in claims[key]["words"]:
                if word in blob:
                    wordhits[key][word] += 1
result = {}
for key in both:
    strong_found = sorted({h[1] for h in hits[key]})
    result[key] = {"strong_terms": sorted(claims[key]["strong"]), "strong_found": strong_found, "word_share": round(len(wordhits[key]) / max(len(claims[key]["words"]), 1), 2), "examples": hits[key][:2], "target": claims[key]["target"][:180]}
json.dump(result, open(OUT / "recheck_unchecked.json", "w"), indent=0)
found = [k for k in both if result[k]["strong_found"]]
print("both unchecked", len(both), "have strong terms", sum(bool(result[k]["strong_terms"]) for k in both), "strong term found in speaker's full logs", len(found))
for key in found:
    r = result[key]
    print(key, r["strong_found"][:4], "|", r["target"][:110].replace("\n", " "))
    for example in r["examples"][:1]:
        print("    ", example[0], example[2][:150].replace("\n", " "))
