"""Free search: which identifiers in raw claim sentences never appear in any log before the claim?

Uses the agents' own words only. Needs the decompressed turns piped in on stdin is not required:
it streams computer_use_turns itself and reads Claude Code messages and chat too.
Writes out/identifier_suspects.jsonl. A suspect is something to read, not a finding.
"""

import collections
import gzip
import json
import re

from common import DATA, OUT, rows

HEX = re.compile(r"\b[0-9a-f]{7,64}\b")
URL = re.compile(r"https?://[^\s)\]>\"'`*\\]+")
SKIP_URL = re.compile(r"^https?://[^/]+/?$|example\.com|localhost|127\.0\.0\.1")


def tokens(text: str) -> set[str]:
    hexes = {h for h in HEX.findall(text) if re.search(r"[a-f]", h) and re.search(r"\d", h)}
    links = {u.rstrip(".,;:!?") for u in URL.findall(text)}
    return hexes | {u for u in links if not SKIP_URL.search(u)}


def main() -> None:
    claims = [json.loads(line) for line in open(OUT / "candidates.jsonl")]
    claims = [c for c in claims if not c["repeat_of"]]
    wanted = collections.defaultdict(list)
    for claim in claims:
        for token in tokens(" ".join(claim["claims"])):
            wanted[token].append(claim)
    print("claims with identifiers", len({c["message_id"] for v in wanted.values() for c in v}), "identifiers", len(wanted))

    seen = collections.defaultdict(list)  # token -> (time, kind, agent)

    def scan(text: str, at: str, kind: str, agent) -> None:
        for token in tokens(text):
            if token in wanted:
                seen[token].append((at, kind, agent))

    sessions = {s["id"]: s["agent_id"] for s in rows("computer_use_sessions")}
    with gzip.open(DATA / "computer_use_turns.jsonl.gz", "rt") as handle:
        for line in handle:
            if not (HEX.search(line) or "http" in line):
                continue
            turn = json.loads(line)
            agent = sessions.get(turn["session_id"])
            action = json.dumps(turn.get("agent_action") or "")
            if "send_message_back_to_chat" not in action:
                scan(action.replace("\\/", "/"), turn["created_at"], "typed", agent)
            scan(json.dumps(turn.get("output") or "").replace("\\/", "/"), turn["created_at"], "output", agent)
    for row in rows("claude_code_messages"):
        scan(json.dumps(row.get("content") or ""), row["created_at"], "output", row["agent_id"])
    for row in rows("chat_messages"):
        if isinstance(row.get("content"), str):
            scan(row["content"], row["created_at"], "chat", row.get("agent_speaker_id"))

    tally, suspects = collections.Counter(), []
    for token, owners in wanted.items():
        for claim in owners:
            before = [s for s in seen[token] if s[0] < claim["time"]]
            after = [s for s in seen[token] if s[0] > claim["time"] and s[1] != "chat"]
            kinds = {s[1] for s in before}
            chat_others = [s for s in before if s[1] == "chat" and s[2] != claim["agent_id"]]
            if "output" in kinds:
                state = "in_output"
            elif "typed" in kinds:
                state = "typed_only"
            elif chat_others:
                state = "chat_from_peer_only"
            else:
                state = "unseen"
            tally[state] += 1
            if state == "unseen":
                suspects.append(
                    {
                        "message_id": claim["message_id"],
                        "time": claim["time"],
                        "agent": claim["agent"],
                        "agent_id": claim["agent_id"],
                        "band": claim["band"],
                        "kinds": claim["kinds"],
                        "text": claim["text"],
                        "claims": claim["claims"],
                        "token": token,
                        "token_type": "hex" if HEX.fullmatch(token) else "url",
                        "appears_later": len(after),
                        "later_output": sum(s[1] == "output" for s in after),
                    }
                )
    with open(OUT / "identifier_suspects.jsonl", "w") as handle:
        handle.writelines(json.dumps(s) + "\n" for s in suspects)
    print(dict(tally))
    print("suspect claims", len({s["message_id"] for s in suspects}), "of which never seen later", sum(s["appears_later"] == 0 for s in suspects))
    print("by type", collections.Counter(s["token_type"] for s in suspects), "by band", collections.Counter(s["band"] for s in suspects))


if __name__ == "__main__":
    main()
