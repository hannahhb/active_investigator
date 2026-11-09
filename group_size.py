"""Agents active per room per day (distinct agent speakers), cached to out/group_size.json. Free."""
import collections, json
from common import OUT, rows

def main():
    seen = collections.defaultdict(set)
    for m in rows("chat_messages"):
        if m.get("agent_speaker_id"):
            seen[f'{m["room_id"]}|{m["created_at"][:10]}'].add(m["agent_speaker_id"])
    json.dump({k: len(v) for k, v in seen.items()}, open(OUT / "group_size.json", "w"))
    print(len(seen), "room-days")

if __name__ == "__main__":
    main()
