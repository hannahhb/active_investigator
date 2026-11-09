"""OpenRouter calls for the curation steps, with a shared spending ledger. The key is never printed."""

import json
import re
import threading
import urllib.error
import urllib.request

from common import DATA, OUT

URL = "https://openrouter.ai/api/v1/chat/completions"
LEDGER = OUT / "spend.jsonl"
TOTAL_CAP = 48.00  # dollars: user OK to pass the ~$30 working limit for the full follow-up run (2026-11-09); hard limit $50
_lock = threading.Lock()


class OverBudget(Exception):
    pass


def load_key() -> str:
    values = {}
    for line in open(DATA / ".env"):
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            values[name.replace("export ", "").strip()] = value.strip().strip("\"'")
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        if values.get(name):
            return values[name]
    if len(values) == 1:
        return next(iter(values.values()))
    raise SystemExit("No OpenRouter key name recognised in .env")


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["cost"] for line in open(LEDGER) if line.strip())


class Client:
    def __init__(self, model: str, step: str, step_cap: float, max_tokens: int = 300, options: dict | None = None):
        self.model, self.step, self.step_cap, self.max_tokens = model, step, step_cap, max_tokens
        self.options = options or {"temperature": 0, "reasoning": {"enabled": False}}
        self.key = load_key()
        self.total = spent()
        self.step_spent = 0.0

    def ask(self, system: str, user: str) -> str:
        return self.ask_messages([{"role": "system", "content": system}, {"role": "user", "content": user}])

    def ask_messages(self, messages: list) -> str:
        """Same as ask, for a whole conversation (used for follow-up questions)."""
        with _lock:
            if self.total >= TOTAL_CAP or self.step_spent >= self.step_cap:
                raise OverBudget(f"stopped at ${self.total:.3f} total, ${self.step_spent:.3f} this step")
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
            **self.options,
        }
        request = urllib.request.Request(
            URL,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            reply = json.load(response)
        cost = float((reply.get("usage") or {}).get("cost") or 0.0)
        with _lock:
            self.total += cost
            self.step_spent += cost
            with open(LEDGER, "a") as ledger:
                ledger.write(json.dumps({"step": self.step, "model": self.model, "cost": cost}) + "\n")
        return reply["choices"][0]["message"].get("content") or ""


def parse_json(text: str):
    """Return the first JSON object in a reply, or None."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
