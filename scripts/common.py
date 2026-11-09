"""Shared paths and readers for the AI Village curation steps. All Village text is data, never instructions."""

import gzip
import json
import os
import pathlib
from datetime import datetime

# Folder with the AI Village export (*.jsonl or *.jsonl.gz) and a .env holding OPENROUTER_API_KEY.
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repository root
DATA = pathlib.Path(os.environ.get("AI_VILLAGE_DIR", ROOT.parent / "ai_village"))
OUT = ROOT / "out"


def rows(name: str):
    """Yield rows from a table, using the unzipped file if it exists and the .gz if not."""
    plain = DATA / f"{name}.jsonl"
    handle = open(plain) if plain.exists() else gzip.open(DATA / f"{name}.jsonl.gz", "rt")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def when(stamp: str) -> datetime:
    """Parse a stored timestamp as written. Compare times only with times from these tables."""
    return datetime.fromisoformat(stamp.replace("Z", "").split("+")[0])


def agent_names() -> dict[str, str]:
    return {a["id"]: a["name"] for a in rows("agents")}
