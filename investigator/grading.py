"""Turn a free-text reply into a graded answer."""

from __future__ import annotations

import pyine.utils.code.output_compare as output_compare
import pyine.utils.parsing as parsing

ANSWER_TAG = "final"


def extract_answer(
    reply: str,
    tag: str = ANSWER_TAG,
) -> str | None:
    """Pull the tagged answer out of a reply; the last tag wins when several are present."""
    result = parsing.extract_tag_blocks(reply, tag)
    selection = parsing.select_tag_block(result, policy="last")
    return None if selection is None else selection[0].strip()


def answer_readings(answer: str) -> list[str]:
    """Plausible readings of a model's answer text, most literal first.

    Hosted models often write a multi-line value as an escaped string literal — the two characters
    backslash and ``n`` instead of a newline — or wrap it in quotes. The truth from the trace is never
    escaped, so those readings are tried in addition to the literal text, never instead of it.
    """
    literal = answer.strip()
    readings = [literal]
    if len(literal) >= 2 and literal[0] == literal[-1] and literal[0] in "'\"":
        readings.append(literal[1:-1])
    for reading in list(readings):
        if "\\n" in reading or "\\t" in reading:
            readings.append(reading.replace("\\n", "\n").replace("\\t", "\t"))
    return readings


def is_correct(
    answer: str | None,
    truth: str,
) -> bool:
    """Compare an extracted answer against the trace's value with PyINE's soft comparison.

    Any plausible reading of the answer may match; the truth itself is never rewritten.
    """
    if answer is None:
        return False
    expected = truth.strip()
    return any(bool(output_compare.compare(reading.strip(), expected)) for reading in answer_readings(answer))
