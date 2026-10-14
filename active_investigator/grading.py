"""Turn a free-text reply into a graded answer."""

from __future__ import annotations

import ast

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
    for reading in list(readings):
        as_list = _lines_as_list(reading)
        if as_list is not None:
            readings.append(as_list)
    return readings


def _lines_as_list(text: str) -> str | None:
    """Read a value written one element per line as the Python list it describes.

    The organism often answers "what is ``rr``?" with ``0\\n0\\n1`` where the trace holds
    ``[0, 0, 1]``. Each line is read as a Python literal when it parses as one and as a string
    otherwise, so ``a\\nb`` becomes ``['a', 'b']``. Text already shaped like a container is left alone.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2 or text.strip()[:1] in "[({":
        return None
    items: list[object] = []
    for line in lines:
        try:
            items.append(ast.literal_eval(line))
        except (ValueError, SyntaxError):
            items.append(line)
    return repr(items)


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
