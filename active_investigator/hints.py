"""Remove docstrings and comments from a program without moving any line.

Follow-up questions refer to lines by number, so the stripped text must keep the original line
structure: docstring lines become blank (or ``pass`` when the docstring was the whole body), and
comments are cut from the end of their line. Execution is unchanged, so every question and its
answer still hold; what changes is that the responder no longer sees the hint while answering.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize

_DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def strip_hints(code: str) -> str:
    """Return ``code`` with docstrings blanked and comments removed, line count preserved."""
    lines = code.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if not (
            isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        ):
            continue
        end = first.end_lineno or first.lineno
        for line_no in range(first.lineno, end + 1):
            indent = re.match(r"[ \t]*", lines[line_no - 1]).group(0)
            only_statement = len(node.body) == 1 and line_no == first.lineno
            lines[line_no - 1] = f"{indent}pass\n" if only_statement else "\n"
    text = "".join(lines)
    out = text.splitlines(keepends=True)
    for tok in reversed(list(tokenize.generate_tokens(io.StringIO(text).readline))):
        if tok.type != tokenize.COMMENT:
            continue
        (row, col_start), (_, col_end) = tok.start, tok.end
        line = out[row - 1]
        kept = line[:col_start].rstrip()
        out[row - 1] = (kept + line[col_end:]) if kept else "\n"
    return "".join(out)
