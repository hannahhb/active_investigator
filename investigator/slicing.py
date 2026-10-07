"""Backward slicing over a traced execution.

Answers: how many earlier steps actually determine the value of ``variable`` at ``step``?
Data dependencies come from the trace (which step wrote what) plus the source (which names a
line reads); control dependencies add the conditional or loop header that let a step run.
The slice size is the question's *locality*: a small slice is cheap to verify and hard to get
wrong for any reason other than never having traced the code.
"""

from __future__ import annotations

import ast
import collections
import functools

import active_investigator.investigator.tracing as tracing

_COMPOUND = (ast.If, ast.While, ast.For)


class SourceIndex:
    """Static facts about a program: names read per line, and the headers guarding each line."""

    def __init__(
        self,
        code: str,
    ) -> None:
        """Parse the program once and index reads and guards by line."""
        self.reads: dict[int, set[str]] = collections.defaultdict(set)
        self.guards: dict[int, list[int]] = collections.defaultdict(list)
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self.reads[node.lineno].add(node.id)
            elif isinstance(node, ast.AugAssign):  # `x += y` reads x as well as writing it
                for name in ast.walk(node.target):
                    if isinstance(name, ast.Name):
                        self.reads[node.lineno].add(name.id)
            elif isinstance(node, _COMPOUND):
                body_nodes = [*node.body, *getattr(node, "orelse", [])]
                for child in body_nodes:
                    for line in range(child.lineno, (child.end_lineno or child.lineno) + 1):
                        self.guards[line].append(node.lineno)


@functools.lru_cache(maxsize=256)
def _index(code: str) -> SourceIndex:
    return SourceIndex(code)


def steps_needed(
    bundle: tracing.TraceBundle,
    variable: str,
    step: int,
) -> list[int]:
    """Return the trace steps whose execution determines ``variable``'s value as of ``step``.

    The step itself is included when it wrote the variable. Control dependencies are counted
    once per guarded step: the most recent prior execution of each enclosing header.
    """
    index = _index(bundle.code)
    known = {name for s in bundle.steps for name in s.writes}
    by_step = {s.step: s for s in bundle.steps}
    needed = {variable}
    guard_requests: collections.Counter[int] = collections.Counter()
    chosen: set[int] = set()
    for current in sorted((s for s in by_step if s <= step), reverse=True):
        this = by_step[current]
        written = set(this.writes)
        wanted_write = written & needed
        wanted_guard = guard_requests[this.line] > 0
        if not wanted_write and not wanted_guard:
            continue
        chosen.add(current)
        if wanted_guard:
            guard_requests[this.line] -= 1
        needed -= wanted_write
        needed |= index.reads[this.line] & known
        for header in index.guards[this.line]:
            guard_requests[header] += 1
        if not needed and not +guard_requests:
            break
    return sorted(chosen)


def locality(
    bundle: tracing.TraceBundle,
    variable: str,
    step: int,
) -> tuple[float, int]:
    """Locality score ``1 / steps_needed`` and the raw step count (at least 1)."""
    count = max(1, len(steps_needed(bundle, variable, step)))
    return 1.0 / count, count
