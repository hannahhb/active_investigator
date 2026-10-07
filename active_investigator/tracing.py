"""Re-execute a record's program and expose its trace as a flat list of write steps.

Every PyINE eval record carries the program, its input and the entrypoint, so a fresh trace can
be produced on demand instead of joining back to the stored trace datasets. Traces are
deterministic, so this is equivalent and far simpler.
"""

from __future__ import annotations

import collections
import dataclasses
import typing

import pyine.data.deltas.dataset_utils as delta_utils
import pyine.utils.code.execution as execution


@dataclasses.dataclass(frozen=True)
class Step:
    """One executed line and the variables it wrote."""

    step: int
    """Trace step index (unique, increasing)."""
    line: int
    """Source line that executed at this step (1-indexed)."""
    next_line: int
    """Source line that executed next; reveals which branch was taken."""
    writes: dict[str, str]
    """Variables assigned by this line, as value representations."""
    occurrence: int
    """How many times this line had already executed before this step (0 for the first time)."""


@dataclasses.dataclass(frozen=True)
class TraceBundle:
    """A traced execution plus the derived step list used for question generation."""

    code: str
    steps: tuple[Step, ...]
    return_value: str | None
    exception: str | None
    entrypoint_step: int | None
    return_step: int | None
    """Trace step index of the final return of the traced code, when there is one."""

    @property
    def source_lines(self) -> list[str]:
        """Source split into 1-indexed-friendly lines (index 0 is unused)."""
        return ["", *self.code.splitlines()]

    def source_line(self, line: int) -> str:
        """The stripped source text at a given line number."""
        lines = self.source_lines
        return lines[line].strip() if 0 < line < len(lines) else ""

    def last_write_before_return(self) -> Step | None:
        """The latest step that wrote at least one variable before the final return."""
        limit = self.return_step if self.return_step is not None else float("inf")
        candidates = [s for s in self.steps if s.writes and s.step <= limit]
        return max(candidates, key=lambda s: s.step) if candidates else None


def trace_code(
    code: str,
    inputs: typing.Any,
    entrypoint: str | None,
    identifier: str = "investigator",
    timeout_seconds: float = 30,
) -> TraceBundle:
    """Execute ``code`` under the PyINE tracer and flatten its deltas into steps.

    Args:
        code: Program source.
        inputs: Inputs exactly as stored on the record (a JSON string or stdin lines).
        entrypoint: Entrypoint name such as ``"Solution.longest"``, or None for scripts.
        identifier: Label used by the tracer for logging.
        timeout_seconds: Execution timeout.

    Returns:
        The traced execution as a ``TraceBundle``.
    """
    trace = execution.execute_and_trace_code(
        code_string=code,
        inputs=inputs,
        entrypoint_name=entrypoint or None,
        identifier=identifier,
        timeout_seconds=timeout_seconds,
    )
    with_deltas = delta_utils.get_deltas_from_trace_steps(
        trace,
        delta_utils.DeltaGeneratorType.SIMPLE,
        include_global_vars=False,
    )
    seen_per_line: collections.Counter[int] = collections.Counter()
    steps: list[Step] = []
    for delta in with_deltas.deltas:
        if delta.curr_trace_key.file != execution.EXEC_TRACE_FILE_NAME:
            continue  # the synthetic parent frame that invokes the entrypoint
        line = delta.curr_trace_key.line
        writes = {name: value for name, value in delta.variables.items() if _is_data_write(name, value)}
        steps.append(
            Step(
                step=delta.trace_step_idx,
                line=line,
                next_line=delta.next_trace_key.line,
                writes=writes,
                occurrence=seen_per_line[line],
            )
        )
        seen_per_line[line] += 1
    events = [event for event in trace.traced_steps if event is not None]
    returns = [event.trace_step_idx for event in events if event.event_type == execution.TraceEventType.RETURN]
    crashed = trace.exception is not None
    return TraceBundle(
        code=code,
        steps=tuple(steps),
        return_value=None if trace.return_value is None else repr(trace.return_value),
        exception=None if trace.exception is None else repr(trace.exception),
        entrypoint_step=trace.entrypoint_step_idx,
        # a crashed run has no meaningful final return; inner returns before the crash must not cap questions
        return_step=max(returns) if returns and not crashed else None,
    )


_NON_DATA_PREFIXES = ("<module", "<function", "<class", "<built-in", "<bound method", "<_io.")


def _is_data_write(
    name: str,
    value: str,
) -> bool:
    """Keep assignments of data; drop dunder bookkeeping and imported modules, functions and classes."""
    return not name.startswith("__") and not value.startswith(_NON_DATA_PREFIXES)


def normalize_entrypoint(value: typing.Any) -> str | None:
    """Records store a missing entrypoint as the string ``"None"``; treat that like no entrypoint."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text in {"", "None", "null"} else text


def trace_record(
    record: typing.Mapping[str, typing.Any],
    timeout_seconds: float = 30,
) -> TraceBundle:
    """Trace the program stored on a PyINE eval record (its ``record`` payload)."""
    return trace_code(
        code=record["code"],
        inputs=record["inputs"],
        entrypoint=normalize_entrypoint(record.get("entrypoint")),
        identifier=str(record.get("sample_id", "record")),
        timeout_seconds=timeout_seconds,
    )
