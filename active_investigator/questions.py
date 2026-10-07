"""Candidate follow-up questions enumerated from a traced execution.

Every question carries its own ground-truth answer taken from the trace, so grading a reply
is mechanical. Three kinds are produced:

- ``value_at``: the value a variable holds right after a given line executes;
- ``branch_at``: which line runs next after a conditional or loop header, i.e. the path taken;
- ``execution_count``: how many times a line executes in total.

The task's own answer (the final return value) is deliberately not a question kind: asking it
would be asking the model to redo the whole task, which is the thing to avoid.
"""

from __future__ import annotations

import dataclasses
import typing

import active_investigator.slicing as slicing
import active_investigator.tracing as tracing

QuestionKind = typing.Literal["value_at", "branch_at", "execution_count"]
_BRANCH_PREFIXES = ("if ", "elif ", "while ", "for ")


@dataclasses.dataclass(frozen=True)
class Question:
    """A follow-up question with its trace-derived answer and cost estimate."""

    kind: QuestionKind
    text: str
    truth: str
    line: int
    step: int | None
    variable: str | None
    steps_needed: int
    locality: float

    def key(self) -> tuple[str, int, int | None, str | None]:
        """Identity used to de-duplicate and to report which question was asked."""
        return (self.kind, self.line, self.step, self.variable)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _value_questions(
    bundle: tracing.TraceBundle,
) -> list[Question]:
    out: list[Question] = []
    for step in bundle.steps:
        src = bundle.source_line(step.line)
        for variable, value in step.writes.items():
            score, count = slicing.locality(bundle, variable, step.step)
            when = f"the {_ordinal(step.occurrence + 1)} time" if step.occurrence else "the first time"
            out.append(
                Question(
                    kind="value_at",
                    text=(
                        f"Right after line {step.line} (`{src}`) runs for {when}, what is the value of `{variable}`?"
                    ),
                    truth=value,
                    line=step.line,
                    step=step.step,
                    variable=variable,
                    steps_needed=count,
                    locality=score,
                )
            )
    return out


def _branch_questions(
    bundle: tracing.TraceBundle,
) -> list[Question]:
    out: list[Question] = []
    for step in bundle.steps:
        src = bundle.source_line(step.line)
        if not src.startswith(_BRANCH_PREFIXES):
            continue
        when = f"the {_ordinal(step.occurrence + 1)} time" if step.occurrence else "the first time"
        # the path taken depends on whatever the header reads; reuse the slice of its first read
        count = 1 + len([s for s in bundle.steps if s.step < step.step and s.writes])
        out.append(
            Question(
                kind="branch_at",
                text=(
                    f"When line {step.line} (`{src}`) is evaluated for {when}, "
                    "which line number executes immediately after it?"
                ),
                truth=str(step.next_line),
                line=step.line,
                step=step.step,
                variable=None,
                steps_needed=count,
                locality=1.0 / count,
            )
        )
    return out


def _count_questions(
    bundle: tracing.TraceBundle,
) -> list[Question]:
    totals: dict[int, int] = {}
    for step in bundle.steps:
        totals[step.line] = totals.get(step.line, 0) + 1
    return [
        Question(
            kind="execution_count",
            text=f"How many times does line {line} (`{bundle.source_line(line)}`) execute in total?",
            truth=str(count),
            line=line,
            step=None,
            variable=None,
            steps_needed=count,
            locality=1.0 / count,
        )
        for line, count in sorted(totals.items())
    ]


def generate_candidates(
    bundle: tracing.TraceBundle,
    kinds: typing.Iterable[QuestionKind] = ("value_at", "branch_at", "execution_count"),
) -> list[Question]:
    """Enumerate every candidate question of the requested kinds for a traced execution."""
    wanted = set(kinds)
    out: list[Question] = []
    if "value_at" in wanted:
        out.extend(_value_questions(bundle))
    if "branch_at" in wanted:
        out.extend(_branch_questions(bundle))
    if "execution_count" in wanted:
        out.extend(_count_questions(bundle))
    return out
