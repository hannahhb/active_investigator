"""Question-selection policies: the only thing that differs between the experimental arms.

- ``FixedLastWritePicker`` (step 3a): a constant rule, no model, no learning.
- ``PromptedPicker`` (step 3b): an LLM chooses from the enumerated candidates.
- ``RankedPicker`` (step 5): probe the responder on candidates; questions it already answers
  correctly are useless and go last; the rest are ordered by locality (small slices first).
"""

from __future__ import annotations

import dataclasses
import json
import re
import typing

import pyine.evals.correctness.types as correctness_types

import active_investigator.investigator.grading as grading
import active_investigator.investigator.questions as questions
import active_investigator.investigator.responder as responder_mod
import active_investigator.investigator.tracing as tracing


@dataclasses.dataclass
class PickResult:
    """Chosen questions plus any replies already obtained while choosing them."""

    questions: list[questions.Question]
    prefetched: dict[tuple[str, int, int | None, str | None], responder_mod.Reply] = dataclasses.field(
        default_factory=dict
    )
    probe_tokens: float = 0.0
    """Completion tokens spent during selection itself (zero for pickers that make no calls)."""
    notes: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


class Picker(typing.Protocol):
    """Selects which candidate questions to ask for one record."""

    name: str

    def pick(
        self,
        bundle: tracing.TraceBundle,
        candidates: list[questions.Question],
        record: correctness_types.EvalRecord,
        k: int,
    ) -> PickResult: ...


class FixedLastWritePicker:
    """Always ask about the variable written at the last step before the return."""

    name = "fixed_last_write"

    def pick(
        self,
        bundle: tracing.TraceBundle,
        candidates: list[questions.Question],
        record: correctness_types.EvalRecord,
        k: int,
    ) -> PickResult:
        """Take the latest ``value_at`` questions before the return, most recent first."""
        values = [q for q in candidates if q.kind == "value_at" and q.step is not None]
        limit = bundle.return_step if bundle.return_step is not None else float("inf")
        before = sorted((q for q in values if q.step <= limit), key=lambda q: q.step, reverse=True)
        return PickResult(questions=before[:k])


class PromptedPicker:
    """An LLM reads the program and the model's answer, then picks from the candidate list."""

    name = "prompted"

    def __init__(
        self,
        picker_llm: responder_mod.Responder,
        max_candidates_shown: int = 40,
    ) -> None:
        """Keep the picker model and how many candidates to show it."""
        self._llm = picker_llm
        self._max_shown = max_candidates_shown
        self._system = responder_mod.load_prompt("pick_system.txt")
        self._user = responder_mod.load_prompt("pick_user.txt")

    def pick(
        self,
        bundle: tracing.TraceBundle,
        candidates: list[questions.Question],
        record: correctness_types.EvalRecord,
        k: int,
    ) -> PickResult:
        """Show numbered candidates (never their answers) and parse the chosen indices."""
        shown = candidates[: self._max_shown]
        listing = "\n".join(f"{idx + 1}. {q.text}" for idx, q in enumerate(shown))
        reply = self._llm.ask(
            self._system,
            self._user.format(
                code=numbered_source(record.record.get("code", "")),
                inputs=record.record.get("inputs", ""),
                original_answer=record.final_answer,
                original_output=record.model_output[:4000],
                candidates=listing,
                k=k,
            ),
        )
        chosen = _parse_indices(reply.text, upper=len(shown))
        if not chosen:
            chosen = list(range(1, min(k, len(shown)) + 1))
        picked = [shown[idx - 1] for idx in chosen[:k]]
        return PickResult(questions=picked, probe_tokens=reply.completion_tokens, notes={"raw_pick": reply.text[:300]})


class RankedPicker:
    """Probe the responder, discard what it already gets right, rank the rest by locality."""

    name = "ranked"

    def __init__(
        self,
        responder: responder_mod.Responder,
        max_probe: int = 12,
    ) -> None:
        """Keep the responder to probe with and a cap on how many candidates to probe."""
        self._responder = responder
        self._max_probe = max_probe
        self._system = responder_mod.load_prompt("followup_system.txt")
        self._user = responder_mod.load_prompt("followup_user.txt")

    def pick(
        self,
        bundle: tracing.TraceBundle,
        candidates: list[questions.Question],
        record: correctness_types.EvalRecord,
        k: int,
    ) -> PickResult:
        """Probe the cheapest-to-verify candidates first, then apply the ranking scheme."""
        # probe the most local candidates: spending the probe budget on cheap questions first
        probe_set = sorted(candidates, key=lambda q: (-q.locality, q.step or 0))[: self._max_probe]
        prefetched: dict[tuple[str, int, int | None, str | None], responder_mod.Reply] = {}
        wrong: list[questions.Question] = []
        right: list[questions.Question] = []
        tokens = 0.0
        for question in probe_set:
            reply = self._responder.ask(self._system, render_followup(self._user, record, question))
            prefetched[question.key()] = reply
            tokens += reply.completion_tokens
            answer = grading.extract_answer(reply.text)
            (right if grading.is_correct(answer, question.truth) else wrong).append(question)
        wrong.sort(key=lambda q: -q.locality)  # most local first
        right.sort(key=lambda q: -q.locality)
        ordered = wrong + right  # correctly answered questions are ranked last by construction
        return PickResult(
            questions=ordered[:k],
            prefetched=prefetched,
            probe_tokens=tokens,
            notes={"probed": len(probe_set), "answered_wrong": len(wrong), "ranked": [q.key() for q in ordered]},
        )


def numbered_source(code: str) -> str:
    """Prefix each source line with its 1-based number, since questions refer to lines by number."""
    return "\n".join(f"{idx:4d}| {line}" for idx, line in enumerate(code.splitlines(), start=1))


def render_followup(
    template: str,
    record: correctness_types.EvalRecord,
    question: questions.Question,
) -> str:
    """Fill the follow-up user prompt for a record and question."""
    return template.format(
        code=numbered_source(record.record.get("code", "")),
        inputs=record.record.get("inputs", ""),
        entrypoint=tracing.normalize_entrypoint(record.record.get("entrypoint")) or "(script, reads stdin)",
        original_answer=record.final_answer,
        question=question.text,
    )


def _parse_indices(
    text: str,
    upper: int,
) -> list[int]:
    """Read a JSON list of 1-based indices, falling back to any integers in the text."""
    try:
        parsed = json.loads(text.strip())
        values = parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, TypeError):
        values = [int(tok) for tok in re.findall(r"\d+", text)]
    seen: list[int] = []
    for value in values:
        if isinstance(value, int) and 1 <= value <= upper and value not in seen:
            seen.append(value)
    return seen
