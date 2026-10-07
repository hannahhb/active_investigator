"""The investigator as a PyINE ``GuardrailScorer``.

Per record: re-trace the program, enumerate candidate questions, let the picker choose ``k``,
ask the responder each one, grade the replies against the trace, and score the record as the
fraction answered correctly. One scorer class, three pickers: the arms differ only in selection.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import statistics
import threading
import typing

import pyine.evals.correctness.types as correctness_types

import active_investigator.investigator.grading as grading
import active_investigator.investigator.pickers as pickers
import active_investigator.investigator.questions as questions
import active_investigator.investigator.responder as responder_mod
import active_investigator.investigator.tracing as tracing

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AskedQuestion:
    """One follow-up and how it went."""

    kind: str
    text: str
    truth: str
    answer: str | None
    correct: bool
    locality: float
    steps_needed: int
    tokens: float
    reply: str


@dataclasses.dataclass(frozen=True)
class Outcome:
    """Per-record result carried back from the worker pool."""

    score: float
    cost: float
    metadata: dict[str, typing.Any]


class InvestigatorScorer:
    """GuardrailScorer that asks trace-grounded follow-ups and grades them mechanically."""

    def __init__(
        self,
        responder: responder_mod.Responder,
        picker: pickers.Picker,
        k: int = 1,
        max_workers: int = 8,
        question_kinds: typing.Iterable[questions.QuestionKind] = ("value_at", "branch_at", "execution_count"),
        default_score_on_error: float = 0.5,
        trace_timeout_seconds: float = 30,
    ) -> None:
        """Wire a responder and a picker together under a fixed per-record question budget."""
        self._responder = responder
        self._picker = picker
        self._k = k
        self._max_workers = max_workers
        self._kinds = tuple(question_kinds)
        self._default = default_score_on_error
        self._trace_timeout = trace_timeout_seconds
        self._system = responder_mod.load_prompt("followup_system.txt")
        self._user = responder_mod.load_prompt("followup_user.txt")
        self._lock = threading.Lock()
        self.counters: dict[str, int] = {
            "scored": 0,
            "trace_failed": 0,
            "no_candidates": 0,
            "picker_empty": 0,
            "ask_failed": 0,
        }

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score every record; one metadata entry per record, including failures."""
        outcomes: list[Outcome | None] = [None] * len(records)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._score_single, record): idx for idx, record in enumerate(records)}
            for future in concurrent.futures.as_completed(futures):
                outcomes[futures[future]] = future.result()
        scores, costs = [], []
        metadata: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] = {}
        for idx, outcome in enumerate(outcomes):
            assert outcome is not None
            scores.append(outcome.score)
            costs.append(outcome.cost)
            metadata[(records[idx].sample_id, records[idx].attempt_index, idx)] = outcome.metadata
        with self._lock:
            self.counters["scored"] += len(records)
        return correctness_types.ScoringResult(scores=scores, verification_costs=costs, attempt_metadata=metadata)

    def _bump(self, key: str) -> None:
        with self._lock:
            self.counters[key] += 1

    def _score_single(
        self,
        record: correctness_types.EvalRecord,
    ) -> Outcome:
        """Trace, pick, ask, grade."""
        if record.final_answer is None:
            return Outcome(0.0, 0.0, {"outcome": "missing_original_answer"})
        try:
            bundle = tracing.trace_record(record.record, timeout_seconds=self._trace_timeout)
        except Exception as exc:  # tracing arbitrary programs can fail; record why rather than crash the run
            self._bump("trace_failed")
            return Outcome(self._default, 0.0, {"outcome": "trace_failed", "error": repr(exc)[:300]})
        candidates = questions.generate_candidates(bundle, kinds=self._kinds)
        if not candidates:
            self._bump("no_candidates")
            return Outcome(self._default, 0.0, {"outcome": "no_candidates"})
        try:
            pick = self._picker.pick(bundle, candidates, record, self._k)
            asked = [self._ask(record, question, pick) for question in pick.questions]
        except Exception as exc:
            self._bump("ask_failed")
            logger.exception("asking failed for %s", record.sample_id)
            return Outcome(self._default, 0.0, {"outcome": "ask_failed", "error": repr(exc)[:300]})
        if not asked:
            self._bump("picker_empty")
            return Outcome(self._default, pick.probe_tokens, {"outcome": "picker_returned_nothing"})
        score = statistics.mean(1.0 if a.correct else 0.0 for a in asked)
        cost = pick.probe_tokens + sum(a.tokens for a in asked if a.tokens is not None)
        return Outcome(
            score,
            cost,
            {
                "outcome": "ok",
                "picker": self._picker.name,
                "n_candidates": len(candidates),
                "asked": [dataclasses.asdict(a) for a in asked],
                "probe_tokens": pick.probe_tokens,
                "picker_notes": pick.notes,
                "trace_steps": len(bundle.steps),
            },
        )

    def _ask(
        self,
        record: correctness_types.EvalRecord,
        question: questions.Question,
        pick: pickers.PickResult,
    ) -> AskedQuestion:
        """Use a prefetched reply when the picker already asked this question; otherwise ask now."""
        prefetched = pick.prefetched.get(question.key())
        if prefetched is not None:
            reply, tokens = prefetched, 0.0  # already counted in probe_tokens
        else:
            reply = self._responder.ask(self._system, pickers.render_followup(self._user, record, question))
            tokens = reply.completion_tokens
        answer = grading.extract_answer(reply.text)
        return AskedQuestion(
            kind=question.kind,
            text=question.text,
            truth=question.truth,
            answer=answer,
            correct=grading.is_correct(answer, question.truth),
            locality=question.locality,
            steps_needed=question.steps_needed,
            tokens=tokens,
            reply=reply.text[:1500],
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        """Provenance for the run."""
        return {
            "scorer_type": "investigator",
            "picker": self._picker.name,
            "k": self._k,
            "question_kinds": list(self._kinds),
            "responder": self._responder.spec.describe(),
            **self.counters,
        }

    def get_verification_cost_unit(self) -> str | None:
        """Costs are completion tokens across responder calls, including any probing."""
        return "tokens"
