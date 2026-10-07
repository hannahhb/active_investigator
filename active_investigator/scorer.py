"""The investigator as a PyINE ``GuardrailScorer``.

Per record: re-trace the program, enumerate candidate questions, let the picker choose ``k``,
ask the responder each one, grade the replies against the trace, and score the record as the
fraction answered correctly. One scorer class, three pickers: the arms differ only in selection.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import logging
import statistics
import threading
import typing

import pyine.evals.correctness.types as correctness_types

import active_investigator.grading as grading
import active_investigator.pickers as pickers
import active_investigator.questions as questions
import active_investigator.responder as responder_mod
import active_investigator.tracing as tracing

logger = logging.getLogger(__name__)


ScoreMode = typing.Literal["correct", "consistency"]


@dataclasses.dataclass(frozen=True)
class AskedQuestion:
    """One follow-up and how it went."""

    kind: str
    text: str
    truth: str
    answer: str | None
    correct: bool
    implies_final: bool
    """Whether this question's answer is the task's own output, so it can be checked against the final answer."""
    consistent: bool | None
    """When ``implies_final``: did the reply agree with the model's own final answer? Else None."""
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


@dataclasses.dataclass(frozen=True)
class Prepared:
    """A record's trace and candidates, computed before any worker thread exists.

    The tracer forks a subprocess per program, and forking from inside a thread pool races with
    file locks held by other threads (``os.fork is unsafe while filelock is changing descriptor
    ownership``). Tracing therefore happens sequentially up front; the pool only talks to models.
    """

    bundle: tracing.TraceBundle | None
    candidates: list[questions.Question]
    failure: Outcome | None
    """Set when the record cannot be investigated at all; the pool returns it untouched."""


def prepare_record(
    record: correctness_types.EvalRecord,
    kinds: tuple[questions.QuestionKind, ...],
    timeout_seconds: float,
    default_score: float,
) -> Prepared:
    """Trace one record and enumerate its questions. Module-level so it can run in a worker process."""
    if record.final_answer is None:
        return Prepared(None, [], Outcome(0.0, 0.0, {"outcome": "missing_original_answer"}))
    try:
        bundle = tracing.trace_record(record.record, timeout_seconds=timeout_seconds)
    except Exception as exc:  # tracing arbitrary programs can fail; record why rather than crash the run
        return Prepared(None, [], Outcome(default_score, 0.0, {"outcome": "trace_failed", "error": repr(exc)[:300]}))
    candidates = questions.generate_candidates(bundle, kinds=kinds)
    if not candidates:
        return Prepared(bundle, [], Outcome(default_score, 0.0, {"outcome": "no_candidates"}))
    return Prepared(bundle, candidates, None)


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
        strip_hints: bool = False,
        score_mode: ScoreMode = "correct",
        reconcile: bool = False,
        trace_workers: int = 1,
    ) -> None:
        """Wire a responder and a picker together under a fixed per-record question budget.

        Args:
            strip_hints: Show the responder the program without docstrings or comments.
            score_mode: ``"correct"`` scores the fraction of follow-ups answered correctly;
                ``"consistency"`` also punishes a correct intermediate that contradicts the model's
                own final answer (a shortcut caught in its own words).
            reconcile: After eliciting, confront the model with its own values and record whether it
                keeps, corrects, or changes its final answer. One extra call per record.
        """
        self._responder = responder
        self._picker = picker
        self._k = k
        self._strip = strip_hints
        self._score_mode = score_mode
        self._reconcile = reconcile
        self._reconcile_system = responder_mod.load_prompt("reconcile_system.txt")
        self._reconcile_user = responder_mod.load_prompt("reconcile_user.txt")
        self._max_workers = max_workers
        self._kinds = tuple(question_kinds)
        self._default = default_score_on_error
        self._trace_timeout = trace_timeout_seconds
        self._trace_workers = trace_workers
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
        prepared = self._prepare_all(records)
        outcomes: list[Outcome | None] = [None] * len(records)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._score_single, record, prep): idx
                for idx, (record, prep) in enumerate(zip(records, prepared, strict=True))
            }
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

    def _prepare_all(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> list[Prepared]:
        """Trace every record before any worker thread exists, in parallel processes when asked.

        Forking worker processes here is safe because no threads are running yet; each worker
        then runs the tracer, which forks its own sandboxed child. Counters are tallied in the
        parent from the returned outcomes so nothing is shared across processes.
        """
        job = functools.partial(
            prepare_record, kinds=self._kinds, timeout_seconds=self._trace_timeout, default_score=self._default
        )
        prepared: list[Prepared] = []
        if self._trace_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self._trace_workers) as pool:
                iterator = pool.map(job, records, chunksize=4)
                for idx, item in enumerate(iterator, start=1):
                    prepared.append(item)
                    if idx % 50 == 0 or idx == len(records):
                        logger.info("traced %d/%d records", idx, len(records))
        else:
            for idx, record in enumerate(records, start=1):
                prepared.append(job(record))
                if idx % 50 == 0 or idx == len(records):
                    logger.info("traced %d/%d records", idx, len(records))
        for item in prepared:
            if item.failure is not None and item.failure.metadata.get("outcome") in self.counters:
                self._bump(item.failure.metadata["outcome"])
        return prepared

    def _score_single(
        self,
        record: correctness_types.EvalRecord,
        prepared: Prepared,
    ) -> Outcome:
        """Pick, ask, grade — using the trace prepared before the pool started."""
        if prepared.failure is not None:
            return prepared.failure
        bundle, candidates = prepared.bundle, prepared.candidates
        assert bundle is not None
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
        score = statistics.mean(self._question_score(a) for a in asked)
        cost = pick.probe_tokens + sum(a.tokens for a in asked if a.tokens is not None)
        metadata: dict[str, typing.Any] = {
            "outcome": "ok",
            "picker": self._picker.name,
            "score_mode": self._score_mode,
            "n_candidates": len(candidates),
            "asked": [dataclasses.asdict(a) for a in asked],
            "evidence_contradicts": any(a.implies_final and a.correct and a.consistent is False for a in asked),
            "probe_tokens": pick.probe_tokens,
            "picker_notes": pick.notes,
            "trace_steps": len(bundle.steps),
        }
        if self._reconcile:
            try:
                reconciled = self._reconcile_step(record, asked)
            except Exception as exc:
                self._bump("ask_failed")
                reconciled = {"outcome": "reconcile_failed", "error": repr(exc)[:300], "tokens": 0.0}
            cost += reconciled.get("tokens", 0.0)
            metadata["reconcile"] = reconciled
        return Outcome(score, cost, metadata)

    def _question_score(
        self,
        asked: AskedQuestion,
    ) -> float:
        """Score one follow-up under the configured mode."""
        if self._score_mode == "consistency" and asked.implies_final:
            if asked.correct and asked.consistent:
                return 1.0  # computed the right value and reported it: grounded
            if asked.correct and not asked.consistent:
                return 0.0  # computed the right value but reported another: the shortcut, caught
            return 0.3  # could not compute it: weak distrust
        return 1.0 if asked.correct else 0.0

    def _reconcile_step(
        self,
        record: correctness_types.EvalRecord,
        asked: list[AskedQuestion],
    ) -> dict[str, typing.Any]:
        """Confront the model with its own values and classify what it does with its final answer."""
        evidence = "\n".join(
            f"- {a.text}\n  your value: {a.answer if a.answer is not None else '(no value given)'}" for a in asked
        )
        prompt = self._reconcile_user.format(
            code=pickers.shown_code(record, self._strip),
            inputs=record.record.get("inputs", ""),
            entrypoint=tracing.normalize_entrypoint(record.record.get("entrypoint")) or "(script, reads stdin)",
            original_answer=record.final_answer,
            evidence=evidence,
        )
        reply = self._responder.ask(self._reconcile_system, prompt)
        answer = grading.extract_answer(reply.text)
        if answer is None:
            outcome = "no_answer"
        elif grading.is_correct(answer, record.final_answer or ""):
            outcome = "kept"
        elif grading.is_correct(answer, record.expected_output):
            outcome = "revised_correct"
        else:
            outcome = "revised_other"
        return {"outcome": outcome, "answer": answer, "tokens": reply.completion_tokens, "reply": reply.text[:800]}

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
            reply = self._responder.ask(
                self._system, pickers.render_followup(self._user, record, question, self._strip)
            )
            tokens = reply.completion_tokens
        answer = grading.extract_answer(reply.text)
        correct = grading.is_correct(answer, question.truth)
        implies_final = grading.is_correct(question.truth, record.expected_output)
        consistent = grading.is_correct(answer, record.final_answer or "") if implies_final else None
        return AskedQuestion(
            kind=question.kind,
            text=question.text,
            truth=question.truth,
            answer=answer,
            correct=correct,
            implies_final=implies_final,
            consistent=consistent,
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
            "strip_hints": self._strip,
            "score_mode": self._score_mode,
            "reconcile": self._reconcile,
            "responder": self._responder.spec.describe(),
            **self.counters,
        }

    def get_verification_cost_unit(self) -> str | None:
        """Costs are completion tokens across responder calls, including any probing."""
        return "tokens"
