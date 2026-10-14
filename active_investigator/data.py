"""Load PyINE eval records and, when needed, regenerate the original predictions.

Two situations:

1. Responder is the PyINE organism: use its recorded predictions from the LMDB exports.
2. Responder is some other model (e.g. hosted on OpenRouter): the recorded predictions were
   made by a different model, so ``regenerate_predictions`` asks the responder to predict
   first, using the organism's own prediction prompt, and relabels the record.
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import logging
import os
import pathlib
import random
import threading
import typing

import dotenv
import pyine.evals.correctness.datamodule_configs as datamodule_configs
import pyine.evals.correctness.types as correctness_types
import pyine.prompts.manager
import pyine.utils.langchain as pyine_langchain

import active_investigator.grading as grading
import active_investigator.responder as responder_mod
import active_investigator.tracing as tracing

logger = logging.getLogger(__name__)
PREDICT_TYPE_BY_INVOCATION = {"callable": "function_return", "script": "program_output"}


def default_lmdb_path() -> str:
    """The organism's export directory under ``PYINE_DATA_ROOT``."""
    dotenv.load_dotenv()
    root = os.environ.get("PYINE_DATA_ROOT")
    if not root:
        raise RuntimeError("PYINE_DATA_ROOT is not set; put it in .env or pass --lmdb explicitly")
    return os.path.join(root, "RL_HT_49", "ckpt-model-org-exports")


def load_records(
    split_source: str,
    lmdb_paths: typing.Sequence[str] | None = None,
    subset: str = "guardrail_test",
    limit: int | None = None,
    seed: int = 0,
    max_problems: int | None = None,
    max_attempts: int | None = None,
) -> list[correctness_types.EvalRecord]:
    """Load one guardrail subset through PyINE's datamodule, optionally restricted to a few problems.

    ``max_problems`` keeps whole problems (every variant and attempt of each), chosen at random
    with ``seed``; ``max_attempts`` then keeps only the first attempts of each sample; ``limit``
    finally caps records, keeping whole samples together.
    """
    config = datamodule_configs.CorrectnessDataModuleConfig(
        lmdb_paths=list(lmdb_paths) if lmdb_paths else [default_lmdb_path()],
        split_config=correctness_types.GuardrailSplitConfig(split_source=split_source),
        eval_subset_names=(subset,),
    )
    datamodule = config.instantiate_datamodule()
    datamodule.prepare_data()
    datamodule.setup()
    records = list(datamodule.get_records_for_subset(subset))
    if max_problems is not None:
        problem_ids = sorted({record.problem_id for record in records})
        random.Random(seed).shuffle(problem_ids)
        keep = set(problem_ids[:max_problems])
        records = [record for record in records if record.problem_id in keep]
        logger.info("kept %d of %d problems (%d records)", len(keep), len(problem_ids), len(records))
    if max_attempts is not None:
        records = [record for record in records if record.attempt_index < max_attempts]
        logger.info("kept the first %d attempts of each sample (%d records)", max_attempts, len(records))
    if limit is None or limit >= len(records):
        return records
    by_sample: dict[str, list[correctness_types.EvalRecord]] = collections.defaultdict(list)
    for record in records:
        by_sample[record.sample_id].append(record)
    sample_ids = sorted(by_sample)
    random.Random(seed).shuffle(sample_ids)
    kept: list[correctness_types.EvalRecord] = []
    for sample_id in sample_ids:  # keep whole samples so attempts of one task stay together
        kept.extend(by_sample[sample_id])
        if len(kept) >= limit:
            break
    return kept


def regenerate_predictions(
    records: list[correctness_types.EvalRecord],
    responder: responder_mod.Responder,
    max_workers: int = 8,
    save_path: pathlib.Path | None = None,
) -> list[correctness_types.EvalRecord]:
    """Ask the responder to predict each task afresh, using the organism's prediction prompt.

    The prompt is ``code_execution`` / ``rl_tagged_answer`` — what the organism was trained with
    — so an external model is asked the task exactly the way the organism was. Labels are
    recomputed against ``expected_output`` with PyINE's soft comparison.

    Predictions run ``max_workers`` at a time. With ``save_path``, each prediction is appended to
    that JSON-lines file as it finishes and predictions already in it are reused, so a resumed run
    keeps the original answers its follow-ups were asked about.
    """
    chain = pyine.prompts.manager.get_prompt_chain(
        model=responder.llm,
        prompt_name="code_execution",
        version="rl_tagged_answer",
        use_chat_template=True,
        include_examples=False,
    )
    saved = _load_predictions(save_path)
    lock = threading.Lock()

    def predict(record: correctness_types.EvalRecord) -> str:
        key = f"{record.sample_id}|{record.attempt_index}"
        if key in saved:
            return saved[key]
        payload = record.record
        variables = {
            "code": payload["code"],
            "predict_type": payload.get("predict_type", "program_output"),
            "entrypoint": tracing.normalize_entrypoint(payload.get("entrypoint")) or "",
            "inputs": payload.get("inputs") or "",
            "description": payload.get("description") or "",
            # declared by the template for the frame-variables task type; unused for program output
            "first_line": payload.get("first_line") or 0,
            "first_line_hit": payload.get("first_line_hit") or 0,
            "last_line": payload.get("last_line") or 0,
            "last_line_hit": payload.get("last_line_hit") or 0,
        }
        handler = pyine_langchain.CaptureLLMHandler()
        model_output = str(chain.invoke(variables, config={"callbacks": [handler]}))
        if save_path is not None:
            with lock, save_path.open("a") as handle:
                handle.write(json.dumps({"key": key, "model_output": model_output}) + "\n")
                handle.flush()
        return model_output

    if saved:
        logger.info("reusing %d saved predictions from %s", len(saved), save_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        outputs = list(pool.map(predict, records))
    out: list[correctness_types.EvalRecord] = []
    for record, model_output in zip(records, outputs, strict=True):
        final_answer = grading.extract_answer(model_output)
        label = grading.is_correct(final_answer, record.expected_output)
        out.append(
            correctness_types.EvalRecord(
                sample_id=record.sample_id,
                problem_id=record.problem_id,
                attempt_index=record.attempt_index,
                model_output=model_output,
                final_answer=final_answer,
                expected_output=record.expected_output,
                label=label,
                code_type=record.code_type,
                tags=list(record.tags),
                difficulty_score=record.difficulty_score,
                record={**record.record, "model_output": model_output, "final_answer": final_answer},
            )
        )
    return out


def _load_predictions(path: pathlib.Path | None) -> dict[str, str]:
    """Predictions saved by an earlier, interrupted run; a line cut short by a kill is skipped."""
    saved: dict[str, str] = {}
    if path is None or not path.exists():
        return saved
    with path.open() as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            saved[row["key"]] = row["model_output"]
    return saved
