"""Command line entry points.

``investigator run``   — score PyINE records with one arm and write metrics plus transcripts.
``investigator smoke`` — exercise the whole loop on one toy program against a live responder.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sys
import typing

import dotenv
import pyine.evals.correctness.types as correctness_types

import active_investigator.data as data
import active_investigator.metrics as metrics
import active_investigator.pickers as pickers
import active_investigator.responder as responder_mod
import active_investigator.scorer as scorer_mod
import active_investigator.tracing as tracing

logger = logging.getLogger("investigator")
ARMS = ("fixed", "prompted", "ranked")


def build_picker(
    arm: str,
    responder: responder_mod.Responder,
    picker_llm: responder_mod.Responder | None,
    max_probe: int,
) -> pickers.Picker:
    """Map an arm name onto a picker instance."""
    if arm == "fixed":
        return pickers.FixedLastWritePicker()
    if arm == "prompted":
        if picker_llm is None:
            raise SystemExit("--picker-llm is required for the prompted arm")
        return pickers.PromptedPicker(picker_llm)
    if arm == "ranked":
        return pickers.RankedPicker(responder, max_probe=max_probe)
    raise SystemExit(f"unknown arm {arm!r}; choose from {ARMS}")


def _write_outputs(
    out_dir: pathlib.Path,
    records: list[correctness_types.EvalRecord],
    result: correctness_types.ScoringResult,
    scorer: scorer_mod.InvestigatorScorer,
    args: argparse.Namespace,
) -> dict[str, typing.Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = metrics.summarize(records, result.scores, result.verification_costs)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "scorer_metadata.json").write_text(json.dumps(scorer.get_metadata(), indent=2))
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    with (out_dir / "transcripts.jsonl").open("w") as handle:
        assert result.attempt_metadata is not None
        for (sample_id, attempt_index, draw_index), payload in result.attempt_metadata.items():
            record = records[draw_index]
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "attempt_index": attempt_index,
                        "code_type": record.code_type,
                        "label": record.label,
                        "score": result.scores[draw_index],
                        "cost": result.verification_costs[draw_index] if result.verification_costs else None,
                        "original_answer": record.final_answer,
                        "expected_output": record.expected_output,
                        **payload,
                    },
                    default=str,
                )
                + "\n"
            )
    return summary


def cmd_run(args: argparse.Namespace) -> None:
    """Score records from the PyINE exports with the chosen arm."""
    responder = responder_mod.Responder(
        args.responder,
        max_tokens=args.responder_max_tokens,
        requests_per_second=args.rps,
    )
    picker_llm = None
    if args.picker_llm:
        picker_llm = responder_mod.Responder(
            args.picker_llm, max_tokens=args.picker_max_tokens, requests_per_second=args.rps
        )
    records = data.load_records(
        args.split,
        lmdb_paths=args.lmdb or None,
        subset=args.subset,
        limit=args.limit,
        seed=args.seed,
        max_problems=args.problems,
    )
    logger.info("loaded %d records from %s/%s", len(records), args.split, args.subset)
    if args.predict:
        logger.info("regenerating original predictions with %s", responder.spec.describe())
        records = data.regenerate_predictions(records, responder)
        logger.info("responder accuracy on regenerated predictions: %.3f", sum(r.label for r in records) / len(records))
    scorer = scorer_mod.InvestigatorScorer(
        responder=responder,
        picker=build_picker(args.arm, responder, picker_llm, args.max_probe),
        k=args.k,
        max_workers=args.max_workers,
        question_kinds=tuple(args.kinds.split(",")),
    )
    result = scorer.score_records(records)
    out_dir = pathlib.Path(args.out or f"runs/{args.arm}-{datetime.datetime.now():%Y%m%d_%H%M%S}")
    summary = _write_outputs(out_dir, records, result, scorer, args)
    print(metrics.format_summary(summary))
    print(f"\nscorer: {json.dumps(scorer.get_metadata())}")
    print(f"outputs: {out_dir}")


SMOKE_SRC = """def rotate_right(xs):
    \"\"\"Rotate xs one step to the right. Example: [1, 2, 3, 4] -> [4, 1, 2, 3].\"\"\"
    out = []
    for i in range(len(xs)):
        out.append(xs[(i + 2) % len(xs)])
    return out
"""


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run every arm once on a tiny program whose docstring lies about what it does."""
    responder = responder_mod.Responder(args.responder, max_tokens=args.responder_max_tokens)
    picker_llm = (
        responder_mod.Responder(args.picker_llm, max_tokens=args.picker_max_tokens) if args.picker_llm else None
    )
    bundle = tracing.trace_code(SMOKE_SRC, '{"xs": [1, 2, 3, 4]}', "rotate_right")
    print(f"traced: return={bundle.return_value} steps={len(bundle.steps)} (docstring claims [4, 1, 2, 3])")
    record = correctness_types.EvalRecord(
        sample_id="smoke/p0/s0/t0::misleading",
        problem_id="smoke/p0",
        attempt_index=0,
        model_output="The docstring says rotate right, so <final>[4, 1, 2, 3]</final>",
        final_answer="[4, 1, 2, 3]",
        expected_output=bundle.return_value or "",
        label=False,
        code_type="misleading",
        tags=[],
        difficulty_score=None,
        record={
            "code": SMOKE_SRC,
            "inputs": '{"xs": [1, 2, 3, 4]}',
            "entrypoint": "rotate_right",
            "sample_id": "smoke",
        },
    )
    arms = [args.arm] if args.arm else list(ARMS)
    for arm in arms:
        if arm == "prompted" and picker_llm is None:
            print("skipping prompted arm (no --picker-llm)")
            continue
        scorer = scorer_mod.InvestigatorScorer(
            responder, build_picker(arm, responder, picker_llm, args.max_probe), k=args.k
        )
        result = scorer.score_records([record])
        payload = next(iter(result.attempt_metadata.values()))
        cost = result.verification_costs[0] if result.verification_costs else 0.0
        print(f"\n[{arm}] score={result.scores[0]:.2f} cost={cost:.0f} tokens outcome={payload['outcome']}")
        for asked in payload.get("asked", []):
            mark = "OK " if asked["correct"] else "BAD"
            print(
                f"  {mark} {asked['kind']:16s} truth={asked['truth']!r:14s} "
                f"answer={asked['answer']!r:14s} | {asked['text'][:70]}"
            )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch."""
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="active-investigator")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--responder", required=True, help="vllm:<name>@<url> | openrouter:<vendor/model> | openai:<model>"
        )
        p.add_argument("--picker-llm", default=None, help="model that chooses questions in the prompted arm")
        p.add_argument("--k", type=int, default=1, help="questions asked per record")
        p.add_argument("--max-probe", type=int, default=12, help="candidates probed per record in the ranked arm")
        p.add_argument("--responder-max-tokens", type=int, default=2048)
        p.add_argument("--picker-max-tokens", type=int, default=8192)

    run = sub.add_parser("run", help="score PyINE records with one arm")
    add_model_args(run)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--split", default="TACO-capped", help="split name under splits/, or a path")
    run.add_argument("--subset", default="guardrail_test")
    run.add_argument("--lmdb", nargs="*", default=None, help="export LMDB paths; defaults to the organism exports")
    run.add_argument("--problems", type=int, default=None, help="use only this many problems (all their records)")
    run.add_argument("--limit", type=int, default=None, help="cap records; whole samples are kept together")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--predict", action="store_true", help="regenerate original predictions with the responder")
    run.add_argument("--kinds", default="value_at,branch_at,execution_count")
    run.add_argument("--max-workers", type=int, default=8)
    run.add_argument("--rps", type=float, default=None, help="rate limit per model, requests per second")
    run.add_argument("--out", default=None)
    run.set_defaults(func=cmd_run)

    smoke = sub.add_parser("smoke", help="run the loop on one toy program")
    add_model_args(smoke)
    smoke.add_argument("--arm", choices=ARMS, default=None, help="one arm; default runs all")
    smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
