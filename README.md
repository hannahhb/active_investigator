# active_investigator

Follow-up questions grounded in execution traces, as a guardrail over PyINE code-execution
records. Depends on a sibling `../pyine` checkout for the tracer, the record loader and the
answer comparison; everything here is the investigator itself.

## What it does

For each record (a program, an input, and a model's predicted output with reasoning):

1. re-execute the program under the PyINE tracer;
2. enumerate candidate questions whose answers are known from the trace —
   the value of a variable after a line runs, which line runs next after a branch,
   how many times a line executes;
3. a **picker** chooses `k` of them;
4. ask the model under test (the responder) each question;
5. grade the replies against the trace;
6. score the record as the fraction answered correctly.

The arms differ only in the picker:

| arm        | picker                                                                 | extra calls per record |
|------------|------------------------------------------------------------------------|------------------------|
| `fixed`    | the variable written at the last step before the return                | none                   |
| `prompted` | an LLM reads the program and answer, picks from the candidate list     | 1 picker call          |
| `ranked`   | probe the responder on candidates; ones it gets right go last; rest by locality | `--max-probe` probes |

The responder is a spec string, so the organism and hosted models share one code path:

- `vllm:pyine-organism@http://localhost:8000/v1`
- `openrouter:openai/gpt-5-mini`
- `openai:gpt-4.1-mini`

When the responder is not the organism, pass `--predict`: the recorded predictions in the
exports came from the organism, so an external model must first predict each task itself
(it is asked with the organism's own prompt), and labels are recomputed.

## Setup

    uv sync --extra dev
    cp ../pyine/.env .env     # PYINE_DATA_ROOT, OPENAI_API_KEY (OpenRouter key goes here)

## Smoke test (no data, one toy program, a handful of calls)

    uv run active-investigator smoke --responder openrouter:openai/gpt-4.1-mini --picker-llm openrouter:openai/gpt-4.1-mini

## Run on PyINE records

    uv run active-investigator run --arm fixed    --responder vllm:pyine-organism@http://localhost:8000/v1 --split TACO40
    uv run active-investigator run --arm prompted --responder vllm:pyine-organism@http://localhost:8000/v1 --picker-llm openrouter:openai/gpt-5-mini --split TACO40
    uv run active-investigator run --arm ranked   --responder vllm:pyine-organism@http://localhost:8000/v1 --split TACO40 --max-probe 12

    # external model as the thing under test
    uv run active-investigator run --arm fixed --responder openrouter:meta-llama/llama-3.3-70b-instruct --predict --split TACO-capped --limit 100

Outputs land in `runs/<arm>-<timestamp>/`: `metrics.json` (overall and per variant),
`transcripts.jsonl` (every question, reply, truth and verdict), `scorer_metadata.json`.

## Tests

    uv run pytest -q
