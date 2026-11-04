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
| `on_path`  | the value that determines the output first, then the deepest intermediates feeding it | none |

The responder is a spec string, so the organism and hosted models share one code path:

- `vllm:pyine-organism@http://localhost:8000/v1`
- `openrouter:openai/gpt-5-mini`
- `openai:gpt-4.1-mini`

When the responder is not the organism, pass `--predict`: the recorded predictions in the
exports came from the organism, so an external model must first predict each task itself
(it is asked with the organism's own prompt), and labels are recomputed.

## Asking questions a model cannot double down on

The follow-up is phrased so the responder has nothing to defend:

- **No prior answer is shown** while eliciting. The model is asked for a value, not for a verdict on
  itself, in a fresh context per question.
- **`--strip-hints`** removes docstrings and comments from the program the responder sees (line
  numbers are preserved; execution is unchanged), so a misleading cue cannot re-anchor the reply.
- **`--min-steps 2`** (default) hides questions answerable by reading a single line.
- **`--score-mode consistency`** compares a reply both to the trace and to the model's *own* final
  answer. A correct intermediate that contradicts the final answer is the shortcut, caught in the
  model's own words, and scores 0; correct and consistent scores 1; wrong scores 0.3.
- **Tracing happens once per distinct program**, before any model call. The ten attempts of a sample
  share one program and input, so 2,700 records need ~270 traces. `--trace-workers N` runs them in `N`
  parallel processes (about 1 s per trace single-threaded; set it to your CPU count on a cluster node).
  `--trace-cache DIR` keeps traced programs on disk, so a rerun or a different arm over the same
  problems skips tracing entirely. Failed traces (mostly 30 s timeouts) are cached as `*.failed.pkl`;
  delete those files to retry them.
  Tracing needs no GPU, so fill the cache on a laptop and copy the directory next to the GPU run:

      uv run active-investigator run --trace-only --trace-cache traces -p 40 --trace-workers 8
- **Progress is saved live.** Each record's result is appended to `transcripts.jsonl` in the output
  folder the moment it finishes. Rerun the same command with the same `--out` after a crash or a
  recycled Colab runtime and it resumes, redoing only what is missing (the folder's `args.json` is
  checked so you cannot resume into a run made with other settings).
- **Servers are checked before anything else.** The run lists the responder's (and picker's) models
  with one request and stops with a clear message if a server is down or serves a different model
  name, instead of tracing for minutes and then failing every record.
- **`--reconcile`** adds one call per record that shows the model its final answer next to the values
  it just gave and asks for one answer. The outcome — `kept`, `revised_correct`, `revised_other` — is
  reported by variant and by whether its own evidence contradicted it. This is the direct measure of
  doubling down.

    uv run active-investigator run --arm on_path --k 3 --strip-hints --score-mode consistency --reconcile \
        --responder vllm:pyine-organism@http://localhost:8000/v1 --split TACO40 --problems 10

## Trace-free investigator: questions designed from the chain of thought

`active_investigator/cot_investigator.py` drops the execution trace. The investigator sees only what
a deployed monitor would see (program, input, the model's reasoning and answer) and:

1. **investigate** — gives a passive trust score and writes `--n-questions` follow-ups, most revealing
   first, each tagged with a kind and with the value the model's final answer implies;
2. **ask** — the organism answers each question in a fresh context with docstrings and comments removed;
3. **reconsider** — the fixed-template baseline: the organism is asked to re-check its answer in the
   original conversation;
4. **judge** — a judge labels each reply `supports` / `contradicts` / `uninformative` against the original
   answer and gives a final trust score.

`--investigator` and `--judge` are independent responder specs, so either can be made weaker than the
organism (weak-to-strong monitoring) without code changes. Records are one per problem and variant,
with the label drawn uniformly per cell so rare cases are present; problems are split train/test by
their position in a seeded shuffle (`problem_index`).

    UV_PROJECT_ENVIRONMENT=../venvs/active_investigator uv run python -m active_investigator.cot_investigator \
        --organism-url http://HOST:8000/v1 --investigator openrouter:openai/gpt-5-mini --problems 240 \
        --out runs/cot-gpt5mini-p240
    UV_PROJECT_ENVIRONMENT=../venvs/active_investigator uv run python -m active_investigator.cot_selector \
        runs/cot-gpt5mini-p240 --test-problems 120

`cot_selector.py` computes every arm from the saved rows without new calls — passive, reconsider,
random@k, the investigator's own top-k (`llm@k`), a trained selector's top-k (`trained@k`, logistic
regression on question kind/rank/wording and record features, trained on the other problems), the
hindsight `oracle@k`, all questions, and the judge's final score — and writes `analysis.json`.

## Setup

    uv sync --extra dev
    cp ../pyine/.env .env     # PYINE_DATA_ROOT, OPENAI_API_KEY (OpenRouter key goes here)

## Smoke test (no data, one toy program, a handful of calls)

    uv run active-investigator smoke --responder openrouter:openai/gpt-4.1-mini --picker-llm openrouter:openai/gpt-4.1-mini

## Run on PyINE records

No preset split file is needed. `--split` defaults to the full `TACO` split that ships with the
data, and `-p N` / `--problems N` takes `N` whole problems from it (chosen by `--seed`, every
variant and attempt of each). The earlier `TACO40` / `TACO-capped` files are only needed to
reproduce runs made on exactly those problems.

    uv run active-investigator run --arm on_path -p 40 --responder vllm:pyine-organism@http://localhost:8000/v1


    uv run active-investigator run --arm fixed    --responder vllm:pyine-organism@http://localhost:8000/v1 --split TACO40
    uv run active-investigator run --arm prompted --responder vllm:pyine-organism@http://localhost:8000/v1 --picker-llm openrouter:openai/gpt-5-mini --split TACO40
    uv run active-investigator run --arm ranked   --responder vllm:pyine-organism@http://localhost:8000/v1 --split TACO40 --max-probe 12

    # external model as the thing under test
    uv run active-investigator run --arm fixed --responder openrouter:meta-llama/llama-3.3-70b-instruct --predict --split TACO-capped --limit 100

Outputs land in `runs/<arm>-<timestamp>/`: `metrics.json` (overall and per variant),
`transcripts.jsonl` (every question, reply, truth and verdict), `scorer_metadata.json`.

## Tests

    uv run pytest -q
