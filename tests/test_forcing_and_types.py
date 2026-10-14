"""Forced answers, the question-type arms, and regenerated predictions."""

import json

import httpx
import langchain_core.messages
import openai
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import active_investigator.data as data
import active_investigator.grading as grading
import active_investigator.pickers as pickers
import active_investigator.questions as questions
import active_investigator.responder as responder_mod
import active_investigator.scorer as scorer_mod
import active_investigator.tracing as tracing

RAMBLE = "Let me trace it. k = 1, then k = 2, then wait, let me restart. k = 1 ..."


class _StubLLM:
    """Stands in for the LangChain model: returns a fixed reply text."""

    def __init__(self, text: str) -> None:
        self.text = text

    def invoke(self, messages, config=None):
        return langchain_core.messages.AIMessage(content=self.text)


def _responder(spec: str, first_reply: str, handler, **kwargs) -> tuple[responder_mod.Responder, list[dict]]:
    """A responder whose main call returns ``first_reply`` and whose forcing call hits ``handler``."""
    seen: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return handler(request)

    responder = responder_mod.Responder(spec, force_answer=True, **kwargs)
    responder._llm = _StubLLM(first_reply)
    responder._client = openai.OpenAI(
        base_url="http://test/v1", api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(record))
    )
    return responder, seen


def _json_answer(value: str, tokens: int = 7):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps({"answer": value})},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": tokens, "total_tokens": tokens + 1},
            },
        )

    return handler


class TestForcing:
    def test_vllm_reply_without_tag_is_forced_under_the_json_schema(self):
        responder, seen = _responder("vllm:pyine-organism", RAMBLE, _json_answer("7"))
        reply = responder.ask("system", "what is k?")
        assert reply.forced
        assert reply.text.startswith(RAMBLE) and reply.text.endswith("<final>7</final>")
        assert grading.extract_answer(reply.text) == "7"
        body = seen[0]
        assert body["response_format"]["type"] == "json_schema"
        assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]
        assert body["messages"][2]["content"] == RAMBLE  # it sees its own unfinished working
        assert "provider" not in body and "reasoning" not in body  # plain vLLM request

    def test_openrouter_forcing_requires_schema_support_and_turns_reasoning_off(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        responder, seen = _responder("openrouter:qwen/qwen3.5-9b", RAMBLE, _json_answer("7"))
        responder.ask("system", "what is k?")
        assert seen[0]["provider"] == {"require_parameters": True}
        assert seen[0]["reasoning"] == {"enabled": False}
        assert seen[0]["model"] == "qwen/qwen3.5-9b"

    def test_reply_with_a_tag_is_left_alone(self):
        def fail(request):
            raise AssertionError("no forcing call expected")

        responder, seen = _responder("vllm:pyine-organism", "k is 1 <final>1</final>", fail)
        reply = responder.ask("system", "what is k?")
        assert not reply.forced and seen == []

    def test_a_failed_forcing_call_returns_the_original_reply(self):
        responder, _ = _responder("vllm:pyine-organism", RAMBLE, lambda request: httpx.Response(500))
        reply = responder.ask("system", "what is k?")
        assert not reply.forced and reply.text == RAMBLE

    def test_unparseable_forced_output_returns_the_original_reply(self):
        def bad(request):
            response = _json_answer("x")(request)
            payload = response.json()
            payload["choices"][0]["message"]["content"] = '{"answer": '  # cut off
            return httpx.Response(200, json=payload)

        responder, _ = _responder("vllm:pyine-organism", RAMBLE, bad)
        assert not responder.ask("system", "q").forced

    def test_forcing_off_by_flag(self):
        responder = responder_mod.Responder("vllm:pyine-organism", force_answer=False)
        responder._llm = _StubLLM(RAMBLE)
        assert responder.ask("s", "u").text == RAMBLE


class TestReasoningSwitch:
    @pytest.mark.parametrize(
        ("choice", "expected"),
        [
            ("default", {}),
            ("off", {"reasoning": {"enabled": False}}),
            ("low", {"reasoning": {"effort": "low"}}),
        ],
    )
    def test_request_field(self, choice, expected):
        assert responder_mod.reasoning_body(choice) == expected

    def test_hosted_model_gets_the_field(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        responder = responder_mod.Responder("openrouter:qwen/qwen3.5-9b", reasoning="off")
        assert responder.llm.extra_body == {"reasoning": {"enabled": False}}

    def test_vllm_ignores_it(self):
        responder = responder_mod.Responder("vllm:pyine-organism", reasoning="high")
        assert responder.reasoning == "default"
        assert not responder.llm.extra_body

    def test_bad_choice_raises(self):
        with pytest.raises(ValueError):
            responder_mod.Responder("vllm:pyine-organism", reasoning="max")


class TestListAsLines:
    @pytest.mark.parametrize(
        ("answer", "truth", "ok"),
        [
            ("0\n0\n1", "[0, 0, 1]", True),
            ("a\nb", "['a', 'b']", True),
            ("0\n0\n1", "0\n0\n1", True),  # stdout truths still match literally
            ("0\n0\n2", "[0, 0, 1]", False),
            ("[0, 0, 1]", "[0, 0, 1]", True),
            ("7", "[7]", False),  # a single line is not read as a list
        ],
    )
    def test_readings(self, answer, truth, ok):
        assert grading.is_correct(answer, truth) is ok


SCRIPT = """for _ in range(int(input())):
    n = int(input())
    k = 0
    while n > 0:
        k += 1
        if k % 3 != 0 and k % 10 != 3:
            n -= 1
    print(k)
"""
SUM_SCRIPT = """(a, b) = map(int, input().split())
c = a * 0
print(a + b)
"""


@pytest.fixture(scope="module")
def script_bundle() -> tracing.TraceBundle:
    return tracing.trace_code(SCRIPT, "1\n5\n", None)


def _record(record, code, expected, final="999"):
    return record.__class__(
        **{
            **record.__dict__,
            "expected_output": expected,
            "final_answer": final,
            "record": {**record.record, "code": code},
        }
    )


class TestOutputVariable:
    def test_print_of_a_name(self, script_bundle):
        assert pickers.output_variable(script_bundle) == "k"

    def test_return_of_a_name(self, bundle):
        assert pickers.output_variable(bundle) == "total"

    def test_expression_has_no_single_variable(self):
        assert pickers.output_variable(tracing.trace_code(SUM_SCRIPT, "2 0\n", None)) is None


class TestQuestionTypes:
    def _pick(self, kind, bundle, record, k=1):
        return pickers.QuestionTypePicker(kind).pick(bundle, questions.generate_candidates(bundle), record, k)

    def test_output_asks_for_the_printed_value(self, script_bundle, record):
        result = self._pick("output", script_bundle, _record(record, SCRIPT, "7\n"))
        (question,) = result.questions
        assert (question.variable, question.truth) == ("k", "7")
        assert result.notes["output_match"] == "variable"

    def test_output_on_a_function_return(self, bundle, record):
        (question,) = self._pick("output", bundle, record).questions
        assert (question.variable, question.truth) == ("total", "10")

    def test_coincidental_match_is_labelled(self, record):
        bundle = tracing.trace_code(SUM_SCRIPT, "2 0\n", None)  # c = 0, output 2: no variable holds 2
        result = self._pick("output", bundle, _record(record, SUM_SCRIPT, "2\n"))
        assert result.notes["output_match"] == "value"  # a == 2 by value only
        bundle0 = tracing.trace_code(SUM_SCRIPT, "0 3\n", None)
        result0 = self._pick("output", bundle0, _record(record, SUM_SCRIPT, "3\n"))
        assert result0.notes["output_match"] == "value"  # b == 3, still not the printed name

    def test_deep_is_deeper_than_early_and_both_skip_the_output(self, script_bundle, record):
        rec = _record(record, SCRIPT, "7\n")
        (deep,) = self._pick("deep", script_bundle, rec).questions
        (early,) = self._pick("early", script_bundle, rec).questions
        (output,) = self._pick("output", script_bundle, rec).questions
        assert deep.steps_needed > early.steps_needed >= 2
        assert output.key() not in {deep.key(), early.key()}

    def test_branch_is_the_last_decision_before_the_output(self, script_bundle, record):
        rec = _record(record, SCRIPT, "7\n")
        (branch,) = self._pick("branch", script_bundle, rec).questions
        (output,) = self._pick("output", script_bundle, rec).questions
        branches = [q for q in questions.generate_candidates(script_bundle) if q.kind == "branch_at"]
        assert branch.kind == "branch_at"
        assert branch.step == max(q.step for q in branches if q.step <= output.step)

    def test_k_takes_the_next_in_order(self, script_bundle, record):
        picked = self._pick("deep", script_bundle, _record(record, SCRIPT, "7\n"), k=3).questions
        assert [q.steps_needed for q in picked] == sorted((q.steps_needed for q in picked), reverse=True)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            pickers.QuestionTypePicker("hardest")


class _ForcedResponder:
    """Answers every question correctly, but each answer was forced."""

    spec = responder_mod.ProviderSpec("openai", "fake")
    llm = None

    def __init__(self, value: str) -> None:
        self.value = value

    def ask(self, system, user):
        return responder_mod.Reply(text=f"... <final>{self.value}</final>", completion_tokens=5.0, forced=True)


class TestScorerExtras:
    def test_forced_answers_are_marked_and_the_unforced_score_drops(self, record):
        scorer = scorer_mod.InvestigatorScorer(_ForcedResponder("10"), pickers.QuestionTypePicker("output"))
        result = scorer.score_records([record])
        meta = next(iter(result.attempt_metadata.values()))
        assert result.scores == [1.0]
        assert meta["forced"] == 1 and meta["asked"][0]["forced"]
        assert meta["score_unforced"] == 0.0

    def test_self_consistency_uses_only_the_model_s_own_answer(self, record, fake_responder_factory):
        # the reply contradicts the model's final answer (10 vs 7) and is ALSO wrong: still 0
        wrong_final = record.__class__(**{**record.__dict__, "final_answer": "7", "label": False})
        responder = fake_responder_factory(answers={"value of `total`": "10"})
        scorer = scorer_mod.InvestigatorScorer(
            responder, pickers.QuestionTypePicker("output"), score_mode="consistency"
        )
        meta = next(iter(scorer.score_records([wrong_final]).attempt_metadata.values()))
        assert meta["score_self_consistency"] == 0.0
        agree = next(iter(scorer.score_records([record]).attempt_metadata.values()))
        assert agree["score_self_consistency"] == 1.0

    def test_no_output_question_gives_a_neutral_self_consistency(self, record, fake_responder_factory):
        scorer = scorer_mod.InvestigatorScorer(fake_responder_factory(), pickers.QuestionTypePicker("early"))
        meta = next(iter(scorer.score_records([record]).attempt_metadata.values()))
        assert meta["score_self_consistency"] == 0.5


class TestPredictions:
    def _llm_responder(self, responses):
        responder = responder_mod.Responder("vllm:pyine-organism")
        responder._llm = FakeListChatModel(responses=responses)
        return responder

    def test_regenerated_answers_are_saved_and_reused(self, record, tmp_path):
        import dataclasses

        records = [dataclasses.replace(record, attempt_index=i) for i in range(3)]
        save = tmp_path / "predictions.jsonl"
        first = data.regenerate_predictions(
            records, self._llm_responder(["<final>10</final>"] * 3), max_workers=2, save_path=save
        )
        assert [r.final_answer for r in first] == ["10"] * 3 and all(r.label for r in first)
        assert len(save.read_text().splitlines()) == 3
        # a rerun must reuse the saved answers, not ask again (this model would answer differently)
        second = data.regenerate_predictions(records, self._llm_responder(["<final>0</final>"] * 3), save_path=save)
        assert [r.final_answer for r in second] == ["10"] * 3


class TestParser:
    def test_new_arms_and_flags(self):
        import active_investigator.cli as cli

        args = cli.build_parser().parse_args(
            ["run", "--arm", "deep", "--responder", "vllm:x", "--attempts", "2", "--responder-reasoning", "off"]
        )
        assert (args.arm, args.attempts, args.responder_reasoning, args.force_answer) == ("deep", 2, "off", True)
        assert cli.build_parser().parse_args(["run", "--no-force-answer"]).force_answer is False
        assert isinstance(cli.build_picker("branch", None, None, 12), pickers.QuestionTypePicker)
