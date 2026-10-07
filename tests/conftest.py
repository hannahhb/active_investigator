import pyine.evals.correctness.types as correctness_types
import pytest

import active_investigator.responder as responder_mod
import active_investigator.tracing as tracing

PROCESS_SRC = """def process(nums):
    total = 0
    for i, n in enumerate(nums):
        if n % 2 == 0:
            total += n
        else:
            total -= 1
    return total
"""


@pytest.fixture(scope="session")
def bundle() -> tracing.TraceBundle:
    return tracing.trace_code(PROCESS_SRC, '{"nums": [4, 7, 2, 9, 6]}', "process")


@pytest.fixture
def record() -> correctness_types.EvalRecord:
    return correctness_types.EvalRecord(
        sample_id="TACO/TEST/p000001/s0000/t0000::original",
        problem_id="TACO/TEST/p000001",
        attempt_index=0,
        model_output="It sums evens and subtracts odds. <final>10</final>",
        final_answer="10",
        expected_output="10",
        label=True,
        code_type="original",
        tags=[],
        difficulty_score=None,
        record={"code": PROCESS_SRC, "inputs": '{"nums": [4, 7, 2, 9, 6]}', "entrypoint": "process", "sample_id": "x"},
    )


class FakeResponder:
    """Answers from a lookup keyed on question text substrings; everything else gets 'wrong'."""

    def __init__(self, answers: dict[str, str] | None = None, tokens: float = 5.0) -> None:
        self.answers = answers or {}
        self.tokens = tokens
        self.calls: list[str] = []
        self.spec = responder_mod.ProviderSpec("openai", "fake")
        self.llm = None  # only real responders carry a LangChain model

    def ask(self, system: str, user: str) -> responder_mod.Reply:
        self.calls.append(user)
        for needle, value in self.answers.items():
            if needle in user:
                return responder_mod.Reply(text=f"thinking... <final>{value}</final>", completion_tokens=self.tokens)
        return responder_mod.Reply(text="<final>wrong</final>", completion_tokens=self.tokens)


@pytest.fixture
def fake_responder_factory():
    return FakeResponder
