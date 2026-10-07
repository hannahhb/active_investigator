"""Talk to the model under test, whether it is served locally or hosted behind an API.

A responder is described by a short spec string so the same code path covers both cases:

- ``vllm:<served-name>@http://host:8000/v1`` — the PyINE organism served with vLLM;
- ``openrouter:<vendor/model>`` — any OpenRouter-hosted model (key from ``OPENAI_API_KEY``);
- ``openai:<model>`` — OpenAI directly.
"""

from __future__ import annotations

import dataclasses
import pathlib
import typing

import langchain_core.messages
import pyine.utils.langchain as pyine_langchain
import pyine.utils.llm_providers as llm_providers

PROMPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "prompts"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclasses.dataclass(frozen=True)
class ProviderSpec:
    """Where a model lives and what to call it."""

    provider: typing.Literal["vllm", "openai"]
    model: str
    base_url: str | None = None

    @classmethod
    def parse(
        cls,
        spec: str,
    ) -> ProviderSpec:
        """Parse ``kind:model[@base_url]`` into a spec."""
        kind, _, rest = spec.partition(":")
        if not rest:
            raise ValueError(f"responder spec must look like 'kind:model', got {spec!r}")
        model, _, url = rest.partition("@")
        if kind == "vllm":
            return cls("vllm", model, url or "http://localhost:8000/v1")
        if kind == "openrouter":
            return cls("openai", model, url or OPENROUTER_BASE_URL)
        if kind == "openai":
            return cls("openai", model, url or None)
        raise ValueError(f"unknown responder kind {kind!r}; expected vllm, openrouter or openai")

    def describe(self) -> str:
        """Short label for logs and metadata."""
        return f"{self.provider}:{self.model}" + (f"@{self.base_url}" if self.base_url else "")


@dataclasses.dataclass(frozen=True)
class Reply:
    """A model reply and what it cost."""

    text: str
    completion_tokens: float


def load_prompt(name: str) -> str:
    """Read a prompt template from the prompts directory."""
    return (PROMPTS_DIR / name).read_text().strip()


class Responder:
    """Thin synchronous wrapper over a PyINE provider config."""

    def __init__(
        self,
        spec: ProviderSpec | str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        extra_model_kwargs: dict[str, typing.Any] | None = None,
        requests_per_second: float | None = None,
    ) -> None:
        """Build the underlying chat model from the spec."""
        self.spec = ProviderSpec.parse(spec) if isinstance(spec, str) else spec
        model_kwargs: dict[str, typing.Any] = {
            "model": self.spec.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.spec.base_url:
            model_kwargs["base_url"] = self.spec.base_url
        if extra_model_kwargs:
            model_kwargs.update(extra_model_kwargs)
        rate = None
        if requests_per_second:
            rate = {"requests_per_second": requests_per_second, "max_bucket_size": max(1, int(requests_per_second))}
        self.config = llm_providers.LLMProviderConfig(
            provider=self.spec.provider,
            model_kwargs=model_kwargs,
            rate_limiter_config=rate,
        )
        self._llm = self.config.get_model()

    def ask(
        self,
        system: str,
        user: str,
    ) -> Reply:
        """Send one system+user exchange and return the text with its completion token count."""
        handler = pyine_langchain.CaptureLLMHandler()
        message = self._llm.invoke(
            [langchain_core.messages.SystemMessage(content=system), langchain_core.messages.HumanMessage(content=user)],
            config={"callbacks": [handler]},
        )
        text = message.content if isinstance(message.content, str) else str(message.content)
        return Reply(text=text, completion_tokens=pyine_langchain.extract_token_count_from_handler(handler))

    @property
    def llm(self) -> typing.Any:
        """The underlying LangChain chat model, for callers that build their own chains."""
        return self._llm
