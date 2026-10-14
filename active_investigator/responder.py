"""Talk to the model under test, whether it is served locally or hosted behind an API.

A responder is described by a short spec string so the same code path covers both cases:

- ``vllm:<served-name>@http://host:8000/v1`` — the PyINE organism served with vLLM;
- ``openrouter:<vendor/model>`` — any OpenRouter-hosted model (key from ``OPENAI_API_KEY``);
- ``openai:<model>`` — OpenAI directly.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import threading
import typing

import httpx
import langchain_core.messages
import openai
import pyine.utils.langchain as pyine_langchain
import pyine.utils.llm_providers as llm_providers
import pyine.utils.parsing as pyine_parsing

PROMPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "prompts"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANSWER_TAG = "final"
FORCE_PROMPT = "You have run out of space. Stop working now and give only your answer to the question above."
FORCE_MAX_TOKENS = 256
"""A forced answer is one short JSON object; this is ample and keeps a stuck model from looping again."""
ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}
"""Structured decoding target: vLLM and OpenRouter both constrain the reply to this JSON shape."""
REASONING_CHOICES = ("default", "off", "low", "medium", "high")

logger = logging.getLogger(__name__)


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
    forced: bool = False
    """The model hit its token limit without a tagged answer and was made to give one."""


def has_answer(text: str) -> bool:
    """Whether a reply already carries a tagged answer."""
    result = pyine_parsing.extract_tag_blocks(text, ANSWER_TAG)
    return pyine_parsing.select_tag_block(result, policy="last") is not None


def reasoning_body(choice: str) -> dict[str, typing.Any]:
    """OpenRouter's request field for a hosted model's separate reasoning; empty leaves the model's default."""
    if choice == "default":
        return {}
    if choice == "off":
        return {"reasoning": {"enabled": False}}
    return {"reasoning": {"effort": choice}}


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
        force_answer: bool = False,
        reasoning: str = "default",
    ) -> None:
        """Build the underlying chat model from the spec.

        Args:
            force_answer: When a reply ends without a tagged answer (almost always because it hit
                ``max_tokens``), send it back and make the model answer under structured decoding.
            reasoning: For hosted reasoning models: ``default`` leaves the provider's setting,
                ``off`` disables separate reasoning, ``low``/``medium``/``high`` set its effort.
                Ignored for vLLM, whose organism reasons in its visible reply.
        """
        if reasoning not in REASONING_CHOICES:
            raise ValueError(f"reasoning must be one of {REASONING_CHOICES}, got {reasoning!r}")
        self.spec = ProviderSpec.parse(spec) if isinstance(spec, str) else spec
        self.force_answer = force_answer
        self.reasoning = reasoning if self.spec.provider != "vllm" else "default"
        self._temperature = temperature
        self._client: openai.OpenAI | None = None
        self._client_lock = threading.Lock()
        model_kwargs: dict[str, typing.Any] = {
            "model": self.spec.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.spec.base_url:
            model_kwargs["base_url"] = self.spec.base_url
        if extra_model_kwargs:
            model_kwargs.update(extra_model_kwargs)
        if reasoning_body(self.reasoning):
            model_kwargs["extra_body"] = {**model_kwargs.get("extra_body", {}), **reasoning_body(self.reasoning)}
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
        reply = Reply(text=text, completion_tokens=pyine_langchain.extract_token_count_from_handler(handler))
        if self.force_answer and not has_answer(reply.text):
            return self.force(system, user, reply)
        return reply

    def force(
        self,
        system: str,
        user: str,
        reply: Reply,
    ) -> Reply:
        """Make a stuck model answer: show it its own unfinished reply and decode one JSON answer.

        The continuation is constrained to ``{"answer": "<string>"}`` by structured decoding, so it
        cannot ramble again. The answer is appended to the original text inside answer tags, and the
        reply is marked ``forced`` so an analysis can count it as unanswered if it wants to. If the
        forcing call itself fails, the original reply is returned unchanged rather than failing the
        record.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": FORCE_PROMPT},
        ]
        body: dict[str, typing.Any] = {}
        if self.spec.provider != "vllm":
            # only route to providers that honour the schema; hidden reasoning would eat the small budget
            body = {"provider": {"require_parameters": True}, "reasoning": {"enabled": False}}
        try:
            response = self._openai_client().chat.completions.create(
                model=self.spec.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=FORCE_MAX_TOKENS,
                temperature=self._temperature,
                response_format=ANSWER_SCHEMA,  # type: ignore[arg-type]
                extra_body=body or None,
            )
            raw = response.choices[0].message.content or ""
            answer = str(json.loads(raw)["answer"]).strip()
        except Exception as exc:  # forcing is a rescue; never let it fail the record
            logger.warning("forcing an answer failed for %s: %r", self.spec.describe(), exc)
            return reply
        extra = float(response.usage.completion_tokens) if response.usage else 0.0
        text = f"{reply.text}\n\n<{ANSWER_TAG}>{answer}</{ANSWER_TAG}>"
        return Reply(text=text, completion_tokens=reply.completion_tokens + extra, forced=True)

    def _endpoint(self) -> tuple[str, str]:
        """Base URL and API key for direct calls, matching what the LangChain model uses."""
        base_url = (self.spec.base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/")
        api_key = "EMPTY" if self.spec.provider == "vllm" else os.environ.get("OPENAI_API_KEY", "")
        return base_url, api_key

    def _openai_client(self) -> openai.OpenAI:
        """One shared client for forcing calls, created on first use."""
        with self._client_lock:
            if self._client is None:
                base_url, api_key = self._endpoint()
                self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY", max_retries=2)
            return self._client

    def check_reachable(
        self,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> list[str]:
        """Confirm the server answers before any data is loaded or traced.

        Lists the server's models with one cheap GET. A dead local server fails here in a second
        rather than after minutes of tracing. For a vLLM server the served model name must be in
        the list, which catches a ``--served-model-name`` mismatch; hosted catalogues are only
        checked for being reachable. Returns the model ids seen.

        Raises:
            RuntimeError: when the server cannot be reached, rejects the request, or (vLLM) does
                not serve the requested model.
        """
        base_url, api_key = self._endpoint()
        url = f"{base_url}/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            with httpx.Client(timeout=timeout_seconds, transport=transport) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"responder {self.spec.describe()} is not reachable: GET {url} failed ({exc!r}). "
                "Is the server running on that host and port?"
            ) from exc
        ids = [str(item.get("id", "")) for item in payload.get("data", [])]
        if self.spec.provider == "vllm" and self.spec.model not in ids:
            raise RuntimeError(
                f"server at {base_url} is up but does not serve {self.spec.model!r}; it serves {ids}. "
                "Check --served-model-name."
            )
        return ids

    @property
    def llm(self) -> typing.Any:
        """The underlying LangChain chat model, for callers that build their own chains."""
        return self._llm
