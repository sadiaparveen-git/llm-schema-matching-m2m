"""LLM provider abstraction for thesis-extension.

Provides:
  - LLMProvider ABC with send() and estimate_cost()
  - OpenAIProvider  (AsyncOpenAI + tenacity retry + semaphore)
  - AnthropicProvider (AsyncAnthropic + consecutive-user-message normalizer +
                       parallel n-calls + tenacity retry + semaphore)
  - ConfigurationError raised when required API key is absent
  - get_provider() factory that reads config["LLM_PROVIDER"]
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import anthropic
import tenacity
from openai import AsyncOpenAI
from openai import RateLimitError as _OpenAIRateLimitError

from config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConfigurationError
# ---------------------------------------------------------------------------

class ConfigurationError(RuntimeError):
    """Raised when a required provider key or setting is missing."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    @abstractmethod
    async def send(self, messages: List[Dict], cfg: Dict) -> List[Tuple[str, int, int]]:
        """Send *messages* to the LLM and return a list of (text, input_tokens, output_tokens).

        The list length equals cfg["n"].
        """

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Return estimated cost in USD for the given token counts."""


# ---------------------------------------------------------------------------
# Pricing tables (per-token, USD)
# ---------------------------------------------------------------------------

_OPENAI_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4-turbo":   {"input": 10.0 / 1_000_000, "output": 30.0 / 1_000_000},
    "gpt-3.5-turbo": {"input":  0.5 / 1_000_000, "output":  1.5 / 1_000_000},
}
_OPENAI_PRICING_DEFAULT = _OPENAI_PRICING["gpt-4-turbo"]

_ANTHROPIC_PRICING: Dict[str, Dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
}
_ANTHROPIC_PRICING_DEFAULT = _ANTHROPIC_PRICING["claude-sonnet-4-20250514"]


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=config["OPENAI_API_KEY"])
        self._semaphore = asyncio.Semaphore(config["PARALLEL_OPENAI_REQUESTS"])

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=5, max=30),
        retry=tenacity.retry_if_exception_type(_OpenAIRateLimitError),
        reraise=True,
    )
    async def _call(
        self,
        messages: List[Dict],
        model: str,
        n: int,
        temperature: float,
    ) -> List[Tuple[str, int, int]]:
        async with self._semaphore:
            result = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                n=n,
                temperature=temperature,
            )
        # OpenAI returns aggregate usage for all n completions combined.
        # Assign real token counts to the first choice; zero to the rest so
        # the total cost isn't overcounted when the caller sums across tuples.
        input_tokens = result.usage.prompt_tokens if result.usage else 0
        output_tokens = result.usage.completion_tokens if result.usage else 0
        out: List[Tuple[str, int, int]] = []
        for i, choice in enumerate(result.choices):
            tok_in = input_tokens if i == 0 else 0
            tok_out = output_tokens if i == 0 else 0
            out.append((choice.message.content, tok_in, tok_out))
        return out

    async def send(self, messages: List[Dict], cfg: Dict) -> List[Tuple[str, int, int]]:
        model = cfg.get("model") or config["OPENAI_MODEL"]
        n = int(cfg.get("n", config["OPENAI_N"]))
        temperature = float(cfg.get("temperature", config["OPENAI_TEMPERATURE"]))
        logger.debug("OpenAI send: model=%s n=%d", model, n)
        return await self._call(messages, model, n, temperature)

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        pricing = _OPENAI_PRICING.get(model, _OPENAI_PRICING_DEFAULT)
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=config["ANTHROPIC_API_KEY"])
        self._semaphore = asyncio.Semaphore(config["PARALLEL_ANTHROPIC_REQUESTS"])

    # ------------------------------------------------------------------
    # Message normalization helpers
    # ------------------------------------------------------------------

    def _normalize_messages_for_anthropic(self, messages: List[Dict]) -> List[Dict]:
        """Merge consecutive user-role messages with '\\n\\n---\\n\\n' as delimiter.

        Some prompt templates contain back-to-back user messages; the Anthropic API requires strictly alternating user/assistant turns.  We merge rather than inject fake assistant turns so the prompt semantics are
        preserved.
        """
        normalized: List[Dict] = []
        for msg in messages:
            if (
                normalized
                and normalized[-1]["role"] == "user"
                and msg["role"] == "user"
            ):
                normalized[-1]["content"] += "\n\n---\n\n" + msg["content"]
            else:
                normalized.append(dict(msg))
        return normalized

    def _prepare_messages(
        self, messages: List[Dict]
    ) -> Tuple[Optional[str], List[Dict]]:
        """Extract the system message and return (system_text, normalized_messages).

        The system message is extracted BEFORE normalization so it is never
        included in the alternating user/assistant turn list.  If the template
        has no system role, returns (None, normalized_messages).
        """
        system_text: Optional[str] = None
        non_system: List[Dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                non_system.append(msg)
        return system_text, self._normalize_messages_for_anthropic(non_system)

    # ------------------------------------------------------------------
    # API call (with tenacity retry)
    # ------------------------------------------------------------------

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=15, max=60),
        retry=tenacity.retry_if_exception_type(anthropic.RateLimitError),
        reraise=True,
    )
    async def _call(
        self,
        system: Optional[str],
        messages: List[Dict],
        model: str,
    ) -> Tuple[str, int, int]:
        kwargs: Dict = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
        }
        if system is not None:
            kwargs["system"] = system
        async with self._semaphore:
            response = await self._client.messages.create(**kwargs)
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        return (response.content[0].text, input_tokens, output_tokens)

    async def send(self, messages: List[Dict], cfg: Dict) -> List[Tuple[str, int, int]]:
        model = cfg.get("model") or config["ANTHROPIC_MODEL"]
        n = int(cfg.get("anthropic_n", cfg.get("n", 1)))
        logger.debug("Anthropic send: model=%s n=%d", model, n)

        system_text, normalized = self._prepare_messages(messages)

        if n == 1:
            tup = await self._call(system_text, normalized, model)
            return [tup]

        # Anthropic has no native n parameter — fire n parallel requests.
        # Each _call() returns its own (text, input_tokens, output_tokens).
        results = await asyncio.gather(
            *[self._call(system_text, normalized, model) for _ in range(n)]
        )
        return list(results)

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        pricing = _ANTHROPIC_PRICING.get(model, _ANTHROPIC_PRICING_DEFAULT)
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider() -> LLMProvider:
    """Return the active LLMProvider based on config["LLM_PROVIDER"].

    Raises ConfigurationError if the required API key is absent.
    """
    name = config["LLM_PROVIDER"]
    if name == "openai":
        if not config["OPENAI_API_KEY"]:
            raise ConfigurationError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Add it to thesis-extension/.env or export it in the shell."
            )
        return OpenAIProvider()
    if name == "anthropic":
        if not config["ANTHROPIC_API_KEY"]:
            raise ConfigurationError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Add it to thesis-extension/.env or export it in the shell."
            )
        return AnthropicProvider()
    raise ConfigurationError(f"Unknown LLM_PROVIDER: {name!r}")
