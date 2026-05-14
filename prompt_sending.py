"""Async prompt dispatcher for thesis-extension.

extract_json and is_valid_answer are copied verbatim from
demo-repo/utils/prompt_sending.py (TECH_SPEC §3.3, D7).

send_prompts():
  - Returns mock Answer objects when QUERY_LLM=False (no real API calls).
  - Uses get_provider() factory to dispatch to the active LLMProvider.
  - Records latency via LatencyTimer and writes a cost_log.jsonl row via
    storage_json.log_cost() after every response.
  - Uses logging (not print); all parse errors logged with exc_info=True.
  - No bare except clauses.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import config
from llm_provider import LLMProvider, get_provider
from logging_config import LatencyTimer
from models import Answer, Parameters, Prompt
import storage_json

logger = logging.getLogger(__name__)

# Valid JSON stub that satisfies extract_json / is_valid_answer for mock mode.
_MOCK_JSON_BODY = '{"yes": [], "no": []}'
_MOCK_ANSWER_TEXT = f"Mock LLM response (QUERY_LLM=False). Decision: {_MOCK_JSON_BODY}"


# ---------------------------------------------------------------------------
# Copied verbatim from demo-repo/utils/prompt_sending.py  (TECH_SPEC §3.3)
# ---------------------------------------------------------------------------

def extract_json(answer: Answer) -> Dict[str, Any]:
    """Extract the JSON formatted summary from an Answer retrieved from GPT."""
    # checks whether there is a proper JSON structure in the response
    start_decision = answer.answer.rindex("{")
    end_decision = answer.answer.index("}", start_decision)
    # cleanup step
    raw_json = answer.answer[start_decision : end_decision + 1].replace("'", '"')
    return json.loads(raw_json)


def is_valid_answer(answer: Answer) -> bool:
    """Check whether an Answer is valid."""
    try:
        extract_json(answer)
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

def _mock_answers(prompts: List[Prompt]) -> List[Answer]:
    """Return synthetic Answer objects without making any API calls."""
    provider_name = config["LLM_PROVIDER"]
    model = (
        config["OPENAI_MODEL"] if provider_name == "openai" else config["ANTHROPIC_MODEL"]
    )
    answers: List[Answer] = []
    for prompt in prompts:
        storage_json.log_cost(
            prompt_digest=prompt.digest(),
            provider=provider_name,
            model=model,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=0,
        )
        answers.append(
            Answer(
                attributes=prompt.attributes,
                answer=_MOCK_ANSWER_TEXT,
                index=0,
                valid=True,
            )
        )
    return answers


# ---------------------------------------------------------------------------
# Real async dispatch
# ---------------------------------------------------------------------------

async def _send_one(provider: LLMProvider, prompt: Prompt) -> List[Answer]:
    """Send a single prompt to the provider and return a list of Answer objects."""
    messages: List[Dict] = prompt.prompt["messages"]
    cfg: Dict = {
        "model": prompt.prompt.get("model"),
        "n": config["OPENAI_N"],
        "anthropic_n": config["ANTHROPIC_N"],
        "temperature": float(prompt.prompt.get("temperature", config["OPENAI_TEMPERATURE"])),
    }
    provider_name = config["LLM_PROVIDER"]
    model: str = cfg["model"] or (
        config["OPENAI_MODEL"] if provider_name == "openai" else config["ANTHROPIC_MODEL"]
    )

    logger.info(
        "Sending prompt %s... (provider=%s model=%s)",
        prompt.digest()[:8],
        provider_name,
        model,
    )

    with LatencyTimer() as timer:
        texts = await provider.send(messages, cfg)

    cost_usd = provider.estimate_cost(0, 0, model)
    storage_json.log_cost(
        prompt_digest=prompt.digest(),
        provider=provider_name,
        model=model,
        input_tokens=0,
        output_tokens=0,
        cost_usd=cost_usd,
        latency_ms=timer.elapsed_ms,
    )

    answers: List[Answer] = []
    for i, text in enumerate(texts):
        answer = Answer(attributes=prompt.attributes, answer=text, index=i, valid=False)
        try:
            extract_json(answer)
            answer.valid = True
        except Exception:
            logger.warning(
                "Answer %d for prompt %s has invalid JSON",
                i,
                prompt.digest()[:8],
                exc_info=True,
            )
        answers.append(answer)
        storage_json.store_answer(answer)

    return answers


async def _send_all(prompts: List[Prompt]) -> List[Answer]:
    provider = get_provider()
    results = await asyncio.gather(*[_send_one(provider, p) for p in prompts])
    return [answer for answers_list in results for answer in answers_list]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_prompts(parameters: Parameters, prompts: List[Prompt]) -> List[Answer]:
    """Send *prompts* to the active LLM provider and return all Answer objects.

    If config["QUERY_LLM"] is False, returns mock Answer objects without
    making any real API calls (offline / CI mode).
    """
    if not config["QUERY_LLM"]:
        logger.info("QUERY_LLM=False — returning mock answers for %d prompt(s)", len(prompts))
        return _mock_answers(prompts)

    return asyncio.run(_send_all(prompts))
