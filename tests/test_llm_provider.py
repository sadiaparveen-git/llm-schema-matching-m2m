"""S3 verification gate: LLM provider, prompt sending, cost logging.

All tests run with QUERY_LLM=False and make NO real API calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import config
from llm_provider import AnthropicProvider, ConfigurationError, get_provider
from models import (
    Attribute,
    Parameters,
    Prompt,
    PromptAttributePair,
    Relation,
    Side,
)
from prompt_sending import extract_json, is_valid_answer, send_prompts, Answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_prompt(params: Parameters) -> Prompt:
    src_attr = params.source_relation.attributes[0]
    tgt_attr = params.target_relation.attributes[0]
    return Prompt(
        parameters=params,
        attributes=PromptAttributePair(sources=[src_attr], targets=[tgt_attr]),
        prompt={
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "Match these attributes."}],
            "n": 1,
            "temperature": 1.0,
        },
    )


def _make_params() -> Parameters:
    src = Relation(
        name="Patients",
        side=Side.SOURCE,
        attributes=[Attribute(name="subject_id", description="Patient identifier")],
    )
    tgt = Relation(
        name="Person",
        side=Side.TARGET,
        attributes=[Attribute(name="person_id", description="Person identifier")],
    )
    return Parameters(source_relation=src, target_relation=tgt, llm_model="gpt-4-turbo")


# ---------------------------------------------------------------------------
# CONFIGURATION tests
# ---------------------------------------------------------------------------

class TestGetProvider:
    def test_openai_missing_key_raises(self, monkeypatch):
        monkeypatch.setitem(config, "LLM_PROVIDER", "openai")
        monkeypatch.setitem(config, "OPENAI_API_KEY", None)
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            get_provider()

    def test_openai_empty_key_raises(self, monkeypatch):
        monkeypatch.setitem(config, "LLM_PROVIDER", "openai")
        monkeypatch.setitem(config, "OPENAI_API_KEY", "")
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            get_provider()

    def test_anthropic_missing_key_raises(self, monkeypatch):
        monkeypatch.setitem(config, "LLM_PROVIDER", "anthropic")
        monkeypatch.setitem(config, "ANTHROPIC_API_KEY", None)
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            get_provider()

    def test_anthropic_empty_key_raises(self, monkeypatch):
        monkeypatch.setitem(config, "LLM_PROVIDER", "anthropic")
        monkeypatch.setitem(config, "ANTHROPIC_API_KEY", "")
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            get_provider()

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setitem(config, "LLM_PROVIDER", "unknown_provider_xyz")
        with pytest.raises(ConfigurationError, match="Unknown LLM_PROVIDER"):
            get_provider()

    def test_openai_returns_openai_provider(self, monkeypatch):
        from llm_provider import OpenAIProvider
        monkeypatch.setitem(config, "LLM_PROVIDER", "openai")
        monkeypatch.setitem(config, "OPENAI_API_KEY", "sk-fake-key-for-test")
        provider = get_provider()
        assert isinstance(provider, OpenAIProvider)

    def test_anthropic_returns_anthropic_provider(self, monkeypatch):
        monkeypatch.setitem(config, "LLM_PROVIDER", "anthropic")
        monkeypatch.setitem(config, "ANTHROPIC_API_KEY", "sk-ant-fake-key")
        provider = get_provider()
        assert isinstance(provider, AnthropicProvider)


# ---------------------------------------------------------------------------
# MESSAGE NORMALIZATION tests (AnthropicProvider)
# ---------------------------------------------------------------------------

@pytest.fixture
def anthropic_provider(monkeypatch):
    """Create an AnthropicProvider with a fake key (no real API calls)."""
    monkeypatch.setitem(config, "ANTHROPIC_API_KEY", "sk-ant-fake-key-for-tests")
    return AnthropicProvider()


class TestAnthropicMessageNormalization:
    def test_merges_consecutive_user_messages(self, anthropic_provider):
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "user", "content": "Second message"},
        ]
        result = anthropic_provider._normalize_messages_for_anthropic(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First message\n\n---\n\nSecond message"

    def test_merges_three_consecutive_user_messages(self, anthropic_provider):
        messages = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = anthropic_provider._normalize_messages_for_anthropic(messages)
        assert len(result) == 1
        assert result[0]["content"] == "A\n\n---\n\nB\n\n---\n\nC"

    def test_does_not_merge_user_assistant_user(self, anthropic_provider):
        messages = [
            {"role": "user", "content": "User question"},
            {"role": "assistant", "content": "Assistant answer"},
            {"role": "user", "content": "User follow-up"},
        ]
        result = anthropic_provider._normalize_messages_for_anthropic(messages)
        assert len(result) == 3
        assert result[0]["content"] == "User question"
        assert result[1]["content"] == "Assistant answer"
        assert result[2]["content"] == "User follow-up"

    def test_single_user_message_unchanged(self, anthropic_provider):
        messages = [{"role": "user", "content": "Hello"}]
        result = anthropic_provider._normalize_messages_for_anthropic(messages)
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_empty_messages_returns_empty(self, anthropic_provider):
        assert anthropic_provider._normalize_messages_for_anthropic([]) == []

    def test_does_not_mutate_input(self, anthropic_provider):
        messages = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        original = [dict(m) for m in messages]
        anthropic_provider._normalize_messages_for_anthropic(messages)
        assert messages[0]["content"] == original[0]["content"]
        assert messages[1]["content"] == original[1]["content"]


class TestAnthropicSystemExtraction:
    def test_extracts_system_message(self, anthropic_provider):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        system, normalized = anthropic_provider._prepare_messages(messages)
        assert system == "You are a helpful assistant."
        assert len(normalized) == 1
        assert normalized[0]["role"] == "user"

    def test_system_not_in_normalized_list(self, anthropic_provider):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User msg 1"},
            {"role": "user", "content": "User msg 2"},
        ]
        system, normalized = anthropic_provider._prepare_messages(messages)
        assert system == "System prompt"
        # Two consecutive user messages merged into one
        assert len(normalized) == 1
        assert all(m["role"] != "system" for m in normalized)

    def test_no_system_message_returns_none(self, anthropic_provider):
        messages = [
            {"role": "user", "content": "Only user message"},
        ]
        system, normalized = anthropic_provider._prepare_messages(messages)
        assert system is None
        assert len(normalized) == 1

    def test_no_system_all_user_messages_merged(self, anthropic_provider):
        messages = [
            {"role": "user", "content": "Part 1"},
            {"role": "user", "content": "Part 2"},
        ]
        system, normalized = anthropic_provider._prepare_messages(messages)
        assert system is None
        assert len(normalized) == 1
        assert "\n\n---\n\n" in normalized[0]["content"]


# ---------------------------------------------------------------------------
# MOCK MODE tests
# ---------------------------------------------------------------------------

class TestMockMode:
    @pytest.fixture(autouse=True)
    def use_mock_mode(self, monkeypatch, tmp_path):
        monkeypatch.setitem(config, "QUERY_LLM", False)
        monkeypatch.setitem(config, "RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setitem(config, "LLM_PROVIDER", "openai")

    def test_returns_one_answer_per_prompt(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        answers = send_prompts(params, prompts)
        assert len(answers) == len(prompts)

    def test_returns_one_answer_per_prompt_multiple(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params) for _ in range(3)]
        # Make prompts distinct by tweaking the message content
        for i, p in enumerate(prompts):
            p.prompt["messages"][0]["content"] = f"Match attributes variant {i}"
        answers = send_prompts(params, prompts)
        assert len(answers) == 3

    def test_mock_answer_is_valid_answer_object(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        answers = send_prompts(params, prompts)
        assert all(isinstance(a, Answer) for a in answers)

    def test_mock_answer_contains_parseable_json(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        answers = send_prompts(params, prompts)
        for answer in answers:
            parsed = extract_json(answer)
            assert isinstance(parsed, dict)

    def test_mock_answer_is_valid_per_is_valid_answer(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        answers = send_prompts(params, prompts)
        assert all(is_valid_answer(a) for a in answers)

    def test_does_not_raise_without_api_keys(self, monkeypatch):
        monkeypatch.setitem(config, "OPENAI_API_KEY", None)
        monkeypatch.setitem(config, "ANTHROPIC_API_KEY", None)
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        # Must not raise — QUERY_LLM=False bypasses get_provider()
        answers = send_prompts(params, prompts)
        assert len(answers) == 1


# ---------------------------------------------------------------------------
# COST LOGGING tests
# ---------------------------------------------------------------------------

class TestCostLogging:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, tmp_path):
        self.results_dir = tmp_path / "results"
        monkeypatch.setitem(config, "QUERY_LLM", False)
        monkeypatch.setitem(config, "RESULTS_DIR", str(self.results_dir))
        monkeypatch.setitem(config, "LLM_PROVIDER", "openai")

    def _cost_log_path(self) -> Path:
        return self.results_dir / "cost_log.jsonl"

    def test_cost_log_written_after_mock_send(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        send_prompts(params, prompts)
        assert self._cost_log_path().exists(), "cost_log.jsonl was not created"

    def test_cost_log_has_at_least_one_line(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        send_prompts(params, prompts)
        lines = self._cost_log_path().read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1

    def test_cost_log_one_line_per_prompt(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params) for _ in range(3)]
        for i, p in enumerate(prompts):
            p.prompt["messages"][0]["content"] = f"variant {i}"
        send_prompts(params, prompts)
        lines = self._cost_log_path().read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_each_logged_line_is_valid_json(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        send_prompts(params, prompts)
        for line in self._cost_log_path().read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            assert isinstance(record, dict)

    def test_logged_line_has_required_keys(self):
        required_keys = {
            "timestamp",
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "latency_ms",
        }
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        send_prompts(params, prompts)
        for line in self._cost_log_path().read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            missing = required_keys - record.keys()
            assert not missing, f"Missing keys in cost log: {missing}"

    def test_logged_timestamp_is_string(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        send_prompts(params, prompts)
        line = self._cost_log_path().read_text(encoding="utf-8").strip().splitlines()[0]
        record = json.loads(line)
        assert isinstance(record["timestamp"], str)
        assert "T" in record["timestamp"]

    def test_logged_numeric_fields_are_numbers(self):
        params = _make_params()
        prompts = [_make_minimal_prompt(params)]
        send_prompts(params, prompts)
        line = self._cost_log_path().read_text(encoding="utf-8").strip().splitlines()[0]
        record = json.loads(line)
        assert isinstance(record["input_tokens"], int)
        assert isinstance(record["output_tokens"], int)
        assert isinstance(record["cost_usd"], float)
        assert isinstance(record["latency_ms"], int)
