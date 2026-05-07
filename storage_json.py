"""JSON file-based persistence for thesis-extension (replaces SQLite).

Layout under config["RESULTS_DIR"]:
    parameters/<digest>.json     — one Parameters per file
    prompts/<digest>.json        — one Prompt per file
    answers/<digest>.json        — one Answer per file
    results/<digest>.json        — one Result per file (keyed by Parameters.digest())
    relatedness/<digest>.json    — one RelationRelatednessResult per file
    cost_log.jsonl               — append-only cost/latency log
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from config import config
from models import (
    Answer,
    Parameters,
    Prompt,
    RelationRelatednessResult,
    Result,
)


def _results_dir() -> Path:
    """Read the results directory at call time so tests can monkeypatch it."""
    return Path(config["RESULTS_DIR"])


def _subdir(name: str) -> Path:
    p = _results_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def store_parameters(params: Parameters) -> Path:
    path = _subdir("parameters") / f"{params.digest()}.json"
    path.write_text(json.dumps(params.to_dict()), encoding="utf-8")
    return path


def get_parameters_by_hash(digest: str) -> Optional[Parameters]:
    path = _results_dir() / "parameters" / f"{digest}.json"
    if not path.exists():
        return None
    return Parameters.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def store_prompt(prompt: Prompt) -> Path:
    path = _subdir("prompts") / f"{prompt.digest()}.json"
    path.write_text(json.dumps(prompt.to_dict()), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

def store_answer(answer: Answer) -> Path:
    path = _subdir("answers") / f"{answer.digest()}.json"
    path.write_text(json.dumps(answer.to_dict()), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

def store_result(result: Result) -> Path:
    """Store a Result keyed by its Parameters digest (so cache lookup is O(1))."""
    path = _subdir("results") / f"{result.parameters.digest()}.json"
    path.write_text(result.to_json(), encoding="utf-8")
    return path


def get_result_by_parameters(params: Parameters) -> Optional[Result]:
    path = _results_dir() / "results" / f"{params.digest()}.json"
    if not path.exists():
        return None
    return Result.from_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Relatedness
# ---------------------------------------------------------------------------

def store_relatedness(result: RelationRelatednessResult) -> Path:
    path = _subdir("relatedness") / f"{result.digest()}.json"
    path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Cost log (append-only JSONL)
# ---------------------------------------------------------------------------

def log_cost(
    prompt_digest: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
) -> None:
    """Append one JSONL record to results/cost_log.jsonl."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_digest": prompt_digest,
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
    }
    base = _results_dir()
    base.mkdir(parents=True, exist_ok=True)
    path = base / "cost_log.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
