from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from pathlib import Path
from types import TracebackType
from typing import Optional, Type


def setup_logging(
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """Configure root logger with console (INFO) and optional file (DEBUG) handlers."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not root.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)


class LatencyTimer:
    """Context manager that measures wall-clock elapsed time in milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: int = 0

    def __enter__(self) -> "LatencyTimer":
        self._start = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.elapsed_ms = int((time.monotonic() - self._start) * 1000)


class CostTracker:
    """Context manager that appends a cost record to cost_log.jsonl on exit."""

    def __init__(
        self,
        cost_log_path: str,
        prompt_digest: str,
        provider: str,
        model: str,
    ) -> None:
        self._path = cost_log_path
        self.prompt_digest = prompt_digest
        self.provider = provider
        self.model = model
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cost_usd: float = 0.0
        self.latency_ms: int = 0
        self._timer = LatencyTimer()

    def __enter__(self) -> "CostTracker":
        self._timer.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._timer.__exit__(exc_type, exc_val, exc_tb)
        if self.latency_ms == 0:
            self.latency_ms = self._timer.elapsed_ms

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "prompt_digest": self.prompt_digest,
        }

        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
