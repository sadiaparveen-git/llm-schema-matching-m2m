"""pytest configuration for thesis-extension tests.

Patches config["TEMPLATE_DIR"] to an absolute path so tests work regardless
of the working directory pytest is invoked from.
"""
from __future__ import annotations

import pathlib

import pytest

# Absolute path to thesis-extension/templates/, derived from this file's location.
_TEMPLATES_DIR = str(pathlib.Path(__file__).parent.parent / "templates")


@pytest.fixture(autouse=True)
def patch_template_dir(monkeypatch):
    """Ensure TEMPLATE_DIR always resolves correctly, and clear the lru_cache
    so _load_template re-reads templates with the patched path."""
    import prompt_building
    from config import config

    monkeypatch.setitem(config, "TEMPLATE_DIR", _TEMPLATES_DIR)
    prompt_building._load_template.cache_clear()
    yield
    prompt_building._load_template.cache_clear()
