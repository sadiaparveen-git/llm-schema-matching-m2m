# llm-schema-matching-m2m

LLM-based schema matching with many-to-many discovery and relation relatedness assessment — extending Parciak et al. (VLDB TaDA 2024)

![Python 3.11](https://img.shields.io/badge/python-3.11-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Overview

This project extends the one-to-one LLM schema matching methodology of Parciak et al. (2024) to support many-to-many attribute group discovery and relation-level relatedness assessment. It targets the MIMIC-III to OMOP CDM mapping domain and evaluates two LLMs  GPT-4-turbo (OpenAI) and Claude Sonnet (Anthropic) against a manually prepared ground truth of 42 entries covering 9 relation pairs and 15 transformation types.


**Research questions addressed:**

- **RQ1:** Can LLMs discover many-to-many schema matches beyond the one-to-one paradigm?
- **RQ2:** Can LLMs determine whether two relations are semantically related before attempting matching?

This work extends the methodology of:
> Parciak, M. et al. "Schema Matching with Large Language Models: an Experimental Study." VLDB TaDA Workshop, 2024.

---

## Repository Structure

```
llm-schema-matching-m2m/
├── models.py                  # Dataclasses: Attribute, Relation, and related types + AttributeGroupPair, ResultGroupPair, Result (extended)
├── ground_truth.py            # Ground truth loader, MatchType/RelationshipType enums, OMOP name normalization
├── pipeline.py                # Two-phase schema_match() orchestrator and compute_residuals()
├── llm_provider.py            # LLMProvider ABC + OpenAIProvider + AnthropicProvider + get_provider() factory
├── prompt_building.py         # build_prompts() (1:1), build_m2m_prompts(), build_relatedness_prompts()
├── prompt_postprocessing.py   # Postprocessors for 1:1, M:M group matches, and relatedness answers
├── prompt_sending.py          # Async dispatcher, JSON extraction, and answer validation utilities
├── storage_json.py            # JSON file-based persistence keyed by digest
├── evaluation.py              # evaluate_against_ground_truth(); per-relationship and per-relation-pair metrics
├── logging_config.py          # Structured logging setup, CostTracker and LatencyTimer context managers
├── config.py                  # Env-driven configuration loaded via python-dotenv
├── templates/
│   ├── oneToN.json            # 1-to-N matching prompt template
│   ├── nToOne.json            # N-to-1 matching prompt template
│   ├── manyToMany.json        # New: M:M group discovery prompt template
│   └── relationRelatedness.json  # New: relation-level relatedness pre-filter template
├── notebooks/
│   ├── generate_claude_baseline.ipynb       # Phase 1: 1:1 baseline with GPT-4 or Claude
│   ├── generate_relatedness_results.ipynb   # Phase 2: relation relatedness assessment (RQ2)
│   ├── generate_m2m_results.ipynb           # Phase 3+4: two-phase M:M pipeline at K=2 and K=3
│   └── analyze_m2m_quality.ipynb           # Analysis: heatmaps, K comparison, cost summary
├── tests/
│   ├── test_ground_truth.py   # Loader tests: 42 entries, typo normalization, OMOP map coverage
│   ├── test_models.py         # Digest stability and JSON round-trip for Result with group_pairs
│   ├── test_pipeline.py       # Mock-mode integration test for Patients->Person two-phase run
│   └── test_evaluation.py    # Hand-computed precision/recall against updated_ground_truth.csv
├── results/                   # Generated experiment outputs (gitignored)
│   ├── parameters/            # One JSON file per Parameters object, keyed by digest
│   ├── prompts/               # One JSON file per rendered Prompt
│   ├── answers/               # One JSON file per raw LLM Answer
│   ├── results/               # One JSON file per aggregated Result
│   ├── relatedness/           # One JSON file per RelationRelatednessResult
│   └── cost_log.jsonl         # Append-only token usage and cost records
├── pyproject.toml             # Package dependencies
├── .env.example               # Environment variable template
└── .gitignore                 # Excludes results/, .env, __pycache__, .pytest_cache
```

The ground truth benchmark lives at the project root:

```
../updated_ground_truth.csv    # 42 entries: 40 one-to-one + 2 many-to-one, 15 relationship types, 9 relation pairs
```

---

## Prerequisites

- Python 3.11
- OpenAI API key (for GPT-4-turbo baseline)
- Anthropic API key (for Claude cross-LLM comparison)
- Schema documentation files from Parciak et al. (2024) — required for attribute descriptions. These are **not included** in this repository. Obtain them from the paper's supplementary materials and place them at:

  ```
  ../schema_documentations/
  ```

---

## Installation

```bash
git clone https://github.com/sadiaparveen-git/llm-schema-matching-m2m
cd llm-schema-matching-m2m
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with the following values:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Key configuration options:

| Variable | Values | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` or `anthropic` | Selects which LLM backend to use |
| `QUERY_LLM` | `False` or `True` | `False` enables mock mode (no API calls); `True` makes real requests |
| `ANTHROPIC_N` | integer | Number of samples per prompt for Claude; use `1` for development, `3` for thesis runs |
| `PHASE_2_ENABLED` | `True` or `False` | `True` enables the M:M discovery phase on residual attributes |
| `MAX_GROUP_SIZE` | `2` or `3` | K value for M:M attribute grouping; controls combinatorial scope |

---

## Running the Experiments

Run notebooks in the following order. Always run in mock mode first (`QUERY_LLM=False` in `.env`) to verify the pipeline structure before making real API calls.

### 1. `notebooks/generate_claude_baseline.ipynb`

Phase 1 — 1:1 matching baseline. Reproduces Parciak et al.'s matching experiment using GPT-4-turbo or Claude as the LLM backend with the `oneToN` and `nToOne` prompt templates. Use this notebook to establish a cross-LLM 1:1 F1 baseline before running M:M experiments.

### 2. `notebooks/generate_relatedness_results.ipynb`

Phase 2 — relation relatedness assessment (RQ2). Runs the `relationRelatedness.json` template over all MIMIC-OMOP relation pair combinations to evaluate whether the LLM correctly identifies which pairs are candidates for attribute-level matching. Reports relation-level precision, recall, and F1.

### 3. `notebooks/generate_m2m_results.ipynb`

Phase 3 and 4 — two-phase M:M discovery pipeline. Phase 1 within this notebook runs 1:1 matching (same as the baseline notebook); Phase 2 runs the `manyToMany.json` template on residual attributes not resolved by Phase 1. Runs at `MAX_GROUP_SIZE=2` (Phase 3) and `MAX_GROUP_SIZE=3` (Phase 4) for comparison.

### 4. `notebooks/analyze_m2m_quality.ipynb`

Analysis — per-relation-pair F1 heatmaps, K=2 vs K=3 comparison, per-relationship-type accuracy breakdown, and cost summary from `results/cost_log.jsonl`.

> **Note:** Always set `QUERY_LLM=False` for the first run of any notebook. This exercises the full code path using mock responses so you can confirm the pipeline structure and output format before incurring API costs.

---

## Running Tests

```bash
pytest tests/ -v
```

All 112 tests must pass before running real experiments. The test suite covers ground truth loading (42 entries, typo normalization, OMOP name mapping), model digest stability and JSON round-trips, the mock-mode two-phase pipeline end-to-end, and hand-computed evaluation metrics.

---

## Key Design Decisions

| Decision | What | Why |
|---|---|---|
| Two-phase pipeline | Phase 1 runs 1:1 matching; Phase 2 runs M:M only on attributes not resolved in Phase 1 | Avoids combinatorial explosion (16,000+ group pairs at K=2 for the largest relation pair) |
| JSON file storage | Results cached as one file per digest under `results/`; no SQLite | Simpler than a database at this experiment volume; git-diffable, no schema migrations |
| Provider abstraction | `OpenAIProvider` and `AnthropicProvider` share a common `LLMProvider` ABC | Anthropic's API differs structurally (separate system message, no native `n>1`); abstraction isolates those quirks |
| Separate `ANTHROPIC_N` | Claude has no native multi-sample parameter; `n` parallel requests are issued | Keeps cost control independent from the OpenAI `n` parameter |

---

## Attribution

This project extends the methodology of:

> Parciak, M. et al. "Schema Matching with Large Language Models: an Experimental Study." VLDB TaDA Workshop, 2024.

Attribution is by paper citation only. No links to Marcel Parciak's repositories are included in this project.

---

## License

MIT
