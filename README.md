# thesis-extension

This package extends Marcel Parciak's LLM-based schema matching work (Phases 2–4 of the thesis).

**Phase 1** (baseline reproduction) runs Marcel's existing notebooks unchanged from `actual-repo/`.
This folder contains all code for:

- **RQ1 — Many-to-Many Discovery**: How can LLMs discover many-to-many schema matchings, where groups of source attributes jointly correspond to groups of target attributes? Evaluated against the 2 many-to-one entries in `updated_ground_truth.csv` using a two-phase pipeline (1:1 first, then M:M on residual attributes).

- **RQ2 — Relation Relatedness**: How can LLMs determine whether two relations should be considered semantically related and are candidates for attribute-level matching?

## Setup

```bash
cd thesis-extension/
cp .env.example .env
# Fill in OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env
pip install -e .
```

## Running tests

```bash
pytest
```

## Running experiments

See the notebooks in `notebooks/` for Phases 2–4. For Phase 1 (Marcel's baseline), run the notebooks in `../actual-repo/` directly.

## Ground truth

The authoritative benchmark is `../updated_ground_truth.csv` (42 entries: 40 one-to-one + 2 many-to-one, 15 relationship types, 9 relation pairs).
