"""S5 verification gate: pipeline.schema_match() + compute_residuals() tests.

All tests run offline (QUERY_LLM=False).  The isolated_results_dir fixture
redirects storage to a fresh tmp_path so the real results/ directory is
never touched.
"""
from __future__ import annotations

from typing import Dict, List

import pytest

import storage_json
from config import config
from models import (
    Attribute,
    AttributeGroupPair,
    AttributePair,
    Decision,
    Parameters,
    Relation,
    Result,
    ResultPair,
    Side,
    Vote,
)
from pipeline import compute_residuals, schema_match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attr(name: str, desc: str = "d", included: bool = True) -> Attribute:
    return Attribute(name=name, description=desc, included=included)


def _patients_relation() -> Relation:
    return Relation(
        name="Patients",
        side=Side.SOURCE,
        attributes=[
            _attr("anchor_year", "the patient's anchor year"),
            _attr("anchor_age", "the patient's anchor age"),
            _attr("anchor_year_group", "the patient's anchor year group"),
            _attr("dod", "date of death"),
            _attr("gender", "patient gender"),
            _attr("subject_id", "unique patient identifier"),
        ],
    )


def _person_relation() -> Relation:
    return Relation(
        name="Person",
        side=Side.TARGET,
        attributes=[
            _attr("year_of_birth", "person's year of birth"),
            _attr("gender_source_value", "gender source value"),
            _attr("person_source_value", "person source value"),
            _attr("death_datetime", "datetime of death"),
        ],
    )


def _patients_person_params() -> Parameters:
    return Parameters(
        source_relation=_patients_relation(),
        target_relation=_person_relation(),
        llm_model="mock-model",
    )


def _result_with_votes(
    params: Parameters,
    pair_votes: Dict,
) -> Result:
    """Build a Result with explicit vote lists for named attribute pairs."""
    result = Result(parameters=params)
    src_by = {a.name: a for a in params.source_relation.attributes}
    tgt_by = {a.name: a for a in params.target_relation.attributes}
    for (sn, tn), votes in pair_votes.items():
        s = src_by[sn]
        t = tgt_by[tn]
        pair = AttributePair(source=s, target=t)
        result.pairs[pair] = ResultPair(
            attributes=pair,
            votes=[Decision(vote=v, explanation="") for v in votes],
        )
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_results_dir(tmp_path, monkeypatch):
    """Redirect all JSON storage to an isolated temp directory.

    Prevents test runs from writing into thesis-extension/results/.
    config["RESULTS_DIR"] is restored automatically by monkeypatch.
    """
    results_dir = tmp_path / "results"
    monkeypatch.setitem(config, "RESULTS_DIR", str(results_dir))
    yield results_dir


# ---------------------------------------------------------------------------
# compute_residuals tests
# ---------------------------------------------------------------------------

class TestComputeResiduals:
    def test_empty_pairs_all_included_are_residuals(self):
        params = _patients_person_params()
        result = Result(parameters=params)

        residual_src, residual_tgt = compute_residuals(result, params, threshold=0.66)

        included_src = {a.name for a in params.source_relation.attributes if a.included}
        included_tgt = {a.name for a in params.target_relation.attributes if a.included}
        assert {a.name for a in residual_src} == included_src
        assert {a.name for a in residual_tgt} == included_tgt

    def test_excluded_attribute_never_residual(self):
        params = Parameters(
            source_relation=Relation(
                name="Src",
                side=Side.SOURCE,
                attributes=[
                    _attr("a", included=True),
                    _attr("b", included=False),
                ],
            ),
            target_relation=Relation(
                name="Tgt",
                side=Side.TARGET,
                attributes=[_attr("x")],
            ),
            llm_model="mock",
        )
        result = Result(parameters=params)

        residual_src, _ = compute_residuals(result, params, threshold=0.66)
        names = {a.name for a in residual_src}

        assert "b" not in names, "excluded attribute must never be a residual"
        assert "a" in names, "included attribute with zero votes must be a residual"

    def test_zero_votes_treated_as_residual(self):
        params = Parameters(
            source_relation=Relation(
                name="Src",
                side=Side.SOURCE,
                attributes=[_attr("a")],
            ),
            target_relation=Relation(
                name="Tgt",
                side=Side.TARGET,
                attributes=[_attr("x")],
            ),
            llm_model="mock",
        )
        result = Result(parameters=params)
        s = params.source_relation.attributes[0]
        t = params.target_relation.attributes[0]
        pair = AttributePair(source=s, target=t)
        result.pairs[pair] = ResultPair(attributes=pair, votes=[])

        residual_src, residual_tgt = compute_residuals(result, params, threshold=0.66)

        assert any(a.name == "a" for a in residual_src)
        assert any(a.name == "x" for a in residual_tgt)

    def test_high_yes_ratio_is_not_residual(self):
        params = Parameters(
            source_relation=Relation(
                name="Src",
                side=Side.SOURCE,
                attributes=[_attr("a")],
            ),
            target_relation=Relation(
                name="Tgt",
                side=Side.TARGET,
                attributes=[_attr("x")],
            ),
            llm_model="mock",
        )
        result = _result_with_votes(
            params, {("a", "x"): [Vote.YES, Vote.YES, Vote.YES]}
        )

        residual_src, residual_tgt = compute_residuals(result, params, threshold=0.66)

        assert not any(a.name == "a" for a in residual_src)
        assert not any(a.name == "x" for a in residual_tgt)

    def test_low_yes_ratio_is_residual(self):
        params = Parameters(
            source_relation=Relation(
                name="Src",
                side=Side.SOURCE,
                attributes=[_attr("a")],
            ),
            target_relation=Relation(
                name="Tgt",
                side=Side.TARGET,
                attributes=[_attr("x")],
            ),
            llm_model="mock",
        )
        result = _result_with_votes(
            params, {("a", "x"): [Vote.NO, Vote.NO, Vote.NO]}
        )

        residual_src, residual_tgt = compute_residuals(result, params, threshold=0.66)

        assert any(a.name == "a" for a in residual_src)
        assert any(a.name == "x" for a in residual_tgt)

    def test_source_and_target_residuals_computed_independently(self):
        """An attribute on one side can be non-residual while the other side is."""
        params = Parameters(
            source_relation=Relation(
                name="Src",
                side=Side.SOURCE,
                attributes=[_attr("a"), _attr("b")],
            ),
            target_relation=Relation(
                name="Tgt",
                side=Side.TARGET,
                attributes=[_attr("x"), _attr("y")],
            ),
            llm_model="mock",
        )
        # "a"→"x" has 100% YES; "b" has no votes; "y" has no votes
        result = _result_with_votes(
            params,
            {("a", "x"): [Vote.YES, Vote.YES, Vote.YES]},
        )

        residual_src, residual_tgt = compute_residuals(result, params, threshold=0.66)
        src_names = {a.name for a in residual_src}
        tgt_names = {a.name for a in residual_tgt}

        assert "a" not in src_names, "high-YES source must not be residual"
        assert "b" in src_names, "zero-vote source must be residual"
        assert "x" not in tgt_names, "high-YES target must not be residual"
        assert "y" in tgt_names, "zero-vote target must be residual"


# ---------------------------------------------------------------------------
# Mock mode end-to-end test (most important)
# ---------------------------------------------------------------------------

class TestMockModeEndToEnd:
    def test_patients_person_two_phase(self, isolated_results_dir, monkeypatch):
        monkeypatch.setitem(config, "QUERY_LLM", False)
        monkeypatch.setitem(config, "PHASE_2_ENABLED", True)
        params = _patients_person_params()

        result = schema_match(params)

        # Phase 1: mock 1:1 pairs populated
        assert result.pairs, "Phase 1 mock result must have non-empty pairs"

        # Phase 2: group_pairs populated
        assert result.group_pairs, "Phase 2 mock result must have non-empty group_pairs"

        # Find the group pair containing anchor_year and anchor_age as sources
        found_group = None
        for agp in result.group_pairs:
            src_names = {a.name for a in agp.sources}
            if "anchor_year" in src_names and "anchor_age" in src_names:
                found_group = agp
                break

        assert found_group is not None, (
            "Expected a group_pair with 'anchor_year' and 'anchor_age' in sources"
        )

        # That group's targets must include year_of_birth
        tgt_names = {a.name for a in found_group.targets}
        assert "year_of_birth" in tgt_names, (
            "Expected 'year_of_birth' in the group_pair's targets"
        )

        # The ResultGroupPair must carry at least one YES vote
        rgp = result.group_pairs[found_group]
        assert any(d.vote == Vote.YES for d in rgp.votes), (
            "Expected at least one YES vote in the ResultGroupPair"
        )


# ---------------------------------------------------------------------------
# Caching test
# ---------------------------------------------------------------------------

class TestCaching:
    def test_second_call_uses_cache_not_rewrite(self, isolated_results_dir, monkeypatch):
        monkeypatch.setitem(config, "QUERY_LLM", False)
        monkeypatch.setitem(config, "PHASE_2_ENABLED", True)
        params = _patients_person_params()

        # Track store_result calls via monkeypatching the module attribute
        store_calls: List[int] = []
        original_store = storage_json.store_result

        def tracking_store(result):
            store_calls.append(1)
            return original_store(result)

        monkeypatch.setattr(storage_json, "store_result", tracking_store)

        schema_match(params)

        # Result file must exist after first call
        json_file = isolated_results_dir / "results" / f"{params.digest()}.json"
        assert json_file.exists(), "Result JSON file must exist after first call"
        assert len(store_calls) == 1, "store_result should be called exactly once on first call"

        schema_match(params)

        # Cache hit: store_result must NOT be called a second time
        assert len(store_calls) == 1, (
            f"Expected store_result called 1 time total; got {len(store_calls)} "
            "(second call should hit cache)"
        )


# ---------------------------------------------------------------------------
# Phase 2 disabled test
# ---------------------------------------------------------------------------

class TestPhase2Disabled:
    def test_group_pairs_empty_when_phase2_disabled(self, isolated_results_dir, monkeypatch):
        monkeypatch.setitem(config, "QUERY_LLM", False)
        monkeypatch.setitem(config, "PHASE_2_ENABLED", False)
        params = _patients_person_params()

        result = schema_match(params)

        assert result.group_pairs == {}, (
            "group_pairs must be empty when PHASE_2_ENABLED=False, "
            "even for the Patients/Person pair"
        )
