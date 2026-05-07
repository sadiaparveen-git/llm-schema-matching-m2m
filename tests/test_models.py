"""S2 verification gate: data model + storage tests."""
from __future__ import annotations

import json

import pytest

from config import config as runtime_config
from models import (
    Attribute,
    AttributeGroupPair,
    AttributePair,
    Decision,
    Parameters,
    Relation,
    Result,
    ResultGroupPair,
    ResultPair,
    Side,
    Vote,
)
from storage_json import (
    get_result_by_parameters,
    log_cost,
    store_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attr(name: str, desc: str = "d") -> Attribute:
    return Attribute(name=name, description=desc, included=True)


def _params() -> Parameters:
    src = Relation(
        name="Patients",
        side=Side.SOURCE,
        attributes=[_attr("subject_id"), _attr("anchor_year"), _attr("anchor_age")],
        description="MIMIC patients",
    )
    tgt = Relation(
        name="Person",
        side=Side.TARGET,
        attributes=[_attr("person_id"), _attr("year_of_birth"), _attr("gender_source_value")],
        description="OMOP person",
    )
    return Parameters(source_relation=src, target_relation=tgt, llm_model="gpt-4-turbo")


# ---------------------------------------------------------------------------
# Digest stability
# ---------------------------------------------------------------------------

def test_attribute_digest_stable_across_two_calls():
    a = _attr("foo", "bar")
    assert a.digest() == a.digest()


def test_two_attributes_with_identical_inputs_share_digest():
    assert _attr("foo", "bar").digest() == _attr("foo", "bar").digest()


def test_attribute_group_pair_digest_is_insertion_order_independent():
    a, b, c = _attr("a"), _attr("b"), _attr("c")
    g1 = AttributeGroupPair(sources=frozenset({a, b}), targets=frozenset({c}))
    # Different iterable (list with reversed order), same elements:
    g2 = AttributeGroupPair(sources=frozenset([b, a]), targets=frozenset([c]))
    assert g1.digest() == g2.digest()


def test_result_group_pair_digest_changes_when_votes_appended():
    a, b, c = _attr("a"), _attr("b"), _attr("c")
    g = AttributeGroupPair(sources=frozenset({a, b}), targets=frozenset({c}))
    rgp = ResultGroupPair(attributes=g, votes=[])
    initial = rgp.digest()
    rgp.votes.append(Decision(vote=Vote.YES, explanation="aggregation"))
    assert rgp.digest() != initial


# ---------------------------------------------------------------------------
# AttributeGroupPair: shape and dict-key behavior
# ---------------------------------------------------------------------------

def test_is_one_to_one_only_for_singleton_sides():
    a, b, c, d = _attr("a"), _attr("b"), _attr("c"), _attr("d")

    assert AttributeGroupPair(
        sources=frozenset({a}), targets=frozenset({b})
    ).is_one_to_one is True

    assert AttributeGroupPair(
        sources=frozenset({a, b}), targets=frozenset({c})
    ).is_one_to_one is False

    assert AttributeGroupPair(
        sources=frozenset({a}), targets=frozenset({b, c})
    ).is_one_to_one is False

    assert AttributeGroupPair(
        sources=frozenset({a, b}), targets=frozenset({c, d})
    ).is_one_to_one is False


def test_attribute_group_pair_eq_and_hash_consistent():
    a, b, c = _attr("a"), _attr("b"), _attr("c")
    g1 = AttributeGroupPair(sources=frozenset({a, b}), targets=frozenset({c}))
    g2 = AttributeGroupPair(sources=frozenset({a, b}), targets=frozenset({c}))
    assert g1 == g2
    assert hash(g1) == hash(g2)


def test_attribute_group_pair_works_as_dict_key():
    a, b, c = _attr("a"), _attr("b"), _attr("c")
    g1 = AttributeGroupPair(sources=frozenset({a, b}), targets=frozenset({c}))
    g2 = AttributeGroupPair(sources=frozenset({b, a}), targets=frozenset({c}))

    d = {g1: "value"}
    assert g2 in d
    assert d[g2] == "value"


# ---------------------------------------------------------------------------
# Result serialization round-trip
# ---------------------------------------------------------------------------

def _result_with_pairs_and_groups() -> tuple[Result, AttributePair, AttributeGroupPair]:
    params = _params()
    src = params.source_relation.attributes
    tgt = params.target_relation.attributes

    pair = AttributePair(source=src[0], target=tgt[0])
    rp = ResultPair(
        attributes=pair,
        votes=[Decision(vote=Vote.YES, explanation="match")],
        score=0.9,
    )

    group = AttributeGroupPair(
        sources=frozenset({src[1], src[2]}),  # anchor_year + anchor_age
        targets=frozenset({tgt[1]}),          # year_of_birth
    )
    rgp = ResultGroupPair(
        attributes=group,
        votes=[Decision(vote=Vote.YES, explanation="aggregation")],
        score=0.7,
    )

    result = Result(
        parameters=params,
        name="patients->person",
        pairs={pair: rp},
        group_pairs={group: rgp},
        meta={"phase": "test"},
    )
    return result, pair, group


def test_result_round_trips_with_both_pairs_and_group_pairs():
    result, pair, group = _result_with_pairs_and_groups()
    restored = Result.from_json(result.to_json())

    assert restored.name == "patients->person"
    assert restored.meta == {"phase": "test"}
    assert pair in restored.pairs
    assert restored.pairs[pair].score == 0.9
    assert restored.pairs[pair].votes[0].vote == Vote.YES

    assert group in restored.group_pairs
    assert restored.group_pairs[group].score == 0.7
    assert restored.group_pairs[group].votes[0].explanation == "aggregation"


def test_group_pairs_keys_reconstructed_after_deserialization():
    result, _, group = _result_with_pairs_and_groups()
    restored = Result.from_json(result.to_json())

    keys = list(restored.group_pairs.keys())
    assert len(keys) == 1
    assert keys[0] == group
    # The reconstructed key must hash and compare-equal to the original.
    assert hash(keys[0]) == hash(group)
    assert keys[0].digest() == group.digest()


def test_digests_stable_across_serialization():
    result, pair, group = _result_with_pairs_and_groups()
    raw = result.to_json()
    restored = Result.from_json(raw)

    # AttributeGroupPair digest is stable
    restored_group = next(iter(restored.group_pairs.keys()))
    assert restored_group.digest() == group.digest()

    # AttributePair digest is stable
    restored_pair = next(iter(restored.pairs.keys()))
    assert restored_pair.digest() == pair.digest()

    # Result digest is stable
    assert restored.digest() == result.digest()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_results_dir(tmp_path, monkeypatch):
    """Redirect storage_json's results directory to a tmp path for the test."""
    monkeypatch.setitem(runtime_config, "RESULTS_DIR", str(tmp_path))
    return tmp_path


def test_store_result_writes_file_at_expected_path(isolated_results_dir):
    params = _params()
    result = Result(parameters=params)

    path = store_result(result)
    expected = isolated_results_dir / "results" / f"{params.digest()}.json"
    assert path == expected
    assert path.exists()


def test_get_result_by_parameters_returns_none_for_unknown_digest(isolated_results_dir):
    assert get_result_by_parameters(_params()) is None


def test_get_result_by_parameters_returns_stored_result(isolated_results_dir):
    result, pair, group = _result_with_pairs_and_groups()
    store_result(result)

    loaded = get_result_by_parameters(result.parameters)
    assert loaded is not None
    assert loaded.name == "patients->person"
    assert pair in loaded.pairs
    assert loaded.pairs[pair].score == 0.9
    assert group in loaded.group_pairs
    assert loaded.group_pairs[group].score == 0.7


def test_log_cost_appends_two_lines_on_two_calls(isolated_results_dir):
    log_cost("digest_one", "openai", "gpt-4", 100, 50, 0.01, 1234)
    log_cost("digest_two", "anthropic", "claude-sonnet-4", 200, 80, 0.02, 2345)

    path = isolated_results_dir / "cost_log.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["prompt_digest"] == "digest_one"
    assert rec1["provider"] == "openai"
    assert rec1["input_tokens"] == 100
    assert rec2["prompt_digest"] == "digest_two"
    assert rec2["provider"] == "anthropic"
    assert rec2["latency_ms"] == 2345
