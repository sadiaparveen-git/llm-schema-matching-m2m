"""S4 verification gate: postprocessing + template rendering tests.

All tests run fully offline — no real API calls are made.
"""
from __future__ import annotations

import json
from typing import List

import pytest

from models import (
    Answer,
    Attribute,
    AttributeGroupPair,
    Decision,
    Parameters,
    PromptAttributePair,
    Relation,
    RelationRelatednessResult,
    Result,
    ResultGroupPair,
    Side,
    Vote,
)
from prompt_building import build_m2m_prompts, build_relatedness_prompts
from prompt_postprocessing import (
    postprocess_m2m_answers,
    postprocess_relatedness_answers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attr(name: str, desc: str = "") -> Attribute:
    return Attribute(name=name, description=desc, included=True)


def _make_patients_person_params() -> Parameters:
    """Minimal Patients -> Person parameters matching the sprint task spec."""
    src = Relation(
        name="Patients",
        side=Side.SOURCE,
        description="MIMIC patients table",
        attributes=[
            _attr("anchor_year", "The year in which the patient turned anchor_age years old."),
            _attr("anchor_age", "Patient's age in the anchor_year."),
            _attr("anchor_year_group", "A range of 3 consecutive years the anchor_year falls in."),
            _attr("dod", "Date of death of the patient."),
            _attr("gender", "Patient gender."),
        ],
    )
    tgt = Relation(
        name="Person",
        side=Side.TARGET,
        description="OMOP CDM person table",
        attributes=[
            _attr("year_of_birth", "The year of birth of the person."),
            _attr("gender_source_value", "The source code for the gender of the person."),
            _attr("person_source_value", "An (encrypted) key derived from the person identifier in the source data."),
        ],
    )
    return Parameters(
        source_relation=src,
        target_relation=tgt,
        llm_model="gpt-4-turbo",
    )


def _mock_m2m_answer(parameters: Parameters, json_body: str) -> Answer:
    """Build a mock Answer whose .answer field contains the given JSON body."""
    all_sources = parameters.source_relation.attributes
    all_targets = parameters.target_relation.attributes
    full_text = f"Let me analyze the schema...\n\n{json_body}"
    return Answer(
        attributes=PromptAttributePair(sources=all_sources, targets=all_targets),
        answer=full_text,
        index=0,
        valid=True,
    )


# ---------------------------------------------------------------------------
# M:M POSTPROCESSING
# ---------------------------------------------------------------------------

class TestPostprocessM2mAnswers:

    def test_single_match_creates_one_group_pair(self):
        params = _make_patients_person_params()
        body = json.dumps({
            "matches": [
                {
                    "source_group": ["anchor_year", "anchor_age"],
                    "target_group": ["year_of_birth"],
                    "type": "many_to_one",
                    "reasoning": "birth year computed from anchor year minus anchor age",
                }
            ]
        })
        answer = _mock_m2m_answer(params, body)
        result = postprocess_m2m_answers(Result(parameters=params), params, [answer])

        assert len(result.group_pairs) == 1

    def test_sources_are_correct_attribute_objects(self):
        params = _make_patients_person_params()
        body = json.dumps({
            "matches": [
                {
                    "source_group": ["anchor_year", "anchor_age"],
                    "target_group": ["year_of_birth"],
                    "type": "many_to_one",
                    "reasoning": "birth year computed from anchor year minus anchor age",
                }
            ]
        })
        answer = _mock_m2m_answer(params, body)
        result = postprocess_m2m_answers(Result(parameters=params), params, [answer])

        group_pair_key = next(iter(result.group_pairs))
        src_names = {a.name for a in group_pair_key.sources}
        assert src_names == {"anchor_year", "anchor_age"}

        # Confirm the objects are the actual Attribute instances from parameters
        param_src_names = {a.name for a in params.source_relation.attributes}
        assert src_names.issubset(param_src_names)

    def test_targets_are_correct_attribute_objects(self):
        params = _make_patients_person_params()
        body = json.dumps({
            "matches": [
                {
                    "source_group": ["anchor_year", "anchor_age"],
                    "target_group": ["year_of_birth"],
                    "type": "many_to_one",
                    "reasoning": "birth year computed from anchor year minus anchor age",
                }
            ]
        })
        answer = _mock_m2m_answer(params, body)
        result = postprocess_m2m_answers(Result(parameters=params), params, [answer])

        group_pair_key = next(iter(result.group_pairs))
        tgt_names = {a.name for a in group_pair_key.targets}
        assert tgt_names == {"year_of_birth"}

    def test_result_group_pair_has_one_yes_vote(self):
        params = _make_patients_person_params()
        body = json.dumps({
            "matches": [
                {
                    "source_group": ["anchor_year", "anchor_age"],
                    "target_group": ["year_of_birth"],
                    "type": "many_to_one",
                    "reasoning": "birth year computed from anchor year minus anchor age",
                }
            ]
        })
        answer = _mock_m2m_answer(params, body)
        result = postprocess_m2m_answers(Result(parameters=params), params, [answer])

        rgp = next(iter(result.group_pairs.values()))
        assert len(rgp.votes) == 1
        assert rgp.votes[0].vote == Vote.YES

    def test_result_group_pair_has_no_relationship_attribute(self):
        params = _make_patients_person_params()
        body = json.dumps({
            "matches": [
                {
                    "source_group": ["anchor_year", "anchor_age"],
                    "target_group": ["year_of_birth"],
                    "type": "many_to_one",
                    "reasoning": "aggregation",
                }
            ]
        })
        answer = _mock_m2m_answer(params, body)
        result = postprocess_m2m_answers(Result(parameters=params), params, [answer])

        rgp = next(iter(result.group_pairs.values()))
        # ResultGroupPair must NOT have a relationship field
        assert not hasattr(rgp, "relationship"), (
            "ResultGroupPair should have no 'relationship' field — "
            "see sprint task spec and TECH_SPEC §4.2"
        )


# ---------------------------------------------------------------------------
# ACCUMULATION
# ---------------------------------------------------------------------------

class TestM2mAccumulation:

    def test_same_group_called_twice_accumulates_two_votes(self):
        params = _make_patients_person_params()
        body = json.dumps({
            "matches": [
                {
                    "source_group": ["anchor_year", "anchor_age"],
                    "target_group": ["year_of_birth"],
                    "type": "many_to_one",
                    "reasoning": "aggregation",
                }
            ]
        })
        answer1 = _mock_m2m_answer(params, body)
        answer2 = _mock_m2m_answer(params, body)

        empty_result = Result(parameters=params)
        result = postprocess_m2m_answers(empty_result, params, [answer1])
        result = postprocess_m2m_answers(result, params, [answer2])

        assert len(result.group_pairs) == 1, "Should not create a duplicate entry"
        rgp = next(iter(result.group_pairs.values()))
        assert len(rgp.votes) == 2

    def test_two_different_groups_create_two_entries(self):
        params = _make_patients_person_params()
        body = json.dumps({
            "matches": [
                {
                    "source_group": ["anchor_year", "anchor_age"],
                    "target_group": ["year_of_birth"],
                    "type": "many_to_one",
                    "reasoning": "aggregation",
                },
                {
                    "source_group": ["gender"],
                    "target_group": ["gender_source_value"],
                    "type": "one_to_one",
                    "reasoning": "direct copy",
                },
            ]
        })
        answer = _mock_m2m_answer(params, body)
        result = postprocess_m2m_answers(Result(parameters=params), params, [answer])

        assert len(result.group_pairs) == 2


# ---------------------------------------------------------------------------
# TEMPLATE RENDERING
# ---------------------------------------------------------------------------

class TestM2mTemplateRendering:

    def _make_5x5_params(self) -> Parameters:
        src_attrs = [_attr(f"src_attr_{i}", f"Source attribute {i}") for i in range(5)]
        tgt_attrs = [_attr(f"tgt_attr_{i}", f"Target attribute {i}") for i in range(5)]
        src = Relation(
            name="SourceTable",
            side=Side.SOURCE,
            description="A source relation with 5 attributes",
            attributes=src_attrs,
        )
        tgt = Relation(
            name="TargetTable",
            side=Side.TARGET,
            description="A target relation with 5 attributes",
            attributes=tgt_attrs,
        )
        return Parameters(source_relation=src, target_relation=tgt, llm_model="gpt-4-turbo")

    def test_build_m2m_prompts_returns_one_prompt(self):
        params = self._make_5x5_params()
        residual_src = params.source_relation.attributes
        residual_tgt = params.target_relation.attributes
        prompts = build_m2m_prompts(params, residual_src, residual_tgt, max_group_size=2)
        assert len(prompts) == 1

    def test_no_unrendered_jinja_variables(self):
        params = self._make_5x5_params()
        residual_src = params.source_relation.attributes
        residual_tgt = params.target_relation.attributes
        prompts = build_m2m_prompts(params, residual_src, residual_tgt, max_group_size=2)
        rendered_text = " ".join(
            m["content"] for m in prompts[0].prompt["messages"]
        )
        assert "{{" not in rendered_text, f"Unrendered Jinja variable found: {rendered_text}"

    def test_max_group_size_appears_in_rendered_text(self):
        params = self._make_5x5_params()
        residual_src = params.source_relation.attributes
        residual_tgt = params.target_relation.attributes
        prompts = build_m2m_prompts(params, residual_src, residual_tgt, max_group_size=2)
        rendered_text = " ".join(
            m["content"] for m in prompts[0].prompt["messages"]
        )
        assert "2" in rendered_text

    def test_only_residual_attributes_appear(self):
        params = self._make_5x5_params()
        # Use only first 3 as residuals; attrs 3 and 4 should NOT appear
        residual_src = params.source_relation.attributes[:3]
        residual_tgt = params.target_relation.attributes[:3]
        prompts = build_m2m_prompts(params, residual_src, residual_tgt, max_group_size=2)
        rendered_text = " ".join(
            m["content"] for m in prompts[0].prompt["messages"]
        )
        for attr in residual_src:
            assert attr.name in rendered_text
        # Attributes NOT in residuals should not appear (they are excluded)
        for attr in params.source_relation.attributes[3:]:
            assert attr.name not in rendered_text

    def test_non_residual_attributes_excluded(self):
        params = self._make_5x5_params()
        # Only include the first 2 source / first 2 target as residuals
        residual_src = params.source_relation.attributes[:2]
        residual_tgt = params.target_relation.attributes[:2]
        prompts = build_m2m_prompts(params, residual_src, residual_tgt, max_group_size=2)
        rendered_text = " ".join(
            m["content"] for m in prompts[0].prompt["messages"]
        )
        for attr in params.source_relation.attributes[2:]:
            assert attr.name not in rendered_text, (
                f"Non-residual attribute {attr.name!r} should not appear in rendered prompt"
            )


# ---------------------------------------------------------------------------
# RELATEDNESS POSTPROCESSING
# ---------------------------------------------------------------------------

class TestPostprocessRelatednessAnswers:

    def _make_relatedness_answer(self, json_body: str) -> Answer:
        dummy_attr = _attr("dummy")
        return Answer(
            attributes=PromptAttributePair(sources=[dummy_attr], targets=[dummy_attr]),
            answer=f"Step by step analysis...\n\n{json_body}",
            index=0,
            valid=True,
        )

    def test_related_true_high_confidence(self):
        body = '{"related": true, "confidence": "high", "reasoning": "test"}'
        answer = self._make_relatedness_answer(body)
        results = postprocess_relatedness_answers([answer])

        assert len(results) == 1
        assert results[0].related is True
        assert results[0].confidence == "high"
        assert results[0].reasoning == "test"

    def test_related_false_low_confidence(self):
        body = '{"related": false, "confidence": "low", "reasoning": "unrelated"}'
        answer = self._make_relatedness_answer(body)
        results = postprocess_relatedness_answers([answer])

        assert len(results) == 1
        assert results[0].related is False
        assert results[0].confidence == "low"
        assert results[0].reasoning == "unrelated"

    def test_returns_relation_relatedness_result_type(self):
        body = '{"related": true, "confidence": "medium", "reasoning": "partial match"}'
        answer = self._make_relatedness_answer(body)
        results = postprocess_relatedness_answers([answer])

        assert isinstance(results[0], RelationRelatednessResult)

    def test_empty_answers_returns_empty_list(self):
        results = postprocess_relatedness_answers([])
        assert results == []

    def test_invalid_json_answer_is_skipped(self):
        dummy_attr = _attr("dummy")
        bad_answer = Answer(
            attributes=PromptAttributePair(sources=[dummy_attr], targets=[dummy_attr]),
            answer="This has no JSON at all.",
            index=0,
            valid=False,
        )
        results = postprocess_relatedness_answers([bad_answer])
        assert results == []

    def test_multiple_answers_parsed_independently(self):
        bodies = [
            '{"related": true, "confidence": "high", "reasoning": "yes"}',
            '{"related": false, "confidence": "medium", "reasoning": "no"}',
        ]
        answers = [self._make_relatedness_answer(b) for b in bodies]
        results = postprocess_relatedness_answers(answers)

        assert len(results) == 2
        assert results[0].related is True
        assert results[1].related is False
