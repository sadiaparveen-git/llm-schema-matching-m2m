"""S6 verification gate: evaluation.py tests.

All tests run offline (QUERY_LLM=False).  The real updated_ground_truth.csv
is loaded for group and per-relation-pair tests.  Synthetic Results are built
directly — no LLM calls, no schema_match() invocations.
"""
from __future__ import annotations

import pathlib

import pytest

from config import config
from evaluation import EvaluationReport, _pr_f1, evaluate_against_ground_truth
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

# Absolute path to updated_ground_truth.csv at the project root.
_GT_PATH = str(pathlib.Path(__file__).parent.parent.parent / "updated_ground_truth.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attr(name: str, desc: str = "d") -> Attribute:
    return Attribute(name=name, description=desc)


def _yes() -> Decision:
    return Decision(vote=Vote.YES, explanation="")


def _no() -> Decision:
    return Decision(vote=Vote.NO, explanation="")


def _patients_relation() -> Relation:
    return Relation(
        name="Patients",
        side=Side.SOURCE,
        attributes=[
            _attr("anchor_year", "anchor year"),
            _attr("anchor_age", "anchor age"),
            _attr("gender", "patient gender"),
            _attr("subject_id", "patient id"),
            _attr("dod", "date of death"),
        ],
    )


def _person_relation() -> Relation:
    return Relation(
        name="Person",
        side=Side.TARGET,
        attributes=[
            _attr("year_of_birth", "year of birth"),
            _attr("gender_source_value", "gender source"),
            _attr("person_source_value", "person source"),
            _attr("gender_concept_id", "gender concept"),
        ],
    )


def _patients_person_params() -> Parameters:
    return Parameters(
        source_relation=_patients_relation(),
        target_relation=_person_relation(),
        llm_model="mock",
    )


def _result_with_pairs(params: Parameters, pairs: dict) -> Result:
    """Build a Result with explicit (vote_list) per (src_name, tgt_name)."""
    result = Result(parameters=params)
    src = {a.name: a for a in params.source_relation.attributes}
    tgt = {a.name: a for a in params.target_relation.attributes}
    for (sn, tn), votes in pairs.items():
        ap = AttributePair(source=src[sn], target=tgt[tn])
        result.pairs[ap] = ResultPair(attributes=ap, votes=votes)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_results_dir(tmp_path, monkeypatch):
    """Redirect storage to a temp directory so thesis-extension/results/ is untouched."""
    results_dir = tmp_path / "results"
    monkeypatch.setitem(config, "RESULTS_DIR", str(results_dir))
    yield results_dir


# ---------------------------------------------------------------------------
# _pr_f1 helper tests
# ---------------------------------------------------------------------------

class TestPrF1:
    def test_perfect_prediction(self):
        pred = {("a", "x"), ("b", "y")}
        gt = {("a", "x"), ("b", "y")}
        p, r, f = _pr_f1(pred, gt)
        assert p == 1.0
        assert r == 1.0
        assert f == 1.0

    def test_empty_predicted_returns_zeros(self):
        p, r, f = _pr_f1(set(), {("a", "x")})
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_both_empty_returns_zeros(self):
        p, r, f = _pr_f1(set(), set())
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_partial_overlap(self):
        pred = {("a", "x"), ("b", "y"), ("c", "z")}
        gt   = {("a", "x"), ("b", "y"), ("d", "w")}
        p, r, f = _pr_f1(pred, gt)
        assert pytest.approx(p) == 2 / 3
        assert pytest.approx(r) == 2 / 3
        assert pytest.approx(f) == 2 / 3

    def test_no_overlap_returns_zeros(self):
        pred = {("a", "x")}
        gt   = {("b", "y")}
        p, r, f = _pr_f1(pred, gt)
        assert p == 0.0
        assert f == 0.0

    def test_threshold_at_boundary_gte(self):
        """Precision=0 path: no TPs, recall is still returned correctly."""
        pred = {("a", "x")}
        gt   = {("b", "y")}
        p, r, f = _pr_f1(pred, gt)
        assert p == 0.0
        assert r == 0.0  # 0 TP out of 1 GT item
        assert f == 0.0


# ---------------------------------------------------------------------------
# 1:1 evaluation tests
# ---------------------------------------------------------------------------

class TestOneToOneEvaluation:
    def test_perfect_1to1_match(self):
        """Both gender->gender_source_value and subject_id->person_source_value are in GT."""
        params = _patients_person_params()
        result = _result_with_pairs(params, {
            ("gender", "gender_source_value"): [_yes(), _yes(), _yes()],  # score 1.0
            ("subject_id", "person_source_value"): [_yes(), _yes(), _yes()],  # score 1.0
        })
        report = evaluate_against_ground_truth(result, _GT_PATH)
        # Both predicted pairs are in GT → precision = 1.0
        assert report.precision_1to1 == pytest.approx(1.0)
        # total_gt_1to1 = 35: the GT's 41 one_to_one rows deduplicate to 35 unique
        # (attr_name, attr_name) pairs because subject_id→person_id appears 7 times.
        assert report.total_gt_1to1 == 35
        assert report.recall_1to1 == pytest.approx(2 / 35)
        assert report.f1_1to1 > 0.0
        assert report.total_predicted_1to1 == 2

    def test_pair_below_threshold_not_counted(self):
        """score 0.333 < 0.66 → NOT a true positive even though it is in GT."""
        params = _patients_person_params()
        result = _result_with_pairs(params, {
            ("gender", "gender_source_value"): [_yes(), _yes(), _yes()],  # 1.0
            ("subject_id", "person_source_value"): [_yes(), _no(), _no()],  # 0.333
        })
        report = evaluate_against_ground_truth(result, _GT_PATH)
        # Only gender->gender_source_value crosses threshold
        assert report.total_predicted_1to1 == 1
        assert report.precision_1to1 == pytest.approx(1.0)

    def test_pair_at_threshold_counted(self):
        """score exactly 0.66 (2/3) must be counted (>= not >)."""
        params = _patients_person_params()
        # 2 YES out of 3 → 0.6666... which is >= 0.66
        result = _result_with_pairs(params, {
            ("gender", "gender_source_value"): [_yes(), _yes(), _no()],  # 0.667
        })
        report = evaluate_against_ground_truth(result, _GT_PATH)
        assert report.total_predicted_1to1 == 1
        assert report.precision_1to1 == pytest.approx(1.0)

    def test_empty_result_returns_all_zeros(self):
        params = _patients_person_params()
        result = Result(parameters=params)
        report = evaluate_against_ground_truth(result, _GT_PATH)
        assert report.precision_1to1 == 0.0
        assert report.recall_1to1 == 0.0
        assert report.f1_1to1 == 0.0
        assert report.precision_group == 0.0
        assert report.recall_group == 0.0
        assert report.f1_group == 0.0
        assert report.total_predicted_1to1 == 0
        assert report.total_predicted_group == 0

    def test_hand_computed_precision_recall_f1(self):
        """Three predictions: 2 TPs, 1 FP.  Hand-computed values asserted."""
        params = _patients_person_params()
        # gender->gender_source_value: in GT ✓
        # gender->gender_concept_id: in GT ✓
        # dod->person_source_value: NOT in GT ✗
        result = _result_with_pairs(params, {
            ("gender", "gender_source_value"): [_yes(), _yes(), _yes()],  # 1.0
            ("gender", "gender_concept_id"): [_yes(), _yes(), _yes()],    # 1.0
            ("dod", "person_source_value"): [_yes(), _yes(), _yes()],      # 1.0 but not in GT
        })
        report = evaluate_against_ground_truth(result, _GT_PATH)
        # precision = 2/3 (2 TPs out of 3 predictions)
        # recall = 2/35 (GT deduplicates to 35 unique pairs; see test_perfect_1to1_match)
        assert report.precision_1to1 == pytest.approx(2 / 3)
        assert report.recall_1to1 == pytest.approx(2 / 35)
        expected_f1 = 2 * (2 / 3) * (2 / 35) / ((2 / 3) + (2 / 35))
        assert report.f1_1to1 == pytest.approx(expected_f1)


# ---------------------------------------------------------------------------
# Group evaluation tests (most important per spec)
# ---------------------------------------------------------------------------

class TestGroupEvaluation:
    def _build_group_result(self, params: Parameters) -> Result:
        """Build a Result with the known Patients/Person group pair."""
        result = Result(parameters=params)
        src = {a.name: a for a in params.source_relation.attributes}
        tgt = {a.name: a for a in params.target_relation.attributes}
        anchor_year = src["anchor_year"]
        anchor_age = src["anchor_age"]
        year_of_birth = tgt["year_of_birth"]
        group = AttributeGroupPair(
            sources=frozenset({anchor_year, anchor_age}),
            targets=frozenset({year_of_birth}),
        )
        result.group_pairs[group] = ResultGroupPair(
            attributes=group,
            votes=[_yes(), _yes()],  # 2 YES out of 2 → score 1.0
        )
        return result

    def test_known_group_is_true_positive(self):
        """anchor_year+anchor_age → year_of_birth must be counted as TP."""
        params = _patients_person_params()
        result = self._build_group_result(params)
        report = evaluate_against_ground_truth(result, _GT_PATH)
        # GT has 2 group entries; we predict 1 (the Patients/Person entry)
        assert report.recall_group > 0.0
        assert report.f1_group > 0.0

    def test_group_precision_recall_f1_values(self):
        """Hand-computed: 1 correct prediction out of 2 GT group entries."""
        params = _patients_person_params()
        result = self._build_group_result(params)
        report = evaluate_against_ground_truth(result, _GT_PATH)
        # precision = 1/1 = 1.0 (only one group predicted)
        assert report.precision_group == pytest.approx(1.0)
        # recall = 1/2 = 0.5 (2 GT group entries, 1 matched)
        assert report.recall_group == pytest.approx(0.5)
        # F1 = 2*1.0*0.5 / 1.5 = 2/3
        assert report.f1_group == pytest.approx(2 / 3)
        assert report.total_predicted_group == 1
        assert report.total_gt_group == 2

    def test_group_below_threshold_not_counted(self):
        """A group with score < 0.5 must not be a true positive."""
        params = _patients_person_params()
        result = Result(parameters=params)
        src = {a.name: a for a in params.source_relation.attributes}
        tgt = {a.name: a for a in params.target_relation.attributes}
        group = AttributeGroupPair(
            sources=frozenset({src["anchor_year"], src["anchor_age"]}),
            targets=frozenset({tgt["year_of_birth"]}),
        )
        # 0 YES out of 2 → score 0.0 < 0.5
        result.group_pairs[group] = ResultGroupPair(
            attributes=group,
            votes=[_no(), _no()],
        )
        report = evaluate_against_ground_truth(result, _GT_PATH)
        assert report.total_predicted_group == 0
        assert report.recall_group == 0.0
        assert report.f1_group == 0.0

    def test_group_at_threshold_counted(self):
        """score exactly 0.5 (1 YES out of 2) must be counted (>= not >)."""
        params = _patients_person_params()
        result = Result(parameters=params)
        src = {a.name: a for a in params.source_relation.attributes}
        tgt = {a.name: a for a in params.target_relation.attributes}
        group = AttributeGroupPair(
            sources=frozenset({src["anchor_year"], src["anchor_age"]}),
            targets=frozenset({tgt["year_of_birth"]}),
        )
        result.group_pairs[group] = ResultGroupPair(
            attributes=group,
            votes=[_yes(), _no()],  # score = 0.5 exactly
        )
        report = evaluate_against_ground_truth(result, _GT_PATH)
        assert report.total_predicted_group == 1
        assert report.recall_group > 0.0

    def test_custom_threshold_overrides_config(self):
        """match_score_threshold=0.9 rejects a group with score 0.5."""
        params = _patients_person_params()
        result = Result(parameters=params)
        src = {a.name: a for a in params.source_relation.attributes}
        tgt = {a.name: a for a in params.target_relation.attributes}
        group = AttributeGroupPair(
            sources=frozenset({src["anchor_year"], src["anchor_age"]}),
            targets=frozenset({tgt["year_of_birth"]}),
        )
        result.group_pairs[group] = ResultGroupPair(
            attributes=group,
            votes=[_yes(), _no()],  # score 0.5
        )
        report = evaluate_against_ground_truth(result, _GT_PATH, match_score_threshold=0.9)
        assert report.total_predicted_group == 0


# ---------------------------------------------------------------------------
# Per relation-pair breakdown tests
# ---------------------------------------------------------------------------

class TestPerRelationPair:
    def test_all_gt_relation_pairs_present_as_keys(self):
        """per_relation_pair must contain exactly the keys present in the GT."""
        params = _patients_person_params()
        result = Result(parameters=params)
        report = evaluate_against_ground_truth(result, _GT_PATH)
        keys = set(report.per_relation_pair.keys())
        # GT has 9 relation pairs; all must appear
        assert "Patients->Person" in keys
        assert "Admissions->Visit_Occurrence" in keys
        assert "Admissions->Death" in keys
        assert "Prescriptions->Drug_Exposure" in keys
        assert "Diagnoses_ICD->Condition_Occurrence" in keys
        assert "Transfers->Care_Site" in keys
        assert "Transfers->Visit_Detail" in keys
        assert "Admissions->Visit_Detail" in keys
        assert "Services->Visit_Detail" in keys
        assert len(keys) == 9

    def test_pair_with_no_predictions_has_zero_metrics(self):
        """Pairs not covered by the result have P=R=F1=0."""
        params = _patients_person_params()
        result = Result(parameters=params)
        report = evaluate_against_ground_truth(result, _GT_PATH)
        # Admissions->Visit_Occurrence: not the result's pair → no predictions
        m = report.per_relation_pair["Admissions->Visit_Occurrence"]
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_result_pair_gets_its_predictions(self):
        """The result's own pair gets non-zero recall when predictions are correct."""
        params = _patients_person_params()
        result = _result_with_pairs(params, {
            ("gender", "gender_source_value"): [_yes(), _yes(), _yes()],
        })
        report = evaluate_against_ground_truth(result, _GT_PATH)
        m = report.per_relation_pair["Patients->Person"]
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] > 0.0
        assert m["f1"] > 0.0


# ---------------------------------------------------------------------------
# Per match-type breakdown tests
# ---------------------------------------------------------------------------

class TestPerMatchType:
    def test_all_four_match_types_present(self):
        params = _patients_person_params()
        result = Result(parameters=params)
        report = evaluate_against_ground_truth(result, _GT_PATH)
        assert "one_to_one" in report.per_match_type
        assert "many_to_one" in report.per_match_type
        assert "one_to_many" in report.per_match_type
        assert "many_to_many" in report.per_match_type

    def test_one_to_one_metrics_match_overall(self):
        """per_match_type[one_to_one] must equal the overall 1:1 metrics."""
        params = _patients_person_params()
        result = _result_with_pairs(params, {
            ("gender", "gender_source_value"): [_yes(), _yes(), _yes()],
        })
        report = evaluate_against_ground_truth(result, _GT_PATH)
        m = report.per_match_type["one_to_one"]
        assert m["precision"] == pytest.approx(report.precision_1to1)
        assert m["recall"] == pytest.approx(report.recall_1to1)
        assert m["f1"] == pytest.approx(report.f1_1to1)

    def test_many_to_one_type_reflects_group_match(self):
        """When a group match is a true positive, many_to_one recall > 0."""
        params = _patients_person_params()
        result = Result(parameters=params)
        src = {a.name: a for a in params.source_relation.attributes}
        tgt = {a.name: a for a in params.target_relation.attributes}
        group = AttributeGroupPair(
            sources=frozenset({src["anchor_year"], src["anchor_age"]}),
            targets=frozenset({tgt["year_of_birth"]}),
        )
        result.group_pairs[group] = ResultGroupPair(
            attributes=group,
            votes=[_yes(), _yes()],
        )
        report = evaluate_against_ground_truth(result, _GT_PATH)
        m = report.per_match_type["many_to_one"]
        assert m["recall"] > 0.0
        assert m["f1"] > 0.0

    def test_one_to_many_and_many_to_many_zeros_when_no_gt(self):
        """GT has no one_to_many or many_to_many entries → recall must be 0."""
        params = _patients_person_params()
        result = Result(parameters=params)
        report = evaluate_against_ground_truth(result, _GT_PATH)
        assert report.per_match_type["one_to_many"]["recall"] == 0.0
        assert report.per_match_type["many_to_many"]["recall"] == 0.0


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_result_all_zeros(self):
        params = _patients_person_params()
        result = Result(parameters=params)
        report = evaluate_against_ground_truth(result, _GT_PATH)
        assert report.precision_1to1 == 0.0
        assert report.recall_1to1 == 0.0
        assert report.f1_1to1 == 0.0
        assert report.precision_group == 0.0
        assert report.recall_group == 0.0
        assert report.f1_group == 0.0
        assert report.total_predicted_1to1 == 0
        assert report.total_predicted_group == 0
        assert report.total_gt_1to1 == 35  # 41 raw one_to_one rows deduplicate to 35 unique pairs
        assert report.total_gt_group == 2

    def test_score_exactly_at_1to1_threshold_is_counted(self):
        """2/3 ≈ 0.6667 which is >= 0.66 → must be counted."""
        params = _patients_person_params()
        result = _result_with_pairs(params, {
            ("gender", "gender_source_value"): [_yes(), _yes(), _no()],  # 0.667
        })
        report = evaluate_against_ground_truth(result, _GT_PATH)
        assert report.total_predicted_1to1 == 1

    def test_score_just_below_1to1_threshold_is_not_counted(self):
        """1/3 ≈ 0.333 < 0.66 → must NOT be counted."""
        params = _patients_person_params()
        result = _result_with_pairs(params, {
            ("gender", "gender_source_value"): [_yes(), _no(), _no()],  # 0.333
        })
        report = evaluate_against_ground_truth(result, _GT_PATH)
        assert report.total_predicted_1to1 == 0
