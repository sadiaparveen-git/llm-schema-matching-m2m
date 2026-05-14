"""Evaluation metrics for schema matching results (TECH_SPEC §6.6).

evaluate_against_ground_truth() computes precision/recall/F1 for both 1:1 and
group (many-to-one / many-to-many) predictions against updated_ground_truth.csv.

NOTE: relationship types are semantic labels only — this module does NOT compute
relationship classification accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set, Tuple

from config import config
from ground_truth import GTGroupEntry, GTOneToOneEntry, MatchType, load_updated_ground_truth
from models import Result, ResultGroupPair, ResultPair, Vote

# A PairKey represents any match as (frozenset of source attr names,
# frozenset of target attr names).  1:1 matches use singleton frozensets.
_PairKey = Tuple[FrozenSet[str], FrozenSet[str]]


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    precision_1to1: float
    recall_1to1: float
    f1_1to1: float
    precision_group: float
    recall_group: float
    f1_group: float
    # keys are MatchType string values; inner dict has precision/recall/f1
    per_match_type: Dict[str, Dict[str, float]]
    # keys are "SourceRelation->TargetRelation"; inner dict has precision/recall/f1
    per_relation_pair: Dict[str, Dict[str, float]]
    total_predicted_1to1: int
    total_predicted_group: int
    total_gt_1to1: int
    total_gt_group: int


# ---------------------------------------------------------------------------
# Helper: precision / recall / F1
# ---------------------------------------------------------------------------

def _pr_f1(predicted: set, ground_truth: set) -> Tuple[float, float, float]:
    """Compute (precision, recall, F1) for exact-match prediction sets.

    Returns:
        (0.0, 0.0, 0.0) when both sets are empty.
        (0.0, 0.0, 0.0) when predicted is empty.
        (0.0, recall, 0.0) when precision is 0 — avoids div/0 in F1.
    """
    if not predicted:
        return 0.0, 0.0, 0.0
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth) if ground_truth else 0.0
    if precision == 0.0:
        return 0.0, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Private scoring helpers
# ---------------------------------------------------------------------------

def _score_pair(rp: ResultPair) -> float:
    """YES votes / total votes for a 1:1 ResultPair."""
    if not rp.votes:
        return 0.0
    return sum(1 for d in rp.votes if d.vote == Vote.YES) / len(rp.votes)


def _score_group(rgp: ResultGroupPair) -> float:
    """YES votes / total votes for a ResultGroupPair."""
    if not rgp.votes:
        return 0.0
    return sum(1 for d in rgp.votes if d.vote == Vote.YES) / len(rgp.votes)


# ---------------------------------------------------------------------------
# Private path helpers
# ---------------------------------------------------------------------------

def _attr_name(path: str) -> str:
    """Extract the attribute name (last dot-segment) from a dotted path.

    Works for both simple paths ('MIMIC.Patients.gender' → 'gender') and
    compound paths ('MIMIC.X.a + MIMIC.X.b' → 'b', last dot-segment).
    """
    return path.rsplit(".", 1)[-1]


def _relation_name(path: str) -> str:
    """Extract the relation (table) name from a dotted path.

    For compound sources, uses the first path component before '+'.
    'MIMIC.Patients.gender' → 'Patients'
    'MIMIC.Diagnoses_ICD.a + MIMIC.Diagnoses_ICD.b' → 'Diagnoses_ICD'
    """
    first = path.split("+")[0].strip()
    parts = first.split(".")
    return parts[1] if len(parts) >= 2 else first


# ---------------------------------------------------------------------------
# Private set builders
# ---------------------------------------------------------------------------

def _predicted_1to1(result: Result, threshold: float) -> Set[_PairKey]:
    return {
        (frozenset({ap.source.name}), frozenset({ap.target.name}))
        for ap, rp in result.pairs.items()
        if _score_pair(rp) >= threshold
    }


def _predicted_groups(result: Result, threshold: float) -> Set[_PairKey]:
    return {
        (
            frozenset(a.name for a in agp.sources),
            frozenset(a.name for a in agp.targets),
        )
        for agp, rgp in result.group_pairs.items()
        if _score_group(rgp) >= threshold
    }


def _gt_1to1_set(entries: list[GTOneToOneEntry]) -> Set[_PairKey]:
    return {
        (frozenset({_attr_name(e.source)}), frozenset({_attr_name(e.target)}))
        for e in entries
    }


def _gt_groups_set(entries: list[GTGroupEntry]) -> Set[_PairKey]:
    return {
        (
            frozenset(_attr_name(s) for s in e.sources),
            frozenset(_attr_name(t) for t in e.targets),
        )
        for e in entries
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_against_ground_truth(
    result: Result,
    ground_truth_path: str,
    match_score_threshold: Optional[float] = None,
) -> EvaluationReport:
    """Evaluate *result* against updated_ground_truth.csv and return an EvaluationReport.

    Args:
        result: The schema matching result to evaluate.
        ground_truth_path: Path to updated_ground_truth.csv.
        match_score_threshold: Override for the group-match threshold; defaults
            to config["MATCH_SCORE_THRESHOLD"] (0.5).
    """
    gt_1to1, gt_groups = load_updated_ground_truth(ground_truth_path)

    thr_1to1 = config["PHASE_1_CONFIDENCE"]
    thr_group = (
        match_score_threshold
        if match_score_threshold is not None
        else config["MATCH_SCORE_THRESHOLD"]
    )

    # ---- 1:1 evaluation ----
    pred_1to1 = _predicted_1to1(result, thr_1to1)
    gt_1to1_s = _gt_1to1_set(gt_1to1)
    p1, r1, f1_1 = _pr_f1(pred_1to1, gt_1to1_s)

    # ---- Group evaluation ----
    pred_grp = _predicted_groups(result, thr_group)
    gt_grp_s = _gt_groups_set(gt_groups)
    pg, rg, f1_g = _pr_f1(pred_grp, gt_grp_s)

    # ---- Per match-type breakdown ----
    per_match_type: Dict[str, Dict[str, float]] = {}

    # one_to_one: same as overall 1:1
    p_, r_, f_ = _pr_f1(pred_1to1, gt_1to1_s)
    per_match_type[str(MatchType.one_to_one)] = {"precision": p_, "recall": r_, "f1": f_}

    for mt in (MatchType.many_to_one, MatchType.one_to_many, MatchType.many_to_many):
        mt_gt = _gt_groups_set([e for e in gt_groups if e.match_type == mt])
        p_, r_, f_ = _pr_f1(pred_grp, mt_gt)
        per_match_type[str(mt)] = {"precision": p_, "recall": r_, "f1": f_}

    # ---- Per relation-pair breakdown ----
    # Build GT sets keyed by "SourceRelation->TargetRelation".
    rp_gt: Dict[str, Set[_PairKey]] = {}

    for entry in gt_1to1:
        key = f"{_relation_name(entry.source)}->{_relation_name(entry.target)}"
        rp_gt.setdefault(key, set()).add(
            (frozenset({_attr_name(entry.source)}), frozenset({_attr_name(entry.target)}))
        )

    for entry in gt_groups:
        key = f"{_relation_name(entry.sources[0])}->{_relation_name(entry.targets[0])}"
        rp_gt.setdefault(key, set()).add(
            (
                frozenset(_attr_name(s) for s in entry.sources),
                frozenset(_attr_name(t) for t in entry.targets),
            )
        )

    # The result covers exactly one relation pair; build its unified predictions.
    result_key = (
        f"{result.parameters.source_relation.name}"
        f"->{result.parameters.target_relation.name}"
    )
    result_pred: Set[_PairKey] = pred_1to1 | pred_grp

    per_relation_pair: Dict[str, Dict[str, float]] = {}
    for key, gt_set in rp_gt.items():
        preds = result_pred if key == result_key else set()
        p_, r_, f_ = _pr_f1(preds, gt_set)
        per_relation_pair[key] = {"precision": p_, "recall": r_, "f1": f_}

    return EvaluationReport(
        precision_1to1=p1,
        recall_1to1=r1,
        f1_1to1=f1_1,
        precision_group=pg,
        recall_group=rg,
        f1_group=f1_g,
        per_match_type=per_match_type,
        per_relation_pair=per_relation_pair,
        total_predicted_1to1=len(pred_1to1),
        total_predicted_group=len(pred_grp),
        total_gt_1to1=len(gt_1to1_s),
        total_gt_group=len(gt_grp_s),
    )
