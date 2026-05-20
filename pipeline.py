"""Two-phase schema matching pipeline.

schema_match() orchestrates Phase 1 (1:1 matching)
and Phase 2 (M:M on residuals). Results are cached
to disk by Parameters digest so repeated calls with
the same parameters are cheap.

compute_residuals() identifies attributes whose best
Phase 1 YES-vote ratio falls below a confidence
threshold; those attributes feed Phase 2.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from config import config
from models import (
    Attribute,
    AttributeGroupPair,
    AttributePair,
    Decision,
    Parameters,
    PromptDesign,
    Relation,
    Result,
    ResultGroupPair,
    ResultPair,
    Vote,
)
from prompt_building import build_m2m_prompts, build_prompts
from prompt_postprocessing import postprocess_answers, postprocess_m2m_answers
from prompt_sending import send_prompts
import storage_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def schema_match(parameters: Parameters) -> Result:
    """Run the two-phase schema matching pipeline and return a Result.

    Step 1: persist parameters.
    Step 2: return cached result if available.
    Step 3: in mock mode, build synthetic result and cache it.
    Step 4: Phase 1 — 1:1 matching via oneToN + nToOne prompts.
    Step 5: Phase 2 — M:M matching on residual attributes (if enabled).
    Step 6: persist result.
    Step 7: return result.
    """
    # Step 1
    storage_json.store_parameters(parameters)

    # Step 2 — cache hit
    cached = storage_json.get_result_by_parameters(parameters)
    if cached is not None:
        logger.info("Cache hit for parameters %s", parameters.digest()[:8])
        return cached

    # Step 3 — mock mode (QUERY_LLM=False)
    if not config["QUERY_LLM"]:
        logger.info("QUERY_LLM=False — generating mock result")
        result = _generate_mock_result(parameters)
        if config["PHASE_2_ENABLED"]:
            result.group_pairs = _generate_mock_m2m_result(parameters)
        storage_json.store_result(result)
        return result

    # Step 4 — Phase 1: 1:1 matching
    prompts_p1 = build_prompts(
        parameters,
        modes=[PromptDesign.oneToN, PromptDesign.nToOne],
    )
    if len(prompts_p1) > 10:
        n = (
            config["ANTHROPIC_N"]
            if config["LLM_PROVIDER"] == "anthropic"
            else config["OPENAI_N"]
        )
        estimated_calls = len(prompts_p1) * n
        logger.warning(
            "Phase 1 generated %d prompts for %s->%s. "
            "With current n setting this means ~%d API calls. "
            "Consider reducing OPENAI_N or ANTHROPIC_N in .env.",
            len(prompts_p1),
            parameters.source_relation.name,
            parameters.target_relation.name,
            estimated_calls,
        )
    answers_p1 = send_prompts(parameters, prompts_p1)
    result = postprocess_answers(parameters, answers_p1)

    # Step 5 — Phase 2: M:M on residuals
    if config["PHASE_2_ENABLED"]:
        residual_src, residual_tgt = compute_residuals(
            result, parameters, threshold=config["PHASE_1_CONFIDENCE"]
        )
        if residual_src and residual_tgt:
            prompts_p2 = build_m2m_prompts(
                parameters,
                residual_src,
                residual_tgt,
                max_group_size=config["MAX_GROUP_SIZE"],
            )
            answers_p2 = send_prompts(parameters, prompts_p2)
            result = postprocess_m2m_answers(result, parameters, answers_p2)

    # Step 6
    storage_json.store_result(result)
    # Step 7
    return result


def compute_residuals(
    result: Result,
    parameters: Parameters,
    threshold: float,
) -> Tuple[List[Attribute], List[Attribute]]:
    """Return (residual_sources, residual_targets) for Phase 2.

    An attribute is residual if its best YES-vote ratio across all 1:1 pairs
    is strictly below *threshold*.

    Edge cases:
    - included=False  → never a residual, skipped unconditionally.
    - zero total votes → max_yes_ratio = 0.0 → treated as residual.
    - empty result.pairs → all included attributes are residuals.
    """
    residual_sources: List[Attribute] = []
    for s in parameters.source_relation.attributes:
        if not s.included:
            continue
        pair_ratios: List[float] = []
        for t in parameters.target_relation.attributes:
            rp = result.pairs.get(AttributePair(source=s, target=t))
            if rp is None or len(rp.votes) == 0:
                continue
            yes = sum(1 for d in rp.votes if d.vote == Vote.YES)
            pair_ratios.append(yes / len(rp.votes))
        max_yes_ratio = max(pair_ratios) if pair_ratios else 0.0
        if max_yes_ratio < threshold:
            residual_sources.append(s)

    residual_targets: List[Attribute] = []
    for t in parameters.target_relation.attributes:
        if not t.included:
            continue
        pair_ratios = []
        for s in parameters.source_relation.attributes:
            rp = result.pairs.get(AttributePair(source=s, target=t))
            if rp is None or len(rp.votes) == 0:
                continue
            yes = sum(1 for d in rp.votes if d.vote == Vote.YES)
            pair_ratios.append(yes / len(rp.votes))
        max_yes_ratio = max(pair_ratios) if pair_ratios else 0.0
        if max_yes_ratio < threshold:
            residual_targets.append(t)

    return residual_sources, residual_targets


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_attr(relation: Relation, name: str) -> Optional[Attribute]:
    """Return the Attribute with the given name from *relation*, or None."""
    for attr in relation.attributes:
        if attr.name == name:
            return attr
    return None


def _generate_mock_result(parameters: Parameters) -> Result:
    """Synthetic 1:1 Result for offline / CI use (QUERY_LLM=False).

    For Patients→Person and Prescriptions→DrugExposure, return plausible
    mock pairs.  For all other relation pairs, return an empty-pairs Result.
    """
    result = Result(parameters=parameters)
    src_name = parameters.source_relation.name
    tgt_name = parameters.target_relation.name

    if src_name == "Patients" and tgt_name == "Person":
        candidate_pairs = [
            ("gender", "gender_source_value"),
            ("subject_id", "person_source_value"),
            ("dod", "death_datetime"),
        ]
    elif src_name == "Prescriptions" and tgt_name == "DrugExposure":
        candidate_pairs = [
            ("route", "route_concept_id"),
            ("dose_val_rx", "quantity"),
        ]
    else:
        return result

    for src_name_attr, tgt_name_attr in candidate_pairs:
        s = _find_attr(parameters.source_relation, src_name_attr)
        t = _find_attr(parameters.target_relation, tgt_name_attr)
        if s and t:
            pair = AttributePair(source=s, target=t)
            result.pairs[pair] = ResultPair(
                attributes=pair,
                votes=[
                    Decision(
                        vote=Vote.YES,
                        explanation=f"mock: {src_name_attr} -> {tgt_name_attr}",
                    )
                ],
                score=1.0,
            )

    return result


def _generate_mock_m2m_result(
    parameters: Parameters,
) -> Dict[AttributeGroupPair, ResultGroupPair]:
    """Synthetic M:M group matches for offline / CI use (QUERY_LLM=False).

    For Patients→Person returns the known aggregation:
        {anchor_year, anchor_age} → {year_of_birth}

    For all other relation pairs returns an empty dict.
    """
    src_name = parameters.source_relation.name
    tgt_name = parameters.target_relation.name

    if src_name == "Patients" and tgt_name == "Person":
        anchor_year = _find_attr(parameters.source_relation, "anchor_year")
        anchor_age = _find_attr(parameters.source_relation, "anchor_age")
        year_of_birth = _find_attr(parameters.target_relation, "year_of_birth")
        if anchor_year and anchor_age and year_of_birth:
            group = AttributeGroupPair(
                sources=frozenset({anchor_year, anchor_age}),
                targets=frozenset({year_of_birth}),
            )
            return {
                group: ResultGroupPair(
                    attributes=group,
                    votes=[
                        Decision(
                            vote=Vote.YES,
                            explanation="mock: anchor_year - anchor_age = year_of_birth",
                        )
                    ],
                )
            }

    return {}
