"""Postprocessors for thesis-extension.

postprocess_answers()          — 1:1 / oneToN / nToOne vote aggregation
                                  (adapted from Marcel's demo-repo logic)
postprocess_m2m_answers()      — parses {"matches": [...]} JSON, populates
                                  result.group_pairs
postprocess_relatedness_answers() — parses {"related": bool, ...} JSON
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List

from models import (
    Answer,
    Attribute,
    AttributeGroupPair,
    AttributePair,
    Decision,
    Parameters,
    PromptDesign,
    RelationRelatednessResult,
    Result,
    ResultGroupPair,
    ResultPair,
    Vote,
)
from prompt_sending import extract_json, is_valid_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_outermost_json(text: str) -> dict:
    """Extract the first (outermost) JSON object from *text*.

    Marcel's extract_json uses rindex("{") which finds the innermost brace for
    nested JSON.  This function instead finds the first "{" and matches it to
    its corresponding "}" via brace counting, so nested structures like
    {"matches": [...]} are extracted correctly.
    """
    start = text.index("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                raw = text[start : i + 1].replace("'", '"')
                return json.loads(raw)
    raise ValueError("No complete JSON object found in answer text")


# ---------------------------------------------------------------------------
# 1:1 postprocessor  (vote-aggregation adapted from Marcel's demo-repo)
# ---------------------------------------------------------------------------

def postprocess_answers(
    parameters: Parameters,
    answers: List[Answer],
) -> Result:
    """Aggregate votes from oneToN / nToOne answers into a Result.

    For each valid answer the LLM returns {"yes": [...], "no": [...]} lists of
    attribute names.  A Decision(vote=YES/NO) is recorded for every name
    mentioned.  One ResultPair per (source, target) AttributePair is
    accumulated across all answers.

    Handles only oneToN and nToOne modes (the only modes used in
    thesis-extension Phase 1).
    """
    result = Result(parameters=parameters)

    src_by_name: Dict[str, Attribute] = {
        a.name: a for a in parameters.source_relation.attributes
    }
    tgt_by_name: Dict[str, Attribute] = {
        a.name: a for a in parameters.target_relation.attributes
    }

    for answer in answers:
        if not is_valid_answer(answer):
            logger.warning("Skipping invalid answer (no parseable JSON): %s", answer.digest()[:8])
            continue

        try:
            parsed = extract_json(answer)
        except Exception:
            logger.warning("JSON extraction failed for answer %s", answer.digest()[:8], exc_info=True)
            continue

        # Determine which side is the "single" side for this answer so we know
        # which attribute is the pivot.
        # oneToN prompt: sources list has one attribute; targets list has many.
        # nToOne prompt: sources list has many; targets list has one.
        src_attrs = answer.attributes.sources
        tgt_attrs = answer.attributes.targets

        # In oneToN mode the LLM lists target attribute names under yes/no.
        # In nToOne mode the LLM lists source attribute names under yes/no.
        if len(src_attrs) == 1:
            # oneToN: pivot is the single source; named attrs are targets
            pivot_src = src_attrs[0]
            named_lookup = tgt_by_name
            for vote_key, vote_val in [(Vote.YES, "yes"), (Vote.NO, "no")]:
                for name in parsed.get(vote_val, []):
                    tgt = named_lookup.get(name)
                    if tgt is None:
                        continue
                    pair = AttributePair(source=pivot_src, target=tgt)
                    if pair not in result.pairs:
                        result.pairs[pair] = ResultPair(attributes=pair, votes=[])
                    result.pairs[pair].votes.append(
                        Decision(vote=vote_key, explanation=name, answer=answer)
                    )
        else:
            # nToOne: pivot is the single target; named attrs are sources
            pivot_tgt = tgt_attrs[0]
            named_lookup = src_by_name
            for vote_key, vote_val in [(Vote.YES, "yes"), (Vote.NO, "no")]:
                for name in parsed.get(vote_val, []):
                    src = named_lookup.get(name)
                    if src is None:
                        continue
                    pair = AttributePair(source=src, target=pivot_tgt)
                    if pair not in result.pairs:
                        result.pairs[pair] = ResultPair(attributes=pair, votes=[])
                    result.pairs[pair].votes.append(
                        Decision(vote=vote_key, explanation=name, answer=answer)
                    )

    return result


# ---------------------------------------------------------------------------
# M:M postprocessor
# ---------------------------------------------------------------------------

def postprocess_m2m_answers(
    existing_result: Result,
    parameters: Parameters,
    answers: List[Answer],
) -> Result:
    """Parse {"matches": [...]} JSON and accumulate into existing_result.group_pairs.

    Each match entry must have:
        source_group: list of source attribute names
        target_group: list of target attribute names
        type:         structural match type string (informational only)
        reasoning:    optional explanation string

    For each match a Decision(vote=YES) is recorded.  If the AttributeGroupPair
    already exists in group_pairs the vote is appended (accumulation).

    NOTE: ResultGroupPair has NO relationship field — that label lives in the
    ground truth only and is never predicted by the LLM.
    """
    src_by_name: Dict[str, Attribute] = {
        a.name: a for a in parameters.source_relation.attributes
    }
    tgt_by_name: Dict[str, Attribute] = {
        a.name: a for a in parameters.target_relation.attributes
    }

    for answer in answers:
        try:
            parsed = _extract_outermost_json(answer.answer)
        except Exception:
            logger.warning(
                "M:M JSON extraction failed for answer %s",
                answer.digest()[:8],
                exc_info=True,
            )
            continue

        matches = parsed.get("matches", [])
        if not isinstance(matches, list):
            logger.warning("M:M answer has non-list 'matches' field: %s", answer.digest()[:8])
            continue

        for match in matches:
            src_names = match.get("source_group", [])
            tgt_names = match.get("target_group", [])

            src_attrs = frozenset(
                src_by_name[n] for n in src_names if n in src_by_name
            )
            tgt_attrs = frozenset(
                tgt_by_name[n] for n in tgt_names if n in tgt_by_name
            )

            if not src_attrs or not tgt_attrs:
                logger.warning(
                    "M:M match references unknown attribute names: src=%s tgt=%s",
                    src_names,
                    tgt_names,
                )
                continue

            group = AttributeGroupPair(sources=src_attrs, targets=tgt_attrs)
            reasoning = match.get("reasoning") or ""

            decision = Decision(vote=Vote.YES, explanation=reasoning)

            if group not in existing_result.group_pairs:
                existing_result.group_pairs[group] = ResultGroupPair(
                    attributes=group,
                    votes=[],
                )
            existing_result.group_pairs[group].votes.append(decision)

    return existing_result


# ---------------------------------------------------------------------------
# Relatedness postprocessor
# ---------------------------------------------------------------------------

def postprocess_relatedness_answers(
    answers: List[Answer],
) -> List[RelationRelatednessResult]:
    """Parse {"related": bool, "confidence": str, "reasoning": str} JSON.

    Returns one RelationRelatednessResult per valid answer.
    Answers that cannot be parsed are logged and skipped.
    """
    results: List[RelationRelatednessResult] = []

    for answer in answers:
        try:
            parsed = _extract_outermost_json(answer.answer)
        except Exception:
            logger.warning(
                "Relatedness JSON extraction failed for answer %s",
                answer.digest()[:8],
                exc_info=True,
            )
            continue

        related = parsed.get("related")
        confidence = parsed.get("confidence", "low")
        reasoning = parsed.get("reasoning", "")

        if not isinstance(related, bool):
            # LLMs occasionally return "true"/"false" strings
            if isinstance(related, str):
                related = related.strip().lower() == "true"
            else:
                logger.warning(
                    "Relatedness answer has unexpected 'related' value: %r (answer %s)",
                    related,
                    answer.digest()[:8],
                )
                continue

        # Derive relation names from the PromptAttributePair metadata when
        # available.  The answer carries attributes but not relation names
        # directly, so we use a placeholder that callers can override.
        results.append(
            RelationRelatednessResult(
                source_relation_name="",
                target_relation_name="",
                related=related,
                confidence=confidence,
                reasoning=reasoning,
            )
        )

    return results
