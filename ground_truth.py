from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Tuple


class MatchType(StrEnum):
    oneToOne = "one_to_one"
    manyToOne = "many_to_one"


class RelationshipType(StrEnum):
    corresponds = "corresponds"
    copied_as_source_value = "copied_as_source_value"
    linked_via_person_source_value_to_person_id = "linked_via_person_source_value_to_person_id"
    cast_to_date = "cast_to_date"
    standardized_via_vocabulary = "standardized_via_vocabulary"
    standardized_via_vocabulary_when_not_null = "standardized_via_vocabulary_when_not_null"
    linked_via_visit_source_value_to_visit_occurrence_id = "linked_via_visit_source_value_to_visit_occurrence_id"
    computed_as_min_datetime = "computed_as_min_datetime"
    computed_as_min_datetime_then_cast_to_date = "computed_as_min_datetime_then_cast_to_date"
    aggregation = "aggregation"
    transformed = "transformed"
    mapped_via_vocabulary = "mapped_via_vocabulary"
    standardized_to_route_concept_id = "standardized_to_route_concept_id"
    standardized_to_condition_concept_id = "standardized_to_condition_concept_id"
    standardized_to_source_concept_id = "standardized_to_source_concept_id"


@dataclass
class GTEntry:
    source: str
    target: str
    relationship: RelationshipType


@dataclass
class GTGroupEntry:
    sources: List[str]
    target: str
    relationship: RelationshipType


# CamelCase OMOP relation name → underscore-separated name used in schema documentation.
# Applied to the relation-name segment of every target path during loading.
# Any CamelCase token NOT listed here triggers a ValueError at load time.
OMOP_NAME_NORMALIZATION = {
    "VisitOccurrence": "Visit_Occurrence",
    "DrugExposure": "Drug_Exposure",
    "ConditionOccurrence": "Condition_Occurrence",
    "VisitDetail": "Visit_Detail",
    "CareSite": "Care_Site",
}


def _is_camel_case(token: str) -> bool:
    """Return True if token is CamelCase: no underscores and has uppercase after position 0."""
    return "_" not in token and any(c.isupper() for c in token[1:])


def _normalize_target(target: str) -> str:
    """Normalize the relation-name segment in an OMOP target path.

    Input:  'OMOP.VisitOccurrence.person_id'
    Output: 'OMOP.Visit_Occurrence.person_id'
    """
    parts = target.split(".")
    if len(parts) != 3:
        return target
    schema, relation, attribute = parts
    if _is_camel_case(relation):
        if relation not in OMOP_NAME_NORMALIZATION:
            raise ValueError(
                f"Unmapped CamelCase OMOP relation name {relation!r} in target {target!r}. "
                f"Add it to OMOP_NAME_NORMALIZATION in ground_truth.py."
            )
        relation = OMOP_NAME_NORMALIZATION[relation]
    return f"{schema}.{relation}.{attribute}"


def load_updated_ground_truth(path: str) -> Tuple[List[GTEntry], List[GTGroupEntry]]:
    """Load updated_ground_truth.csv and return (one_to_one entries, many_to_one entries).

    Normalizations applied:
    - 'ont_to_one' typo → 'one_to_one'
    - '+'-delimited sources split for many_to_one rows only
    - CamelCase OMOP relation names mapped via OMOP_NAME_NORMALIZATION
    """
    one_to_one: List[GTEntry] = []
    many_to_one: List[GTGroupEntry] = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_type = row["type"].strip()
            # Defensive normalization of known typo
            if raw_type == "ont_to_one":
                raw_type = "one_to_one"

            match_type = MatchType(raw_type)
            source_raw = row["source"].strip()
            target_raw = row["target"].strip()
            relationship = RelationshipType(row["relationship"].strip())
            normalized_target = _normalize_target(target_raw)

            if match_type == MatchType.manyToOne:
                sources = [s.strip() for s in source_raw.split("+")]
                many_to_one.append(GTGroupEntry(
                    sources=sources,
                    target=normalized_target,
                    relationship=relationship,
                ))
            else:
                one_to_one.append(GTEntry(
                    source=source_raw,
                    target=normalized_target,
                    relationship=relationship,
                ))

    return one_to_one, many_to_one
