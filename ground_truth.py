from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Tuple


class MatchType(StrEnum):
    one_to_one = "one_to_one"
    one_to_many = "one_to_many"
    many_to_one = "many_to_one"
    many_to_many = "many_to_many"


class RelationshipType(StrEnum):
    """Semantic relationship labels from updated_ground_truth.csv.

    Documentation only — these labels are NOT predicted by the LLM and NOT
    compared against ground truth during evaluation. The enum exists solely to
    parse and validate the CSV's `relationship` column.
    """
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
class GTOneToOneEntry:
    """A 1:1 ground truth entry. Match type is implicit (always one_to_one)."""
    source: str
    target: str
    relationship: RelationshipType


@dataclass
class GTGroupEntry:
    """A non-1:1 ground truth entry: one_to_many, many_to_one, or many_to_many."""
    sources: List[str]
    targets: List[str]
    match_type: MatchType
    relationship: RelationshipType


# Maps CamelCase OMOP relation names to their
# underscore-separated form used in schema
# documentation files. Any CamelCase token not
# listed here raises ValueError at load time.
OMOP_NAME_NORMALIZATION = {
    "VisitOccurrence": "Visit_Occurrence",
    "DrugExposure": "Drug_Exposure",
    "ConditionOccurrence": "Condition_Occurrence",
    "VisitDetail": "Visit_Detail",
    "CareSite": "Care_Site",
}


def _is_camel_case(token: str) -> bool:
    """True if token is CamelCase: no underscores and has uppercase after position 0."""
    return "_" not in token and any(c.isupper() for c in token[1:])


def _normalize_relation_path(path: str) -> str:
    """Apply OMOP_NAME_NORMALIZATION to the relation-name segment of a dotted path.

    Input:  'OMOP.VisitOccurrence.person_id'
    Output: 'OMOP.Visit_Occurrence.person_id'

    Paths that don't match the SCHEMA.Relation.attribute shape are returned
    unchanged. Single-word relation names (e.g., 'Person', 'Patients') pass
    through. CamelCase relation names not in the map raise ValueError.
    """
    parts = path.split(".")
    if len(parts) != 3:
        return path
    schema, relation, attribute = parts
    if _is_camel_case(relation):
        if relation not in OMOP_NAME_NORMALIZATION:
            raise ValueError(
                f"Unmapped CamelCase relation name {relation!r} in path {path!r}. "
                f"Add it to OMOP_NAME_NORMALIZATION in ground_truth.py."
            )
        relation = OMOP_NAME_NORMALIZATION[relation]
    return f"{schema}.{relation}.{attribute}"


def load_updated_ground_truth(path: str) -> Tuple[List[GTOneToOneEntry], List[GTGroupEntry]]:
    """Load updated_ground_truth.csv and return (one-to-one entries, group entries).

    Behavior:
    - The `type` column drives classification by MatchType.
    - For non-one_to_one rows (one_to_many, many_to_one, many_to_many), both the
      `source` and `target` columns are split on '+' to yield individual attribute
      paths. one_to_one rows are stored verbatim (no splitting).
    - OMOP_NAME_NORMALIZATION is applied to the relation-name segment of every
      source and target path. An unmapped CamelCase token raises ValueError.
    - No typo normalization is performed; the CSV is assumed clean.
    """
    one_to_one: List[GTOneToOneEntry] = []
    groups: List[GTGroupEntry] = []

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            match_type = MatchType(row["type"].strip())
            source_raw = row["source"].strip()
            target_raw = row["target"].strip()
            relationship = RelationshipType(row["relationship"].strip())

            if match_type == MatchType.one_to_one:
                one_to_one.append(GTOneToOneEntry(
                    source=_normalize_relation_path(source_raw),
                    target=_normalize_relation_path(target_raw),
                    relationship=relationship,
                ))
            else:
                sources = [_normalize_relation_path(s.strip()) for s in source_raw.split("+")]
                targets = [_normalize_relation_path(t.strip()) for t in target_raw.split("+")]
                groups.append(GTGroupEntry(
                    sources=sources,
                    targets=targets,
                    match_type=match_type,
                    relationship=relationship,
                ))

    return one_to_one, groups
