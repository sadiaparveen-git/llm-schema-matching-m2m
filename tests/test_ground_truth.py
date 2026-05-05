"""S1 verification gate: ground_truth loader tests."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from ground_truth import (
    OMOP_NAME_NORMALIZATION,
    GTGroupEntry,
    MatchType,
    RelationshipType,
    _is_camel_case,
    _normalize_target,
    load_updated_ground_truth,
)

# Path to the authoritative benchmark (one level above thesis-extension/)
GT_CSV = Path(__file__).parent.parent.parent / "updated_ground_truth.csv"


# ---------------------------------------------------------------------------
# Basic load tests
# ---------------------------------------------------------------------------

def test_load_returns_43_entries():
    # The CSV has 43 data rows: 41 one_to_one + 2 many_to_one.
    # (TECH_SPEC §4.3 originally estimated 42; the prepared CSV has one additional
    # one-to-one entry covering the Services->VisitDetail pair.)
    one_to_one, many_to_one = load_updated_ground_truth(str(GT_CSV))
    assert len(one_to_one) == 41, f"Expected 41 one-to-one entries, got {len(one_to_one)}"
    assert len(many_to_one) == 2, f"Expected 2 many-to-one entries, got {len(many_to_one)}"
    assert len(one_to_one) + len(many_to_one) == 43


# ---------------------------------------------------------------------------
# Typo normalization
# ---------------------------------------------------------------------------

def test_ont_to_one_typo_normalized():
    """Loader must convert 'ont_to_one' to 'one_to_one' without failing."""
    rows = [
        {"type": "one_to_one", "source": "MIMIC.Patients.subject_id",
         "relationship": "corresponds", "target": "OMOP.Person.person_source_value"},
        # Row with the typo — target uses a single-word OMOP name (no normalization needed)
        {"type": "ont_to_one", "source": "MIMIC.Transfers.careunit",
         "relationship": "mapped_via_vocabulary", "target": "OMOP.Person.person_source_value"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as fh:
        tmp_path = fh.name
        writer = csv.DictWriter(fh, fieldnames=["type", "source", "relationship", "target"])
        writer.writeheader()
        writer.writerows(rows)

    one_to_one, many_to_one = load_updated_ground_truth(tmp_path)
    assert len(one_to_one) == 2, "Both rows should be loaded as one_to_one after typo fix"
    assert len(many_to_one) == 0
    # Verify no entry has the raw typo as its type
    for entry in one_to_one:
        # GTEntry has a RelationshipType; the MatchType processing succeeded if we got here
        assert isinstance(entry.relationship, RelationshipType)


# ---------------------------------------------------------------------------
# OMOP normalization map coverage
# ---------------------------------------------------------------------------

def test_omop_map_covers_all_camel_case_relation_names():
    """All CamelCase OMOP relation names in the GT must be keys in OMOP_NAME_NORMALIZATION."""
    with open(GT_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        camel_case_names: set[str] = set()
        for row in reader:
            target = row["target"].strip()
            parts = target.split(".")
            if len(parts) == 3:
                relation = parts[1]
                if _is_camel_case(relation):
                    camel_case_names.add(relation)

    unmapped = camel_case_names - set(OMOP_NAME_NORMALIZATION.keys())
    assert not unmapped, (
        f"CamelCase OMOP relation names not in OMOP_NAME_NORMALIZATION: {unmapped}"
    )
    # Confirm the 5 expected names are all there
    expected = {"VisitOccurrence", "DrugExposure", "ConditionOccurrence", "VisitDetail", "CareSite"}
    assert camel_case_names == expected, (
        f"Expected CamelCase names {expected}, found {camel_case_names}"
    )


def test_loading_does_not_raise_for_any_target():
    """Successful load proves all 9 relation pairs are handled without unmapped tokens."""
    one_to_one, many_to_one = load_updated_ground_truth(str(GT_CSV))
    all_targets = [e.target for e in one_to_one] + [ge.target for ge in many_to_one]
    for target in all_targets:
        parts = target.split(".")
        assert len(parts) == 3
        relation = parts[1]
        # After normalization, no internal uppercase letters remain in relation names
        assert not _is_camel_case(relation), (
            f"CamelCase relation name survived normalization: {relation!r} in {target!r}"
        )


def test_unmapped_camel_case_raises():
    """An unmapped CamelCase OMOP relation name must raise ValueError."""
    rows = [
        {"type": "one_to_one", "source": "MIMIC.Foo.bar",
         "relationship": "corresponds", "target": "OMOP.UnknownTable.some_col"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as fh:
        tmp_path = fh.name
        writer = csv.DictWriter(fh, fieldnames=["type", "source", "relationship", "target"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Unmapped CamelCase"):
        load_updated_ground_truth(tmp_path)


# ---------------------------------------------------------------------------
# Many-to-one source lists
# ---------------------------------------------------------------------------

def test_many_to_one_source_lists():
    one_to_one, many_to_one = load_updated_ground_truth(str(GT_CSV))
    assert len(many_to_one) == 2

    # Build a dict keyed by target attribute name for easy lookup
    by_target: dict[str, GTGroupEntry] = {
        ge.target.split(".")[-1]: ge for ge in many_to_one
    }

    # Entry 1: anchor_year + anchor_age -> year_of_birth
    assert "year_of_birth" in by_target, "Missing year_of_birth many-to-one entry"
    yob = by_target["year_of_birth"]
    src_attrs = [s.split(".")[-1] for s in yob.sources]
    assert set(src_attrs) == {"anchor_year", "anchor_age"}, (
        f"Unexpected sources for year_of_birth: {src_attrs}"
    )
    assert yob.relationship == RelationshipType.aggregation

    # Entry 2: subject_id + hadm_id -> visit_occurrence_id (Prescriptions -> DrugExposure)
    assert "visit_occurrence_id" in by_target, "Missing visit_occurrence_id many-to-one entry"
    vid = by_target["visit_occurrence_id"]
    src_attrs2 = [s.split(".")[-1] for s in vid.sources]
    assert set(src_attrs2) == {"subject_id", "hadm_id"}, (
        f"Unexpected sources for visit_occurrence_id: {src_attrs2}"
    )
    assert vid.relationship == RelationshipType.linked_via_visit_source_value_to_visit_occurrence_id


# ---------------------------------------------------------------------------
# All 15 RelationshipType values present in loaded data
# ---------------------------------------------------------------------------

def test_all_15_relationship_types_present():
    one_to_one, many_to_one = load_updated_ground_truth(str(GT_CSV))
    found = {e.relationship for e in one_to_one} | {ge.relationship for ge in many_to_one}
    all_types = set(RelationshipType)
    missing = all_types - found
    assert not missing, (
        f"These RelationshipType values were not found in the loaded data: {missing}"
    )
    assert len(found) == 15, f"Expected 15 distinct relationship types, found {len(found)}"
