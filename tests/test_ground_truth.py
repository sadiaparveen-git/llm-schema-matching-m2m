"""S1 verification gate: ground_truth loader tests."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from ground_truth import (
    OMOP_NAME_NORMALIZATION,
    GTGroupEntry,
    GTOneToOneEntry,
    MatchType,
    RelationshipType,
    _is_camel_case,
    _normalize_relation_path,
    load_updated_ground_truth,
)

# Path to the authoritative benchmark (one level above thesis-extension/)
GT_CSV = Path(__file__).parent.parent.parent / "updated_ground_truth.csv"


def _read_raw_rows():
    with open(GT_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Loader: count consistency
# ---------------------------------------------------------------------------

def test_loaded_total_equals_csv_row_count():
    """Sum of returned entries across all match types equals the raw CSV row count."""
    one_to_one, groups = load_updated_ground_truth(str(GT_CSV))
    raw_rows = _read_raw_rows()
    assert len(one_to_one) + len(groups) == len(raw_rows)


def test_one_to_one_count_matches_csv():
    """Number of GTOneToOneEntry returned equals the count of one_to_one CSV rows."""
    one_to_one, _ = load_updated_ground_truth(str(GT_CSV))
    csv_one_to_one = sum(1 for r in _read_raw_rows() if r["type"].strip() == "one_to_one")
    assert len(one_to_one) == csv_one_to_one


def test_group_counts_match_csv_per_match_type():
    """Each non-1:1 match-type bucket size equals the CSV row count of that type."""
    _, groups = load_updated_ground_truth(str(GT_CSV))
    for mt in (MatchType.one_to_many, MatchType.many_to_one, MatchType.many_to_many):
        csv_count = sum(1 for r in _read_raw_rows() if r["type"].strip() == mt.value)
        loaded_count = sum(1 for g in groups if g.match_type == mt)
        assert loaded_count == csv_count, f"count mismatch for {mt.value}"


# ---------------------------------------------------------------------------
# Match-type classification
# ---------------------------------------------------------------------------

def test_one_to_one_entries_are_correct_class():
    """Every entry in the first list is a GTOneToOneEntry (implicit match_type = one_to_one)."""
    one_to_one, _ = load_updated_ground_truth(str(GT_CSV))
    for entry in one_to_one:
        assert isinstance(entry, GTOneToOneEntry)


def test_group_entries_have_valid_match_type():
    """Every GTGroupEntry has match_type ∈ {one_to_many, many_to_one, many_to_many}."""
    _, groups = load_updated_ground_truth(str(GT_CSV))
    valid = {MatchType.one_to_many, MatchType.many_to_one, MatchType.many_to_many}
    for entry in groups:
        assert isinstance(entry, GTGroupEntry)
        assert entry.match_type in valid, (
            f"GTGroupEntry has unexpected match_type: {entry.match_type}"
        )


# ---------------------------------------------------------------------------
# Source / target splitting per match type
# ---------------------------------------------------------------------------

def test_many_to_one_sources_split_into_list():
    """many_to_one entries have multiple sources and exactly one target."""
    _, groups = load_updated_ground_truth(str(GT_CSV))
    m2o = [g for g in groups if g.match_type == MatchType.many_to_one]
    assert len(m2o) >= 1, "Expected at least one many_to_one entry in the CSV"
    for entry in m2o:
        assert isinstance(entry.sources, list)
        assert len(entry.sources) > 1, (
            f"many_to_one sources should be split into >1 elements, got {entry.sources}"
        )
        assert len(entry.targets) == 1


def test_one_to_many_targets_split_into_list_if_any():
    """If any one_to_many entries exist, they have multiple targets and exactly one source."""
    _, groups = load_updated_ground_truth(str(GT_CSV))
    o2m = [g for g in groups if g.match_type == MatchType.one_to_many]
    for entry in o2m:
        assert len(entry.targets) > 1, (
            f"one_to_many targets should be split into >1 elements, got {entry.targets}"
        )
        assert len(entry.sources) == 1


def test_many_to_many_both_sides_split_if_any():
    """If any many_to_many entries exist, both sources and targets are split."""
    _, groups = load_updated_ground_truth(str(GT_CSV))
    m2m = [g for g in groups if g.match_type == MatchType.many_to_many]
    for entry in m2m:
        assert len(entry.sources) > 1
        assert len(entry.targets) > 1


def test_synthetic_csv_covers_all_four_match_types():
    """End-to-end: a synthetic CSV with all four match types is parsed correctly,
    confirming the splitting logic works for one_to_many and many_to_many even
    though the current real CSV may not contain such rows."""
    rows = [
        {"type": "one_to_one", "source": "MIMIC.Patients.gender",
         "relationship": "corresponds",
         "target": "OMOP.Person.gender_source_value"},
        {"type": "one_to_many", "source": "MIMIC.Admissions.admission_type",
         "relationship": "copied_as_source_value",
         "target": "OMOP.Person.person_id + OMOP.Person.gender_source_value"},
        {"type": "many_to_one",
         "source": "MIMIC.Patients.anchor_year + MIMIC.Patients.anchor_age",
         "relationship": "aggregation",
         "target": "OMOP.Person.year_of_birth"},
        {"type": "many_to_many",
         "source": "MIMIC.Admissions.admittime + MIMIC.Admissions.dischtime",
         "relationship": "cast_to_date",
         "target": "OMOP.Person.gender_source_value + OMOP.Person.year_of_birth"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as fh:
        tmp = fh.name
        writer = csv.DictWriter(fh, fieldnames=["type", "source", "relationship", "target"])
        writer.writeheader()
        writer.writerows(rows)

    one_to_one, groups = load_updated_ground_truth(tmp)
    assert len(one_to_one) == 1
    assert len(groups) == 3

    by_type = {g.match_type: g for g in groups}
    assert MatchType.one_to_many in by_type
    assert MatchType.many_to_one in by_type
    assert MatchType.many_to_many in by_type

    assert len(by_type[MatchType.one_to_many].sources) == 1
    assert len(by_type[MatchType.one_to_many].targets) == 2

    assert len(by_type[MatchType.many_to_one].sources) == 2
    assert len(by_type[MatchType.many_to_one].targets) == 1

    assert len(by_type[MatchType.many_to_many].sources) == 2
    assert len(by_type[MatchType.many_to_many].targets) == 2


# ---------------------------------------------------------------------------
# OMOP normalization
# ---------------------------------------------------------------------------

def test_omop_map_covers_all_camel_case_in_csv():
    """Every CamelCase relation name in the CSV (source or target column) must
    be a key in OMOP_NAME_NORMALIZATION."""
    camel_case_names: set[str] = set()
    for row in _read_raw_rows():
        for col in ("source", "target"):
            for path in row[col].split("+"):
                parts = path.strip().split(".")
                if len(parts) == 3 and _is_camel_case(parts[1]):
                    camel_case_names.add(parts[1])

    unmapped = camel_case_names - set(OMOP_NAME_NORMALIZATION.keys())
    assert not unmapped, (
        f"CamelCase relation names not in OMOP_NAME_NORMALIZATION: {unmapped}"
    )


def test_no_camel_case_relation_names_after_loading():
    """After loading, no relation-name segment in any source/target path is CamelCase."""
    one_to_one, groups = load_updated_ground_truth(str(GT_CSV))

    paths: list[str] = []
    for e in one_to_one:
        paths.extend([e.source, e.target])
    for e in groups:
        paths.extend(e.sources)
        paths.extend(e.targets)

    for path in paths:
        parts = path.split(".")
        if len(parts) == 3:
            relation = parts[1]
            assert not _is_camel_case(relation), (
                f"CamelCase relation survived normalization: {relation!r} in {path!r}"
            )


def test_unmapped_camel_case_raises():
    """An unmapped CamelCase OMOP relation name must raise ValueError at load time."""
    rows = [
        {"type": "one_to_one", "source": "MIMIC.Foo.bar",
         "relationship": "corresponds", "target": "OMOP.UnknownTable.some_col"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as fh:
        tmp = fh.name
        writer = csv.DictWriter(fh, fieldnames=["type", "source", "relationship", "target"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Unmapped CamelCase"):
        load_updated_ground_truth(tmp)


# ---------------------------------------------------------------------------
# Enum membership: every CSV value must be a valid enum member
# ---------------------------------------------------------------------------

def test_all_csv_match_type_values_are_valid_enum_members():
    """Every distinct value in the CSV `type` column constructs a valid MatchType."""
    csv_values = {row["type"].strip() for row in _read_raw_rows()}
    for v in csv_values:
        # Will raise ValueError if v is not a valid MatchType — proves coverage.
        MatchType(v)


def test_all_csv_relationship_values_are_valid_enum_members():
    """Every distinct value in the CSV `relationship` column constructs a valid RelationshipType."""
    csv_values = {row["relationship"].strip() for row in _read_raw_rows()}
    for v in csv_values:
        RelationshipType(v)


# ---------------------------------------------------------------------------
# Direct unit checks for normalization helper
# ---------------------------------------------------------------------------

def test_normalize_relation_path_handles_known_camel_case():
    assert (_normalize_relation_path("OMOP.VisitOccurrence.person_id")
            == "OMOP.Visit_Occurrence.person_id")
    assert (_normalize_relation_path("OMOP.DrugExposure.drug_concept_id")
            == "OMOP.Drug_Exposure.drug_concept_id")
    # Already-normalized and single-word relations pass through unchanged
    assert (_normalize_relation_path("OMOP.Person.person_id")
            == "OMOP.Person.person_id")
    assert (_normalize_relation_path("OMOP.Care_Site.care_site_name")
            == "OMOP.Care_Site.care_site_name")
