"""Shared utilities for thesis-extension experiment notebooks.

Provides:
- load_relation(): load a Relation from actual-repo/schema_documentations/
- RELATION_PAIRS: the 9 ground truth relation pairs
- ALL_MIMIC_TABLES / ALL_OMOP_TABLES: all tables discoverable in schema_documentations/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_NOTEBOOKS_DIR = Path(__file__).resolve().parent
_THESIS_ROOT = _NOTEBOOKS_DIR.parent
_PROJECT_ROOT = _THESIS_ROOT.parent
_SCHEMA_DOCS = _PROJECT_ROOT / "actual-repo" / "schema_documentations"

# Ensure thesis-extension is importable
_thesis_str = str(_THESIS_ROOT)
if _thesis_str not in sys.path:
    sys.path.insert(0, _thesis_str)

from models import Attribute, Relation, Side  # noqa: E402

# ---------------------------------------------------------------------------
# The 9 ground truth relation pairs
# Format: (source_name, source_schema, target_name, target_schema)
# Names must match evaluation keys derived from updated_ground_truth.csv
# ---------------------------------------------------------------------------

RELATION_PAIRS: List[Tuple[str, str, str, str]] = [
    ("Patients", "MIMIC", "Person", "OMOP"),
    ("Admissions", "MIMIC", "Visit_Occurrence", "OMOP"),
    ("Admissions", "MIMIC", "Death", "OMOP"),
    ("Prescriptions", "MIMIC", "Drug_Exposure", "OMOP"),
    ("Diagnoses_ICD", "MIMIC", "Condition_Occurrence", "OMOP"),
    ("Transfers", "MIMIC", "Care_Site", "OMOP"),
    ("Transfers", "MIMIC", "Visit_Detail", "OMOP"),
    ("Admissions", "MIMIC", "Visit_Detail", "OMOP"),
    ("Services", "MIMIC", "Visit_Detail", "OMOP"),
]

# ---------------------------------------------------------------------------
# All MIMIC tables present in schema_documentations/
# Names lowercase to the schema_documentations prefix: mimic_{name.lower()}
# ---------------------------------------------------------------------------

ALL_MIMIC_TABLES: List[str] = [
    "Admissions",
    "D_HCPCS",
    "D_ICD_Diagnoses",
    "D_ICD_Procedures",
    "D_LabItems",
    "Diagnoses_ICD",
    "DRGCodes",
    "EMAR",
    "EMAR_Detail",
    "HCPCSEvents",
    "LabEvents",
    "MicrobiologyEvents",
    "OMR",
    "Patients",
    "Pharmacy",
    "POE",
    "POE_Detail",
    "Prescriptions",
    "Procedures_ICD",
    "Provider",
    "Services",
    "Transfers",
]

# ---------------------------------------------------------------------------
# All OMOP tables present in schema_documentations/
# Names uppercase to the schema_documentations prefix: omop_{name.upper()}
# ---------------------------------------------------------------------------

ALL_OMOP_TABLES: List[str] = [
    "Attribute_Definition",
    "Care_Site",
    "CDM_Source",
    "Cohort_Definition",
    "Concept",
    "Concept_Ancestor",
    "Concept_Class",
    "Concept_Relationship",
    "Concept_Synonym",
    "Condition_Era",
    "Condition_Occurrence",
    "Cost",
    "Death",
    "Device_Exposure",
    "Domain",
    "Dose_Era",
    "Drug_Era",
    "Drug_Exposure",
    "Drug_Strength",
    "Fact_Relationship",
    "Location",
    "Measurement",
    "Metadata",
    "Note",
    "Note_NLP",
    "Observation",
    "Observation_Period",
    "Payer_Plan_Period",
    "Person",
    "Procedure_Occurrence",
    "Provider",
    "Relationship",
    "Source_To_Concept_Map",
    "Specimen",
    "Visit_Detail",
    "Visit_Occurrence",
    "Vocabulary",
]

# ---------------------------------------------------------------------------
# Minimal mock attributes per schema (fallback when actual-repo not found)
# ---------------------------------------------------------------------------

_MOCK_ATTRS: dict = {
    "MIMIC": [
        Attribute("subject_id", "A unique patient identifier."),
        Attribute("hadm_id", "A unique hospital admission identifier."),
        Attribute("charttime", "The time when the observation was charted."),
    ],
    "OMOP": [
        Attribute("person_id", "A unique OMOP person identifier."),
        Attribute(
            "visit_occurrence_id", "A unique OMOP visit occurrence identifier."
        ),
        Attribute("concept_id", "A standard OMOP concept identifier."),
    ],
}


def _file_prefix(schema: str, name: str) -> str:
    """Return the filename prefix for a relation in schema_documentations/.

    MIMIC: 'mimic_' + name.lower()   e.g. 'mimic_patients'
    OMOP:  'omop_'  + name.upper()   e.g. 'omop_PERSON', 'omop_VISIT_OCCURRENCE'
    """
    s = schema.upper()
    if s == "MIMIC":
        return "mimic_" + name.lower()
    if s == "OMOP":
        return "omop_" + name.upper()
    raise ValueError(f"Unknown schema {schema!r}. Expected 'MIMIC' or 'OMOP'.")


def load_relation(
    name: str,
    schema: str,
    side: Side,
    docs_dir: Optional[Path] = None,
) -> Relation:
    """Load a Relation from actual-repo/schema_documentations/.

    Each .txt file whose name starts with ``{prefix}_`` becomes one Attribute:
    the attribute name is the filename stem after stripping the prefix, and
    the file content is the description.

    Falls back to a minimal mock Relation (2-3 attributes) when the default
    schema_documentations/ directory is not found, so notebooks can run
    offline / in mock mode without the actual-repo data.

    Args:
        name: Relation name matching the ground truth, e.g. ``"Patients"``
              or ``"Visit_Occurrence"``.
        schema: ``"MIMIC"`` or ``"OMOP"``.
        side: ``Side.SOURCE`` or ``Side.TARGET``.
        docs_dir: Override the default path.  If provided and missing, raises
                  FileNotFoundError.

    Raises:
        FileNotFoundError: When *docs_dir* is explicitly provided but absent,
                           or when *docs_dir* is the default and exists but
                           contains no files for the requested relation.
    """
    if docs_dir is not None:
        if not docs_dir.exists():
            raise FileNotFoundError(
                f"Schema documentations directory not found: {docs_dir}"
            )
        dir_path = docs_dir
    else:
        dir_path = _SCHEMA_DOCS
        if not dir_path.exists():
            # Fallback: actual-repo not available — use mock data
            return _mock_relation(name, schema, side)

    prefix = _file_prefix(schema, name)
    files = sorted(dir_path.glob(f"{prefix}_*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No attribute files found for relation {name!r} (schema={schema!r}) "
            f"in {dir_path}. Expected files matching '{prefix}_*.txt'."
        )

    prefix_with_sep = prefix + "_"
    attributes = []
    for f in files:
        attr_name = f.stem[len(prefix_with_sep) :]
        description = f.read_text(encoding="utf-8").strip()
        attributes.append(
            Attribute(name=attr_name, description=description or None)
        )

    table_desc = _read_table_description(dir_path, schema, name)
    return Relation(
        name=name, side=side, attributes=attributes, description=table_desc
    )


def _read_table_description(
    docs_dir: Path, schema: str, name: str
) -> Optional[str]:
    """Read the table-level description file if present."""
    if schema.upper() == "MIMIC":
        table_file = docs_dir / f"mimic_table_{name.lower()}.txt"
    else:
        table_file = docs_dir / f"omop_table_{name.upper()}.txt"
    if table_file.exists():
        return table_file.read_text(encoding="utf-8").strip() or None
    return None


def _mock_relation(name: str, schema: str, side: Side) -> Relation:
    """Return a minimal 3-attribute mock Relation for offline use."""
    base_attrs = _MOCK_ATTRS.get(schema.upper(), _MOCK_ATTRS["MIMIC"])
    attrs = [Attribute(a.name, f"[mock] {a.description}") for a in base_attrs]
    return Relation(
        name=name,
        side=side,
        attributes=attrs,
        description=f"[mock] {name} (actual-repo not found)",
    )
