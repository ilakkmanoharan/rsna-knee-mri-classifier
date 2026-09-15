"""Competition constants. Runtime schema from sample_submission.csv is authoritative."""

from __future__ import annotations

DEFAULT_TARGETS: list[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

STUDY_ID_COL = "StudyInstanceUID"
SERIES_ID_COL = "SeriesInstanceUID"
REPORT_COL = "Report"
PLANE_COL = "Anatomical_Plane"

PLANES = ("sagittal", "coronal", "axial", "unknown")

EVIDENCE_STATES = (
    "positive",
    "negative",
    "uncertain",
    "historical",
    "unmentioned",
)

KAGGLE_INPUT_CANDIDATES = (
    "/kaggle/input/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormality-detection-2025",
)
