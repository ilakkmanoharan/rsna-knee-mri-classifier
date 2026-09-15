"""Tests for DICOM ordering, plane resolution, and sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.dicom import SliceMeta, plane_from_orientation, resolve_plane, sort_slices
from src.data.sampling import sample_volume, uniform_sample_indices


def test_uniform_sample_includes_endpoints():
    idx = uniform_sample_indices(100, 10)
    assert idx[0] == 0
    assert idx[-1] == 99
    assert len(idx) == 10


def test_uniform_sample_small():
    idx = uniform_sample_indices(3, 10)
    assert list(idx) == [0, 1, 2]


def test_sample_volume():
    vol = np.arange(20 * 4 * 4).reshape(20, 4, 4).astype(np.float32)
    out = sample_volume(vol, 5)
    assert out.shape[0] == 5


def test_sort_by_position():
    metas = [
        SliceMeta(Path("c.dcm"), 3, (0, 0, 30), None, 16, 16),
        SliceMeta(Path("a.dcm"), 1, (0, 0, 10), None, 16, 16),
        SliceMeta(Path("b.dcm"), 2, (0, 0, 20), None, 16, 16),
    ]
    ordered = sort_slices(metas)
    assert [m.path.name for m in ordered] == ["a.dcm", "b.dcm", "c.dcm"]


def test_sort_fallback_instance_number():
    metas = [
        SliceMeta(Path("c.dcm"), 3, None, None, 16, 16),
        SliceMeta(Path("a.dcm"), 1, None, None, 16, 16),
        SliceMeta(Path("b.dcm"), 2, None, None, 16, 16),
    ]
    ordered = sort_slices(metas)
    assert [m.instance_number for m in ordered] == [1, 2, 3]


def test_plane_from_orientation_axial():
    # Row along x, col along y => normal z => axial
    orient = (1, 0, 0, 0, 1, 0)
    assert plane_from_orientation(orient) == "axial"


def test_plane_from_orientation_sagittal():
    orient = (0, 1, 0, 0, 0, -1)
    assert plane_from_orientation(orient) == "sagittal"


def test_resolve_plane_prefers_metadata():
    plane, src = resolve_plane((1, 0, 0, 0, 1, 0), description="sag", metadata_plane="Coronal")
    assert plane == "coronal"
    assert src == "metadata"
