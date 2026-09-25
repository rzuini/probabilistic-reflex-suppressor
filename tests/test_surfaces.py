"""Unit tests for model-agnostic mirror and glass mask handling."""

import numpy as np
import pytest

from probabilistic_reflex_suppressor.models import BoundingBox
from probabilistic_reflex_suppressor.surfaces import SurfaceSegmentation


def test_surface_overlap_and_dominant_kind_are_computed_from_box_area() -> None:
    mirror = np.zeros((20, 30), dtype=bool)
    glass = np.zeros((20, 30), dtype=bool)
    mirror[5:15, 20:30] = True
    segmentation = SurfaceSegmentation(mirror, glass, source="synthetic")

    assert segmentation.overlap_fraction(BoundingBox(20, 5, 10, 10)) == pytest.approx(1.0)
    assert segmentation.dominant_kind(BoundingBox(20, 5, 10, 10)) == "mirror"
    assert segmentation.overlap_fraction(BoundingBox(0, 0, 10, 10)) == pytest.approx(0.0)
    assert segmentation.dominant_kind(BoundingBox(0, 0, 10, 10)) is None


def test_pair_evidence_requires_one_box_inside_and_one_outside() -> None:
    mirror = np.zeros((20, 30), dtype=bool)
    mirror[:, 20:30] = True
    segmentation = SurfaceSegmentation(mirror, np.zeros_like(mirror))

    outside = BoundingBox(0, 5, 10, 10)
    inside = BoundingBox(20, 5, 10, 10)

    assert segmentation.pair_evidence(outside, inside) == pytest.approx(1.0)
    assert segmentation.pair_evidence(inside, inside) == pytest.approx(0.0)


def test_glass_overlap_is_reported_but_not_used_as_mirror_pair_evidence() -> None:
    glass = np.zeros((20, 30), dtype=bool)
    glass[:, 20:30] = True
    segmentation = SurfaceSegmentation(np.zeros_like(glass), glass)

    assert segmentation.overlap_fraction(BoundingBox(20, 5, 10, 10)) == pytest.approx(1.0)
    assert segmentation.dominant_kind(BoundingBox(20, 5, 10, 10)) == "glass"
    assert segmentation.pair_evidence(BoundingBox(0, 5, 10, 10), BoundingBox(20, 5, 10, 10)) == 0.0
