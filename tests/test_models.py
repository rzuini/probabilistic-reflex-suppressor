"""Unit tests for stable data objects."""

import pytest

from probabilistic_reflex_suppressor.models import BoundingBox, Detection


def test_bounding_box_iou_is_symmetric() -> None:
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(5, 0, 10, 10)

    assert first.iou(second) == pytest.approx(1 / 3)
    assert second.iou(first) == pytest.approx(1 / 3)


def test_invalid_model_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        BoundingBox(0, 0, 0, 10)
    with pytest.raises(ValueError, match="between 0 and 1"):
        Detection(BoundingBox(0, 0, 10, 10), 1.1)
