"""Tests for visual tags used to audit counted people and reflections."""

import numpy as np

from probabilistic_reflex_suppressor.annotation import annotate_image
from probabilistic_reflex_suppressor.models import BoundingBox, Detection, TaggedDetection


def test_annotation_uses_distinct_colors_for_counted_and_reflected_boxes() -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    tagged = [
        TaggedDetection(Detection(BoundingBox(10, 30, 40, 70), 0.9), "counted_person", True),
        TaggedDetection(
            Detection(BoundingBox(150, 30, 40, 70), 0.8),
            "mirror_reflection",
            False,
            paired_index=0,
        ),
    ]

    annotated = annotate_image(image, tagged)

    assert np.any(np.all(annotated == (40, 190, 40), axis=2))
    assert np.any(np.all(annotated == (40, 40, 220), axis=2))
    assert not np.array_equal(annotated, image)


def test_annotation_rejects_grayscale_images() -> None:
    with np.testing.assert_raises(ValueError):
        annotate_image(np.zeros((20, 20), dtype=np.uint8), [])
