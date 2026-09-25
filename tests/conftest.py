"""Reusable deterministic test fixtures."""

import cv2
import numpy as np
import pytest

from probabilistic_reflex_suppressor.models import BoundingBox, Detection


@pytest.fixture
def synthetic_mirror_image() -> np.ndarray:
    """Create two visually similar regions around a central mirror axis."""
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    person_patch = np.zeros((80, 40, 3), dtype=np.uint8)
    cv2.rectangle(person_patch, (12, 4), (28, 24), (190, 190, 190), -1)
    cv2.rectangle(person_patch, (6, 25), (34, 76), (120, 120, 120), -1)
    image[40:120, 20:60] = person_patch
    image[40:120, 180:220] = cv2.flip(person_patch, 1)
    cv2.line(image, (120, 0), (120, 159), (180, 180, 180), 2)
    cv2.line(image, (0, 128), (239, 128), (180, 180, 180), 2)
    return image


@pytest.fixture
def mirrored_detections() -> list[Detection]:
    """Return equal-sized detections mirrored around x=120."""
    return [
        Detection(BoundingBox(20, 40, 40, 80), 0.90),
        Detection(BoundingBox(180, 40, 40, 80), 0.75),
    ]
