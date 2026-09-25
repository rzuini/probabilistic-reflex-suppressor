"""Focused tests for reflection suppression decisions."""

import cv2
import numpy as np

from probabilistic_reflex_suppressor.models import BoundingBox, Detection
from probabilistic_reflex_suppressor.reflection import ReflectionSuppressor


def test_symmetric_pair_keeps_higher_confidence(
    synthetic_mirror_image, mirrored_detections
) -> None:
    decision = ReflectionSuppressor().suppress(synthetic_mirror_image, mirrored_detections)

    assert len(decision.detections) == 1
    assert decision.detections[0].confidence == 0.90
    assert len(decision.suppressed) == 1
    assert decision.suppressed[0].suppressed_index == 1


def test_vertical_mismatch_does_not_suppress() -> None:
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    detections = [
        Detection(BoundingBox(20, 10, 40, 80), 0.9),
        Detection(BoundingBox(180, 90, 40, 80), 0.8),
    ]

    decision = ReflectionSuppressor().suppress(image, detections)

    assert len(decision.detections) == 2
    assert decision.suppressed == ()


def test_symmetric_pair_without_mirror_evidence_is_retained() -> None:
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    detections = [
        Detection(BoundingBox(20, 40, 40, 80), 0.9),
        Detection(BoundingBox(180, 40, 40, 80), 0.8),
    ]

    decision = ReflectionSuppressor().suppress(image, detections)

    assert len(decision.detections) == 2
    assert decision.suppressed == ()


def test_off_center_mirror_pair_is_suppressed() -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    cv2.line(image, (160, 0), (160, 179), (180, 180, 180), 2)
    detections = [
        Detection(BoundingBox(40, 45, 50, 90), 0.9),
        Detection(BoundingBox(230, 45, 50, 90), 0.8),
    ]

    decision = ReflectionSuppressor().suppress(image, detections)

    assert len(decision.detections) == 1
    assert decision.suppressed[0].suppressed_index == 1


def test_same_side_people_are_not_reflection_pairs() -> None:
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    detections = [
        Detection(BoundingBox(20, 40, 40, 80), 0.9),
        Detection(BoundingBox(70, 40, 40, 80), 0.8),
    ]

    decision = ReflectionSuppressor().suppress(image, detections)

    assert len(decision.detections) == 2
