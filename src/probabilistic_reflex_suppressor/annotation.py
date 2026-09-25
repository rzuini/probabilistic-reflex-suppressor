"""Render detection tags and bounding boxes onto an image."""

from collections.abc import Sequence

import cv2
import numpy as np

from .models import TaggedDetection


def annotate_image(
    image: np.ndarray,
    tagged_detections: Sequence[TaggedDetection],
) -> np.ndarray:
    """Return a copy with counted people and reflections visibly tagged.

    Green boxes represent detections included in the count. Red boxes represent
    likely mirror duplicates that remain visible for auditability but are not
    included in the count. Amber boxes overlap a segmented surface and remain
    counted conservatively because overlap alone is not proof of reflection.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty color image with shape (height, width, 3)")

    annotated = image.copy()
    for index, item in enumerate(tagged_detections):
        box = item.detection.box
        left = max(0, int(box.x))
        top = max(0, int(box.y))
        right = min(image.shape[1] - 1, int(box.x + box.width))
        bottom = min(image.shape[0] - 1, int(box.y + box.height))
        if not item.counted:
            color = (40, 40, 220)
        elif item.tag == "surface_overlap_uncertain":
            color = (0, 190, 220)
        else:
            color = (40, 190, 40)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 1)
        if not item.counted:
            label = f"{index}: MIRROR REFLECTION (NOT COUNTED)"
        elif item.tag == "surface_overlap_uncertain":
            label = f"{index}: SURFACE OVERLAP (COUNTED UNCERTAINLY)"
        else:
            label = f"{index}: COUNTED PERSON"
        baseline_y = max(24, top - 8)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_top = max(0, baseline_y - text_height - 8)
        label_right = min(image.shape[1] - 1, left + text_width + 8)
        cv2.rectangle(annotated, (left, label_top), (label_right, baseline_y), color, -1)
        cv2.putText(
            annotated,
            label,
            (left + 4, baseline_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated
