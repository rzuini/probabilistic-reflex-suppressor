"""Test doubles shared by pipeline and CLI tests."""

import numpy as np

from probabilistic_reflex_suppressor.models import Detection


class FakePersonDetector:
    """Detector double that returns preconfigured observations."""

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return a copy so the pipeline cannot mutate fixture state."""
        return list(self.detections)
