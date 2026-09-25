"""Interfaces that keep the counting pipeline independent of a model vendor."""

from typing import Protocol

import numpy as np

from ..models import Detection


class PersonDetector(Protocol):
    """Protocol implemented by any detector that emits person observations."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect people in a BGR image represented as a NumPy array."""
