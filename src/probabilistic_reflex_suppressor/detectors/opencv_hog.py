"""OpenCV's dependency-light, CPU-friendly person detector."""

import cv2
import numpy as np

from ..models import BoundingBox, Detection


class OpenCVHOGPersonDetector:
    """Detect people with OpenCV's built-in HOG + linear SVM model.

    HOG is useful as a portable baseline, not as a claim of state-of-the-art
    accuracy. A deployment can replace this class with an injected detector.
    """

    def __init__(
        self,
        score_threshold: float = 0.0,
        min_edge_density: float = 0.015,
        min_contrast: float = 0.08,
    ) -> None:
        """Create a detector with OpenCV's default pedestrian SVM weights."""
        if not -1.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between -1 and 1")
        if not 0.0 <= min_edge_density <= 1.0:
            raise ValueError("min_edge_density must be between 0 and 1")
        if not 0.0 <= min_contrast <= 1.0:
            raise ValueError("min_contrast must be between 0 and 1")
        self.score_threshold = score_threshold
        self.min_edge_density = min_edge_density
        self.min_contrast = min_contrast
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # type: ignore[attr-defined]

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return HOG detections after converting raw SVM scores to 0..1."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a color image with shape (height, width, 3)")
        minimum_width, minimum_height = self._hog.winSize
        if image.shape[1] < minimum_width or image.shape[0] < minimum_height:
            return []

        boxes, weights = self._hog.detectMultiScale(
            image,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
        )
        detections: list[Detection] = []
        for raw_box, raw_weight in zip(boxes, weights, strict=True):
            score = float(np.asarray(raw_weight).reshape(-1)[0])
            if score < self.score_threshold:
                continue
            # The SVM score is uncalibrated; sigmoid keeps the public confidence bounded.
            confidence = 1.0 / (1.0 + float(np.exp(-score)))
            x, y, width, height = (float(value) for value in raw_box)
            if not self._has_visual_support(image, x, y, width, height):
                continue
            detections.append(Detection(BoundingBox(x, y, width, height), confidence))
        return detections

    def _has_visual_support(
        self,
        image: np.ndarray,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> bool:
        """Reject boxes whose pixels resemble a low-information background patch."""
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(image.shape[1], int(x + width))
        bottom = min(image.shape[0], int(y + height))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        contrast = float(gray.std()) / 255.0
        return edge_density >= self.min_edge_density and contrast >= self.min_contrast
