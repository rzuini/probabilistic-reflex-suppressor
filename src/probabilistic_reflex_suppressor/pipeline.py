"""Application pipeline for image loading, counting, and result creation."""

from pathlib import Path

import cv2
import numpy as np

from .detectors.base import PersonDetector
from .models import Detection, DetectionResult, DetectionTag, SuppressedReflection, TaggedDetection
from .reflection import ReflectionSuppressor
from .surfaces import SurfaceSegmentation, SurfaceSegmenter


class PersonCountingPipeline:
    """Compose an injected detector with de-duplication and reflection suppression."""

    def __init__(
        self,
        detector: PersonDetector,
        suppressor: ReflectionSuppressor | None = None,
        nms_threshold: float = 0.35,
        surface_segmenter: SurfaceSegmenter | None = None,
        surface_overlap_threshold: float = 0.5,
    ) -> None:
        """Create a pipeline with explicit model and heuristic dependencies."""
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")
        if not 0.0 <= surface_overlap_threshold <= 1.0:
            raise ValueError("surface_overlap_threshold must be between 0 and 1")
        self.detector = detector
        self.suppressor = suppressor or ReflectionSuppressor()
        self.nms_threshold = nms_threshold
        self.surface_segmenter = surface_segmenter
        self.surface_overlap_threshold = surface_overlap_threshold

    def analyze_array(self, image: np.ndarray, image_name: str | None = None) -> DetectionResult:
        """Analyze a BGR color array without performing filesystem I/O."""
        self._validate_image(image)
        raw_detections = self.detector.detect(image)
        detections = self._non_max_suppression(raw_detections)
        segmentation = (
            self.surface_segmenter.segment(image) if self.surface_segmenter is not None else None
        )
        if segmentation is not None and segmentation.surface_mask.shape != image.shape[:2]:
            raise ValueError("surface masks must have the same height and width as the image")
        decision = self.suppressor.suppress(
            image,
            detections,
            surface_segmentation=segmentation,
        )
        suppressed_by_index = {item.suppressed_index: item for item in decision.suppressed}
        paired_by_index: dict[int, int] = {}
        for item in decision.suppressed:
            paired_by_index[item.suppressed_index] = item.retained_index
            paired_by_index[item.retained_index] = item.suppressed_index
        tagged_detections = tuple(
            self._tag_detection(
                detection=detection,
                index=index,
                suppressed_by_index=suppressed_by_index,
                paired_by_index=paired_by_index,
                segmentation=segmentation,
            )
            for index, detection in enumerate(detections)
        )
        return DetectionResult(
            image=image_name,
            count=len(decision.detections),
            detections=decision.detections,
            suppressed_reflections=decision.suppressed,
            tagged_detections=tagged_detections,
            surface_segmentation=segmentation.summary if segmentation is not None else None,
        )

    def _tag_detection(
        self,
        *,
        detection: Detection,
        index: int,
        suppressed_by_index: dict[int, SuppressedReflection],
        paired_by_index: dict[int, int],
        segmentation: SurfaceSegmentation | None,
    ) -> TaggedDetection:
        """Create an auditable tag without discarding ambiguous detections."""
        surface_overlap = (
            segmentation.overlap_fraction(detection.box) if segmentation is not None else 0.0
        )
        surface_kind = (
            segmentation.dominant_kind(detection.box) if segmentation is not None else None
        )
        tag: DetectionTag
        if index in suppressed_by_index:
            tag = "mirror_reflection"
            counted = False
        elif surface_overlap >= self.surface_overlap_threshold:
            tag = "surface_overlap_uncertain"
            counted = True
        else:
            tag = "counted_person"
            counted = True
        return TaggedDetection(
            detection=detection,
            tag=tag,
            counted=counted,
            paired_index=paired_by_index.get(index),
            surface_overlap=surface_overlap,
            surface_kind=surface_kind,
        )

    def analyze_path(self, image_path: str | Path) -> DetectionResult:
        """Load an image path and analyze it, raising a useful error if unreadable."""
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read a color image from: {path}")
        return self.analyze_array(image, str(path))

    def _non_max_suppression(self, detections: list[Detection]) -> list[Detection]:
        """Remove overlapping boxes while retaining higher-confidence observations."""
        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        kept: list[Detection] = []
        for candidate in ordered:
            if all(candidate.box.iou(existing.box) <= self.nms_threshold for existing in kept):
                kept.append(candidate)
        return kept

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Validate the array contract before passing it to a model backend."""
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("image must be a non-empty color array with shape (height, width, 3)")
