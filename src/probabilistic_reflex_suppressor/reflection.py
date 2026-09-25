"""Auditable, single-image heuristics for likely mirror reflections."""

from dataclasses import dataclass

import cv2
import numpy as np

from .models import BoundingBox, Detection, SuppressedReflection
from .surfaces import SurfaceSegmentation


@dataclass(frozen=True, slots=True)
class ReflectionDecision:
    """The retained detections and audit records produced by suppression."""

    detections: tuple[Detection, ...]
    suppressed: tuple[SuppressedReflection, ...]


class ReflectionSuppressor:
    """Collapse highly symmetric detection pairs around a candidate mirror axis.

    The axis is inferred between a candidate pair and must have visible mirror
    boundary evidence. This still cannot distinguish every reflection from a
    symmetric real-person arrangement with certainty.
    """

    def __init__(
        self,
        *,
        minimum_score: float = 0.72,
        vertical_tolerance: float = 0.18,
        appearance_weight: float = 0.35,
        mirror_evidence_threshold: float = 0.28,
    ) -> None:
        """Configure pair matching and the conservative mirror-evidence gate."""
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")
        if not 0.0 < vertical_tolerance <= 1.0:
            raise ValueError("vertical_tolerance must be between 0 and 1")
        if not 0.0 <= appearance_weight <= 1.0:
            raise ValueError("appearance_weight must be between 0 and 1")
        if not 0.0 <= mirror_evidence_threshold <= 1.0:
            raise ValueError("mirror_evidence_threshold must be between 0 and 1")
        self.minimum_score = minimum_score
        self.vertical_tolerance = vertical_tolerance
        self.appearance_weight = appearance_weight
        self.mirror_evidence_threshold = mirror_evidence_threshold

    def suppress(
        self,
        image: np.ndarray,
        detections: list[Detection],
        surface_segmentation: SurfaceSegmentation | None = None,
    ) -> ReflectionDecision:
        """Return detections after removing the weaker likely reflection in each pair."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a color image with shape (height, width, 3)")
        active = list(detections)
        suppressed: list[SuppressedReflection] = []
        image_height = float(image.shape[0])
        removed: set[int] = set()

        for left_index, left in enumerate(detections):
            if left_index in removed:
                continue
            for right_index in range(left_index + 1, len(detections)):
                if right_index in removed:
                    continue
                right = detections[right_index]
                score = self._pair_score(
                    left,
                    right,
                    image,
                    image_height,
                    surface_segmentation,
                )
                if score < self.minimum_score:
                    continue
                retained, removed_index = self._choose_retained(
                    left_index, left, right_index, right
                )
                removed.add(removed_index)
                suppressed.append(
                    SuppressedReflection(
                        suppressed_index=removed_index,
                        retained_index=retained,
                        score=score,
                        reason=(
                            "segmented mirror evidence plus symmetric candidate-axis "
                            "pair with similar appearance"
                            if surface_segmentation is not None
                            else "mirror boundary evidence plus symmetric candidate-axis "
                            "pair with similar appearance"
                        ),
                    )
                )

        kept = tuple(detection for index, detection in enumerate(active) if index not in removed)
        return ReflectionDecision(kept, tuple(suppressed))

    def _pair_score(
        self,
        first: Detection,
        second: Detection,
        image: np.ndarray,
        image_height: float,
        surface_segmentation: SurfaceSegmentation | None,
    ) -> float:
        """Combine normalized geometry and mirrored crop similarity into one score."""
        first_box, second_box = first.box, second.box
        if first_box.center_x == second_box.center_x:
            return 0.0
        left_box, right_box = sorted((first_box, second_box), key=lambda box: box.center_x)
        if left_box.x + left_box.width > right_box.x:
            return 0.0
        mirror_axis = (first_box.center_x + second_box.center_x) / 2
        vertical_gap = abs(first_box.center_y - second_box.center_y) / image_height
        if vertical_gap > self.vertical_tolerance:
            return 0.0
        mirror_evidence = self._mirror_evidence_score(image, first_box, second_box, mirror_axis)
        surface_evidence = (
            surface_segmentation.pair_evidence(first_box, second_box)
            if surface_segmentation is not None
            else 0.0
        )
        mirror_evidence = max(mirror_evidence, surface_evidence)
        if mirror_evidence < self.mirror_evidence_threshold:
            return 0.0
        size_ratio = min(first_box.area, second_box.area) / max(first_box.area, second_box.area)
        geometry_score = max(0.0, size_ratio * (1.0 - vertical_gap))
        appearance_score = self._appearance_similarity(image, first_box, second_box)
        return (
            1.0 - self.appearance_weight
        ) * geometry_score + self.appearance_weight * appearance_score

    @staticmethod
    def _mirror_evidence_score(
        image: np.ndarray,
        first_box: BoundingBox,
        second_box: BoundingBox,
        mirror_axis: float,
    ) -> float:
        """Estimate mirror context from a long edge near a candidate pair axis.

        This gate deliberately favors precision over recall: without a visible
        planar boundary, a symmetric pair is retained as two real detections.
        It is a scene heuristic, not a mirror segmentation model.
        """
        half_height = max(first_box.height, second_box.height) * 0.65
        left = max(0, int(min(first_box.x, second_box.x) - half_height))
        right = min(
            image.shape[1],
            int(max(first_box.x + first_box.width, second_box.x + second_box.width) + half_height),
        )
        top = max(0, int(min(first_box.y, second_box.y) - half_height))
        bottom = min(
            image.shape[0],
            int(
                max(first_box.y + first_box.height, second_box.y + second_box.height) + half_height
            ),
        )
        context = image[top:bottom, left:right]
        if context.size == 0:
            return 0.0
        gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        axis_in_context = int(round(mirror_axis - left))
        corridor_half_width = max(2, int(min(first_box.width, second_box.width) * 0.2))
        corridor_left = max(0, axis_in_context - corridor_half_width)
        corridor_right = min(edges.shape[1], axis_in_context + corridor_half_width + 1)
        if corridor_left >= corridor_right:
            return 0.0
        vertical_support = np.mean(
            edges[:, corridor_left:corridor_right] > 0,
            axis=0,
        )
        return float(vertical_support.max())

    @staticmethod
    def _appearance_similarity(
        image: np.ndarray,
        first_box: BoundingBox,
        second_box: BoundingBox,
    ) -> float:
        """Compare normalized grayscale crops after mirroring the second crop."""
        first_crop = ReflectionSuppressor._crop(image, first_box)
        second_crop = ReflectionSuppressor._crop(image, second_box)
        if first_crop.size == 0 or second_crop.size == 0:
            return 0.0
        target_size = (32, 64)
        first_resized = cv2.resize(first_crop, target_size).astype(np.float32)
        second_resized = cv2.resize(cv2.flip(second_crop, 1), target_size).astype(np.float32)
        first_gray = cv2.cvtColor(first_resized, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.cvtColor(second_resized, cv2.COLOR_BGR2GRAY)
        first_gray = (first_gray - first_gray.mean()) / (first_gray.std() + 1e-6)
        second_gray = (second_gray - second_gray.mean()) / (second_gray.std() + 1e-6)
        correlation = float(np.mean(first_gray * second_gray))
        return max(0.0, min(1.0, (correlation + 1.0) / 2.0))

    @staticmethod
    def _crop(image: np.ndarray, box: BoundingBox) -> np.ndarray:
        """Crop a box while safely clipping coordinates to the image."""
        x = max(0, int(box.x))
        y = max(0, int(box.y))
        right = min(image.shape[1], int(box.x + box.width))
        bottom = min(image.shape[0], int(box.y + box.height))
        return image[y:bottom, x:right]

    @staticmethod
    def _choose_retained(
        first_index: int,
        first: Detection,
        second_index: int,
        second: Detection,
    ) -> tuple[int, int]:
        """Keep the more confident box and use index order as a stable tie-breaker."""
        if first.confidence >= second.confidence:
            return first_index, second_index
        return second_index, first_index
