"""Vendor-neutral contracts and geometry helpers for mirror/glass masks."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .models import BoundingBox, SurfaceSummary

SurfaceKind = Literal["mirror", "glass"]


@dataclass(frozen=True, slots=True)
class SurfaceSegmentation:
    """Binary mirror and glass masks produced for one image.

    The masks intentionally contain no model-specific details. A future neural
    segmenter can therefore replace the deterministic test provider without
    changing the counting pipeline or its JSON contract.
    """

    mirror_mask: np.ndarray
    glass_mask: np.ndarray
    source: str = "unknown"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate mask shape and confidence before the masks enter the pipeline."""
        object.__setattr__(self, "mirror_mask", np.asarray(self.mirror_mask, dtype=bool))
        object.__setattr__(self, "glass_mask", np.asarray(self.glass_mask, dtype=bool))
        if self.mirror_mask.ndim != 2 or self.glass_mask.ndim != 2:
            raise ValueError("surface masks must be two-dimensional")
        if self.mirror_mask.shape != self.glass_mask.shape:
            raise ValueError("mirror and glass masks must have the same shape")
        if self.mirror_mask.size == 0:
            raise ValueError("surface masks must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("surface segmentation confidence must be between 0 and 1")

    @classmethod
    def empty(cls, image_shape: tuple[int, int], source: str = "none") -> "SurfaceSegmentation":
        """Create an explicit no-surface result for an image shape."""
        height, width = image_shape
        if height <= 0 or width <= 0:
            raise ValueError("image shape must contain positive dimensions")
        empty: np.ndarray = np.zeros((height, width), dtype=bool)
        return cls(empty, empty.copy(), source=source, confidence=0.0)

    @property
    def surface_mask(self) -> np.ndarray:
        """Return the union of mirror and glass pixels."""
        surface_mask: np.ndarray = np.logical_or(self.mirror_mask, self.glass_mask)
        return surface_mask

    @property
    def summary(self) -> SurfaceSummary:
        """Return compact metadata suitable for the JSON result."""
        return SurfaceSummary(
            source=self.source,
            confidence=self.confidence,
            mirror_pixels=int(np.count_nonzero(self.mirror_mask)),
            glass_pixels=int(np.count_nonzero(self.glass_mask)),
        )

    def overlap_fraction(self, box: BoundingBox) -> float:
        """Return the fraction of a detection box covered by either surface."""
        clipped = self._clip_box(box)
        if clipped is None:
            return 0.0
        left, top, right, bottom = clipped
        return float(np.mean(self.surface_mask[top:bottom, left:right]))

    def dominant_kind(self, box: BoundingBox, minimum_overlap: float = 0.05) -> SurfaceKind | None:
        """Return the surface type covering most of a box, if coverage is meaningful."""
        clipped = self._clip_box(box)
        if clipped is None:
            return None
        left, top, right, bottom = clipped
        mirror_fraction = float(np.mean(self.mirror_mask[top:bottom, left:right]))
        glass_fraction = float(np.mean(self.glass_mask[top:bottom, left:right]))
        if max(mirror_fraction, glass_fraction) < minimum_overlap:
            return None
        return "mirror" if mirror_fraction >= glass_fraction else "glass"

    def pair_evidence(self, first: BoundingBox, second: BoundingBox) -> float:
        """Score whether exactly one detection lies on a segmented surface.

        A reflection candidate should overlap the mirror mask while its paired
        real-world candidate remains outside it. Glass is deliberately excluded
        from automatic suppression in phase 1 because a person visible through
        a window can be real rather than a reflection. This is evidence, not
        proof, and is intentionally bounded to [0, 1].
        """
        first_overlap = self._mask_overlap(self.mirror_mask, first)
        second_overlap = self._mask_overlap(self.mirror_mask, second)
        strongest = max(first_overlap, second_overlap)
        separation = abs(first_overlap - second_overlap)
        return strongest * separation

    def _mask_overlap(self, mask: np.ndarray, box: BoundingBox) -> float:
        """Return the fraction of a box covered by one specific mask."""
        clipped = self._clip_box(box)
        if clipped is None:
            return 0.0
        left, top, right, bottom = clipped
        return float(np.mean(mask[top:bottom, left:right]))

    def _clip_box(self, box: BoundingBox) -> tuple[int, int, int, int] | None:
        """Convert a floating-point box to a safe, non-empty mask slice."""
        height, width = self.surface_mask.shape
        left = max(0, int(np.floor(box.x)))
        top = max(0, int(np.floor(box.y)))
        right = min(width, int(np.ceil(box.x + box.width)))
        bottom = min(height, int(np.ceil(box.y + box.height)))
        if left >= right or top >= bottom:
            return None
        return left, top, right, bottom


class SurfaceSegmenter(Protocol):
    """Protocol implemented by any mirror/glass segmentation backend."""

    def segment(self, image: np.ndarray) -> SurfaceSegmentation:
        """Return mirror and glass masks aligned with the input image."""


@dataclass(frozen=True, slots=True)
class StaticSurfaceSegmenter:
    """Deterministic provider useful for tests and externally generated masks."""

    segmentation: SurfaceSegmentation

    def segment(self, image: np.ndarray) -> SurfaceSegmentation:
        """Return the configured masks after checking image alignment."""
        if image.shape[:2] != self.segmentation.surface_mask.shape:
            raise ValueError("surface masks must have the same height and width as the image")
        return self.segmentation
