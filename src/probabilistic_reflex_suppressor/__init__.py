"""Vendor-agnostic image person counting with auditable reflection heuristics."""

from .annotation import annotate_image
from .models import BoundingBox, Detection, DetectionResult, SurfaceSummary, TaggedDetection
from .pipeline import PersonCountingPipeline
from .surfaces import StaticSurfaceSegmenter, SurfaceSegmentation, SurfaceSegmenter

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "PersonCountingPipeline",
    "StaticSurfaceSegmenter",
    "SurfaceSegmenter",
    "SurfaceSegmentation",
    "SurfaceSummary",
    "TaggedDetection",
    "annotate_image",
]
