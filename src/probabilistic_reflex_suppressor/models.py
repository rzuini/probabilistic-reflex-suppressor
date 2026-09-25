"""Typed data objects shared by detectors, heuristics, and renderers."""

from dataclasses import dataclass
from typing import Any, Literal

DetectionTag = Literal[
    "counted_person",
    "mirror_reflection",
    "surface_overlap_uncertain",
]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An image-space rectangle represented by its top-left corner and size."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        """Reject boxes that cannot represent a visible region."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("BoundingBox width and height must be positive")

    @property
    def center_x(self) -> float:
        """Return the horizontal center coordinate."""
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        """Return the vertical center coordinate."""
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        """Return the box area in square pixels."""
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        """Return intersection-over-union with another box."""
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    """A person observation emitted by a detector."""

    box: BoundingBox
    confidence: float
    label: str = "person"

    def __post_init__(self) -> None:
        """Keep confidence values meaningful for downstream ranking."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SuppressedReflection:
    """Audit record for a detection removed as a likely reflection."""

    suppressed_index: int
    retained_index: int
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class SurfaceSummary:
    """Compact audit metadata for a mirror/glass segmentation result."""

    source: str
    confidence: float
    mirror_pixels: int
    glass_pixels: int


@dataclass(frozen=True, slots=True)
class TaggedDetection:
    """A detection annotated for JSON output and visual rendering."""

    detection: Detection
    tag: DetectionTag
    counted: bool
    paired_index: int | None = None
    surface_overlap: float = 0.0
    surface_kind: str | None = None

    def to_dict(self, index: int) -> dict[str, Any]:
        """Convert the tagged observation into an auditable JSON object."""
        box = self.detection.box
        return {
            "index": index,
            "tag": self.tag,
            "counted": self.counted,
            "paired_index": self.paired_index,
            "surface_overlap": round(self.surface_overlap, 6),
            "surface_kind": self.surface_kind,
            "label": self.detection.label,
            "confidence": round(self.detection.confidence, 6),
            "box": {
                "x": round(box.x, 3),
                "y": round(box.y, 3),
                "width": round(box.width, 3),
                "height": round(box.height, 3),
            },
        }


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Stable output schema for terminal, file, and library consumers."""

    image: str | None
    count: int
    detections: tuple[Detection, ...]
    suppressed_reflections: tuple[SuppressedReflection, ...] = ()
    warnings: tuple[str, ...] = ()
    tagged_detections: tuple[TaggedDetection, ...] = ()
    surface_segmentation: SurfaceSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into JSON-compatible primitive values."""
        tagged = self.tagged_detections or tuple(
            TaggedDetection(detection, "counted_person", True) for detection in self.detections
        )
        return {
            "image": self.image,
            "count": self.count,
            "detections": [item.to_dict(index) for index, item in enumerate(tagged)],
            "suppressed_reflections": [
                {
                    "suppressed_index": item.suppressed_index,
                    "retained_index": item.retained_index,
                    "score": round(item.score, 6),
                    "reason": item.reason,
                }
                for item in self.suppressed_reflections
            ],
            "warnings": list(self.warnings),
            "surface_segmentation": (
                {
                    "source": self.surface_segmentation.source,
                    "confidence": round(self.surface_segmentation.confidence, 6),
                    "mirror_pixels": self.surface_segmentation.mirror_pixels,
                    "glass_pixels": self.surface_segmentation.glass_pixels,
                }
                if self.surface_segmentation is not None
                else None
            ),
        }
