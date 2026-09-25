"""Unit and integration tests for pipeline composition."""

from pathlib import Path

import cv2
import numpy as np

from probabilistic_reflex_suppressor.models import BoundingBox, Detection
from probabilistic_reflex_suppressor.pipeline import PersonCountingPipeline
from probabilistic_reflex_suppressor.surfaces import StaticSurfaceSegmenter, SurfaceSegmentation

from .fakes import FakePersonDetector


def test_pipeline_deduplicates_overlap_before_reflection_logic() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    detector = FakePersonDetector(
        [
            Detection(BoundingBox(10, 10, 40, 80), 0.9),
            Detection(BoundingBox(12, 12, 40, 80), 0.7),
        ]
    )

    result = PersonCountingPipeline(detector).analyze_array(image, "synthetic")

    assert result.count == 1
    assert result.image == "synthetic"
    assert result.suppressed_reflections == ()


def test_pipeline_exposes_tags_for_all_post_nms_detections(
    synthetic_mirror_image, mirrored_detections
) -> None:
    result = PersonCountingPipeline(FakePersonDetector(mirrored_detections)).analyze_array(
        synthetic_mirror_image
    )

    payload = result.to_dict()

    assert result.count == 1
    assert [item["tag"] for item in payload["detections"]] == [
        "counted_person",
        "mirror_reflection",
    ]
    assert [item["counted"] for item in payload["detections"]] == [True, False]


def test_pipeline_reads_image_path(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    result = PersonCountingPipeline(FakePersonDetector([])).analyze_path(image_path)

    assert result.count == 0
    assert result.image == str(image_path)


def test_pipeline_uses_surface_mask_as_reflection_evidence(
    synthetic_mirror_image, mirrored_detections
) -> None:
    mirror_mask = np.zeros(synthetic_mirror_image.shape[:2], dtype=bool)
    mirror_mask[:, 120:] = True
    segmentation = SurfaceSegmentation(
        mirror_mask=mirror_mask,
        glass_mask=np.zeros_like(mirror_mask),
        source="synthetic-mask",
    )
    pipeline = PersonCountingPipeline(
        FakePersonDetector(mirrored_detections),
        surface_segmenter=StaticSurfaceSegmenter(segmentation),
    )

    result = pipeline.analyze_array(synthetic_mirror_image)

    assert result.count == 1
    assert result.surface_segmentation is not None
    assert result.surface_segmentation.source == "synthetic-mask"
    assert result.tagged_detections[1].tag == "mirror_reflection"
    assert result.tagged_detections[1].surface_kind == "mirror"


def test_pipeline_keeps_unpaired_surface_overlap_counted() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mirror_mask = np.zeros((100, 100), dtype=bool)
    mirror_mask[10:90, 10:50] = True
    detection = Detection(BoundingBox(10, 10, 40, 80), 0.9)
    pipeline = PersonCountingPipeline(
        FakePersonDetector([detection]),
        surface_segmenter=StaticSurfaceSegmenter(
            SurfaceSegmentation(mirror_mask, np.zeros_like(mirror_mask), source="synthetic-mask")
        ),
    )

    result = pipeline.analyze_array(image)

    assert result.count == 1
    assert result.tagged_detections[0].tag == "surface_overlap_uncertain"
    assert result.tagged_detections[0].counted is True
