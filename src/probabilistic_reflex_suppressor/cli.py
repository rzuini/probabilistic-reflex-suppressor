"""Command-line interface for image analysis."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .annotation import annotate_image
from .detectors.opencv_hog import OpenCVHOGPersonDetector
from .pipeline import PersonCountingPipeline
from .reflection import ReflectionSuppressor
from .surfaces import StaticSurfaceSegmenter, SurfaceSegmentation


def build_parser() -> argparse.ArgumentParser:
    """Build the parser separately so CLI behavior can be tested directly."""
    parser = argparse.ArgumentParser(
        prog="probabilistic-reflex-suppressor",
        description="Count people in an image and audit likely mirror reflections.",
    )
    parser.add_argument("image", type=Path, help="Path to a readable color image")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path where the JSON result will also be written",
    )
    parser.add_argument(
        "--annotated-output",
        type=Path,
        help="Optional path for an image with tagged detection boxes",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Raw OpenCV HOG score threshold (default: 0.0)",
    )
    parser.add_argument(
        "--min-edge-density",
        type=float,
        default=0.015,
        help="Minimum edge density inside a HOG box (default: 0.015)",
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=0.08,
        help="Minimum normalized contrast inside a HOG box (default: 0.08)",
    )
    parser.add_argument(
        "--mirror-evidence-threshold",
        type=float,
        default=0.28,
        help=("Minimum local mirror-boundary evidence for reflection suppression (default: 0.28)"),
    )
    parser.add_argument(
        "--mirror-mask",
        type=Path,
        help="Optional grayscale mask image whose non-zero pixels identify mirrors",
    )
    parser.add_argument(
        "--glass-mask",
        type=Path,
        help="Optional grayscale mask image whose non-zero pixels identify glass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-compatible exit code."""
    args = build_parser().parse_args(argv)
    try:
        detector = OpenCVHOGPersonDetector(
            score_threshold=args.score_threshold,
            min_edge_density=args.min_edge_density,
            min_contrast=args.min_contrast,
        )
        suppressor = ReflectionSuppressor(mirror_evidence_threshold=args.mirror_evidence_threshold)
        surface_segmenter = _build_surface_segmenter(args.mirror_mask, args.glass_mask)
        pipeline = PersonCountingPipeline(
            detector,
            suppressor=suppressor,
            surface_segmenter=surface_segmenter,
        )
        result = pipeline.analyze_path(args.image)
        if args.annotated_output is not None:
            source_image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if source_image is None:
                raise OSError(f"Could not read the image for annotation: {args.image}")
            annotated = annotate_image(source_image, result.tagged_detections)
            args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.annotated_output), annotated):
                raise OSError(f"Could not write annotated image: {args.annotated_output}")
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"error: {error}")
        return 2

    payload = result.to_dict()
    if args.annotated_output is not None:
        payload["annotated_image"] = str(args.annotated_output)
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _build_surface_segmenter(
    mirror_mask_path: Path | None,
    glass_mask_path: Path | None,
) -> StaticSurfaceSegmenter | None:
    """Build a deterministic mask provider from optional grayscale mask files."""
    if mirror_mask_path is None and glass_mask_path is None:
        return None
    mirror_mask = _read_mask(mirror_mask_path) if mirror_mask_path is not None else None
    glass_mask = _read_mask(glass_mask_path) if glass_mask_path is not None else None
    if mirror_mask is None and glass_mask is not None:
        mirror_mask = np.zeros_like(glass_mask, dtype=bool)
    if glass_mask is None and mirror_mask is not None:
        glass_mask = np.zeros_like(mirror_mask, dtype=bool)
    if mirror_mask is None or glass_mask is None:
        raise ValueError("at least one surface mask must be provided")
    if mirror_mask.shape != glass_mask.shape:
        raise ValueError("mirror and glass mask files must have the same dimensions")
    return StaticSurfaceSegmenter(
        SurfaceSegmentation(
            mirror_mask=mirror_mask,
            glass_mask=glass_mask,
            source="cli-mask",
            confidence=1.0,
        )
    )


def _read_mask(path: Path) -> np.ndarray:
    """Read a grayscale image and convert all non-zero pixels to a boolean mask."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read a grayscale mask from: {path}")
    binary_mask: np.ndarray = mask > 0
    return binary_mask
