"""CLI boundary tests using a patched detector to stay model-independent."""

import json
from pathlib import Path

import cv2
import numpy as np

from probabilistic_reflex_suppressor import cli
from probabilistic_reflex_suppressor.models import BoundingBox, Detection


class EmptyDetector:
    """Minimal detector for a valid empty-scene CLI path."""

    def detect(self, image: np.ndarray) -> list:
        """Return no observations."""
        return []


class SingleDetector:
    """Detector double used to verify CLI mask wiring."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return one detection covering the left half of the fixture."""
        return [Detection(BoundingBox(0, 0, 20, 40), 0.9)]


def test_cli_writes_json_output(monkeypatch, tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "empty.png"
    output_path = tmp_path / "nested" / "result.json"
    annotated_path = tmp_path / "nested" / "annotated.png"
    assert cv2.imwrite(str(image_path), np.zeros((40, 40, 3), dtype=np.uint8))
    monkeypatch.setattr(cli, "OpenCVHOGPersonDetector", lambda **kwargs: EmptyDetector())

    exit_code = cli.main(
        [
            str(image_path),
            "--output",
            str(output_path),
            "--annotated-output",
            str(annotated_path),
        ]
    )

    assert exit_code == 0
    terminal_result = json.loads(capsys.readouterr().out)
    file_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert terminal_result == file_result
    assert terminal_result["count"] == 0
    assert terminal_result["annotated_image"] == str(annotated_path)
    assert annotated_path.exists()


def test_cli_reports_missing_image(capsys) -> None:
    exit_code = cli.main(["does-not-exist.png"])

    assert exit_code == 2
    assert "could not read" in capsys.readouterr().out.lower()


def test_cli_accepts_external_surface_mask(monkeypatch, tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "scene.png"
    mask_path = tmp_path / "mirror-mask.png"
    assert cv2.imwrite(str(image_path), np.zeros((40, 40, 3), dtype=np.uint8))
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[:, :20] = 255
    assert cv2.imwrite(str(mask_path), mask)
    monkeypatch.setattr(cli, "OpenCVHOGPersonDetector", lambda **kwargs: SingleDetector())

    exit_code = cli.main([str(image_path), "--mirror-mask", str(mask_path)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["surface_segmentation"]["source"] == "cli-mask"
    assert payload["detections"][0]["tag"] == "surface_overlap_uncertain"
