"""Tests for the portable OpenCV detector adapter."""

import numpy as np

from probabilistic_reflex_suppressor.detectors.opencv_hog import OpenCVHOGPersonDetector


def test_hog_detector_handles_images_smaller_than_its_window() -> None:
    detector = OpenCVHOGPersonDetector()

    assert detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []


def test_hog_detector_rejects_low_information_regions() -> None:
    detector = OpenCVHOGPersonDetector()
    image = np.zeros((160, 120, 3), dtype=np.uint8)

    assert not detector._has_visual_support(image, 20, 20, 60, 100)
