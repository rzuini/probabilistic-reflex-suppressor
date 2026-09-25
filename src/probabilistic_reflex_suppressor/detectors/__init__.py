"""Detector implementations and interfaces."""

from .base import PersonDetector
from .opencv_hog import OpenCVHOGPersonDetector

__all__ = ["OpenCVHOGPersonDetector", "PersonDetector"]
