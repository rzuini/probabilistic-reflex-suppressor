# Testing and calibration

The test suite separates deterministic counting behavior from model behavior. Core tests use injected fake detections and synthetic arrays, so they are fast, reproducible, and safe to run offline. OpenCV model smoke tests can be added separately when a deployment has the required model assets.

The current baseline is the probabilistic reflection fallback: it uses person detections, local mirror-boundary evidence, geometry, and crop appearance. Surface masks are covered only as an experimental extension and are not required for the default CLI path.

## Recommended additions

1. Add synthetic scenes with a central mirror, off-center mirror, partial reflection, two real people, and an empty room.
2. Store expected boxes and reflection labels as small JSON fixtures rather than binary images when possible.
3. Add openly licensed validation images only with provenance and consent documentation.
4. Measure precision/recall for both person detection and reflection suppression independently.
5. Tune thresholds on a validation split and keep a fixed regression split untouched.

## What to monitor

- false suppression of two real, similarly positioned people;
- missed reflections when the mirror is off-center or tilted;
- detector confidence changes across OpenCV versions;
- image-size and crop-quality effects on appearance similarity;
- the rate of `warnings` and low-confidence decisions in production logs.
