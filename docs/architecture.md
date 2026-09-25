# Architecture

## Data flow

The CLI loads an image with OpenCV and passes the resulting BGR NumPy array to `PersonCountingPipeline`. The pipeline owns composition but not model details:

```text
image path -> image loader -> PersonDetector -> NMS -> ReflectionSuppressor -> tagged DetectionResult -> JSON/annotated-image renderers
```

`PersonDetector` is a `typing.Protocol`. The default `OpenCVHOGPersonDetector` is deliberately replaceable because model choice, licenses, accelerator support, and accuracy requirements vary by deployment.

The baseline deliberately does not require a surface-segmentation model. A separate `SurfaceSegmenter` protocol exists as an experimental extension for synthetic tests and externally generated grayscale masks; it is not part of the default execution path.

## Reflection heuristic

The suppressor first applies a mirror-evidence gate, then evaluates candidate pairs only when:

- a long straight edge near the candidate axis between the pair is compatible with a mirror boundary or panel;
- the boxes are on opposite sides of that candidate axis;
- their centers have similar vertical positions;
- their dimensions are similar;
- their horizontal centers are close to mirror positions around the candidate axis; and
- resized image crops have similar appearance after horizontal flipping.

When an experimental surface mask is available, the pair score rewards the expected pattern that one candidate overlaps a mirror region while its paired candidate remains outside. Glass overlap is retained for auditability but does not trigger automatic suppression because a person visible through a window may be real. Surface overlap by itself never silently removes a detection: it receives the `surface_overlap_uncertain` tag and remains counted until association evidence supports suppression.

The result is a decision aid, not ground truth. It stores both the retained detection and the reason a candidate was suppressed. The default policy keeps the higher-confidence detection and breaks exact ties deterministically by index.

Each post-NMS detection receives a `counted_person`, `surface_overlap_uncertain`, or `mirror_reflection` tag. All tags remain available to the JSON renderer and annotated-image renderer; only detections with `counted: true` contribute to `count`.

If the mirror-evidence gate is not met, the pair remains counted as two detections. This intentionally reduces false reflection suppression in ordinary scenes at the cost of missing some reflections. The threshold is configurable through `ReflectionSuppressor` for calibration against representative data. This implementation does not claim to discover every mirror region; a future adapter can provide a dedicated mirror segmentation model or a calibrated scene prior while preserving the same interface.

## Extension points

- Add a detector in `detectors/` implementing `PersonDetector`.
- Add a model-backed segmenter implementing `SurfaceSegmenter` as a separately evaluated extension.
- Add a separate model-backed reflection scorer implementing the suppressor protocol.
- Add a renderer without changing the core result dataclasses.
- Add video tracking above the pipeline to use temporal consistency, which is usually stronger evidence than one frame.
