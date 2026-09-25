# QA skill

Every bug fix gets a regression test. Keep unit tests isolated from model weights and network access, then add integration tests at the CLI boundary. Cover happy paths, malformed input, empty detections, overlapping detections, reflection pairs, and plausible non-reflection pairs.

Prefer test doubles that implement the same public protocol as production components.
