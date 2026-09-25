---
name: docker
description: Run this project's development, validation, and CLI commands in disposable Docker containers instead of relying on host Python tooling.
---

# Docker Ephemeral Runtime

Use this skill whenever a project command, test, formatter, linter, type checker, or CLI smoke test must run in `probabilistic-reflex-suppressor`.

## Required behavior

- Use the repository `Makefile`; it builds the `development` target from `Dockerfile` and runs commands with `docker run --rm`.
- Do not install project dependencies into the host interpreter or create a host virtual environment for normal workflows.
- Keep the working tree mounted at `/workspace` so formatter changes and test fixtures remain available to the user.
- Keep caches and coverage data in the container's `/tmp` paths. Do not add generated caches to the repository.
- Use `make run ARGS="..."` for CLI execution and `make check` for the full quality gate.

## Image boundaries

- `Dockerfile` target `production` contains only runtime dependencies and the CLI entrypoint.
- `Dockerfile` target `development` contains development tools and has no entrypoint, allowing Make targets to invoke Python tools directly.
- Every command container is disposable. Rebuild with `make docker-build` when dependency or Dockerfile inputs change.

## Failure handling

If Docker is unavailable, stop and report the daemon or executable error. Do not silently fall back to host Python, because that would invalidate the reproducibility guarantee.
