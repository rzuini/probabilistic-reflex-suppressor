# Agent-neutral project guidance

This repository is designed to be usable by any coding agent or human contributor. These rules are intentionally tool-agnostic and apply to the entire repository.

## Scope and language

- Product code, tests, documentation, command output, and commit messages use English.
- Keep the public API small, typed, and modular.
- Prefer deterministic, offline tests. Do not add real-person images without permission and a clear license.

## Change workflow

1. Read the relevant module, tests, and `docs/architecture.md` before editing.
2. Make the smallest change that solves the problem at its root.
3. Add or update tests for behavior changes, especially reflection edge cases.
4. Update `README.md` and documentation when user-visible behavior or assumptions change.
5. Run `make check` before handing off. The Makefile is Docker-first: project commands run in disposable containers, not on the host Python installation.
6. If Docker is unavailable, report the exact skipped command and daemon error.

## Quality gates

- `make format-check` must pass.
- `make lint` must pass without suppressing a finding broadly.
- `make typecheck` must pass, or a narrowly scoped dependency limitation must be documented.
- `make test` must pass.
- Keep tests extensible: prefer injected detector implementations over monkeypatching internals.
- Use `make run ARGS="..."` for CLI smoke tests so the runtime is also ephemeral.

## Computer-vision safety

- A single image cannot prove that a person is real or that a visible figure is a reflection.
- Reflection suppression is a probabilistic heuristic and must remain visible in result metadata.
- Never describe a heuristic result as identity, surveillance, or certainty.
- Preserve raw detections in APIs where practical so downstream users can audit suppression decisions.

## Code style

- Use type hints and useful docstrings for public modules, classes, and functions.
- Comments should explain decisions and assumptions, not restate syntax.
- Avoid one-letter names except for conventional coordinates or loop indices in small scopes.
- Keep I/O at the CLI/application boundary; keep the core pipeline testable with arrays and injected interfaces.
