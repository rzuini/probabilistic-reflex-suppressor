"""Allow the package to run with ``python -m probabilistic_reflex_suppressor``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
