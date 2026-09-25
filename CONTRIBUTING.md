# Contributing

Use English for code, tests, documentation, and command-line output. Start with `AGENTS.md`, then read the relevant document under `docs/`.

Keep pull requests focused. A behavior change should include a regression test and, when it changes a user-visible assumption, a README update. Use synthetic or openly licensed fixtures only.

Before opening a pull request, run:

```bash
make check
```

If the change affects the CLI, also run the command against a small local image and inspect the JSON output. Do not commit generated outputs, model weights, virtual environments, or private images.
