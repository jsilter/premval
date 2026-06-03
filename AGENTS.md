# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`premval` (Protein ensemble evaluation tooling). Python 3.12+, src-layout, built with `poetry-core`. The package lives at `src/premval/` and is declared via `[tool.poetry] packages = [{include = "premval", from = "src"}]` in `pyproject.toml` (required because the src layout is not auto-discovered by poetry-core).

## Commands

A project virtualenv at `.venv/` has `premval`, dev tools, and Modal client all
installed. Prefer `.venv/bin/<tool>` over the shell PATH; the global tools (e.g.
the `uv tool`-installed `modal`) run under their own isolated Python and will
not see project deps. Example: `.venv/bin/modal run inference/esmflow_modal.py`,
`.venv/bin/pytest`, `.venv/bin/mypy`.

Install in editable mode with dev tools:

```bash
pip install -e ".[dev]"
```

Run the full test suite, a single file, or a single test:

```bash
.venv/bin/pytest
.venv/bin/pytest tests/test_version.py
.venv/bin/pytest tests/test_version.py::test_version
```

Lint and type-check:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/mypy
```

`mypy` is configured with `strict = true` over `src` and `tests`; new code is expected to type-check cleanly under strict mode.

## Conventions

- Ruff: line length 100, target `py312`, lint rules `E, F, I, W, B, UP`.
- The package ships a `py.typed` marker; keep public APIs annotated so downstream consumers get type info.

## Coding standards

Read the full rulebook (DRY, YAGNI, KISS, SOLID, docstring style, and
agent-specific guardrails) before non-trivial work:

@CODING_STANDARDS.md
