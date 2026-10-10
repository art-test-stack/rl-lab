# Repository Guidelines

## Project Structure and Module Organization

This is a Python 3.14+ package using a `src` layout. Production code lives in `src/rl_lab/`, organized by feature: `engines/` contains inference backends, `tasks/` benchmark adapters, `eval/` evaluation logic, and `metrics/`, `tokenizer/`, `tools/`, and `utils/` supporting modules.

Tests live under `tests/` and mirror those areas, including `tests/engines/`, `tests/tasks/`, and `tests/eval/`. Put user-facing technical notes in `docs/`; runnable examples and utilities belong in `scripts/`.

## Build, Test, and Development Commands

- `uv sync --locked --group dev` installs the exact locked runtime and test dependencies.
- `uv run --locked pytest` runs the complete CPU test suite with Hub access disabled by the fixtures.
- `uv run --locked pytest tests/engines/test_router.py` runs a focused test module.

The package uses Hatchling as its build backend. Keep `uv.lock` synchronized when changing dependencies.

## Coding Style and Naming Conventions

Follow the surrounding Python style: four-space indentation, descriptive `snake_case` for modules, functions, and variables, and `CapWords` for classes.

Keep engine-specific dependencies inside their backend modules so importing the package or selecting another backend does not require them. Use normal top-level imports inside a selected backend module; keep backend selection lazy at the factory or package boundary.

Prefer small, explicit interfaces and document behavioral limits in docstrings or `docs/`. No formatter or linter is configured in `pyproject.toml`; match nearby code and keep changes focused.

## Development Principles

- Implement changes in small, independently understandable steps with a clear purpose.
- Prefer simple, direct implementations before introducing abstractions.
- Avoid speculative abstractions or machinery without a demonstrated requirement.
- Keep behavioral, API, dependency, and workflow changes explicit and scoped.

## Testing Guidelines

Use pytest. Name files `test_<feature>.py` and tests `test_<behavior>`. Add or update tests alongside changes to engine routing, prompt handling, task scoring, or resource cleanup.

The current suite uses small local models and mocks optional runtime calls. GPU integration checks should remain separate from the locked CPU suite. Keep tests focused on the behavior changed and its documented limits.

## Commit and Pull Request Guidelines

Use short, descriptive commit subjects, usually imperative, such as `Add tool completion to the Transformers engine`. Keep each commit focused.

Pull requests should explain the behavior changed, link a related issue when available, and include the relevant test command and result. Add documentation or example updates when a public API or supported behavior changes. Include screenshots only when a change has a visual component.

## Configuration and Compatibility

Keep Python and core dependency requirements consistent with `pyproject.toml` and `uv.lock`. Do not commit credentials, local model files, or generated evaluation artifacts. State any backend or hardware requirements in the relevant documentation.
