# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.14+ package using a `src` layout. Production code lives in `src/rl_lab/`, organized by feature: `engines/` contains inference backends, `tasks/` benchmark adapters, `eval/` evaluation logic, and `metrics/`, `tokenizer/`, `tools/`, and `utils/` supporting modules. Tests live under `tests/` and mirror those areas (`tests/engines/`, `tests/tasks/`, `tests/eval/`). Put user-facing technical notes in `docs/`; runnable examples and utilities belong in `scripts/`.

## Build, Test, and Development Commands

- `uv sync --locked --group dev` installs the exact locked runtime and test dependencies.
- `uv run --locked pytest` runs the complete CPU test suite with Hub access disabled by the fixtures.
- `uv run --locked pytest tests/engines/test_router.py` runs a focused test module.

The package uses Hatchling as its build backend. Keep `uv.lock` synchronized when changing dependencies; optional inference backends such as vLLM and SGLang are installed separately and are not part of the base test setup.

## Coding Style & Naming Conventions

Follow the surrounding Python style: four-space indentation, descriptive `snake_case` for modules, functions, and variables, and `CapWords` for classes. Keep engine-specific dependencies inside their backend modules so importing the package or selecting another backend does not require them. Prefer small, explicit interfaces and document behavioral limits in docstrings or `docs/`. No formatter or linter is configured in `pyproject.toml`; match nearby code and keep changes focused.

## Design Checkpoints

Before implementing a feature or fix, identify choices that would change a public API, behavior, dependencies, or user workflow. Briefly describe the concrete options, their tradeoffs, and a recommendation, then ask the user to choose. Continue investigation that does not depend on the choice, but wait before implementing the dependent design. Skip this checkpoint for straightforward changes or when the user has already specified the design. After the user chooses, implement that choice without reopening settled decisions unless new constraints appear.

## Testing Guidelines

Use pytest. Name files `test_<feature>.py` and tests `test_<behavior>`. Add or update tests alongside changes to engine routing, prompt handling, task scoring, or resource cleanup. The current suite uses small local models and mocks optional runtime calls; GPU integration checks should remain separate from the locked CPU suite.

When a change has meaningful testing choices—such as unit versus integration coverage, mocks versus real components, or where a behavior should be asserted—describe the options, tradeoffs, and recommendation before writing tests, then wait for the user's choice. For straightforward coverage or when the user has specified the test approach, proceed directly. Keep test design aligned with the selected implementation behavior and avoid adding coverage beyond the agreed scope.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive subjects, often imperative (for example, `Add offline SGLang generation engine and factory routing`). Keep each commit focused. Pull requests should explain the behavior changed, link a related issue when available, and include the relevant test command and result. Add documentation or example updates when a public API or supported behavior changes; include screenshots only when a change has a visual component.

## Configuration & Compatibility

Keep Python and core dependency requirements consistent with `pyproject.toml` and `uv.lock`. Do not commit credentials, local model files, or generated evaluation artifacts. State any backend or hardware requirements in the relevant documentation.
