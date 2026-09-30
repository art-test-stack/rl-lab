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

Implement work in small, independently understandable steps, each with a clear purpose. Before each step that changes behavior, API, dependencies, or workflow, present only the next concrete choice, its options and tradeoffs, and a recommendation; wait for the user's decision before implementing it. Do not jump ahead to later steps or add speculative abstractions. Prefer a naive, direct first implementation and introduce complexity only when a demonstrated need or user choice justifies it. Once the user settles a choice, follow it without reopening it unless a new constraint appears. Continue independent inspection while a decision is pending.

For the Transformers engine rewrite, first build and manually run `scripts/tf_engines.py` against a tiny Hugging Face causal LM using direct Transformers calls. Use its observed behavior to guide the wrapper only in later, separately approved steps. For this initial rewrite, use a simple `model.eval()` / `model.train()` toggle; do not add per-submodule mode snapshots or restoration machinery.

## Testing Guidelines

Use pytest. Name files `test_<feature>.py` and tests `test_<behavior>`. Add or update tests alongside changes to engine routing, prompt handling, task scoring, or resource cleanup. The current suite uses small local models and mocks optional runtime calls; GPU integration checks should remain separate from the locked CPU suite.

For each implementation step, keep tests scoped to that step and its stated purpose. When test approach itself has meaningful choices—such as unit versus integration coverage, mocks versus real components, or where behavior should be asserted—describe the options and recommendation before writing tests, then wait for the user's choice. Proceed directly when the user has already specified the approach. Avoid broad test suites or extra cases that do not verify the agreed behavior.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive subjects, often imperative (for example, `Add offline SGLang generation engine and factory routing`). Keep each commit focused. Pull requests should explain the behavior changed, link a related issue when available, and include the relevant test command and result. Add documentation or example updates when a public API or supported behavior changes; include screenshots only when a change has a visual component.

## Configuration & Compatibility

Keep Python and core dependency requirements consistent with `pyproject.toml` and `uv.lock`. Do not commit credentials, local model files, or generated evaluation artifacts. State any backend or hardware requirements in the relevant documentation.
