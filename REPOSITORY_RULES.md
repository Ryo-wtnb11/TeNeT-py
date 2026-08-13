# Repository rules

Process rules for `TeNeT-py`. Design rules live in `README.md` (authoritative)
and `AGENTS.md` (condensed).

## Source of truth

- `README.md` is the design document. Code follows it; if implementation
  experience shows the design is wrong, update `README.md` in the same PR and
  say why.
- API is unstable pre-1.0. Breaking changes are allowed but must be
  intentional and noted in the commit/PR message.

## Branches and commits

- `main` must stay green (CI passing).
- Work on feature branches; merge to `main` via PR when collaborating.
  Solo direct commits to `main` are acceptable while the project is
  single-author, but CI must pass.
- Commit messages: imperative summary line ≤ 72 chars; body explains *why*
  when non-obvious. Reference the README milestone when relevant
  (e.g. `M1: add SU2 fusion provider`).

## Code

- Python ≥ 3.12, src layout (`src/tenet/`).
- Formatting and linting: `ruff` (config in `pyproject.toml`). No other
  formatters.
- Type annotations on all public functions and classes.
- Structural/categorical types are immutable.
- Dependencies: keep minimal. Core depends on `numpy` and `autoray` only.
  JAX/PyTorch are optional extras (`tenet-py[jax]`), never hard requirements;
  core never imports them.

## Tests

- Framework: `pytest`, under `tests/`, mirroring `src/tenet/`.
- Every feature PR includes tests. Bug fixes include a regression test.
- Non-Abelian features must be tested with SU(2), not only Trivial/U(1).
- Where feasible, verify against explicit dense expansion (`to_dense()`
  reference results).
- No network access, no GPU requirements in CI tests. Backend-specific tests
  (JAX/PyTorch) are skipped automatically when the backend is not installed;
  CI installs CPU JAX so those tests do run there.

## CI

- Workflow: `.github/workflows/ci.yml`.
- Required checks: `ruff check`, `ruff format --check`, `pytest`.
- A PR/commit that turns `main` red gets fixed or reverted immediately.
