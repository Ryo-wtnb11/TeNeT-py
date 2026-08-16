# Repository rules

Process rules for `TeNeT-py`. Design rules live in `docs/design.md`
(authoritative) and `AGENTS.md` (condensed).

## Source of truth

- `docs/design.md` is the design document. Code follows it; if implementation
  experience shows the design is wrong, update `docs/design.md` in the same PR
  and say why.
- `README.md` is the user-facing page, not a design document. It may fall out
  of date with the design only in the sense of omitting things, never of
  contradicting them.
- API is unstable pre-1.0. Breaking changes are allowed but must be
  intentional and noted in the commit/PR message.

## Branches and commits

- `main` must stay green (CI passing).
- Work on feature branches; merge to `main` via PR when collaborating.
  Solo direct commits to `main` are acceptable while the project is
  single-author, but CI must pass.
- Commit messages: imperative summary line ≤ 72 chars; body explains *why*
  when non-obvious. Reference the `docs/design.md` milestone when relevant
  (e.g. `M1: add SU2 fusion provider`).

## Code

- Python ≥ 3.12, src layout (`src/tenet/`).
- Formatting and linting: `ruff` (config in `pyproject.toml`). No other
  formatters.
- Type annotations on all public functions and classes.
- Structural/categorical types are immutable.
- Dependencies: keep minimal. Core depends on `numpy`, `autoray` and
  `opt-einsum` only. JAX/PyTorch are optional extras (`tenet-py[jax]`), never
  hard requirements; core never imports them.
  - `opt-einsum` was added for M8 (#67), which needs a contraction-path finder
    for `einsum` over three or more operands. It is a 72 KB pure-Python wheel
    with zero runtime dependencies (it does not import NumPy), BSD-3, and
    already arrives transitively with JAX. The alternatives were worse: an
    in-tree path finder is exactly the reinvention this rule exists to prevent,
    and an extra would put a *packaging* cliff in the middle of the primary
    tensor-network API — `einsum` working for two operands and refusing three
    depending on what else is installed, which is a refusal that is not a
    categorical statement, unlike every other refusal in that module. It is
    imported lazily, inside the three-or-more-operand branch, so `import tenet`
    and the pairwise path never pay for it.
  - `scipy` was **not** added for `expm` (#86), and the contrast with `opt-einsum`
    is the reason: it is a large compiled, platform-specific wheel with its own
    dependency tree, i.e. it fails every criterion that admitted `opt-einsum`.
    `tenet.linalg.expm` on the NumPy backend therefore raises an `ImportError`
    naming `pip install scipy` — autoray resolves NumPy's `linalg.expm` to
    `scipy.linalg.expm`, and NumPy ships no matrix exponential — while the JAX
    backend needs nothing extra. That is a packaging cliff of the kind the
    `opt-einsum` entry above refused, and it is accepted here only because it is
    one function rather than the primary contraction API, and because the message
    names the fix. `scipy` sits in the dev group (it already arrives transitively
    via JAX and quimb) so the NumPy-backend `expm` tests run in CI.

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
