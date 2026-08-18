# AGENTS.md

Guide for coding agents (and humans) working in this repository.

## What this project is

`TeNeT-py`: non-Abelian symmetric tensors with ndarray-style Python APIs and
backend-native numerical execution. **`docs/design.md` is the design document.**
Read the relevant section of it before implementing anything; it defines the
architecture, milestones, and invariants. `REPOSITORY_RULES.md` defines
process rules.

Three-layer architecture (never blur these):

```text
mathematical model   TensorMap semantics  (Hom(D,C), duality, fusion, F/R)
programming model    SymmetricTensor      (ordered ndarray-like legs)
execution model      backend-native reduced arrays (NumPy / JAX / PyTorch via autoray)
```

## Commands

```bash
uv sync                  # install (creates .venv, includes dev deps)
uv run pytest            # run tests
uv run ruff check .      # lint
uv run ruff format .     # format
```

CI runs exactly: `ruff check`, `ruff format --check`, `pytest`. All must pass.

## Layout

```text
src/tenet/       library code (src layout; see "Proposed package structure" in docs/design.md)
tests/           pytest tests, mirroring src/tenet/ structure
```

## Design invariants (condensed from docs/design.md — full list there is authoritative)

1. Every `SymmetricTensor` has exact TensorMap semantics: `domain`/`codomain`
   derived from leg metadata.
2. `side` (IN/OUT) and `dual` are independent. Never identify input with dual.
3. Public axis order is independent of domain/codomain grouping.
4. Fusion trees / intermediate sectors / multiplicities are tensor-level
   structure, never stored on individual `Leg`s.
5. Categorical operations are never *defined* by backend array operations;
   backend ops only implement lowered plans.
6. Reduced blocks contain only degeneracy indices; sector labels are metadata.
7. Reduced block axis order follows public tensor axis order, not
   codomain × domain.
8. Numerical leaves are backend-native arrays: `blocks: tuple[Array, ...]`
   ordered by `structure.block_order`. `TensorStructure` is immutable,
   hashable, and array-free — F/R and other coefficient arrays never go in
   structural fields. Do not subclass `numpy.ndarray` / `jax.Array` /
   `torch.Tensor`.
9. No implicit densification. Dense expansion only via explicit `to_dense()`.
10. Structural planning (categorical analysis) is separated from numerical
    execution; plans are static and cacheable. Structural logic runs in
    Python on static metadata (so JAX tracing / torch AD see only array
    ops); structure-changing operations (truncation, sector selection) live
    outside JIT/compile boundaries or use static shapes/masks.
11. Expose an ndarray-style operation only when its categorical meaning is
    defined; unsupported operations fail loudly.
12. Never invent an implicit braid convention. If an expression does not
    uniquely specify a braid, require explicit input.

## Coding rules

- Symmetry-specific math goes through capability protocols — `FusionRules` for
  structure, the `*Data` protocols (`FMatrixData`, `BendingCoefficients`, ...)
  for the coefficients a category is defined by, `*Provider` for a concrete
  symmetry. No `if symmetry == "su2":` branching in core code.
- Test non-Abelian behavior with SU(2) from the start; Abelian-only tests can
  hide wrong assumptions (external sectors ≠ complete fusion basis).
- Where feasible, validate numerical operations against explicit dense
  expansion in tests.
- A PR that adds a symmetry provider or a public operation updates
  `tests/COVERAGE.md` in the same diff: a provider names its column and the
  cells it fills; an operation names its row and the providers and modes it
  runs under. Any cell left empty carries one of the four classifications —
  genuine gap (with a follow-up), structurally redundant (with the argument),
  out of contract (with the refusal test), or blocked (with what blocks it) —
  in one line.
- Structural/categorical types are immutable (frozen dataclasses or similar).
- Backend abstraction is `autoray` only; keep it thin, don't build a second
  numerical framework. Framework-specific optimization (`jit`, `grad`,
  `torch.compile`) is application-level (symmray/quimb model).
- Integration surface: `get_params`/`set_params` (quimb-compatible) is core;
  JAX PyTree registration is opt-in via `tenet.pytree` (import-guarded, core
  never imports jax/torch).
- Follow the milestone order in docs/design.md ("Initial implementation strategy")
  unless told otherwise.
