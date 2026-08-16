# DMRG — a U(1) Heisenberg chain against exact diagonalization

Source: [`examples/dmrg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/dmrg.py).
**Oracle:** exact diagonalization of the same finite chain, plus the thermodynamic limit
`1/4 - ln 2` (Bethe 1931; Hulthén 1938) that `main()` reports against. The file is
executed by `tests/integration/test_dmrg.py`, so the code you read there is code that
runs — read it in the repository rather than a copy pasted here.

```sh
uv run python examples/dmrg.py
```

## What the library owns, and what the example owns

`tenet.network` owns the MPS container and its canonical form, the MPO builder, the
directed-bond environment cache with its invalidation, the Krylov step, the two-site
sweep and the convergence loop — `MPS`, `MPO`, `Env`, `lanczos`, `sweep`, `dmrg`. Every
one of those is identical for every finite-MPS algorithm.

The example owns the **physics**: the 5×5 Heisenberg `W` and its channel constants, the
`2 Sᶻ ∈ {-1, +1}` grading, and the reachable-charge bond spaces. One sentence draws the
line: *the library takes bond spaces; this file computes which spaces are reachable.*

## What it demonstrates

Nothing outside the core install — no `scipy`, no `quimb`, no `jax`.

- **A U(1) MPS whose target sector is fixed by the boundary legs alone.** The physical
  charge is `t = 2 Sᶻ ∈ {-1, +1}` and both boundary legs carry the unit sector with
  degeneracy 1. Invariance of every site tensor then forces `Σᵢ 2 Sᶻᵢ = 0`, i.e.
  `Sᶻ_tot = 0` — the sector an even chain's ground state lives in, enforced structurally
  and for free: no penalty term, no projector, no `project=` argument. The general recipe
  is a `D=1` boundary leg carrying `U1Sector(q)`, targeting `Sᶻ_tot = q/2`.
- **An MPO built by `SymmetricTensor.from_dense`** at the default relative `atol`, from
  the 5×5 `W` written out in the carrier basis on a graded MPO bond. A wrong grading makes
  `from_dense` *raise*, and that refusal — asserted in the integration test and again in
  `tests/network/test_mpo.py` — is the proof the grading is right. A passing `allclose`
  would not be.
- **`tenet.linalg.svd_truncated` deciding a bond `GradedSpace` at every bond of every
  sweep**, with the discarded weight reported by Pythagoras — the mirror image of CTMRG's
  frozen bond.
- **An iterative Krylov eigensolver written over `SymmetricTensor` as a vector.**
  `tenet.network.lanczos` needs only `tenet.add`/`subtract`, scalar multiply and divide,
  `tenet.norm` and an inner product. No `scipy.sparse.linalg`, no dense reshaping of the
  local problem.

## Why there is no `jit` and no `grad` here

DMRG is a fixed-point solver whose control flow is data-dependent at every level: the
truncation re-decides the bond space each sweep (a `StructureChangingError` under a trace,
by design), `lanczos`'s happy breakdown tests a norm against `tol`, and `dmrg`'s loop
exits on a measured energy change. Every one of those is precisely what tenet refuses to
trace, and correctly. So this example runs on the eager NumPy backend and makes no
differentiability claim — and neither does `tenet.network`, which is outside `jit`/`grad`
by construction (see [Design](../design.md), M11).

The `svd_truncated`-outside / `svd(bond=)`-inside pairing has two legitimate halves, and
this example uses one: [CTMRG](ctmrg.md) needs the inside half because it differentiates
through its sweeps; DMRG needs only the outside half because it does not.

## Deliberate limits

- **Two-site DMRG only.** Single-site plus subspace expansion (Hubig–McCulloch–
  Schollwöck–Wall, PRB 91, 155115 (2015)) needs `tenet.linalg.left_null`, a mixing factor,
  its own schedule and a second `heff1` contraction chain. Named upgrade path.
- **Hand-written pairwise contraction orders**, not `optimize=` on a five-operand einsum:
  `opt_einsum` costs a graded network from *physical* leg sizes, and a U(1) MPS bond with
  unevenly filled sectors is exactly where that estimate is wrong.
- **The MPO is written out, not generated.** `tenet.network.MPO.from_w` takes the array;
  an `Hterm`-style generator is the right API for arbitrary Hamiltonians and the wrong
  thing for a file whose Hamiltonian is one line of physics.

## Reference

- [`tenet.network`](../api/network.md) — `MPS`, `MPO`, `Env`, `lanczos`, `sweep`, `dmrg`
- [`tenet.linalg`](../api/linalg.md) — `svd_truncated`
