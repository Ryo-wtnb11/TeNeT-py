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
sweep and the convergence loop — `MPS`, `MPO`, `Env`, `lanczos`, `sweep_`, `dmrg_`. Every
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

## The same Hamiltonian, listed as terms

`MPO.from_terms` takes `(coefficient, [(operator, site), …])` tuples, with identities
implied on every untouched site. Each operator is rank 3 — `local_op` puts the charge it
emits on a third `D=1` leg, which is what makes `S⁺` expressible at all — and the MPO bond
spaces are then *derived*: each term is a bond-1 MPO, `direct_sum` stacks them, and one
`svd_truncated` sweep collapses the result to the operator Schmidt rank.

```python
sp = network.local_op(sp_dense, phys=PHYS, charge=U1Sector(-2))
sm = network.local_op(sm_dense, phys=PHYS, charge=U1Sector(2))
terms = [(0.5, [(sp, i), (sm, i + 1)]) for i in range(n_sites - 1)] + ...
h = network.MPO.from_terms(n_sites, terms)
```

The two routes agree as operators, and `from_terms` recovers the hand-written `MPO_BOND`
sector for sector — `{0: 3, +2: 1, −2: 1}` — with no grading written down anywhere. The
example keeps `from_w` as its primary route because the 5×5 `W` and its channel table are
what teach what an MPO *is*.

## The same chain under SU(2)

Under SU(2) there is no `S_z` and no `S⁺`: they are not invariant, so they are not
operators you can write. What *is* invariant is the whole term. Hand it over as one array
— `local_op` with no `charge` takes `(d**k, d**k)` or `(d,)*2k`, which is the layout
`np.kron` already produces — and `MPO.from_terms` splits it with `svd_truncated`:

```python
PHYS = GradedSpace.new(SU2, {SU2Sector(1): 1})  # one spin-1/2 doublet
ss = network.local_op(np.kron(sz, sz) + (np.kron(sp, sm) + np.kron(sm, sp)) / 2, phys=PHYS)
h = network.MPO.from_terms(n_sites, [(1.0, [(ss, (i, i + 1))]) for i in range(n_sites - 1)])
```

Nothing about the recoupling is written down, and there is no coupling tree to name: it is
already inside the array's blocks, and the MPO bond comes out of the SVD. Hand it
`np.kron(sz, sz)` alone and `from_dense` *raises* — the DSL cannot express a
symmetry-breaking term. The bond spaces this derives:

| | SU(2) | U(1) |
|---|---|---|
| one `S·S` term's bond | `{2: 1}` — **1 block**, dense 3 | `{0: 1, ±2: 1}` — 3 blocks, dense 3 |
| bulk MPO bond | `{0: 2, 2: 1}` — **3 blocks**, dense 5 | `MPO_BOND` — 5 blocks, dense 5 |

The MPO's *dense* bond is 5 either way; what SU(2) buys there is three blocks instead of
five, matching MPSKit's finite Jordan form. The compression is on the MPS side: `dmrg_` on
this MPO reaches the same N=12 energy at a mid-chain bond of 12 multiplets where U(1)
needs 32 states, because `max_bond` bounds the *dense* dimension `Σ_c qdim(c)·m_c`.

## Why there is no `jit` and no `grad` here

DMRG is a fixed-point solver whose control flow is data-dependent at every level: the
truncation re-decides the bond space each sweep (a `StructureChangingError` under a trace,
by design), `lanczos`'s happy breakdown tests a norm against `tol`, and `dmrg_`'s loop
exits on a measured energy change. Every one of those is precisely what tenet refuses to
trace, and correctly. So this example runs on the eager NumPy backend and makes no
differentiability claim — and neither does `tenet.network`, which is outside `jit`/`grad`
by construction (see [Design](../design.md), M11).

The `svd_truncated`-outside / `svd(bond=)`-inside pairing has two legitimate halves, and
this example uses one: [CTMRG](ctmrg.md) needs the inside half because it differentiates
through its sweeps; DMRG needs only the outside half because it does not.

## What to do with the converged state

`dmrg_` hands back a `DMRG_out` whose `psi` is an ordinary `MPS`, and four calls cover
what a user wants from it on the first day — checkpoint it, read it back, measure a local
observable, and trade bond dimension for size:

```python
import tenet
from tenet import IN, OUT, Leg
from tenet.network import MPS, expectation_1site

out = dmrg(8, chi=32)  # examples/dmrg.py

out.psi.save("ground-state")  # a directory: 000.npz .. 007.npz plus mps.json
psi = MPS.load("ground-state")  # NumPy blocks; .to_backend("jax") per site to restore

sz = tenet.SymmetricTensor.from_dense(  # S^z on this example's physical space
    np.diag([-0.5, 0.5]), (Leg(PHYS, OUT), Leg(PHYS, IN))
)
print([expectation_1site(psi, sz, n) for n in range(len(psi))])  # <S^z_n>, normalized

discarded = psi.compress_(chi=8)  # in place; the total discarded weight, sqrt(sum_bond dw)
```

`expectation_1site` divides by `<psi|psi>`, so it is an expectation value on any state,
canonical or not — unlike `Env.measure()`, which returns the unnormalized `<psi|H|psi>`.
`compress_` returns the **total** discarded weight, where `sweep_` returns the per-bond
**maximum**: one answers "how much of my state did I throw away", the other "which bond is
the convergence diagnostic".

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

- [`tenet.network`](../api/network.md) — `MPS`, `MPO`, `Env`, `lanczos`, `sweep_`, `dmrg_`,
  `expectation_1site`, `expectation_2site`
- [`tenet.linalg`](../api/linalg.md) — `svd_truncated`
