# DMRG — a U(1) Heisenberg chain against exact diagonalization

Source: [`examples/heisenberg_walkthrough.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/heisenberg_walkthrough.py)
— the library route, with the symmetry input spelled out. The same chain with the
algorithm written out by hand instead of called is
[`examples/toy_codes/dmrg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/dmrg.py),
whose page is [Toy DMRG](../examples/toy-dmrg.md).
**Oracle:** exact diagonalization of the same finite chain, plus the thermodynamic limit
`1/4 - ln 2` (Bethe 1931; Hulthén 1938) that `main()` reports against. The file is
executed by `tests/test_examples.py` and `tests/integration/test_dmrg.py`, so the code you
read there is code that runs — read it in the repository rather than a copy pasted here.

```sh
uv run python examples/heisenberg_walkthrough.py
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

That convention matters when you write a schedule. SUNDMRG.jl's `m` counts SU(N)
*multiplets*; tenet's `chi` bounds the dense dimension, and `svd_truncated`'s own
docstring says of the two: "For U(1) and fermionic parity these coincide; for SU(2) they
do not, and that will surprise people." A `Sweep(chi=64)` on SU(2) legs therefore keeps
roughly a third the multiplets it keeps on U(1) legs — the same state, held in fewer,
larger irreps — so an SU(2) ramp written in U(1) habits is a tighter ramp than it looks.

## Schedules and noise

`dmrg_`'s flat `chi`/`cutoff` kwargs are one spelling of a one-entry schedule. The other
spelling is `schedule=`, a list of `Sweep(chi, cutoff, noise)` entries — one record per
sweep, **whose last entry repeats** until convergence or `max_sweeps` (both block2 and
SUNDMRG.jl pad this way, independently):

```python
from tenet.network import Sweep, dmrg_

dmrg_(psi, h, chi=64)                              # today, unchanged
dmrg_(psi, h, schedule=[Sweep(chi=64)])            # the same run, exactly
dmrg_(psi, h, schedule=[Sweep(32, noise=1e-4)] * 4
              + [Sweep(64, noise=1e-5)] * 4
              + [Sweep(64)], max_sweeps=20)        # ramp, cool down, converge
```

Passing `schedule` together with `chi` or `cutoff` raises — silently letting one win is
how a run reports a `chi` it did not use. `DMRG_out.schedule` records the *realized*
schedule, one entry per sweep run, so `zip(out.schedule, out.history)` is exact and
`out.schedule` answers whether the run actually reached its final `chi`. A `callback=`
reports each `DMRG_out` while the run happens; it has no early-stop protocol.

`noise` is **wavefunction noise** — block2's `NoiseTypes::Wavefunction`, the one mixer an
SVD-splitting sweep admits: a random symmetric tensor over the two-site tensor's own legs,
added at relative strength `noise` after the eigensolver and before the split, then
renormalized. It fills every *structurally allowed* coupled sector of the two-site map,
including the ones the eigensolver left numerically empty and which the truncation
therefore dropped from the bond — the local minimum a symmetric DMRG falls into, since a
sector that is zero stays zero forever otherwise. It cannot reach outside
`bond_l ⊗ phys`; no wavefunction noise can, and neither can two-site DMRG itself. Noise is
not variational — a noisy sweep's energy can sit above its clean twin — so `dmrg_` never
declares convergence on a sweep that is still noisy or still inside the schedule: the
exit requires the last entry *and* `noise == 0.0`, exactly block2's guard. Taper it:
`1e-4` early, `1e-5` in the middle, `0.0` at the end.

## Targeting a sector: `MPS.product`

The general recipe above — a `D=1` boundary leg carrying `U1Sector(q)` targets
`Sᶻ_tot = q/2` — stops being prose with `MPS.product`: one sector per site, bonds
*derived* backwards from the charges, the total landing on bond 0 where it is printable
and assertable.

```python
neel = MPS.product(PHYS, [U1Sector(1), U1Sector(-1)] * 4)   # bond 0: {U1Sector(0): 1}
up   = MPS.product(PHYS, [U1Sector(1)] * 4)                 # bond 0: {U1Sector(4): 1}
```

The result has a single dense amplitude of exactly 1.0, and `dmrg_` grows the `D=1` seed
by a factor of `d` per sweep exactly as it grows the random seed. `MPS.product` is
Abelian-only, permanently: a single sector is not a non-Abelian multiplet, so under SU(2)
it refuses, and the route is `MPS.random` with a charged boundary leg.

## Restarting

Restart needs no argument. `MPS.save` the state, `MPS.load` it back, and re-enter the
schedule at a slice:

```python
out = dmrg_(psi, h, schedule=schedule[:2])   # ... interrupted after two sweeps
psi = MPS.load("checkpoint")
out = dmrg_(psi, h, schedule=schedule[2:])   # matches the uninterrupted run
```

There is no `forward=` to restore because a tenet sweep is a full round trip, and no
`sweep_start=` because the slice *is* the position. `tests/network/test_dmrg.py` verifies
the resumed run matches an uninterrupted one.

## Extrapolating in the discarded weight

The truncation error is linear in the discarded weight near convergence (Schollwöck 2011),
so a fit of energy against discarded weight extrapolates to the `chi → ∞` energy. Do
**not** fit the forward run's history — block2's tutorial is explicit that those energies
are not converged at their bond dimensions. The protocol is a *reverse* schedule on the
converged state, and the schedule feature is what makes it expressible:

```python
out = dmrg_(psi, h, chi=64)                                  # converge first
rev = dmrg_(psi, h, schedule=[Sweep(16)] * 2 + [Sweep(12)] * 2 + [Sweep(8)] * 2,
            energy_tol=0.0, max_sweeps=6)                    # nothing exits early
dws      = [h[3] for h in rev.history]
energies = [h[0] for h in rev.history]
e_extrapolated = np.polyfit(dws, energies, 1)[1]
```

block2's conventions, worth copying verbatim: zero noise throughout the reverse run;
`energy_tol=0.0` so no sweep exits early; an even number of sweeps per `chi` (odd and
even half-sweeps report different discarded weights); the first reverse `chi` slightly
below the last forward one; and the error bar quoted as one fifth of the extrapolation
distance, as a convention. The protocol has one hard prerequisite: **every sweep must be
two-site**, because a one-site sweep at zero noise reports near-zero discarded weights —
and tenet is two-site only, so the deliberate limitation is, for this one feature, an
outright advantage. `tests/integration/test_dmrg.py` runs the recipe at N=12 and the fit
lands two orders of magnitude closer to the exact energy than the `chi=8` run.

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

One piece *inside* the sweep is traceable, and `dmrg_` takes a callable for it. The
two-site matvec is a pure function of a fixed contraction structure, so it can be handed
to `jax.jit` — the caller supplies the callable and the `jax` extra, since this layer
names no accelerator:

```python
import jax                       # pip install "tenet-py[jax]"
import tenet.pytree              # registers SymmetricTensor as a JAX pytree

dmrg_(psi, h, chi=64)                    # the plain run, NumPy, no extra
dmrg_(psi, h, chi=64, compile=jax.jit)   # the same run, matvec compiled
```

**Read the caveat before reaching for it.** The compiled matvec is real — measured 10.6x
faster per call on this lattice model, and 1.8–3.0x on ab initio integrals, where the
work is already inside BLAS. But the *sweep* around it is not traceable, so on the JAX
backend a compiled run is today slower end to end than the plain NumPy one; and `Env`
rebuilds its prepared operator at every bond visit and so calls `compile` again each
time, one XLA trace per bond per sweep. The measured grid, and what it says about where
the constant factor actually lives, is in [Design](../design.md), M54.

## What to do with the converged state

`dmrg_` hands back a `DMRG_out` whose `psi` is an ordinary `MPS`, and four calls cover
what a user wants from it on the first day — checkpoint it, read it back, measure a local
observable, and trade bond dimension for size:

```python
import tenet
from tenet import IN, OUT, Leg
from tenet.network import MPS, expectation_1site

out = dmrg(8, chi=32)  # examples/heisenberg_walkthrough.py

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

### Measuring the converged state

Four more calls take the state where the ones above take one site of it:

```python
from tenet.network import correlation_function, expectation_profile, measure_mpo, overlap

print(expectation_profile(psi, sz))        # <S^z_n> for every n, in ONE pass over the chain
print(overlap(phi, psi))                   # <phi|psi>, undivided
print(measure_mpo(phi, h, psi))            # <phi|H|psi>, undivided
print(correlation_function(psi, sz3, sz3, pairs=[(0, j) for j in range(1, len(psi))]))
```

- **`expectation_profile` is the one to reach for over a list comprehension.** Writing
  `[expectation_1site(psi, sz, n) for n in range(len(psi))]` costs two full-chain transfer
  passes *per site*; the profile moves the orthogonality centre once along the chain and
  reads the operator off it, which a canonical MPS makes exact. Same numbers, `O(N)`
  instead of `O(N²)`.
- **`overlap` and `measure_mpo` do not divide**, matching `Env.measure()`. A fidelity is
  `overlap(phi, psi) / (phi.norm() * psi.norm())` and the caller writes the division; the
  divided readings are the ones whose names say `expectation`.
- **`correlation_function` takes the rank-3 charged operators** `MPO.from_terms` takes —
  the form a fermionic `c` has to have — and returns `{(i, j): value}` for `i < j`, every
  pair by default. The Jordan-Wigner string across the sites between `i` and `j` is the
  fZ2 braiding the term builder already inserts, so a fermionic correlator at a distance is
  correct rather than refused. It costs one MPO build and one pass **per pair**, so pass
  `pairs=` for the row or the distance you actually want rather than taking all `N²`.

`Env(psi, h, bra=phi)` is the object all of this stands on, and `measure_mpo(phi, h, psi)`
is its one-line spelling. `Env.heff2` refuses on a two-state environment — the prepared
matvec reads the `IdL`/`IdR` channels as gauge identities, true of a canonical chain
against itself and false of a mixed transfer — which is why measurement and the sweep are
different entry points into the same cache.

### The entanglement profile

The fifth call is the one a DMRG user plots. The entanglement entropy across each cut is
what says whether the chain is critical or gapped, what a central-charge fit consumes, and
what tells you the bond dimension is saturating:

```python
entropy = out.psi.entanglement_entropy()       # {bond: S}, keyed by the bond's left site
renyi2 = out.psi.entanglement_entropy(alpha=2)  # the Renyi family, same keys
values = out.psi.schmidt_values()              # the spectrum the entropy is derived from
sectors = out.psi.schmidt_sectors()[3]         # bond 3's spectrum, split by symmetry sector
```

Three things worth knowing before reading a number off them:

- **The unit is nats**, not bits. `S = (c/6) log(x)` on an open chain wants the natural
  log; divide by `log(2)` for bits. The two references disagree here — YASTN's
  `get_entropy` is base 2 — so the convention is stated on `tenet.network.entropy`.
- **The key is the bond's left site**, `0 .. N-2`, the same key `sweep_`'s `schmidt` dict
  uses. The two trivial boundary cuts of a finite open chain are zero and are not returned.
- **`schmidt_sectors` is the read a graded bond is for.** On an SU(2) bond a single `j`
  multiplet stands for `2j + 1` dense Schmidt values, and the entropy accounts for that —
  which is why the two-site singlet reports `log 2` under SU(2) and under U(1) alike, while
  `-sum p log p` over the *flattened* SU(2) spectrum would report `0`.

Each of the three readers canonizes a **copy** of the state and runs its own SVD sweep, so
they never re-gauge the state you hand them and a non-canonical `psi` is not silently read
in the wrong gauge. Keep the result rather than calling twice on a large state.

`DMRG_out` carries no spectrum field: `out.psi` answers for itself, exactly and in any
gauge, where the sweep's own per-bond dict is a truncated convergence diagnostic taken at
whichever direction visited the bond last.

## Deliberate limits

- **Two-site DMRG only.** Single-site plus subspace expansion (Hubig–McCulloch–
  Schollwöck–Wall, PRB 91, 155115 (2015)) needs `tenet.linalg.left_null`, a mixing factor,
  its own schedule and a second `heff1` contraction chain. Named upgrade path — and note
  that the extrapolation recipe above *requires* two-site sweeps, so it is a limit with a
  payoff.
- **Wavefunction noise only.** Density-matrix and perturbative mixers (block2's default,
  tenpy's `DensityMatrixMixer`, SUNDMRG's α) all perturb `ρ = tr aa aa†` and need an
  `eigh` split; tenet's sweep splits with `svd_truncated`, which is what decides the bond
  `GradedSpace` sector by sector. A stronger mixer is a different split and its own issue.
- **Hand-written pairwise contraction orders**, not `optimize=` on a five-operand einsum:
  `opt_einsum` costs a graded network from *physical* leg sizes, and a U(1) MPS bond with
  unevenly filled sectors is exactly where that estimate is wrong.
- **The MPO is written out, not generated.** `tenet.network.MPO.from_w` takes the array;
  an `Hterm`-style generator is the right API for arbitrary Hamiltonians and the wrong
  thing for a file whose Hamiltonian is one line of physics.

## The usage lane

This page's toy code writes the `W` matrix and its bond spaces out by hand — that is what
it teaches. To see the same chain *called* through the library instead, run
[`examples/heisenberg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/heisenberg.py)
(`local_op` → `MPO.from_terms` → `MPS.product` → `dmrg_` → `expectation_2site`, no `W`
anywhere) and
[`examples/su2_heisenberg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/su2_heisenberg.py),
which runs the same chain under SU(2) and prints the multiplet-vs-dense table above as
computed numbers. Both run on a core install and are executed by `tests/test_examples.py`;
their committed output is on the [Heisenberg, U(1)](../examples/heisenberg.md) and
[Heisenberg, SU(2)](../examples/su2-heisenberg.md) example pages.

## Reference

- [`tenet.network`](../api/network.md) — `MPS`, `MPO`, `Env`, `Sweep`, `lanczos`,
  `sweep_`, `dmrg_`, `expectation_1site`, `expectation_2site`
- [`tenet.linalg`](../api/linalg.md) — `svd_truncated`
