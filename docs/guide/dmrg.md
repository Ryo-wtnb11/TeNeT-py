# DMRG

[`dmrg_`][tenet.network.dmrg_] sweeps a state to the ground state of an operator, in
place, with two-site updates. Everything on this page runs on the core install: no
`scipy`, no `quimb`, no `jax`.

```python
>>> from tenet.models import spin_half
>>> from tenet.network import MPO, MPS, dmrg_
>>> from tenet.symmetry import U1Sector
>>> site, n = spin_half(), 8
>>> terms = []
>>> for i in range(n - 1):
...     terms.append((1.0, [(site.ops["Sz"], i), (site.ops["Sz"], i + 1)]))
...     terms.append((0.5, [(site.ops["S+"], i), (site.ops["S-"], i + 1)]))
...     terms.append((0.5, [(site.ops["S-"], i), (site.ops["S+"], i + 1)]))
>>> h = MPO.from_terms(n, terms)
>>> psi = MPS.product(site.phys, [U1Sector(1 if i % 2 else -1) for i in range(n)])
>>> out = dmrg_(psi, h, chi=32)
>>> round(out.energy, 9)
-3.374932599

```

## The driver

`dmrg_` right-canonicalizes `psi` first, so a freshly seeded random MPS or a product
state is the expected input. It mutates `psi` and returns a
[`DMRG_out`][tenet.network.DMRG_out] whose `psi` is that same object:

| field | what it holds |
|---|---|
| `psi` | the converged state — the object you passed in |
| `sweeps` | how many sweeps ran |
| `energy`, `denergy` | the last sweep's energy and energy change |
| `max_dSchmidt` | the last sweep's worst-cut Schmidt change |
| `max_discarded_weight` | the last sweep's maximum per-bond discarded weight |
| `history` | one `(energy, denergy, dSchmidt, discarded)` tuple per sweep |
| `schedule` | the **realized** schedule, one `Sweep` per sweep run |

`zip(out.schedule, out.history)` is exact, so `out.schedule` alone answers whether a run
reached its final `chi` or converged earlier.

```python
>>> len(out.history) == out.sweeps == len(out.schedule)
True

```

## Targeting a sector

The state's boundary legs fix the symmetry sector, and the site tensors' invariance keeps
it there. [`MPS.product`][tenet.network.MPS.product] takes one physical sector per site
and derives the bonds backwards from those charges, so the total lands on bond 0 where it
is printable:

```python
>>> neel = MPS.product(site.phys, [U1Sector(1), U1Sector(-1)] * 4)
>>> neel[0].legs[0].space.sectors          # S^z_tot = 0
((U1Sector(charge=0), 1),)
>>> up = MPS.product(site.phys, [U1Sector(1)] * 4)
>>> up[0].legs[0].space.sectors            # S^z_tot = 2
((U1Sector(charge=4), 1),)

```

The general recipe is a `D=1` boundary leg carrying `U1Sector(q)`, targeting
`S^z_tot = q/2`; the boundary charge and the tensors' invariance hold it. `MPS.product` is
Abelian-only: a single sector is not a non-Abelian multiplet, so under SU(2) the route is
[`MPS.random`][tenet.network.MPS.random] with a charged boundary leg.

## Schedules

`dmrg_`'s flat `chi`/`cutoff` keywords are one spelling of a one-entry schedule. The
other is `schedule=`, a list of [`Sweep`][tenet.network.Sweep] entries — one record per
sweep, **whose last entry repeats** until convergence or `max_sweeps`:

```python
from tenet.network import Sweep, dmrg_

dmrg_(psi, h, chi=64)                              # flat
dmrg_(psi, h, schedule=[Sweep(chi=64)])            # the same run, exactly
dmrg_(psi, h, schedule=[Sweep(32, noise=1e-4)] * 4
              + [Sweep(64, noise=1e-5)] * 4
              + [Sweep(64)], max_sweeps=20)        # ramp, cool down, converge
```

A `Sweep` carries `chi`, `cutoff`, `noise` and `noise_type`. The loop tolerances —
`energy_tol`, `schmidt_tol`, `max_sweeps`, `ncv` — are properties of the loop, not of a
sweep, and stay flat keywords on `dmrg_`.

Passing `schedule` together with `chi` or `cutoff` raises, and so does an empty
`schedule`.

A `callback=` is invoked once per sweep with that sweep's `DMRG_out`, after `history` is
appended, so it sees the sweep that just finished. Its return value is ignored: there is
no early-stop protocol.

## Convergence

The loop stops when **both** criteria are met in one sweep: the energy change is below
`energy_tol` (default `1e-12`) and the worst-cut Schmidt change is below `schmidt_tol`
(default `1e-8`). The Schmidt criterion is the sensitive one, and it is what catches a
run whose energy has plateaued on a wrong bond structure.

Convergence is never declared on a sweep that is still noisy or still inside the
schedule: the exit requires the schedule's last entry *and* `noise == 0.0`. An energy
that stopped moving under noise at a ramp's intermediate `chi` has converged to the wrong
thing.

### Converged, or only plateaued?

Both criteria are **change** tests, and a change test is satisfied by a run stuck on a
wrong bond structure: nothing moved because nothing could. The check that is not a change
test is the energy variance:

```python
>>> round(h.variance(out.psi), 9)
0.0

```

`<psi|H^2|psi> / <psi|psi> - E^2` is zero for an exact eigenstate. Run it at two bond
dimensions: a state converging on an eigenstate has a variance falling towards zero as
`chi` grows; a state plateaued on the wrong structure has one that does not.

### Extrapolating in the discarded weight

The truncation error is linear in the discarded weight near convergence, so a fit of
energy against discarded weight extrapolates to the `chi → ∞` energy. Fit a **reverse**
schedule run on the already-converged state, not the forward run's history — the forward
energies are not converged at their bond dimensions:

```python
out = dmrg_(psi, h, chi=64)                                  # converge first
rev = dmrg_(psi, h, schedule=[Sweep(16)] * 2 + [Sweep(12)] * 2 + [Sweep(8)] * 2,
            energy_tol=0.0, max_sweeps=6)                    # nothing exits early
dws      = [record[3] for record in rev.history]
energies = [record[0] for record in rev.history]
e_extrapolated = np.polyfit(dws, energies, 1)[1]
```

The conventions worth keeping: zero noise throughout the reverse run; `energy_tol=0.0` so
no sweep exits early; an even number of sweeps per `chi`, since odd and even half-sweeps
report different discarded weights; the first reverse `chi` slightly below the last
forward one; and the error bar quoted as one fifth of the extrapolation distance. The
recipe needs two-site sweeps, which is what `dmrg_` runs.

## Noise

`noise` on a `Sweep` mixes a perturbation in at each split, at relative strength `noise`.
`noise_type` says which perturbation, and therefore which split runs:

| `(noise, noise_type)` | the split |
|---|---|
| `noise == 0.0`, any `noise_type` | `svd_truncated` of the two-site tensor |
| `> 0`, `"wavefunction"` | `svd_truncated` of a perturbed two-site tensor |
| `> 0`, `"perturbative"` | `eigh` of a perturbed density matrix |

Nothing else decides it — no bond width, no `chi`, no runtime probe. `noise=0.0` draws no
random number and builds no density matrix.

**Wavefunction noise** (the default) adds a random symmetric tensor over the two-site
tensor's own legs after the eigensolver and before the split, then renormalizes. It fills
every structurally allowed coupled sector of the two-site map, including the ones the
eigensolver left numerically empty and which the truncation therefore dropped from the
bond. That is the local minimum a symmetric DMRG falls into: a sector that is zero stays
zero otherwise. It cannot reach outside `bond_l ⊗ phys`.

**Perturbative noise** builds the density matrix and splits with `eigh` instead. Squaring
the two-site tensor into `rho` resolves a singular value `sigma` through `sigma**2`, so
the split's accuracy floor is the square root of machine epsilon, which is why a
noiseless sweep — including the cooling tail of a ramp — takes the SVD split.

Noise is not variational: a noisy sweep's energy can sit above its clean twin. Taper it —
`1e-4` early, `1e-5` in the middle, `0.0` at the end.

`seed=` makes the draw at bond `n` reproducible as `seed + n`, distinctly per sweep.

## Excited states

`orthogonal_to=` takes already-converged states and holds `psi` orthogonal to them,
turning the run into an excited-state search:

```python
ground = dmrg_(psi1, h, chi=64)
first  = dmrg_(psi2, h, chi=64, orthogonal_to=[ground.psi])
```

The given states are not modified and need no particular gauge. The machinery is one
two-state [`Env`][tenet.network.Env] per given state over
[`MPO.identity`][tenet.network.MPO.identity], swept alongside the main environment; what
each contributes at a bond is a projection vector handed to the eigensolver. The reported
energy is the projected operator's own Ritz value, so it is the excited energy directly,
with no shift to subtract.

Sector targeting composes with it: a charged `D=1` boundary leg fixes the sector,
orthogonality walks up the spectrum inside it, and a converged state whose boundary legs
put it in a *different* sector is dropped from the projection — the symmetry already made
it orthogonal.

## Restarting

Save the state, load it back, and re-enter the schedule at a slice:

```python
out = dmrg_(psi, h, schedule=schedule[:2])   # ... interrupted after two sweeps
psi = MPS.load("checkpoint")
out = dmrg_(psi, h, schedule=schedule[2:])   # matches the uninterrupted run
```

The slice is the position, and a sweep is a full round trip, so no direction or
sweep-index argument is needed. [Saving and loading](saving-and-loading.md) covers the
files.

## Measuring the converged state

```python
>>> import numpy as np
>>> import tenet
>>> from tenet import IN, OUT, Leg
>>> from tenet.network import expectation_profile, expectation_1site, overlap
>>> sz = tenet.SymmetricTensor.from_dense(
...     np.diag([-0.5, 0.5]), (Leg(site.phys, OUT), Leg(site.phys, IN))
... )
>>> profile = expectation_profile(out.psi, sz)
>>> max(abs(v) for v in profile) < 1e-9
True

```

| call | what it returns |
|---|---|
| [`expectation_1site`][tenet.network.expectation_1site] | `<psi\|o_n\|psi> / <psi\|psi>` at one site |
| [`expectation_2site`][tenet.network.expectation_2site] | the same for a rank-4 operator on `(n, n+1)` |
| [`expectation_profile`][tenet.network.expectation_profile] | `<psi\|o_n\|psi> / <psi\|psi>` at **every** site, in one pass |
| [`overlap`][tenet.network.overlap] | `<phi\|psi>`, undivided |
| [`measure_mpo`][tenet.network.measure_mpo] | `<phi\|H\|psi>`, undivided |
| [`correlation_function`][tenet.network.correlation_function] | `{(i, j): value}` for the pairs you ask for |

Three things to know:

- **`expectation_profile` is the one to reach for over a list comprehension.** A
  per-site loop costs two full-chain transfer passes per site; the profile moves the
  orthogonality centre once along the chain and reads the operator off it. Same numbers,
  `O(N)` instead of `O(N²)`.
- **`overlap` and `measure_mpo` do not divide.** A fidelity is
  `overlap(phi, psi) / (phi.norm() * psi.norm())`, and you write the division. The
  divided readings are the ones whose names say `expectation`.
- **`correlation_function` takes the rank-3 charged operators** `MPO.from_terms` takes,
  which is the form a fermionic `c` has. It returns `{(i, j): value}` for `i < j`, every
  pair by default, and the Jordan-Wigner string between `i` and `j` is the fZ2 braiding
  the term builder inserts. It costs one MPO build and one pass **per pair**, so pass
  `pairs=` for the row or the distance you want.

[`Env(psi, h, bra=phi)`][tenet.network.Env] is the object all of this stands on, and
`measure_mpo(phi, h, psi)` is its one-line spelling. `Env.heff2` refuses on a two-state
environment — the prepared matvec reads the `IdL`/`IdR` channels as gauge identities,
true of a canonical chain against itself and false of a mixed transfer — which is why
measurement and the sweep are different entry points into one cache.
[`Env.measure`][tenet.network.Env.measure] returns the unnormalized `<psi|H|psi>`.

### The entanglement profile

```python
>>> entropy = out.psi.entanglement_entropy()          # {bond: S}, keyed by the bond's left site
>>> sorted(entropy) == list(range(n - 1))
True
>>> renyi2 = out.psi.entanglement_entropy(alpha=2)    # the Renyi family, same keys
>>> values = out.psi.schmidt_values()                 # the spectrum the entropy comes from
>>> sectors = out.psi.schmidt_sectors()[3]            # bond 3's spectrum, by symmetry sector

```

- **The unit is nats.** `S = (c/6) log(x)` on an open chain wants the natural log; divide
  by `log(2)` for bits.
- **The key is the bond's left site**, `0 .. N-2` — the same key the sweep's `schmidt`
  dict uses. The two trivial boundary cuts are zero and are not returned.
- **`schmidt_sectors` is the read a graded bond is for.** On an SU(2) bond a single `j`
  multiplet stands for `2j + 1` dense Schmidt values, and the entropy accounts for that,
  which is why a two-site singlet reports `log 2` under SU(2) and under U(1) alike.

Each of the three readers canonizes a **copy** of the state and runs its own SVD sweep,
so they never re-gauge the state you hand them. Keep the result rather than calling twice
on a large state.

`DMRG_out` carries no spectrum field: the spectrum is a property of the state, and
`out.psi` answers for it exactly and in any gauge.

## Compressing

```python
>>> discarded = out.psi.compress_(chi=8)
>>> out.psi[4].legs[0].space.dim <= 8
True

```

[`compress_`][tenet.network.MPS.compress_] truncates in place and returns the **total**
discarded weight, `sqrt(sum_bond dw)`, where a sweep reports the per-bond **maximum**.
One answers "how much of my state did I throw away", the other "which bond is the
convergence diagnostic".

## `compile=`

The two-site matvec is a pure function of a fixed contraction structure, so it can be
handed to `jax.jit`. `dmrg_` takes the callable; this layer names no accelerator, so you
supply it and the `jax` extra:

```python
import jax                       # pip install "tenet-py[jax]"
import tenet

tenet.enable_jax()               # registers SymmetricTensor as a JAX pytree

dmrg_(psi, h, chi=64)                    # the plain run, NumPy
dmrg_(psi, h, chi=64, compile=jax.jit)   # the same run, matvec compiled
```

It changes the run's performance regime, not its accuracy. Read the shape of it before
reaching for it: the *sweep* around the matvec is not traceable, because the truncating
SVD re-decides each bond space every sweep, so on the JAX backend a compiled run is
slower end to end than the plain NumPy one; and `Env` rebuilds its prepared operator at
every bond visit and so invokes `compile` again each time, one XLA trace per bond per
sweep.

## The sweep's fixed choices

Every sweep is a **two-site** update, which is what the extrapolation recipe above
consumes. The split is `svd_truncated`, or `eigh` of a density matrix under perturbative
noise, per the table above. The contractions inside the sweep run in hand-written pairwise
orders, which is what keeps them right on a U(1) bond with unevenly filled sectors, where
a cost model reading physical leg sizes misprices the network.

## Where next

- [Building a Hamiltonian](hamiltonians.md) — the operator `dmrg_` sweeps.
- [Truncation](truncation.md) — what `chi` and `cutoff` mean on a graded bond.
- [DMRG end to end](../tutorials/dmrg.md) — a full lattice problem against exact
  diagonalization.
- [`tenet.network`](../api/network.md) — the reference.
