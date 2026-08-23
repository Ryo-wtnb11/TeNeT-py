# DMRG end to end — a Heisenberg chain

A spin-1/2 Heisenberg chain, `H = Σ_i S_i · S_{i+1}`, open boundaries, U(1)-graded by
`2 S^z`. This is [`examples/heisenberg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/heisenberg.py)
walked through; the file runs standalone and its output is committed on the
[Heisenberg, U(1)](../examples/heisenberg.md) page.

```sh
uv run python examples/heisenberg.py
```

Nothing outside the core install is involved.

## The site

```python
from tenet.models import spin_half

SITE = spin_half()
PHYS = SITE.phys                    # U(1), charge 2 S^z, the doublet {-1, +1}
```

`SITE.ops` holds `Sz`, `S+` and `S-` as rank-3 term operators — two physical legs plus a
`D=1` leg carrying the charge each emits. `SITE.matrices` holds the dense forms, plus
`S.S`, the invariant two-site operator, which has no rank-3 form.

## The Hamiltonian

```python
from tenet.network import MPO

def heisenberg_mpo(n_sites: int) -> MPO:
    op_sz, op_sp, op_sm = (SITE.ops[name] for name in ("Sz", "S+", "S-"))
    terms = []
    for i in range(n_sites - 1):
        terms.append((1.0, [(op_sz, i), (op_sz, i + 1)]))
        terms.append((0.5, [(op_sp, i), (op_sm, i + 1)]))
        terms.append((0.5, [(op_sm, i), (op_sp, i + 1)]))
    return MPO.from_terms(n_sites, terms)
```

No MPO bond space is written down. Each term is a bond-1 MPO, a direct sum stacks them,
and two compressing SVD sweeps collapse the stack to the operator Schmidt rank. The
grading falls out of the operators' own charges: the bulk bond comes out
`{-2: 1, 0: 3, +2: 1}`, dense dimension 5.

If you want to see the same operator with its `W` matrix and grading written out by hand,
[`examples/heisenberg_walkthrough.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/heisenberg_walkthrough.py)
builds it three ways — `from_w`, `from_entries`, `from_terms` — and lands all three on the
same twelve digits. Its page is
[Heisenberg, U(1) walkthrough](../examples/heisenberg-walkthrough.md).

## The seed fixes the sector

```python
from tenet.network import MPS
from tenet.symmetry import U1Sector

psi = MPS.product(PHYS, [U1Sector(1 if n % 2 else -1) for n in range(n_sites)])
```

A Néel product state: one physical sector per site, and the bonds derived backwards from
those charges. Both boundary legs then carry the unit sector with degeneracy 1, and
invariance of every site tensor forces `Σ_i 2 S^z_i = 0` — the sector an even chain's
ground state lives in. The sector is held structurally, by the seed's charges and the
tensors' invariance, for the whole run.

The general recipe: a `D=1` boundary leg carrying `U1Sector(q)` targets `S^z_tot = q/2`.

## Sweeping

```python
from tenet.network import Sweep, dmrg_

schedule = [Sweep(16, noise=1e-4)] * 3 + [Sweep(32, noise=1e-5)] * 3 + [Sweep(64)]
out = dmrg_(psi, heisenberg_mpo(n_sites), schedule=schedule)
```

A ramp with noise that cools to a clean `chi=64` tail. On this chain a flat
`dmrg_(psi, h, chi=64)` reaches the same energy: a degeneracy-1 U(1) seed already grows by
a factor of `d` per sweep. The ramp is here because writing one is what you do on a
problem where the seed cannot ramp itself — see [DMRG](../guide/dmrg.md) for when noise
pays.

At `N=20` this converges in nine sweeps to `E = -8.682473334398`, which is the recorded
exact-diagonalization energy of the same finite chain to twelve digits.

Inside a sweep: `svd_truncated` decides the bond `GradedSpace` at every bond, reporting
the discarded weight by Pythagoras; `lanczos` solves the two-site eigenproblem over
`SymmetricTensor` as a vector, needing only `add`, `subtract`, scalar multiply, `norm`
and an inner product — no `scipy.sparse.linalg`, no dense reshaping of the local problem.

## Reading the state

```python
from tenet.network import expectation_1site, expectation_2site, local_op

ss = local_op(SITE.matrices["S.S"], phys=PHYS)
profile = [expectation_2site(out.psi, ss, n) for n in range(n_sites - 1)]
```

The 19 bond energies sum to `out.energy` to twelve digits: `expectation_2site` and
`dmrg_` weigh the same state on the same scale.

```python
op_sz = local_op(SITE.matrices["Sz"], phys=PHYS)
max_sz = max(abs(expectation_1site(out.psi, op_sz, n)) for n in range(n_sites))
```

`max_n |<S^z_n>|` comes out at float noise, `~5e-13`, because the sector is enforced by
the seed's own charges.

For a whole profile in one pass, use
[`expectation_profile`][tenet.network.expectation_profile] instead of the comprehension:
same numbers, `O(N)` rather than `O(N²)`. [DMRG](../guide/dmrg.md) has the full
measurement set, the entanglement readers and the variance check.

## Why there is no `jit` and no `grad` here

DMRG's control flow is data-dependent at every level: the truncation re-decides the bond
space each sweep, `lanczos`'s happy breakdown tests a norm against a tolerance, and the
loop exits on a measured energy change. Each of those is what a traced region cannot do,
and `svd_truncated` says so by raising `StructureChangingError` under a trace. So this
runs on the eager NumPy backend.

One piece inside the sweep is traceable, and `dmrg_` takes a callable for it:
`compile=jax.jit` wraps the two-site matvec. [JAX and backends](../guide/jax-and-backends.md)
and [DMRG](../guide/dmrg.md#compile) describe what that changes.

## Where next

- [Fermions and the Hubbard model](fermions.md) — the same driver with `fZ2` sites.
- [SU(2)](su2.md) — the same chain with the non-Abelian grading.
- [Quantum chemistry](quantum-chemistry.md) — `from_arrays(..., symbolic=True)`.
- [`examples/toy_codes/dmrg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/dmrg.py)
  writes the algorithm out by hand on `tenet`'s tensor layer; its page is
  [Toy DMRG](../examples/toy-dmrg.md).
