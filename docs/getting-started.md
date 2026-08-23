# Getting started

## Install

```sh
uv add tenet-py         # or: pip install tenet-py
```

The core install pulls `numpy`, `autoray`, `opt-einsum` and `racah-py`. Every symmetry
— SU(N), SU(2), U(1), Z2, fermion parity and their products — runs on it. Two optional
extras add backends:

```sh
uv add "tenet-py[jax]"      # jax>=0.10 — pytrees, jit, grad
uv add "tenet-py[torch]"    # torch>=2.0 — eager blocks
```

`racah-py` ships abi3-py312 wheels for linux x86_64/aarch64, macOS arm64/x86_64 and
windows x64. On any other platform pip builds it from the sdist, which needs a Rust
toolchain.

## The first example, line by line

A spin-1/2 Heisenberg chain to its ground state. Start with the site:

```python
>>> from tenet.models import spin_half
>>> site = spin_half()
>>> site.phys.dim, sorted(site.ops)
(2, ['S+', 'S-', 'Sz'])

```

A [Site][tenet.models.Site] is a physical [GradedSpace][tenet.GradedSpace] plus a table
of local operators. `spin_half()` grades the doublet by U(1), with the charge `2 S^z`,
so the two basis states carry charges `-1` and `+1`:

```python
>>> site.phys.sectors
((U1Sector(charge=-1), 1), (U1Sector(charge=1), 1))

```

Each entry of `site.ops` is a rank-3 tensor: the two physical legs plus a `D=1` leg
carrying the charge the operator emits. That third leg is what makes `S+` expressible —
it changes `2 S^z` by `-2`, and the leg carries the change.

Now the Hamiltonian. `MPO.from_terms` takes `(coefficient, [(operator, site), ...])`
tuples, with the identity implied on every site a term does not name:

```python
>>> from tenet.network import MPO
>>> n = 12
>>> terms = []
>>> for i in range(n - 1):
...     terms.append((1.0, [(site.ops["Sz"], i), (site.ops["Sz"], i + 1)]))
...     terms.append((0.5, [(site.ops["S+"], i), (site.ops["S-"], i + 1)]))
...     terms.append((0.5, [(site.ops["S-"], i), (site.ops["S+"], i + 1)]))
>>> h = MPO.from_terms(n, terms)
>>> len(h)
12

```

You declared no MPO bond spaces. The builder derives them: every term is a bond-1 MPO,
they are stacked with a direct sum, and two compressing SVD sweeps collapse the stack to
the operator Schmidt rank. The grading comes out of the operators' own charges.

The starting state is a Néel product state, one physical sector per site:

```python
>>> from tenet.network import MPS
>>> from tenet.symmetry import U1Sector
>>> psi = MPS.product(site.phys, [U1Sector(1 if i % 2 else -1) for i in range(n)])
>>> psi[0].legs[0].space.sectors      # bond 0 carries the total charge
((U1Sector(charge=0), 1),)

```

Bond 0 is a `D=1` leg carrying charge 0, which is `S^z_tot = 0`. Every site tensor is
invariant, so nothing in the sweep can move the state out of that sector: the target
sector is structural, not a constraint added on top.

Then sweep:

```python
>>> from tenet.network import dmrg_
>>> out = dmrg_(psi, h, chi=64)
>>> round(out.energy, 9)
-5.142090633

```

`dmrg_` right-canonicalizes `psi`, then runs two-site sweeps until the energy and the
Schmidt values both stop moving. It mutates `psi` and returns a
[DMRG_out][tenet.network.DMRG_out] whose `psi` is that same object:

```python
>>> out.psi is psi
True
>>> len(out.history) == out.sweeps
True

```

## Reading the state

The converged state answers for itself, in any gauge:

```python
>>> import numpy as np
>>> import tenet
>>> from tenet import IN, OUT, Leg
>>> from tenet.network import expectation_profile
>>> sz = tenet.SymmetricTensor.from_dense(
...     np.diag([-0.5, 0.5]), (Leg(site.phys, OUT), Leg(site.phys, IN))
... )
>>> profile = expectation_profile(out.psi, sz)
>>> max(abs(v) for v in profile) < 1e-9
True

```

Every `<S^z_n>` is float noise, because the sector is fixed. The entanglement entropy
across each cut, in nats, keyed by the cut's left site:

```python
>>> entropy = out.psi.entanglement_entropy()
>>> sorted(entropy) == list(range(n - 1))
True
>>> round(entropy[5], 3)
0.537

```

## Checking yourself against dense NumPy

Every tensor in the library expands to an ordinary array with
[to_dense][tenet.SymmetricTensor.to_dense]. It is the wrong tool for computation — it
throws away the structure the library exists to keep — and the right tool for convincing
yourself the structure is what you think it is:

```python
>>> from tenet import GradedSpace, SymmetricTensor
>>> from tenet.symmetry import U1
>>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
>>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
>>> a
SymmetricTensor(ndim=2, shape=(3, 3), dtype=float64, backend='numpy', blocks=2)
>>> d = a.to_dense()
>>> bool(np.allclose(np.linalg.norm(d), float(tenet.norm(a))))
True

```

`a` stores two blocks, one per charge, rather than nine dense entries. The entries the
symmetry forbids are zero, and the two charge sectors occupy disjoint index ranges:

```python
>>> bool(np.allclose(d[2, :2], 0.0)) and bool(np.allclose(d[:2, 2], 0.0))
True

```

Contract, factorize, `to_dense` both sides, compare with dense NumPy: that check works
for every operation in the library.

## Where to go next

- [Tensors, legs and spaces](guide/tensors-legs-spaces.md) — `GradedSpace`, `Leg`,
  `SymmetricTensor`, `TensorStructure`.
- [Symmetries and providers](guide/symmetries-and-providers.md) — sector conventions
  (SU(2) labels by **2j**), capabilities, and what a `CapabilityError` means.
- [Contraction](guide/contraction.md) — `tensordot`, `einsum`, `compose`, and the
  composition rule.
- [Building a Hamiltonian](guide/hamiltonians.md) — sites, the four MPO builders, and
  when to keep the symbolic description.
- [DMRG](guide/dmrg.md) — schedules, noise, excited states, measurement.
- [Tutorials](tutorials/dmrg.md) walk complete problems; the
  [examples](examples/index.md) are runnable files with committed output.
