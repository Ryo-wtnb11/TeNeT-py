# TeNeT-py

[![tests](https://github.com/Ryo-wtnb11/TeNeT-py/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryo-wtnb11/TeNeT-py/actions/workflows/ci.yml)
[![coverage](https://ryo-wtnb11.github.io/TeNeT-py/coverage-badge.svg)](https://github.com/Ryo-wtnb11/TeNeT-py/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/TeNeT-py/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/TeNeT-py)
[![docs](https://img.shields.io/badge/docs-online-blue.svg)](https://ryo-wtnb11.github.io/TeNeT-py/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/pyproject.toml)

**A Python library for symmetric tensor networks: NumPy-ish API on the surface, category theory under the hood.**

`tenet` gives you block-sparse tensors that carry a symmetry exactly — SU(N), SU(2),
U(1), Z2, fermion parity and their products — through exact recoupling coefficients.
A tensor carries legs, contracts with `tenet.einsum` and factorizes with `tenet.linalg`,
the way an ndarray does. Blocks live in NumPy, JAX or PyTorch arrays through
[`autoray`](https://github.com/jcmgray/autoray), so the same tensor runs under
`jax.jit` and `jax.grad` with its symmetry structure intact. On top of the tensor layer,
`tenet.network` ships finite DMRG and CTMRG.

## Install

```sh
uv add tenet-py         # or: pip install tenet-py
```

The core install pulls `numpy`, `autoray`, `opt-einsum` and `racah-py`, and every
symmetry works on it. Two optional extras:

```sh
uv add "tenet-py[jax]"      # jax>=0.10 — pytrees, jit, grad
uv add "tenet-py[torch]"    # torch>=2.0 — eager blocks
```

## First example

A 20-site spin-1/2 Heisenberg chain, U(1)-graded by `2 S^z`, to its ground state:

```python
from tenet.models import spin_half
from tenet.network import MPO, MPS, dmrg_
from tenet.symmetry import U1Sector

site, n = spin_half(), 20
terms = []
for i in range(n - 1):
    terms.append((1.0, [(site.ops["Sz"], i), (site.ops["Sz"], i + 1)]))
    terms.append((0.5, [(site.ops["S+"], i), (site.ops["S-"], i + 1)]))
    terms.append((0.5, [(site.ops["S-"], i), (site.ops["S+"], i + 1)]))

h = MPO.from_terms(n, terms)
psi = MPS.product(site.phys, [U1Sector(1 if i % 2 else -1) for i in range(n)])
out = dmrg_(psi, h, chi=64)

print(out.sweeps, out.energy)          # 6 -8.682473334398...
print(out.psi.entanglement_entropy())  # {bond: S}, in nats
```

The Néel product state's own charges put the run in the `S^z_tot = 0` sector, and the
site tensors' invariance keeps it there — no projector, no penalty term.
[Getting started](https://ryo-wtnb11.github.io/TeNeT-py/getting-started/) reads this
example line by line.

## What it supports

- **Symmetries.** SU(N), SU(2), U(1), Z2, fermion parity (`fZ2`) and Deligne products
  of any of them. Non-Abelian sectors are multiplets with exact Clebsch-Gordan,
  F- and R-symbols; fermionic wires carry their Koszul signs.
- **Tensors.** `SymmetricTensor` over a flat tuple of `Leg`s, one reduced block per
  allowed fusion channel. `einsum`, `tensordot`, `compose`, `transpose`, `fuse`,
  `repartition`, `trace`, and `tenet.linalg`'s `svd`, `qr`, `lq`, `eigh`, `eig`,
  `polar`, `expm`, `left_null`, plus the truncating `svd_truncated` / `eigh_truncated`.
- **Algorithms.** Finite two-site DMRG (`dmrg_`, schedules, noise, excited states) with
  MPS/MPO containers, environment caches and measurement; C4v CTMRG (`ctmrg`,
  `ctmrg_unrolled`) with a differentiable unrolled form.
- **Backends.** NumPy, JAX and PyTorch blocks through `autoray`. `tenet.enable_jax()`
  registers `SymmetricTensor` as a JAX pytree, so `jit`, `grad` and `vmap` reach the
  blocks while the structure stays static metadata.

## Docs

- [Getting started](https://ryo-wtnb11.github.io/TeNeT-py/getting-started/) — install and the first example.
- [User guide](https://ryo-wtnb11.github.io/TeNeT-py/guide/tensors-legs-spaces/) — tensors, symmetries, contraction, Hamiltonians, DMRG, truncation, JAX, files.
- [Tutorials](https://ryo-wtnb11.github.io/TeNeT-py/tutorials/dmrg/) — DMRG, fermions, SU(2), quantum chemistry, CTMRG, VMC.
- [Examples](https://ryo-wtnb11.github.io/TeNeT-py/examples/) — runnable files with their committed output.
- [API reference](https://ryo-wtnb11.github.io/TeNeT-py/api/tenet/) — every public name.
- [`docs/design.md`](https://ryo-wtnb11.github.io/TeNeT-py/design/) — the categorical model underneath.
- [`REPOSITORY_RULES.md`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/REPOSITORY_RULES.md) — process rules for contributing.

## Citation

If you use TeNeT-py in your research, please cite it:

```bibtex
@software{tenet-py,
  author  = {Watanabe, Ryo},
  title   = {{TeNeT-py}: a {Python} library for symmetric tensor networks
             --- a {NumPy}-style {API} on the surface, category theory under
             the hood},
  url     = {https://github.com/Ryo-wtnb11/TeNeT-py},
  license = {Apache-2.0},
  year    = {2026}
}
```

The same metadata lives in
[`CITATION.cff`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/CITATION.cff) —
GitHub's "Cite this repository" button renders it as BibTeX or APA.

## License

Apache License 2.0 — see [`LICENSE`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/LICENSE).
