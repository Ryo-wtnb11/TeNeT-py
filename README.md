# TeNeT-py

[![tests](https://github.com/Ryo-wtnb11/TeNeT-py/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryo-wtnb11/TeNeT-py/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-online-blue.svg)](https://ryo-wtnb11.github.io/TeNeT-py/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/pyproject.toml)

**Non-Abelian symmetric tensors with ndarray-style Python APIs and backend-native numerical execution.**

`tenet` provides block-sparse tensors with **non-Abelian** symmetries — SU(2) and its
products, alongside the Abelian and fermionic cases — using exact recoupling
coefficients rather than a dense embedding. Its tensors follow the NumPy interface where
possible: they carry legs, not a codomain × domain partition, and they contract with
`tenet.einsum` and factorize with `tenet.linalg`. Blocks are stored in NumPy, JAX or
PyTorch through [`autoray`](https://github.com/jcmgray/autoray), so the same tensor runs
under `jax.jit` and `jax.grad` without leaving the symmetry structure behind.

The categorical model underneath — why a tensor is a morphism, why `side` and `dual` stay
independent, why fusion is a primitive and `reshape` is not — is written out in
[`docs/design.md`](https://ryo-wtnb11.github.io/TeNeT-py/design/).

> **Status:** early design and implementation. The API is not stable.

## Install

```sh
uv add tenet-py         # or: pip install tenet-py
```

The core install needs only `numpy`, `autoray` and `opt-einsum`. Three optional extras:

```sh
uv add "tenet-py[jax]"      # jax>=0.10 — first release with the wide-matrix qr JVP
uv add "tenet-py[torch]"    # torch>=2.0 — eager only; tenet.ad stays JAX-only
uv add "tenet-py[sun]"      # racah-py — required for SU(N); see below
```

`tenet.symmetry.sun` (SU(N), SU(3) first) needs `racah-py`, and raises an
`ImportError` naming the extra without it. That refusal is categorical, not a
fallback: SU(N) coefficients *are* the gauge, so a pure-Python second
implementation would be a second source of truth. Everything else — SU(2), U(1),
Z2, fermion parity and their products — needs nothing extra.

## Quickstart

An SU(2) leg, a random invariant tensor, an ndarray-style contraction and an SVD:

```python
import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, SU2Sector

# SU(2) sectors are labelled by 2j, so SU2Sector(1) is spin-1/2. A space maps sector -> degeneracy.
V = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2})
W = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 2})
U = GradedSpace.new(SU2, {SU2Sector(1): 1, SU2Sector(2): 2})

a = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT), Leg(U, OUT)), seed=0)
b = SymmetricTensor.random((Leg(U, OUT, dual=True), Leg(W, IN), Leg(V, IN)), seed=1)

c = tenet.einsum("abc,cde->abde", a, b)          # legs, not codomain x domain
u, s, vh = tenet.linalg.svd(c, axes=((0, 2), (1, 3)))

assert c.legs == (a.legs[0], a.legs[1], b.legs[1], b.legs[2])
print(c.legs, float(tenet.norm(c)))
```

The same tensors are a JAX pytree — the structure is static metadata, the blocks are
traced — so autodiff and `jit` compose straight through:

```python
import jax

import tenet.pytree  # noqa: F401  # registration is the import's side effect

g = jax.jit(jax.grad(lambda t: tenet.norm(t) ** 2))(a)
print(g.legs == a.legs)
```

## Features

- **Non-Abelian symmetries, exactly.** SU(2) — and U(1), Z2, fermion parity, and their
  products — via exact recoupling coefficients. Block-sparse all the way down, never a
  dense embedding.
- **An ndarray-style API.** Tensors carry legs; you write `tenet.einsum` equations and
  `tenet.linalg` factorizations. The categorical machinery stays under the hood.
- **Differentiable and jittable.** Tensors are JAX pytrees, so `jit`, `grad` and `vmap`
  compose with the symmetry structure — including through truncations, and with
  gradients that stay finite at the degenerate spectra symmetric tensors produce.
- **Multi-backend.** The same code runs on NumPy, JAX or PyTorch blocks through
  [`autoray`](https://github.com/jcmgray/autoray) (torch eager-only).
- **Algorithms included.** `tenet.network` ships DMRG and CTMRG — MPS/MPO,
  environments, sweeps — specified by fully worked examples. A Hamiltonian can be written
  out as a `W` matrix (`MPO.from_w`) or listed as terms (`MPO.from_terms`), and the term
  route *derives* the graded MPO bond spaces instead of asking you to declare them. A term
  may be a symmetry-**invariant** *k*-site operator, so the SU(2) Heisenberg chain is one
  `np.kron` and a list comprehension — and DMRG runs on it.

## Examples

Each example is executed by the integration suite and checked against a named exact
oracle; the SU(2) coefficient conventions themselves are pinned by vendored fixtures.

| Example | What it does | Oracle |
| --- | --- | --- |
| [`examples/dmrg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/dmrg.py) | finite two-site DMRG, U(1) Heisenberg chain | exact diagonalization |
| [`examples/ctmrg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/ctmrg.py) | differentiable CTMRG, then a U(1)/SU(2) iPEPS gradient | Onsager's closed-form free energy |
| [`examples/vmc_mps.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/vmc_mps.py) | symmetric MPS → pytree → `jax.grad` → SGD step | variational energy decrease |

## Related projects

- [`symmray`](https://github.com/jcmgray/symmray) — Abelian and fermionic block-sparse
  arrays on the same `autoray` premise; the closest cousin.
- [`yastn`](https://github.com/yastn/yastn) — Abelian symmetric tensors, differentiable,
  with a mature docs site; a direct influence on this repository's CTMRG example.
- [`TensorKit.jl`](https://github.com/QuantumKitHub/TensorKit.jl) — the categorical formulation,
  in Julia, and the source of much of the vocabulary used here.
- [`froSTspin`](https://github.com/ogauthe/frostspin) — SU(2)-symmetric tensors for
  frustrated spin systems.

## Docs

- [`docs/design.md`](https://ryo-wtnb11.github.io/TeNeT-py/design/) — the design document: invariants, the categorical
  model, and the milestone plan.
- [`REPOSITORY_RULES.md`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/REPOSITORY_RULES.md) — process rules for contributing.

## Citation

If you use TeNeT-py in your research, please cite it:

```bibtex
@software{tenet-py,
  author  = {Watanabe, Ryo},
  title   = {{TeNeT-py}: a non-{Abelian} symmetric tensor {Python} library},
  url     = {https://github.com/Ryo-wtnb11/TeNeT-py},
  license = {Apache-2.0},
  year    = {2026}
}
```

The same metadata lives in
[`CITATION.cff`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/CITATION.cff) —
GitHub's "Cite this repository" button renders it as BibTeX or APA. A DOI will be
added once a release is archived on Zenodo.

## License

Apache License 2.0 — see [`LICENSE`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/LICENSE).
