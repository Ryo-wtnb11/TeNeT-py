# CTMRG — 2D Ising against Onsager, then a differentiable iPEPS

Corner transfer matrix renormalization on a C4v environment. Two problems share one core:
the classical 2D Ising partition function, whose free energy per site has Onsager's closed
form, and a single-site iPEPS whose energy you differentiate.

[`examples/ising2d.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/ising2d.py)
is the Ising half through the library, on a core install; its output is committed on the
[2D Ising CTMRG](../examples/ising2d.md) page.
[`examples/toy_codes/ctmrg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/ctmrg.py)
writes the same algorithm out on the tensor layer and adds the iPEPS gradient; it needs
the `jax` extra.

```sh
uv run python examples/ising2d.py
uv run --extra jax python examples/toy_codes/ctmrg.py
```

## The converging sweep

```python
from tenet.network import ctmrg, single_layer_ctm, spectrum

out = ctmrg(*single_layer_ctm(bulk), chi=24)
out.sweeps, out.converged, out.max_dsv
env = out.env                      # CTMEnv: c, e, bond
```

[`single_layer_ctm`][tenet.network.single_layer_ctm] takes a rank-4 bulk tensor and hands
back `(absorber, c, e)` — the model's absorption closures plus a seed corner and edge.
[`double_layer_ctm`][tenet.network.double_layer_ctm] does the same for a rank-5 iPEPS ket,
building the bra by conjugation.

An [`Absorb`][tenet.network.Absorb] is the *definition* of a model's absorption:
`corner(c, e)` builds the enlarged corner and `edge(e, p)` absorbs one edge onto a
projector's new bond. [`move`][tenet.network.move] is the step;
`ctmrg` is the loop, sweeping until the corner spectrum stops
moving.

[`CTMRG_out`][tenet.network.CTMRG_out] carries `sweeps`, `max_dsv`, `converged`, `history`
— the per-sweep corner-spectrum change — and `env`, a
[`CTMEnv`][tenet.network.CTMEnv] holding `c`, `e` and the frozen `bond` the last move
decided. `CTMEnv` unpacks positionally: `c, e, bond = move(...)`.

`chi` is an input, and the realized environment bond is `env.bond`, whose `dim` and
`reduced_dim` say what was actually kept.

At `chi=24` this reproduces Onsager's free energy to float precision off criticality —
relative error `3e-16` at `beta=0.3` and `1e-15` at `beta=0.5` — and to `2e-6` at
`beta_c`, where the correlation length outruns any finite environment.

## Grading the Ising half by Z2

The Boltzmann bulk tensor is Z2-graded. That stops a finite-`chi` environment from
breaking the symmetry spuriously in the ordered phase, which is what lets the run go above
`beta_c` against Onsager at all. Two further things the grading gives:

- **Zero magnetization is structural.** A spin insertion is a Z2-odd tensor, which no
  invariant `SymmetricTensor` can hold, so `from_dense` refuses it. The refusal is the
  statement.
- **The ordered-phase corner spectrum is exactly two-fold degenerate**, the doublet
  spanning the two parity sectors:

```
corner spectrum at beta=0.5: 0.6905 0.6905 0.1486 0.1486 0.0320 0.0320
```

Because that doubling is *cross*-sector, and the broadened SVD rule in `tenet.ad` broadens
*per coupled sector*, a graded run never hands one SVD a degenerate pair.

## The differentiable form

`ctmrg` runs outside `jit` and `grad` and cannot be otherwise: it
loops on a measured spectrum change and re-decides the environment bond each sweep.
[`ctmrg_unrolled`][tenet.network.ctmrg_unrolled] is the traceable form — exactly `k`
fixed-structure moves on a bond you decided outside:

```python
import jax
import tenet

tenet.enable_jax(ad=True)          # pytrees + the broadened SVD/eigh VJPs

def beta_f(beta):
    absorb, c, e = single_layer_ctm(ising_bulk(beta))
    bond = ctmrg(absorb, c, e, chi=chi).env.bond      # decided outside
    c, e = tenet.network.ctmrg_unrolled(c, e, absorb, bond, k=4)
    ...

jax.grad(beta_f)(0.4)
```

That is the same decide-outside / project-inside pairing as
[truncation](../guide/truncation.md): `svd_truncated` chooses a bond space out here,
`svd(..., bond=)` reuses it in there. `move(..., bond=B)` is shape-static for the same
reason; `move(..., chi=...)` is not.

`ctmrg_unrolled` takes `c`, `e` and `bond` as three arguments. `bond` is a `GradedSpace`
— frozen, hashable, array-free metadata — and it enters `jit` as a cache key, which is
what keeps it out of the flattened leaves.

[`normalized`][tenet.network.normalized], [`ring`][tenet.network.ring],
[`layers`][tenet.network.layers], [`single_layer`][tenet.network.single_layer] and
[`double_layer`][tenet.network.double_layer] are the pieces inside the traced region, and
each says on itself which side of the trace it belongs to.

The oracle for that gradient is `d(βf)/dβ` from Onsager's closed form, which
`tests/integration/test_ctmrg.py` checks the unrolled sweeps against.

## The iPEPS half

A single-site U(1) (or SU(2)) iPEPS with a random symmetric two-site `h`, descending
through the same gradient path. It exercises graded truncation, `svd(bond=)` across
sectors and multiplet degeneracies.

It makes **no benchmark-energy claim**, and a one-site unit cell cannot. A single-site AFM
Heisenberg cell needs one sublattice rotated by π about y, which turns `SˣSˣ - SʸSʸ` into
`(S⁺S⁺ + S⁻S⁻)/2` — an operator that changes `Sᶻ_tot` by ±2 and so destroys the U(1) the
ansatz is graded by. The file says so in its own docstring.

## What the library owns

`tenet.network` owns the environment: `Absorb`, the seeds, `move`, the sweep, the
truncation, the traced form. What it does not own, and the example supplies: the bulk
tensor, the ansatz — a library that symmetrized your input would be silently editing your
state — and what to measure, which is a geometry-specific reduced density matrix.

## Where next

- [JAX and backends](../guide/jax-and-backends.md) — `enable_jax(ad=True)` and what the
  broadened VJP assumes.
- [Truncation](../guide/truncation.md) — the pairing this page leans on.
- [`tenet.network`](../api/network.md) — `ctmrg`, `ctmrg_unrolled`, `move`, `Absorb`.
