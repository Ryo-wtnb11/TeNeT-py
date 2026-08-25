# Toy CTMRG

C4v CTMRG written on the tensor layer — corner, edge, absorption, projector, sweep —
then run on two problems: the classical Ising free energy *and its $\beta$-derivative*,
and a variational iPEPS optimization. Nothing is imported from `tenet.network`; needs the
`jax` extra. The bulk tensor it contracts and the Onsager form it is judged against are
[Toy Ising](toy-ising.md).

## The algorithm

One CTMRG move (`move`) is: enlarge the corner by absorbing an edge and a bulk tensor,
factor the enlarged corner $\tilde C = U S V^\dagger$, keep $U$ as the projector, and
absorb one bulk tensor into the edge through it. Iterating to a fixed point gives a corner
$C$ and edge $T$ that stand for a quadrant and a half-row of the infinite lattice,
compressed onto $\chi$ states.

`svd_truncated` picks the surviving sectors from measured singular values, so it runs in
`converge`, **outside** the trace; `svd(bond=)` reuses that frozen `GradedSpace` in
`unrolled`, **inside** it.

## Problem 1 — Ising thermodynamics

$\beta f = -\ln\kappa$ from Baxter's telescoping (`log_kappa`), and

$$
u = \frac{\partial (\beta f)}{\partial\beta}
$$

by `jax.grad` through the unrolled moves. `rel` is $\beta f$ against Onsager;
$\mathrm{d}(\beta f)/\mathrm{d}\beta$ is judged against the same closed form. The gradient
is **truncated backprop through $k = 4$ unrolled moves from a converged initial
condition**, not an implicit fixed-point derivative — the converged $(C, T)$ enters the
traced region as a constant. The second derivative, and what the finite $k$ costs there,
is [2D Ising thermodynamics by AD](ising-thermo.md).

## Problem 2 — variational iPEPS

A single-site ansatz $A$ with a random symmetric two-site $h$, objective

$$
E(A) = \frac{\langle\Psi(A)\vert h\vert\Psi(A)\rangle}
            {\langle\Psi(A)\vert\Psi(A)\rangle},
$$

evaluated as a ratio of two contractions of the same $2\times1$ patch: numerator with the
physical legs held open for $h$ to close, denominator with them closed against each other
(the environment is defined only up to a scale, so the ratio is the observable). `step` is
$A \leftarrow A - \eta\,\nabla_A E$ by `jax.value_and_grad` and `jax.tree.map` — the
gradient is a `SymmetricTensor` with $A$'s own structure, so the step touches only block
values and the updated ansatz is symmetric by construction. Both traces (U(1) and SU(2))
run through identical code and the energy falls on every step.

**No benchmark-energy claim.** A one-site unit cell cannot represent the AFM Heisenberg
ground state without a sublattice rotation that destroys the U(1) the ansatz is graded by;
`h` is a random symmetric operator and this half is a plumbing result. The file says so in
its own docstring.

The Ising half through the library: [2D Ising CTMRG](ising2d.md) and
[2D Ising thermodynamics by AD](ising-thermo.md). Full derivation:
[CTMRG tutorial](../tutorials/ctmrg.md).

## Source

[`examples/toy_codes/ctmrg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/ctmrg.py)

```python linenums="1"
--8<-- "examples/toy_codes/ctmrg.py"
```

## Output

Produced by `ctmrg.main(chi_ising=16, k=4, steps=1)` as run by
`tests/integration/test_ctmrg.py` — the run CI performs; `main()`'s default is `steps=3`,
so `python examples/toy_codes/ctmrg.py` prints three entries per iPEPS trace instead of
the one below.

```text
ising beta=0.30  beta*f=-0.7905590710  onsager=-0.7905590710  rel=1.11e-16  d(beta f)/dbeta=-0.70449907
ising beta=0.40  beta*f=-0.8793638208  onsager=-0.8793638208  rel=5.22e-15  d(beta f)/dbeta=-1.10607920
ising beta=0.50  beta*f=-1.0257928127  onsager=-1.0257928127  rel=5.47e-14  d(beta f)/dbeta=-1.74556458
ipeps u1: -0.31099339
ipeps su2: -0.03847516
```
