# 2D Ising thermodynamics by AD

[2D Ising CTMRG](ising2d.md) contracts the Boltzmann network and reads off one number,
the free energy. This page differentiates that same contraction with respect to
$\beta$ and gets two more.

## What is computed

The classical model is $H = -\sum_{\langle ij\rangle} s_i s_j$ on the square lattice,
$s_i = \pm 1$, with

$$
Z(\beta) = \sum_{\{s\}} e^{-\beta H},
\qquad
\beta f(\beta) = -\frac{1}{N}\ln Z(\beta) .
$$

The two derivatives of $\beta f$ are themselves thermodynamic observables:

$$
u = -\frac{1}{N}\frac{\partial \ln Z}{\partial \beta}
  = \frac{\partial (\beta f)}{\partial \beta},
\qquad
c_V = \frac{\beta^{2}}{N}\frac{\partial^{2}\ln Z}{\partial\beta^{2}}
    = -\beta^{2}\frac{\partial^{2}(\beta f)}{\partial\beta^{2}} .
$$

$u$ is the internal energy per site and $c_V$ the specific heat per site. Onsager's
closed form gives $\beta f$ exactly, so all three quantities have an oracle — and the
derivatives of the oracle are what the derivatives of the code are judged against.

## The tensor network

$Z$ is a translation-invariant contraction of one rank-4 tensor. Split each bond weight
symmetrically, $e^{\beta s s'} = \sum_{\mu} W_{s\mu} W_{s'\mu}$ with

$$
W = \begin{pmatrix}\sqrt{\cosh\beta} & \sqrt{\sinh\beta}\\[2pt]
                   \sqrt{\cosh\beta} & -\sqrt{\sinh\beta}\end{pmatrix},
$$

and sum each site's spin out:

$$
a_{tlbr}(\beta) \;=\; \sum_{s=\pm 1} W_{st}\,W_{sl}\,W_{sb}\,W_{sr},
\qquad
Z = \operatorname{tTr}\bigotimes_{\text{sites}} a .
$$

The columns of $W$ *are* the $\mathbb{Z}_2$ parity basis: column $\mu = 0$ does not
depend on $s$, column $\mu = 1$ is odd under $s \to -s$. So the sum over $s$ doubles
every entry whose four leg parities multiply to even and cancels the other eight. Those
eight have no block in a $\mathbb{Z}_2$-graded `SymmetricTensor` and are never built —
the grading is the statement of the model, not a check applied to it. It is also what
keeps a finite-$\chi$ environment from breaking the symmetry spuriously above $\beta_c$,
which is why Onsager is an oracle on both sides of the transition here.

The infinite contraction is approximated by a corner-and-edge environment
([`EnvCTMc4v`][tenet.network.EnvCTMc4v]): one corner $C$ and one edge $T$ stand for a
quadrant and a half-row of the lattice. Baxter's telescoping then extracts one site's
worth of partition function from three patches,

$$
\kappa = \frac{Z_{(L+1)\times(L+1)}\,Z_{L\times L}}
              {Z_{(L+1)\times L}\,Z_{L\times (L+1)}},
\qquad
\ln\kappa = \frac{1}{N}\ln Z,
$$

because $(L+1)^2 + L^2 - 2L(L+1) = 1$ and every environment tensor and gauge factor
cancels between numerator and denominator. That is `ising2d.log_kappa`.

## Code to mathematics

| object in the file | mathematics |
| --- | --- |
| `traced_bulk(beta)` | $a_{tlbr}(\beta)$, one block per allowed parity assignment |
| `warm(beta)` — `iterate_` | converge $C, T$ and *decide* the environment bond $\chi$ |
| the returned `bond` | the truncated environment `GradedSpace`; static metadata |
| `beta_free_energy` — `update_(bond=...)` | $K$ CTM moves at that frozen bond |
| `log_kappa(env)` | $\ln\kappa$, the three-patch telescoping above |
| `jax.grad(beta_free_energy)` | $u = \partial_\beta(\beta f)$ |
| `jax.grad(jax.grad(...))` | $\partial^2_\beta(\beta f)$, hence $c_V$ |

The whole chain is
$\beta \rightarrow a(\beta) \rightarrow \text{CTM} \rightarrow \ln\kappa \rightarrow
\beta f \rightarrow \partial_\beta (\beta f)$,
and $\beta$ enters only through *block values*. The grading, the block shapes and the
environment bond are structure, so `jax` never sees them change.

## What is approximated, and what is differentiated

Two separate approximations, and the page keeps them apart:

- **The environment is finite.** $\chi = 16$ truncates the corner spectrum. Off
  criticality this is invisible: $\beta f$ matches Onsager to $10^{-12}$ relative.
- **The gradient is truncated backprop through $K$ unrolled CTM moves — not an
  implicit fixed-point derivative.** [`iterate_`][tenet.network.EnvCTM.iterate_] loops on
  a measured spectrum change and re-decides the bond each sweep, so it can never run
  inside a trace; it runs **once, outside**, and hands the traced region a converged
  $(C, T)$ as a *constant* initial condition plus a frozen `GradedSpace` bond. Inside,
  exactly $K$ calls to [`update_`][tenet.network.EnvCTMc4v.update_]`(bond=...)` carry
  the derivative. This is the decide-outside / project-inside pairing of
  [Truncation](../guide/truncation.md).

The cost of the finite $K$ is *measured*, not asserted: because the environment is
already at its fixed point when the traced region starts, $\beta f$ does not depend on
$K$ and $u$ barely does, but $c_V$ does — the $K$ moves must carry the environment's
second-order response to $\beta$ themselves. The last output line is that scan.

## How we know it is right

Every number on the page has an independent oracle, checked in
`tests/test_examples.py`:

- $\beta f$ against Onsager's quadrature, $10^{-12}$ relative;
- $u$ against a central difference of Onsager at $h = 10^{-4}$, $10^{-6}$ relative —
  which is the accuracy of the *oracle*, so the AD value is not measurably worse;
- $c_V$ against a central second difference at $h = 10^{-3}$, $10^{-3}$ relative;
- $|c_V(K) - c_V^{\text{Onsager}}|$ decreasing monotonically in $K$, and by more than a
  factor of ten from $K = 2$ to $K = 8$. That is the truncated-backprop claim made
  checkable.
- `traced_bulk` against `ising2d.ising_bulk` as dense arrays, so the differentiated
  model is literally the model the non-AD page contracts.

Full derivation of the CTMRG side: the [CTMRG tutorial](../tutorials/ctmrg.md). The same
gradient written out on the tensor layer, plus a variational iPEPS optimization through
it: [Toy CTMRG](toy-ctmrg.md).

## Source

```python
--8<-- "examples/ising_thermo.py"
```

## Output

Produced by `ising_thermo.main()` at its defaults — exactly
`python examples/ising_thermo.py` — as run by `tests/test_examples.py`. Onsager's value
is in parentheses after each quantity.

```text
beta=0.30  beta*f=-0.7905590710 (-0.7905590710)  u=-0.70449907 (-0.70449909)  c_V=+0.286290 (+0.286291)
beta=0.50  beta*f=-1.0257928127 (-1.0257928127)  u=-1.74556458 (-1.74556451)  c_V=+0.724844 (+0.724890)
c_V at beta=0.5 vs unrolled moves: K=2:+0.698017  K=4:+0.722584  K=8:+0.724844
```
