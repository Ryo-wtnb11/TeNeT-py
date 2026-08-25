# 2D Ising CTMRG

The free energy per site of the classical 2D Ising model,

$$
H = -\sum_{\langle ij\rangle} s_i s_j,
\qquad
Z(\beta) = \sum_{\{s\}} e^{-\beta H},
\qquad
\beta f = -\frac{1}{N}\ln Z,
$$

computed by contracting the network $Z$ is with a corner-and-edge environment, and
compared with Onsager's closed form. Runs on a core install — no JAX.

**The network.** Splitting each bond weight symmetrically,
$e^{\beta ss'} = \sum_\mu W_{s\mu}W_{s'\mu}$ with $W = \bigl(\begin{smallmatrix}
\sqrt{\cosh\beta} & \sqrt{\sinh\beta}\\ \sqrt{\cosh\beta} & -\sqrt{\sinh\beta}
\end{smallmatrix}\bigr)$, and summing out each site spin gives one rank-4 tensor per site,

$$
a_{tlbr} = \sum_{s=\pm1} W_{st}W_{sl}W_{sb}W_{sr},
$$

so $Z$ is the translation-invariant contraction of copies of `ising_bulk(beta)`. The
columns of $W$ are the $\mathbb{Z}_2$ parity basis, which is why `ising_bulk` grades its
four legs by `Z2`: the eight entries with an odd number of odd legs cancel in the sum over
$s$ and have no block to live in. All four legs are `OUT` and identical — the C4v ansatz's
signature, and what lets one corner and one edge describe the whole environment.

**The approximation.** `EnvCTMc4v.iterate_(max_bond=chi)` converges a corner $C$ and an
edge $T$ standing for a quadrant and a half-row, each truncated to $\chi = 24$ states.
`log_kappa` then reads off $\ln\kappa = \frac1N \ln Z$ by Baxter's telescoping,
$\kappa = Z_{(L+1)^2} Z_{L^2} / Z_{(L+1)L} Z_{L(L+1)}$, in which every environment tensor
and gauge factor cancels.

**Checks.**

- `rel` is $\lvert \beta f_{\mathrm{TN}} / \beta f_{\mathrm{Onsager}} - 1\rvert$: $10^{-16}$
  off criticality, $2\times 10^{-6}$ at $\beta_c$ where the correlation length outruns any
  finite $\chi$.
- In the ordered phase the corner spectrum is exactly two-fold degenerate, the doublet
  spanning the two Z2 parity sectors — spontaneous symmetry breaking read off the
  environment.

Derivation in the [CTMRG tutorial](../tutorials/ctmrg.md); differentiating this same
contraction is [2D Ising thermodynamics by AD](ising-thermo.md).

## Source

```python
--8<-- "examples/ising2d.py"
```

## Output

Produced by `ising2d.main()` at its defaults — exactly `python examples/ising2d.py` — as
run by `tests/test_examples.py`.

```text
beta=0.3000   23 sweeps  beta*f = -0.7905590710  rel 6.7e-16
beta=0.4407  100 sweeps  beta*f = -0.9296933831  rel 2.2e-06
beta=0.5000   98 sweeps  beta*f = -1.0257928127  rel 8.9e-16
corner spectrum at beta=0.5: 0.6905 0.6905 0.1486 0.1486 0.0320 0.0320
```
