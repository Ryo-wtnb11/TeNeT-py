# Toy VMC on an MPS

Variational Monte Carlo written on the tensor layer, on a symmetric MPS: the MPS as a JAX
pytree, the Rayleigh quotient $\langle\psi\vert h
\vert\psi\rangle/\langle\psi\vert\psi\rangle$, `jax.grad`, and an SGD step with `jax.tree.map`.
Needs the `jax` extra.

- Both traces (U(1) and SU(2)) decrease on every one of the 20 steps.

Explained in the [VMC tutorial](../tutorials/vmc.md).

## Source

[`examples/toy_codes/vmc_mps.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/vmc_mps.py)

```python linenums="1"
--8<-- "examples/toy_codes/vmc_mps.py"
```

## Output

Produced by `vmc_mps.main(provider=...)` at its defaults for both providers — exactly
`python examples/toy_codes/vmc_mps.py` — as run by `tests/backends/test_ad.py`.

```text
u1: -0.682692 -0.725133 -0.763486 -0.798024 -0.829057 -0.856907 -0.881888 -0.904301 -0.924424 -0.942510 -0.958786 -0.973455 -0.986697 -0.998672 -1.009520 -1.019363 -1.028312 -1.036461 -1.043895 -1.050687
su2: -1.033559 -1.038339 -1.042817 -1.047020 -1.050971 -1.054691 -1.058199 -1.061512 -1.064645 -1.067612 -1.070424 -1.073094 -1.075631 -1.078045 -1.080343 -1.082534 -1.084624 -1.086621 -1.088529 -1.090355
```
