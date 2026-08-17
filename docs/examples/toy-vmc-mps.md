# Toy VMC on an MPS

What to look at: both traces are strictly decreasing across all 20 SGD steps — `jax.grad`
taken straight through a Rayleigh quotient of `SymmetricTensor` pytrees, for a U(1) and
an SU(2) ansatz through the identical code path. The
[VMC tutorial](../tutorials/vmc.md) walks through the file.

## Source

```python
--8<-- "examples/toy_codes/vmc_mps.py"
```

## Output

Produced by `vmc_mps.main(provider=...)` at its defaults for both providers — exactly
`python examples/toy_codes/vmc_mps.py` — as run by `tests/backends/test_ad.py`.

```text
u1: -0.682692 -0.725133 -0.763486 -0.798024 -0.829057 -0.856907 -0.881888 -0.904301 -0.924424 -0.942510 -0.958786 -0.973455 -0.986697 -0.998672 -1.009520 -1.019363 -1.028312 -1.036461 -1.043895 -1.050687
su2: -1.033559 -1.038339 -1.042817 -1.047020 -1.050971 -1.054691 -1.058199 -1.061512 -1.064645 -1.067612 -1.070424 -1.073094 -1.075631 -1.078045 -1.080343 -1.082534 -1.084624 -1.086621 -1.088529 -1.090355
```
