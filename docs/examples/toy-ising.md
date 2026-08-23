# Toy Ising

The physics the toy CTMRG contracts: the classical 2D Ising partition-function tensor,
Z2-graded, and Onsager's closed form for `beta * f` by direct quadrature.

- The Z2 legs are the statement, not a check applied afterwards: the eight structurally
  zero entries of the bulk tensor have no block to live in and are never built.
- `beta` may be a traced scalar, so the block values are built with `jax.numpy` — this is
  what `jax.grad` differentiates through in [Toy CTMRG](toy-ctmrg.md).
- `onsager` is the oracle both the free energy and its `beta`-derivative are judged
  against, cross-checked against the elliptic form in `tests/integration/test_ctmrg.py`.

Imported by [Toy CTMRG](toy-ctmrg.md), which prints the comparison.

## Source

[`examples/toy_codes/ising.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/ising.py)

```python linenums="1"
--8<-- "examples/toy_codes/ising.py"
```
