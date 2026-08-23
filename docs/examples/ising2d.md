# 2D Ising CTMRG

The classical 2D Ising partition function through `ctmrg(*single_layer_ctm(bulk))`.

- `rel` is the free energy against Onsager's exact result.
- In the ordered phase the corner spectrum is exactly two-fold degenerate: the doublet
  spans the two Z2 parity sectors.

Explained in the [CTMRG tutorial](../tutorials/ctmrg.md).

## Source

```python
--8<-- "examples/ising2d.py"
```

## Output

Produced by `ising2d.main()` at its defaults — exactly `python examples/ising2d.py` — as
run by `tests/test_examples.py`.

```text
beta=0.3000   23 sweeps  beta*f = -0.7905590710  rel 3.3e-16
beta=0.4407  100 sweeps  beta*f = -0.9296935333  rel 2.0e-06
beta=0.5000   93 sweeps  beta*f = -1.0257928127  rel 1.3e-15
corner spectrum at beta=0.5: 0.6905 0.6905 0.1486 0.1486 0.0320 0.0320
```
