# Heisenberg, U(1)

What to look at: the sum of the 19 bond energies reproduces `out.energy` to twelve
digits — `expectation_2site` and `dmrg_` weigh the same state on the same scale — and
`max_n |<S^z_n>|` is float noise, because the `S^z_tot = 0` sector is enforced by the
Néel seed's own charges, not by a projector. See the [DMRG tutorial](../tutorials/dmrg.md)
for the physics.

## Source

```python
--8<-- "examples/heisenberg.py"
```

## Output

Produced by `heisenberg.main()` at its defaults — exactly `python examples/heisenberg.py`
— as run by `tests/test_examples.py`.

```text
N=20  9 sweeps  E = -8.682473334397713  mid bond: 64 states
bond energies: -0.6534 -0.2943 -0.5664 -0.3370 -0.5401 -0.3540 -0.5286 -0.3616 -0.5239 -0.3638 -0.5239 -0.3616 -0.5286 -0.3540 -0.5401 -0.3370 -0.5664 -0.2943 -0.6534
sum of bond energies = -8.682473334397695  vs  out.energy = -8.682473334397713
max_n |<S^z_n>| = 4.7e-13
```
