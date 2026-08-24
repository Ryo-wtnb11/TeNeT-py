# Heisenberg, SU(2)

The same chain under SU(2): one invariant $\mathbf{S} \cdot \mathbf{S}$ operator per bond, a
random multiplet seed, and the U(1) run beside it.

- Both runs give the same energy.
- The SU(2) bond holds 22 multiplets where the U(1) bond holds 64 states.

Explained in the [SU(2) tutorial](../tutorials/su2.md).

## Source

```python
--8<-- "examples/su2_heisenberg.py"
```

## Output

Produced by `su2_heisenberg.main()` at its defaults — exactly
`python examples/su2_heisenberg.py` — as run by `tests/test_examples.py`.

```text
N=20  9 sweeps  E = -8.682473334397713  mid bond: 64 states
bond energies: -0.6534 -0.2943 -0.5664 -0.3370 -0.5401 -0.3540 -0.5286 -0.3616 -0.5239 -0.3638 -0.5239 -0.3616 -0.5286 -0.3540 -0.5401 -0.3370 -0.5664 -0.2943 -0.6534
sum of bond energies = -8.682473334397695  vs  out.energy = -8.682473334397713
max_n |<S^z_n>| = 4.7e-13
U(1) : 9 sweeps  E = -8.682473334398  mid bond 64 states
SU(2): 5 sweeps  E = -8.682473334396  mid bond 22 multiplets, 62 dense
|E_su2 - E_u1| = 1.6e-12
```
