# Heisenberg, SU(2)

What to look at: the `U(1) :` / `SU(2):` pair near the end. The SU(2) run lands on the
same energy as the U(1) run it computes in-process (whose own four lines print first),
from a mid-chain bond counted in **multiplets** — 22 of them against the 64 dense states
U(1) keeps — which is the whole payoff of the non-Abelian grading. See the
[DMRG tutorial](../tutorials/dmrg.md), "The same chain under SU(2)".

## Source

```python
--8<-- "examples/su2_heisenberg.py"
```

## Output

Produced by `su2_heisenberg.main()` at its defaults — exactly
`python examples/su2_heisenberg.py` — as run by `tests/test_examples.py`.

```text
N=20  9 sweeps  E = -8.682473334397688  mid bond: 64 states
bond energies: -0.6534 -0.2943 -0.5664 -0.3370 -0.5401 -0.3540 -0.5286 -0.3616 -0.5239 -0.3638 -0.5239 -0.3616 -0.5286 -0.3540 -0.5401 -0.3370 -0.5664 -0.2943 -0.6534
sum of bond energies = -8.682473334397699  vs  out.energy = -8.682473334397688
max_n |<S^z_n>| = 4.7e-13
U(1) : 9 sweeps  E = -8.682473334398  mid bond 64 states
SU(2): 5 sweeps  E = -8.682473334396  mid bond 22 multiplets, 62 dense
|E_su2 - E_u1| = 2.1e-12
```
