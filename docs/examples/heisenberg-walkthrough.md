# Heisenberg, U(1) — the walkthrough

What to look at: the two MPO routes. `MPO.from_w` takes the hand-graded MPO bond the file
writes out; `MPO.from_terms` derives one from the operators' own charges — and the two
gradings printed below are the same grading, sector for sector, so the two runs land on
the same twelve digits. Below them, the seed's bond dimensions are the *reachable* U(1)
charges at each cut (`1 2 3 4 …`, the whole `S^z_tot = 0` statement) and the final ones
are what `svd_truncated` decided sweep by sweep. `examples/heisenberg.py` is the same
chain with none of this spelled out; `examples/toy_codes/dmrg.py` writes the algorithm
underneath it. The [DMRG tutorial](../tutorials/dmrg.md) walks through the physics.

## Source

```python
--8<-- "examples/heisenberg_walkthrough.py"
```

## Output

Produced by `heisenberg_walkthrough.main()` at its defaults — exactly
`python examples/heisenberg_walkthrough.py` — as run by `tests/test_examples.py`.

```text
from_w      N=12 chi=64  E=-5.142090632841  exact=-5.142090632841
  sweep  1  E=-5.074270839002  dE=inf  dS=inf  dw=4.441e-16
  sweep  2  E=-5.142090098965  dE=6.782e-02  dS=4.787e-01  dw=6.883e-15
  sweep  3  E=-5.142090632840  dE=5.339e-07  dS=3.736e-03  dw=8.882e-15
  sweep  4  E=-5.142090632841  dE=1.794e-13  dS=1.358e-06  dw=5.329e-15
  sweep  5  E=-5.142090632841  dE=8.882e-16  dS=3.492e-10  dw=5.551e-15
from_terms  N=12 chi=64  E=-5.142090632841
  |E(from_w) - E(from_terms)| = 1.776e-15
  MPO bond, hand-graded: -2:1 +0:3 +2:1
  MPO bond, derived:     -2:1 +0:3 +2:1
  seed bond dims:  [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
  final bond dims: [1, 2, 4, 8, 16, 32, 40, 32, 16, 8, 4, 2, 1]
```
