# Heisenberg, U(1) — the walkthrough

The same chain built three ways and cross-checked: `MPO.from_w` with a hand-written
graded `W`, `MPO.from_entries` with the same `W` as eight sparse entries, and
`MPO.from_terms` from the operators.

- All three give the same energy.
- The two printed bond gradings are identical sector for sector.
- The `edge description` lines show which operators keep the symbolic description
  (`symbolic=True`); see [Building a Hamiltonian](../guide/hamiltonians.md#symbolictrue).

## Source

```python
--8<-- "examples/heisenberg_walkthrough.py"
```

## Output

Produced by `heisenberg_walkthrough.main()` at its defaults — exactly
`python examples/heisenberg_walkthrough.py` — as run by `tests/test_examples.py`.

```text
from_w      N=12 chi=64  E=-5.142090632841  exact=-5.142090632841
  sweep  1  E=-5.074270839002  dE=inf  dS=inf  dw=2.220e-16
  sweep  2  E=-5.142090098965  dE=6.782e-02  dS=4.787e-01  dw=7.105e-15
  sweep  3  E=-5.142090632840  dE=5.339e-07  dS=3.736e-03  dw=8.882e-15
  sweep  4  E=-5.142090632841  dE=1.856e-13  dS=1.358e-06  dw=5.329e-15
  sweep  5  E=-5.142090632841  dE=3.553e-15  dS=3.492e-10  dw=5.773e-15
from_terms  N=12 chi=64  E=-5.142090632841
  |E(from_w) - E(from_terms)| = 8.882e-16
from_entries N=12 chi=64  E=-5.142090632841
  |E(from_w) - E(from_entries)| = 5.329e-15
  carries an edge description: {'from_w': False, 'from_entries': False, 'from_terms': False}
  the same two under symbolic=True: {'from_entries': True, 'from_terms': True}
  MPO bond, hand-graded: -2:1 +0:3 +2:1
  MPO bond, derived:     -2:1 +0:3 +2:1
  seed bond dims:  [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
  final bond dims: [1, 2, 4, 8, 16, 32, 40, 32, 16, 8, 4, 2, 1]
```
