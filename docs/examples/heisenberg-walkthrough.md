# Heisenberg, U(1) — the walkthrough

The same chain as [Heisenberg, U(1)](heisenberg.md), with the symmetry input spelled out:
the hand-graded 5x5 `W`, the reachable-charge bond spaces, and three MPO routes
cross-checked against each other.

What to look at: the three routes. `MPO.from_w` takes the hand-graded MPO bond the file
writes out; `MPO.from_entries` names the same `W`'s eight non-zero entries and declares no
grading; `MPO.from_terms` derives one from the operators' own charges. The two gradings
printed below are the same grading, sector for sector, so all three runs land on the same
twelve digits.

The two lines after them say which route carries an **edge description** — the
finite-state machine that decides whether `Env.heff2` takes the prepared, symbolic engine
path or contracts the site tensors. At the builders' default none of them does, which is
the lattice model's path; `symbolic=True` keeps it, and writing a `W` by hand does not
cost you that option. See
[Building a Hamiltonian](../guide/hamiltonians.md#symbolictrue).

Below them, the seed's bond dimensions are the *reachable* U(1) charges at each cut
(`1 2 3 4 …`, the whole `S^z_tot = 0` statement) and the final ones are what
`svd_truncated` decided sweep by sweep.

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
