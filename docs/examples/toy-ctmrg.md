# Toy CTMRG

C4v CTMRG written on the tensor layer — corner, edge,
absorption, projector, sweep — followed by an iPEPS gradient through it with `jax.grad`.
Needs the `jax` extra.

- `move` is one CTMRG step: enlarge the corner, build the projector, absorb an edge.
  `svd_truncated` chooses the bond outside the trace; `svd(bond=)` reuses it inside.
- `rel` is the free energy against Onsager's exact result; `d(beta f)/dbeta` is the
  gradient through the unrolled sweeps, checked against the same result.
- The two iPEPS traces (U(1) and SU(2)) descend through the same code.

The same algorithm through the library: [2D Ising CTMRG](ising2d.md).

## Source

```python
--8<-- "examples/toy_codes/ctmrg.py"
```

## Output

Produced by `ctmrg.main(chi_ising=16, k=4, steps=1)` as run by
`tests/integration/test_ctmrg.py` — the run CI performs; `main()`'s default is `steps=3`,
so `python examples/toy_codes/ctmrg.py` prints three entries per iPEPS trace instead of
the one below.

```text
ising beta=0.30  beta*f=-0.7905590710  onsager=-0.7905590710  rel=1.11e-16  d(beta f)/dbeta=-0.70449907
ising beta=0.40  beta*f=-0.8793638208  onsager=-0.8793638208  rel=5.22e-15  d(beta f)/dbeta=-1.10607920
ising beta=0.50  beta*f=-1.0257928127  onsager=-1.0257928127  rel=5.47e-14  d(beta f)/dbeta=-1.74556458
ipeps u1: -0.31099339
ipeps su2: -0.03847516
```
