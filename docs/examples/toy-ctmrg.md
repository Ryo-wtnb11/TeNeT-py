# Toy CTMRG

What to look at: `move`, the file's one CTMRG step, which builds the enlarged corner,
takes a projector from it and absorbs one edge — with `svd_truncated` outside the trace
deciding a bond and `svd(bond=)` inside it reusing one. Then the output: the `rel` column —
the CTMRG free energy against Onsager's closed form — next to `d(beta f)/dbeta`, which is
`jax.grad` taken through the unrolled sweeps and checked against the same oracle; then the
two iPEPS traces, a U(1) and an SU(2) ansatz descending through the identical gradient
path. The [CTMRG tutorial](../tutorials/ctmrg.md) walks through the file.

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
