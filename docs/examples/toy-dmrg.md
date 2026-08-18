# Toy DMRG

What to look at: the N=12 energy against the exact `-5.142090632840532` printed beside
it, the sweep table's monotone `dE`, and the last line, where N=32 at `chi=64` sits just
above the Bethe-ansatz `e_inf` with the discarded weight that explains the gap. In the
source, the thing to read is that there is no `tenet.network` import: the MPS list, the
canonical form, the environment cache, the Lanczos step and the two-site sweep are all
written out on `SymmetricTensor`. The [DMRG tutorial](../tutorials/dmrg.md) walks through
the same physics through the library, and
[Heisenberg, U(1) walkthrough](heisenberg-walkthrough.md) is this chain called rather than
written.

## Source

```python
--8<-- "examples/toy_codes/dmrg.py"
```

## Output

Produced by `dmrg.main()` at its defaults — exactly `python examples/toy_codes/dmrg.py` —
as run by `tests/integration/test_dmrg.py`.

```text
N=12 chi=64  E=-5.142090632841  exact=-5.142090632840532
  sweep  1  E=-5.074270839002  dE=inf  dS=inf  dw=4.441e-16
  sweep  2  E=-5.142090098965  dE=6.782e-02  dS=4.787e-01  dw=6.883e-15
  sweep  3  E=-5.142090632840  dE=5.339e-07  dS=3.736e-03  dw=8.882e-15
  sweep  4  E=-5.142090632841  dE=1.794e-13  dS=1.358e-06  dw=5.329e-15
  sweep  5  E=-5.142090632841  dE=8.882e-16  dS=3.492e-10  dw=5.551e-15
N=32 chi=64  E=-13.997315618007  E/N=-0.437416113063  e_inf=-0.443147180560  sweeps=8  max_dw=6.650e-12
```
