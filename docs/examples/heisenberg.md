# Heisenberg, U(1)

A 20-site spin-1/2 Heisenberg chain to its ground state: `tenet.models.spin_half`,
`MPO.from_terms`, `MPS.product`, `dmrg_`, then bond energies with `expectation_2site`.

- The bond energies sum to `out.energy`.
- `<S^z_n>` is zero on every site: the Néel seed fixes `S^z_tot = 0` and the sweep keeps it.

Explained step by step in the [DMRG tutorial](../tutorials/dmrg.md).

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
