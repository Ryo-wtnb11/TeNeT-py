# Examples

Each page shows one file from the repository and the output it prints. The output is
committed and checked by CI, so every page reflects the current code.

Two kinds of example:

- `examples/` — scripts that **call the library**: `tenet.models`, `MPO`, `dmrg_`, `ctmrg`.
- `examples/toy_codes/` — **naive implementations** of DMRG, CTMRG and VMC written with
  `tenet`'s basic tensor operations only (`SymmetricTensor`, `tenet.einsum`,
  `tenet.linalg`). They import nothing from `tenet.network`. Read them to see how the
  algorithms are built from the tensor layer.

## Calling the library

| page | file | checked against |
| --- | --- | --- |
| [Heisenberg, U(1)](heisenberg.md) | `examples/heisenberg.py` | exact diagonalization, N=20 |
| [Heisenberg, U(1) walkthrough](heisenberg-walkthrough.md) | `examples/heisenberg_walkthrough.py` | exact diagonalization, N=12 |
| [Heisenberg, SU(2)](su2-heisenberg.md) | `examples/su2_heisenberg.py` | the U(1) run in the same script |
| [2D Ising CTMRG](ising2d.md) | `examples/ising2d.py` | Onsager's free energy |

## Naive implementations

| page | file | checked against |
| --- | --- | --- |
| [Toy DMRG](toy-dmrg.md) | `examples/toy_codes/dmrg.py` | exact diagonalization |
| [Toy CTMRG](toy-ctmrg.md) | `examples/toy_codes/ctmrg.py` (needs `jax`) | Onsager's free energy |
| [Toy VMC on an MPS](toy-vmc-mps.md) | `examples/toy_codes/vmc_mps.py` (needs `jax`) | the energy decreases every step |

`examples/bench_dmrg.py` times the two-site matvec; it is a script, not a test.
