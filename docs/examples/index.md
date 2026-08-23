# Examples

Each page shows one file from the repository and the output it prints. The output is
committed and checked by CI, so every page reflects the current code.

Two kinds of example:

- `examples/` — scripts that **call the library**: `tenet.models`, `MPO`, `dmrg_`, `ctmrg`.
- `examples/toy_codes/` — DMRG, CTMRG and VMC **written on the tensor layer**: the
  algorithm is in the file, built from `SymmetricTensor`, `tenet.einsum` and
  `tenet.linalg`, with nothing imported from `tenet.network`. Read them to see how each
  algorithm is assembled from the tensor operations.

## Calling the library

| page | file | checked against |
| --- | --- | --- |
| [Heisenberg, U(1)](heisenberg.md) | `examples/heisenberg.py` | exact diagonalization, N=20 |
| [Heisenberg, U(1) walkthrough](heisenberg-walkthrough.md) | `examples/heisenberg_walkthrough.py` | exact diagonalization, N=12 |
| [Heisenberg, SU(2)](su2-heisenberg.md) | `examples/su2_heisenberg.py` | the U(1) run in the same script |
| [2D Ising CTMRG](ising2d.md) | `examples/ising2d.py` | Onsager's free energy |

## Written on the tensor layer

| page | file | checked against |
| --- | --- | --- |
| [Toy DMRG](toy-dmrg.md) | `examples/toy_codes/dmrg.py` | exact diagonalization |
| [Toy CTMRG](toy-ctmrg.md) | `examples/toy_codes/ctmrg.py` (needs `jax`) | Onsager's free energy |
| [Toy VMC on an MPS](toy-vmc-mps.md) | `examples/toy_codes/vmc_mps.py` (needs `jax`) | the energy decreases every step |

`examples/bench_dmrg.py` times the two-site matvec; it is a script, not a test.
