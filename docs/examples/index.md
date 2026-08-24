# Examples

Each page shows one file from the repository and the output it prints. The output is
committed and checked by CI, so every page reflects the current code.

Two kinds of example:

- `examples/` — scripts that **call the library**: `tenet.models`, `MPO`, `dmrg_`, `ctmrg`.
- `examples/toy_codes/` — TEBD, DMRG, CTMRG and VMC **written on the tensor layer**: the
  algorithm is in the file, built from `SymmetricTensor`, `tenet.einsum` and
  `tenet.linalg`, with nothing imported from `tenet.network`. Read them to see how each
  algorithm is assembled from the tensor operations. One module holds one concept and
  imports its neighbours, so each has a page of its own; the table below is in reading
  order — the state, then the model, then the algorithms that consume it.

## Calling the library

| page | file | checked against |
| --- | --- | --- |
| [Heisenberg, U(1)](heisenberg.md) | `examples/heisenberg.py` | exact diagonalization, N=20 |
| [Heisenberg, U(1) walkthrough](heisenberg-walkthrough.md) | `examples/heisenberg_walkthrough.py` | exact diagonalization, N=12 |
| [Heisenberg, SU(2)](su2-heisenberg.md) | `examples/su2_heisenberg.py` | the U(1) run in the same script |
| [Heisenberg, SU(3)](su3-heisenberg.md) | `examples/su3_heisenberg.py` | dense ED in the same script, N=6 |
| [2D Ising CTMRG](ising2d.md) | `examples/ising2d.py` | Onsager's free energy |

## Written on the tensor layer

| page | file | holds | checked against |
| --- | --- | --- | --- |
| [Toy MPS](toy-mps.md) | `examples/toy_codes/mps.py` | the state, canonical form, entropy, expectation values | through `tebd.py` and `dmrg.py` |
| [Toy model](toy-model.md) | `examples/toy_codes/model.py` | the Heisenberg chain as gates *and* as an MPO | the dense operator, in `tests/integration/test_dmrg.py` |
| [Toy TEBD](toy-tebd.md) | `examples/toy_codes/tebd.py` | imaginary-time evolution from the gates | `exact.py`, approached from above |
| [Toy DMRG](toy-dmrg.md) | `examples/toy_codes/dmrg.py` | environments and the two-site sweep | exact diagonalization |
| [Toy Lanczos](toy-lanczos.md) | `examples/toy_codes/lanczos.py` | the Krylov ground-eigenpair step | `numpy.linalg.eigvalsh`, in `tests/integration/test_dmrg.py` |
| [Toy exact](toy-exact.md) | `examples/toy_codes/exact.py` | dense ED of the same chain | the recorded open-chain energies |
| [Toy Ising](toy-ising.md) | `examples/toy_codes/ising.py` (needs `jax`) | the Boltzmann bulk tensor and Onsager | through `ctmrg.py` |
| [Toy CTMRG](toy-ctmrg.md) | `examples/toy_codes/ctmrg.py` (needs `jax`) | the moves, the sweep, the iPEPS gradient | Onsager's free energy |
| [Toy VMC on an MPS](toy-vmc-mps.md) | `examples/toy_codes/vmc_mps.py` (needs `jax`) | the Rayleigh quotient and an SGD step | the energy decreases every step |

`examples/bench_dmrg.py` times the two-site matvec; it is a script, not a test.
