# Examples

Every page below carries a file's source, included from the repository, and the output it
printed — committed here and checked against the run CI performs, so a stale page fails
the test suite.

Two lanes. `examples/` **calls** the library, so its files are named for the model.
`examples/toy_codes/` **writes the algorithm out** on `tenet`'s tensor layer —
`SymmetricTensor`, `tenet.einsum`, `tenet.linalg` — and imports nothing from
`tenet.network`, so its files are named for the algorithm. The rule is a test with no
exemptions.

## Calling the library

Each runs on a core install.

| page | file | oracle |
| --- | --- | --- |
| [Heisenberg, U(1)](heisenberg.md) | `examples/heisenberg.py` | the recorded N=20 exact-diagonalization energy |
| [Heisenberg, U(1) walkthrough](heisenberg-walkthrough.md) | `examples/heisenberg_walkthrough.py` | the recorded N=12 exact-diagonalization energy |
| [Heisenberg, SU(2)](su2-heisenberg.md) | `examples/su2_heisenberg.py` | the U(1) run it computes alongside |
| [2D Ising CTMRG](ising2d.md) | `examples/ising2d.py` | Onsager's closed-form free energy |

## Writing the algorithm out

| page | file | oracle |
| --- | --- | --- |
| [Toy DMRG](toy-dmrg.md) | `examples/toy_codes/dmrg.py` | exact diagonalization |
| [Toy CTMRG](toy-ctmrg.md) | `examples/toy_codes/ctmrg.py` (needs `jax`) | Onsager's closed-form free energy |
| [Toy VMC on an MPS](toy-vmc-mps.md) | `examples/toy_codes/vmc_mps.py` (needs `jax`) | the variational energy decreasing |

## Benchmarks

`examples/bench_dmrg.py` times the two-site matvec. It is a script, not a test: it asserts
nothing and prints for a PR body. `benchmarks/` holds the rest, on no CI path.
