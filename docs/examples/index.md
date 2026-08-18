# Examples

`examples/toy_codes/` teaches what the library does not own; `examples/` uses what it
does. Every page below carries the file's source, included from the repository, and the
output it printed — committed here and checked against the run CI already performs, so a
stale page fails the test suite.

**The usage lane** — library calls end to end, on a core install:

- [Heisenberg, U(1)](heisenberg.md) — `examples/heisenberg.py`
- [Heisenberg, U(1) walkthrough](heisenberg-walkthrough.md) — `examples/heisenberg_walkthrough.py`,
  the same chain with the MPO grading and the reachable bond spaces written out
- [Heisenberg, SU(2)](su2-heisenberg.md) — `examples/su2_heisenberg.py`
- [2D Ising CTMRG](ising2d.md) — `examples/ising2d.py`

**The teaching lane** — the algorithms written out by hand over `tenet`'s tensors, on
`SymmetricTensor`/`tenet.einsum`/`tenet.linalg` and importing nothing from
`tenet.network`:

- [Toy DMRG](toy-dmrg.md) — `examples/toy_codes/dmrg.py`
- [Toy CTMRG](toy-ctmrg.md) — `examples/toy_codes/ctmrg.py` (needs JAX; the one file
  still calling `tenet.network`, recorded as an exemption in
  [`examples/toy_codes/README.md`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/README.md))
- [Toy VMC on an MPS](toy-vmc-mps.md) — `examples/toy_codes/vmc_mps.py` (needs JAX)
