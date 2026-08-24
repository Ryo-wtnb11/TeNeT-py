# Toy codes

Standalone teaching implementations. Each file writes its algorithm out on `tenet`'s
**tensor** layer — `SymmetricTensor`, `tenet.einsum`, `tenet.linalg` — and imports
**nothing from `tenet.network`**, so you can read how the algorithm works instead of
watching it be called. That is the whole rule, and `tests/test_examples.py` asserts it
file by file. A toy code may use the symmetric-tensor layer because that is the library's
subject matter, not the algorithm being taught; the split follows `tenpy_toycodes`.

One module holds one concept and imports its neighbours, in reading order:

| file | holds |
| --- | --- |
| `mps.py` | the MPS list, product/random seeds, canonical form, Schmidt values, entropy, expectation values |
| `model.py` | the U(1) Heisenberg chain as two-site gates *and* as an MPO — one physics, two representations |
| `tebd.py` | imaginary-time TEBD from the gates — runs `main()` |
| `dmrg.py` | environments and the two-site sweep, from the MPO — runs `main()` |
| `lanczos.py` | the Krylov ground-eigenpair step, on any space with an inner product |
| `exact.py` | dense ED of the same chain, the reference the other two print against — runs `main()` |
| `ising.py` | the classical Ising bulk tensor and Onsager's free energy |
| `ctmrg.py` | corner, edge, projector, the move, the iPEPS gradient — runs `main()` |
| `vmc_mps.py` | the Rayleigh quotient, `jax.grad` and an SGD step — runs `main()` |

The library's own version of each algorithm lives in `tenet.network`, and the usage lane
(`examples/`) calls it: `examples/heisenberg_walkthrough.py` is `dmrg.py`'s chain through
`MPS`/`MPO`/`dmrg_`, and `examples/ising2d.py` is `ctmrg.py`'s Ising through `ctmrg()`.

**No exemptions.** `ctmrg.py` was the last file with one — #114 promoted its CTMRG core
into the library and the file kept calling it — and #187 rewrote it on the tensor layer,
so the rule above is now true of every file here with nothing excused.

For the lane rule and the file table, see the root [README](../../README.md#examples).
