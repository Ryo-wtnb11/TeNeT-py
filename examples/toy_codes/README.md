# Toy codes

Standalone teaching implementations. Each file writes its algorithm out on `tenet`'s
**tensor** layer — `SymmetricTensor`, `tenet.einsum`, `tenet.linalg` — and imports
**nothing from `tenet.network`**, so you can read how the algorithm works instead of
watching it be called. That is the whole rule, and `tests/test_examples.py` asserts it
file by file. A toy code may use the symmetric-tensor layer because that is the library's
subject matter, not the algorithm being taught; the split follows `tenpy_toycodes`.

The library's own version of each algorithm lives in `tenet.network`, and the usage lane
(`examples/`) calls it: `examples/heisenberg_walkthrough.py` is `dmrg.py`'s chain through
`MPS`/`MPO`/`dmrg_`, and `examples/ising2d.py` is `ctmrg.py`'s Ising through `ctmrg()`.

**One recorded exemption:** `ctmrg.py` still imports `tenet.network` — #114 promoted its
CTMRG core into the library and the file kept calling it. It is exempted by name in
`tests/test_examples.py::_LANE_EXEMPT` and is rewritten in #187, not widened into
here. `vmc_mps.py` and `dmrg.py` obey the rule.

For the lane rule and the file table, see the root [README](../../README.md#examples).
