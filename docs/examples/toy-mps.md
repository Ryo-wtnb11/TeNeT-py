# Toy MPS

The state the other algorithms share: the MPS as a plain list of `SymmetricTensor`s, the
two ways to seed it, right-canonical form from `tenet.linalg.lq`, and the measurements
that read numbers back off it.

- Site `A_n` is `(left bond OUT, physical OUT, right bond IN)`; both end bonds are the
  unit sector, which makes $S^z_{\mathrm{tot}} = 0$ structural rather than imposed.
- `product_mps` is the Néel state with every bond `D=1` — each site has exactly one
  structurally allowed entry, so filling the blocks that exist *is* the basis state, with
  no dense basis written anywhere. It is [Toy TEBD](toy-tebd.md)'s starting point.
- `bond_spaces` is the only thing about this chain a generic MPS container cannot know,
  which is why the library takes bond *spaces* and not a `chi`.
- `expectation` measures a one- or two-site operator by one left-to-right pass of the
  transfer matrix, and `entropy` turns a bond's Schmidt values into the von Neumann
  entanglement entropy. Nothing here knows the Hamiltonian, which is why the same
  container serves both algorithms.

The physical space comes from [Toy model](toy-model.md). The same container through the
library is `tenet.network.MPS`.

## Source

[`examples/toy_codes/mps.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/mps.py)

```python linenums="1"
--8<-- "examples/toy_codes/mps.py"
```
