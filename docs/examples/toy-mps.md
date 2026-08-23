# Toy MPS

The state half of the toy DMRG: the MPS as a plain list of `SymmetricTensor`s, the U(1)
bond spaces the chain can reach, right-canonical form from `tenet.linalg.lq`, and the
Schmidt values on a bond read back through `tenet.to_matrices`.

- Site `A_n` is `(left bond OUT, physical OUT, right bond IN)`; both end bonds are the
  unit sector, which makes `S^z_tot = 0` structural rather than imposed.
- `bond_spaces` is the only thing about this chain a generic MPS container cannot know,
  which is why the library takes bond *spaces* and not a `chi`.

Imported by [Toy MPO](toy-mpo.md) and [Toy DMRG](toy-dmrg.md). The same container through
the library is `tenet.network.MPS`.

## Source

[`examples/toy_codes/mps.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/mps.py)

```python linenums="1"
--8<-- "examples/toy_codes/mps.py"
```
