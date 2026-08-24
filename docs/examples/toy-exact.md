# Toy exact

Dense exact diagonalization of the same chain, in the $S^z_{\mathrm{tot}} = 0$ sector: the
reference [Toy TEBD](toy-tebd.md) and [Toy DMRG](toy-dmrg.md) both print against.

- Deliberately **not** written on the tensor layer. It takes `model.h_bonds()` — the same
  two-site operator TEBD exponentiates — reads it out dense, and builds the many-body
  matrix with numpy and nothing else. An oracle that shared the machinery it judges would
  not be one.
- The basis is the $\binom{N}{N/2}$ bitstrings with $N/2$ ones; the `(down, up)` order of the
  dense two-site block is `model.PHYS`'s own sector order, so a bit pair indexes the block
  directly.
- At $N = 12$ that is $924 \times 924$ and `eigvalsh` costs milliseconds. Above
  $N \approx 16$ the upgrade path named in the file is `scipy.sparse.linalg.eigsh` on the
  same matrix.

## Source

[`examples/toy_codes/exact.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/exact.py)

```python linenums="1"
--8<-- "examples/toy_codes/exact.py"
```

## Output

Produced by `exact.main()` at its defaults — exactly `python examples/toy_codes/exact.py` —
as run by `tests/test_examples.py`. `e_inf` is the Bethe-ansatz thermodynamic limit
$1/4 - \ln 2$, which a finite open chain sits above.

```text
N= 8  dim=   70  E=-3.374932598688  E/N=-0.421866574836  e_inf=-0.443147180560
N=10  dim=  252  E=-4.258035207283  E/N=-0.425803520728  e_inf=-0.443147180560
N=12  dim=  924  E=-5.142090632841  E/N=-0.428507552737  e_inf=-0.443147180560
```
