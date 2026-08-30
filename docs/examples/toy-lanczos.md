# Toy Lanczos

The ground eigenpair of a Hermitian operator by a three-term Krylov recurrence, written
against nothing but `tenet.add`, `tenet.subtract`, scalar multiply and divide,
`tenet.norm` and `tenet.inner`.

- A Krylov method needs a vector space and nothing else, and a `SymmetricTensor` is one —
  which is why this is a file of its own and not a private detail of
  [Toy DMRG](toy-dmrg.md).
- `matvec` is a callable, so the same function serves the two-site effective Hamiltonian,
  a plain matrix on a rank-2 tensor, or anything else carrying those five operations.
- The happy breakdown ($\beta < \texttt{tol}$) drops the row and keeps the space rather than
  dividing by it.

## Source

[`examples/toy_codes/lanczos.py`](https://github.com/Ryo-wtnb11/symtenet/blob/main/examples/toy_codes/lanczos.py)

```python linenums="1"
--8<-- "examples/toy_codes/lanczos.py"
```
