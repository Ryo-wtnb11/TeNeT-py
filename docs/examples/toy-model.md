# Toy model

The U(1) Heisenberg chain, stated once in both of the forms an algorithm can consume:
`h_bonds()` hands it over as two-site gates, `mpo()` as a matrix product operator. Two
algorithms, one model — which is the reason the model is a file of its own rather than a
section of either.

- `h_bond()` is six blocks, and they are the six numbers a textbook writes in the
  `{uu, ud, du, dd}` basis. The other ten entries of the $4 \times 4$ matrix change
  $S^z_{\mathrm{tot}}$ and so have no block to live in.
- `mpo()` is the same operator on a graded bond: $S^{\pm}$ moves the bond charge by $\mp 2$, so
  each is a block of its own, and a wrong bond grading is a refusal from `from_blocks`,
  not a silent projection onto another operator.
- Both halves carry the same six numbers, which is the thing a reader is meant to check
  by eye.

Consumed by [Toy TEBD](toy-tebd.md) (gates), [Toy DMRG](toy-dmrg.md) (MPO) and
[Toy exact](toy-exact.md) (gates again, read out dense). Deriving the MPO from a term list
instead is [Heisenberg, U(1) walkthrough](heisenberg-walkthrough.md).

## Source

[`examples/toy_codes/model.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/model.py)

```python linenums="1"
--8<-- "examples/toy_codes/model.py"
```
