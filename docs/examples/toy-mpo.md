# Toy MPO

The Hamiltonian half of the toy DMRG: the open-boundary Heisenberg MPO written out one
block per allowed sector tuple, on a bond graded by the charge each channel carries.

- Site `W_n` is `(wl IN, p OUT, p IN, wr OUT)`, rank 4 at *every* site — the first and
  last carry a `D=1` boundary bond, so there is no boundary-vector special case.
- `S^±` moves the MPO bond charge by `∓2`, so each is a block of its own; a wrong bond
  grading is a refusal from `from_blocks`, not a silent projection onto another operator.

Takes its physical space from [Toy MPS](toy-mps.md); consumed by [Toy DMRG](toy-dmrg.md).
Deriving the same operator from a term list instead is
[Heisenberg, U(1) walkthrough](heisenberg-walkthrough.md).

## Source

[`examples/toy_codes/mpo.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/mpo.py)

```python linenums="1"
--8<-- "examples/toy_codes/mpo.py"
```
