# Toy TEBD

Imaginary-time TEBD on the tensor layer: exponentiate the model's two-site term into a
gate with `tenet.linalg.expm`, apply it to a pair of neighbouring sites, split the pair
back with `svd_truncated`. The simplest of the three algorithms here, and the reason
[Toy model](toy-model.md) states the Hamiltonian twice — TEBD never sees an MPO.

- $e^{-\tau H}$ suppresses every excited state relative to the ground state, so the state
  is normalized after every split and the energy falls towards the exact one **from
  above**: imaginary time can only lower it and the truncation can only raise it.
- A step is one left-to-right pass and one right-to-left pass over the bonds — the same
  two-direction loop [Toy DMRG](toy-dmrg.md) runs — with the gate carrying `dt/2` each
  way. That keeps the chain canonical, which is what makes the truncated SVD the best
  rank-`chi` approximation of the *state*, and makes the Trotter error
  $O(\mathrm{d}t^3)$ per step without a second gate.
- At $N = 12$, `chi=32` the final energy sits `1.2e-11` above
  [Toy exact](toy-exact.md)'s `-5.142090632841`, and the largest discarded weight along
  the way is `6e-13`.

Needs SciPy, which is what `tenet.linalg.expm` calls on the NumPy backend.

## Source

[`examples/toy_codes/tebd.py`](https://github.com/Ryo-wtnb11/symtenet/blob/main/examples/toy_codes/tebd.py)

```python linenums="1"
--8<-- "examples/toy_codes/tebd.py"
```

## Output

Produced by `tebd.main()` at its defaults — exactly `python examples/toy_codes/tebd.py` —
as run by `tests/test_examples.py`. `S(N/2)` is the half-chain entanglement entropy, read
off the Schmidt values at the moment the middle bond was last cut.

```text
  dt=0.5     steps=100  E=-5.142075287339  dw=5.884e-13
  dt=0.05    steps=200  E=-5.142090631261  dw=5.773e-15
  dt=0.01    steps=300  E=-5.142090632829  dw=1.332e-15
N=12 chi=32  E=-5.142090632829  exact=-5.142090632841  S(N/2)=0.539180
```
