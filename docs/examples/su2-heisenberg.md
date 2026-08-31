# Heisenberg, SU(2)

The same Hamiltonian as [Heisenberg, U(1)](heisenberg.md),

$$
H = \sum_i \mathbf{S}_i\cdot\mathbf{S}_{i+1},
$$

declared under its full SU(2) symmetry instead of the U(1) subgroup it contains.

**What changes.** $S^+$, $S^-$ and $S^z$ do not appear: none of them is an SU(2)-invariant
tensor, and the rank-3 charge-leg form has no leg to emit a spin-1 sector onto. What
exists is the invariant two-site operator $\mathbf{S}\cdot\mathbf{S}$, one whole bond term
— which is exactly what `tenet.models.heisenberg(N, SU2)` hands `MPO.from_terms`, and the
builder splits it with an SVD — so the MPO bond grading comes out of the
operator's own blocks and no coupling tree is named anywhere.

**What it buys.** A U(1) bond stores $\chi$ states; an SU(2) bond stores *multiplets*, and
a $j$ multiplet is worth $2j+1$ dense states at the cost of one degeneracy index. The
Wigner-Eckart theorem is what makes this exact rather than an approximation: within a
multiplet the coefficients are fixed by the symmetry and only the reduced matrix element is
data.

**Checks.**

- Both runs reach the same energy to $10^{-10}$ — the same physical state, two
  declarations.
- The SU(2) mid-bond holds 22 multiplets spanning 62 dense states where the U(1) mid-bond
  holds 64 states, and `reduced_dim < dim` is asserted rather than printed.

Explained in the [SU(2) tutorial](../tutorials/su2.md); what `max_bond` bounds on a
multiplet bond is [Truncation](../guide/truncation.md).

## Source

```python
--8<-- "examples/su2_heisenberg.py"
```

## Output

Produced by `su2_heisenberg.main()` at its defaults — exactly
`python examples/su2_heisenberg.py` — as run by `tests/test_examples.py`.

```text
N=20  ~9 sweeps  E = -8.682473334397722  mid bond: 64 states
bond energies: -0.6534 -0.2943 -0.5664 -0.3370 -0.5401 -0.3540 -0.5286 -0.3616 -0.5239 -0.3638 -0.5239 -0.3616 -0.5286 -0.3540 -0.5401 -0.3370 -0.5664 -0.2943 -0.6534
sum of bond energies = -8.682473334397697  vs  out.energy = -8.682473334397722
max_n |<S^z_n>| = 3.9e-13
U(1) : ~9 sweeps  E = -8.682473334398  mid bond 64 states
SU(2): ~5 sweeps  E = -8.682473334397  mid bond 22 multiplets, 62 dense
|E_su2 - E_u1| = 4.5e-13
```
