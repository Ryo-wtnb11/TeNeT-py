# Toy VMC on an MPS

Gradient-based variational optimization of a symmetric MPS, written on the tensor layer.
Needs the `jax` extra.

**Objective.** The Rayleigh quotient of a two-site operator $h$ on the MPS
$\Psi_{s_1\cdots s_N} = A^{s_1}\cdots A^{s_N}$,

$$
E(\{A\}) = \frac{\langle\psi\vert h\vert\psi\rangle}{\langle\psi\vert\psi\rangle},
$$

both numerator and denominator built as a left-to-right chain of *pairwise*
`tenet.einsum` calls — three or more operands would bring in a contraction path, which is
a separate concern from the gradient.

**Parameters.** Once `tenet.enable_jax()` has run, a `SymmetricTensor` is a JAX pytree
whose leaves are its reduced blocks and whose treedef is its `TensorStructure`. So the
variational parameters are exactly the *reduced* coefficients — the independent numbers
the symmetry leaves free — and the update

$$
A \;\leftarrow\; A - \eta\,\nabla_{A} E
$$

is one `jax.tree.map`. Nothing projects the state back onto the symmetric manifold,
because a gradient with respect to block values never leaves it: equivariance is a
property of the structure, and the structure is the treedef, which the optimizer does not
touch.

**Checks.** Both traces (U(1) and SU(2)) decrease on every one of the 20 steps, through
identical code. `h` is a random symmetric operator, so this is a plumbing result, not a
physics one — build a physical operator when you want a physical number.

Explained in the [VMC tutorial](../tutorials/vmc.md).

## Source

[`examples/toy_codes/vmc_mps.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/toy_codes/vmc_mps.py)

```python linenums="1"
--8<-- "examples/toy_codes/vmc_mps.py"
```

## Output

Produced by `vmc_mps.main(provider=...)` at its defaults for both providers — exactly
`python examples/toy_codes/vmc_mps.py` — as run by `tests/backends/test_ad.py`.

```text
u1: -0.682692 -0.725133 -0.763486 -0.798024 -0.829057 -0.856907 -0.881888 -0.904301 -0.924424 -0.942510 -0.958786 -0.973455 -0.986697 -0.998672 -1.009520 -1.019363 -1.028312 -1.036461 -1.043895 -1.050687
su2: -1.033559 -1.038339 -1.042817 -1.047020 -1.050971 -1.054691 -1.058199 -1.061512 -1.064645 -1.067612 -1.070424 -1.073094 -1.075631 -1.078045 -1.080343 -1.082534 -1.084624 -1.086621 -1.088529 -1.090355
```
