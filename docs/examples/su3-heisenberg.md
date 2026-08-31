# Heisenberg, SU(3)

$H = \sum_i P_{i,i+1}$ on a chain of fundamental $\mathbf{3}$s. The two-site exchange is
$+1$ on the symmetric $\mathbf{6}$ of $\mathbf{3} \otimes \mathbf{3}$ and $-1$ on the
antisymmetric $\bar{\mathbf{3}}$, so the rank-4 term tensor is two blocks named with
`SymmetricTensor.from_blocks` — no Clebsch-Gordan array is written out — and
`MPO.from_terms` takes it whole.

- The dense form of the built term is the permutation matrix, to machine precision.
- DMRG at $N = 6$ reproduces the script's own numpy-only dense ED.
- The $N = 24$ bond carries ten multiplets over 89 dense states, all of zero triality.
- Sutherland's Bethe-ansatz energy per site for the infinite SU($N$) fundamental chain is
  $1 + \frac{2}{N}\left(\gamma + \psi(1/N)\right)$, which at $N = 3$ is
  $1 - \ln 3 - \pi/(3\sqrt{3}) = -0.703212\ldots$

## Source

```python
--8<-- "examples/su3_heisenberg.py"
```

## Output

Produced by `su3_heisenberg.main()` at its defaults — exactly
`python examples/su3_heisenberg.py` — as run by `tests/test_examples.py`.

```text
P: one block per coupled sector of 3 x 3, [(0, 1), (2, 0)]
      max |P_dense - permutation matrix| = 4.4e-16
N= 6  ~3 sweeps  E = -4.069784138805  ED = -4.069784138805
      |E_dmrg - E_ed| = 1.5e-13
N=24  ~8 sweeps  E = -16.686244544237  mid bond 10 multiplets, 89 dense
      mid bond: (0, 0)x2 (0, 3)x1 (1, 1)x5 (2, 2)x1 (3, 0)x1
      E/N = -0.695260  bulk bond <P> = -0.747250
      Sutherland (infinite chain) = -0.703212
```
