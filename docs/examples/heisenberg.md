# Heisenberg, U(1)

The ground state of the spin-1/2 Heisenberg chain on $N = 20$ sites, open boundaries:

$$
H = \sum_{i=0}^{N-2}\mathbf{S}_i\cdot\mathbf{S}_{i+1}
  = \sum_{i=0}^{N-2}\left[S^z_iS^z_{i+1}
    + \tfrac12\left(S^+_iS^-_{i+1} + S^-_iS^+_{i+1}\right)\right],
\qquad
E_0 = \min_{\psi}\frac{\langle\psi\vert H\vert\psi\rangle}{\langle\psi\vert\psi\rangle}.
$$

**Representation.** `spin_half()` grades the physical doublet by U(1) with charge $2S^z$,
so $S^\pm$ are rank-3 tensors carrying $\mp 2$ on a `D=1` charge leg. The three summands
above are the three `MPO.from_terms` entries, coefficient for coefficient. The state is an
MPS, $\Psi_{s_1\cdots s_N} = A^{s_1}\cdots A^{s_N}$, seeded as the Néel product state,
whose `D=1` bond-0 leg carries charge $0$ — that is $S^z_{\mathrm{tot}} = 0$, structurally.

**Approximation.** `dmrg_` sweeps two-site updates, each a Lanczos ground eigenpair of the
two-site effective Hamiltonian followed by a truncated SVD back onto the chosen bond. The
schedule ramps $\chi$ — three sweeps at 16 with noise $10^{-4}$, three at 32 with
$10^{-5}$, then 64 noiseless — because noise repopulates bond sectors a `D=1` product seed
left empty, which a rescaling update can never reach on its own. The energy is
variational, so it approaches $E_0$ from above.

**Checks.**

- $E$ matches the recorded $N = 20$ exact-diagonalization energy to $10^{-10}$.
- The bond energies $\langle\mathbf{S}_i\cdot\mathbf{S}_{i+1}\rangle$, measured with a single
  invariant $\mathbf{S}\cdot\mathbf{S}$ operator rather than the three-term sum the MPO was
  built from, sum to `out.energy` — the measurement and the sweep agree.
- $\langle S^z_n\rangle$ is float noise on every site: the sector is fixed by the boundary
  leg and the tensors' invariance, not by a penalty term.

Step by step in the [DMRG tutorial](../tutorials/dmrg.md); the semantics of every call in
[DMRG](../guide/dmrg.md).

## Source

```python
--8<-- "examples/heisenberg.py"
```

## Output

Produced by `heisenberg.main()` at its defaults — exactly `python examples/heisenberg.py`
— as run by `tests/test_examples.py`.

```text
N=20  9 sweeps  E = -8.682473334397713  mid bond: 64 states
bond energies: -0.6534 -0.2943 -0.5664 -0.3370 -0.5401 -0.3540 -0.5286 -0.3616 -0.5239 -0.3638 -0.5239 -0.3616 -0.5286 -0.3540 -0.5401 -0.3370 -0.5664 -0.2943 -0.6534
sum of bond energies = -8.682473334397695  vs  out.energy = -8.682473334397713
max_n |<S^z_n>| = 4.7e-13
```
