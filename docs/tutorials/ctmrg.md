# CTMRG — classical Ising against Onsager, then a U(1)/SU(2) iPEPS gradient

Source: [`examples/ctmrg.py`](https://github.com/Ryo-wtnb11/TeNeT-py/blob/main/examples/ctmrg.py).
**Oracle:** Onsager's closed-form free energy per site, and its `d(βf)/dβ` as the check on
`jax.grad` taken through the unrolled sweeps. The file is executed by
`tests/integration/test_ctmrg.py`; read the code there rather than a copy.

```sh
uv run --extra jax python examples/ctmrg.py
```

## Two physical problems, one CTMRG core

The core is the library's: `tenet.network.ctmrg` owns `Absorb`,
`single_layer`/`double_layer`, `move`, `converge` and `unrolled`, with the
`svd_truncated`-outside / `svd(bond=)`-inside pairing, the leg conventions and the four
environment ceilings (truncated backprop, no checkpointing, no pre-QR, `svd` rather than
`eigh`) documented on it.

What stays in the example is what the library must **not** decide: which bulk tensor
(`ising_bulk`), which ansatz (`c4v` — a library that symmetrized its input would be
silently editing the user's state), and what to measure (`_halves`/`energy` are a
C4v-and-1×1-and-2×1 reduced density matrix with one geometry and one caller, i.e. a
measurement API; `log_kappa`'s Baxter telescoping is classical-partition-function physics
with no meaning for an iPEPS).

The two halves:

- the **classical 2D Ising partition function**, whose free energy per site has a closed
  form (Onsager) and whose internal energy `d(βf)/dβ` is therefore an oracle for
  `jax.grad` through the unrolled sweeps;
- a **single-site U(1) (or SU(2)) iPEPS** with a random symmetric two-site `h`, which
  exercises graded truncation, `svd(bond=)` across sectors and multiplet degeneracies.

## The Ising half is Z2-graded

For the same reason YASTN's CTMRG Ising example passes `sym='Z2'`: it stops a finite-χ
environment from breaking the symmetry spuriously in the ordered phase, which is what lets
the example run at `β > β_c` against Onsager at all. Two further things the grading buys,
both asserted in the integration test:

- **Zero magnetization becomes structural.** A spin insertion is a Z2-odd tensor, which no
  invariant `SymmetricTensor` can hold, so `from_dense` refuses it — and the refusal is
  the statement.
- **The ordered-phase corner spectrum acquires exact two-fold degeneracy** across the
  parity sectors. Because that doubling is *cross*-sector and `tenet.ad` broadens *per
  coupled sector*, the graded run never hands one SVD a degenerate pair: grading removes
  the `NaN`, it does not create it.

It changed no arithmetic either.

## The iPEPS half is a plumbing result

It makes **no benchmark-energy claim**, and cannot with a one-site unit cell. Liao et al.
get a single-site AFM Heisenberg cell by rotating one sublattice by π about y, which turns
`SˣSˣ - SʸSʸ` into `(S⁺S⁺ + S⁻S⁻)/2` — an operator that changes `Sᶻ_tot` by ±2 and so
destroys the U(1) the ansatz is graded by. The alternatives are a two-site unit cell (out
of scope) or dropping the symmetry, which deletes the reason this half exists. So it
follows [`vmc_mps.py`](vmc.md): random symmetric `h`, no comparison against
`-0.669437(5)`, said out loud in the file itself.

`tenet.cast` is mentioned there and deliberately not used: building an SU(2) ansatz and
casting it to U(1) would be a third concept in a file that already has two models. The
SU(2) provider is instead run through the same iPEPS path via a `provider` parameter.

## Reference

- [`tenet.network`](../api/network.md) — `converge`, `unrolled`, `double_layer`, `move`
- [`tenet.ad`](../api/ad.md) — the broadened SVD rule the gradient goes through
