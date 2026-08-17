"""U(1) Heisenberg ground state through ``tenet.network``, end to end.

Run it standalone::

    uv run python examples/heisenberg.py

What ``examples/toy_codes/dmrg.py`` writes out by hand -- the 5x5 ``W`` matrix, its
channel table, the reachable bond spaces -- this file never mentions: the Hamiltonian is
a term list, ``MPO.from_terms`` derives the graded MPO bond, and the Neel product state
seeds the ``S^z_tot = 0`` sector by its own charges. The schedule below ramps ``chi``
with noise because writing one is what a user does; on this chain it buys nothing -- the
flat ``chi=64`` run reaches the same energy, since a degeneracy-1 U(1) seed already
ramps for free (see docs/tutorials/dmrg.md, "Schedules and noise", for when it pays).
"""

import numpy as np

from tenet import GradedSpace
from tenet.network import MPO, MPS, Sweep, dmrg_, expectation_1site, expectation_2site, local_op
from tenet.symmetry import U1, U1Sector

# Physical space: charge t = 2 S^z, so the spin doublet is {-1, +1}.
PHYS = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})

SZ = np.diag([-0.5, 0.5])
SP = np.array([[0.0, 0.0], [1.0, 0.0]])  # |down> -> |up>, raises 2 S^z by +2


def heisenberg_mpo(n_sites: int) -> MPO:
    """``H = sum_i S_i . S_{i+1}``, J = 1, open boundaries, as a term list."""
    op_sz = local_op(SZ, phys=PHYS, charge=U1Sector(0))
    op_sp = local_op(SP, phys=PHYS, charge=U1Sector(-2))
    op_sm = local_op(SP.T, phys=PHYS, charge=U1Sector(2))
    terms = []
    for i in range(n_sites - 1):
        terms.append((1.0, [(op_sz, i), (op_sz, i + 1)]))
        terms.append((0.5, [(op_sp, i), (op_sm, i + 1)]))
        terms.append((0.5, [(op_sm, i), (op_sp, i + 1)]))
    return MPO.from_terms(n_sites, terms)


def main(n_sites: int = 20, chi: int = 64):
    """Ground state at the defaults CI runs; returns the DMRG_out and the bond profile."""
    psi = MPS.product(PHYS, [U1Sector(1 if n % 2 else -1) for n in range(n_sites)])
    schedule = [Sweep(16, noise=1e-4)] * 3 + [Sweep(32, noise=1e-5)] * 3 + [Sweep(chi)]
    out = dmrg_(psi, heisenberg_mpo(n_sites), schedule=schedule)
    mid = out.psi[n_sites // 2].legs[0].space
    print(f"N={n_sites}  {out.sweeps} sweeps  E = {out.energy:.15f}  mid bond: {mid.dim} states")

    ss = local_op(np.kron(SZ, SZ) + (np.kron(SP, SP.T) + np.kron(SP.T, SP)) / 2, phys=PHYS)
    profile = [expectation_2site(out.psi, ss, n) for n in range(n_sites - 1)]
    print("bond energies:", " ".join(f"{e:+.4f}" for e in profile))
    print(f"sum of bond energies = {sum(profile):.15f}  vs  out.energy = {out.energy:.15f}")

    op_sz = local_op(SZ, phys=PHYS)
    max_sz = max(abs(expectation_1site(out.psi, op_sz, n)) for n in range(n_sites))
    print(f"max_n |<S^z_n>| = {max_sz:.1e}")
    return out, profile


if __name__ == "__main__":
    main()
