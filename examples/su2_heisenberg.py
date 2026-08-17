"""The same Heisenberg chain under SU(2): one invariant term, multiplet compression.

Run it standalone::

    uv run python examples/su2_heisenberg.py

Under SU(2), ``S^z`` alone is symmetry-forbidden -- only the invariant two-site
``S . S`` is a term, so :func:`tenet.network.local_op` takes the ``np.kron`` array with
``charge=None`` and ``MPO.from_terms`` splits it, deriving the graded MPO bond from the
operator's own blocks. The seed is :meth:`MPS.random`: ``MPS.product`` refuses
non-Abelian symmetries by construction (a single spin-up is not an SU(2) multiplet).
The point is the printed table -- same energy as the U(1) run this file computes by
importing ``examples/heisenberg.py``, from a mid-chain bond of far fewer multiplets
than the dense states ``chi`` counts.
"""

import heisenberg
import numpy as np

from tenet import GradedSpace
from tenet.network import MPO, MPS, dmrg_, local_op
from tenet.symmetry import SU2, SU2Sector

PHYS = GradedSpace.new(SU2, {SU2Sector(1): 1})  # one spin-1/2 multiplet, dense dim 2


def su2_mpo(n_sites: int) -> MPO:
    """``H = sum_i S_i . S_{i+1}`` as one invariant two-site term per bond."""
    ss = (
        np.kron(heisenberg.SZ, heisenberg.SZ)
        + (np.kron(heisenberg.SP, heisenberg.SP.T) + np.kron(heisenberg.SP.T, heisenberg.SP)) / 2
    )
    op = local_op(ss, phys=PHYS)  # charge=None: an invariant k-site term
    return MPO.from_terms(n_sites, [(1.0, [(op, (i, i + 1))]) for i in range(n_sites - 1)])


def bond_spaces(n_sites: int) -> list[GradedSpace]:
    """A small full-rank-enough seed: singlet boundaries, {j=0,1/2,1} in the middle."""
    tri = GradedSpace.new(SU2, {SU2Sector(0): 1})
    mid = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2, SU2Sector(2): 1})
    return [tri] + [mid] * (n_sites - 1) + [tri]


def main(n_sites: int = 20, chi: int = 64):
    """Run SU(2) and U(1) side by side; returns both DMRG_outs and the SU(2) mid bond."""
    psi = MPS.random(PHYS, bond_spaces(n_sites), seed=0)
    su2 = dmrg_(psi, su2_mpo(n_sites), chi=chi)
    u1, _ = heisenberg.main(n_sites, chi)

    mid = su2.psi[n_sites // 2].legs[0].space
    mid_u1 = u1.psi[n_sites // 2].legs[0].space
    print(f"U(1) : {u1.sweeps} sweeps  E = {u1.energy:.12f}  mid bond {mid_u1.dim} states")
    print(
        f"SU(2): {su2.sweeps} sweeps  E = {su2.energy:.12f}  "
        f"mid bond {mid.reduced_dim} multiplets, {mid.dim} dense"
    )
    print(f"|E_su2 - E_u1| = {abs(su2.energy - u1.energy):.1e}")
    return su2, u1, mid


if __name__ == "__main__":
    main()
