"""The same Heisenberg chain under SU(2): one invariant term, multiplet compression.

Run it standalone::

    uv run python examples/su2_heisenberg.py

Under SU(2), ``S^z`` alone is symmetry-forbidden -- only the invariant two-site
``S . S`` is a term, and it is the *entire* operator set
:func:`tenet.models.spin_half` returns for this grading -- which is what
:func:`tenet.models.heisenberg` hands ``MPO.from_terms`` under this symmetry. The
builder splits that term, deriving the graded MPO bond from the operator's own blocks.
The seed is :meth:`MPS.random`: ``MPS.product`` refuses non-Abelian symmetries by
construction (a single spin-up is not an SU(2) multiplet).
The point is the printed table -- same energy as the U(1) run this file computes by
importing ``examples/heisenberg.py``, from a mid-chain bond of far fewer multiplets
than the dense states ``chi`` counts.
"""

import heisenberg

from tenet import GradedSpace
from tenet.models import heisenberg as heisenberg_mpo
from tenet.models import spin_half
from tenet.network import MPS, dmrg_
from tenet.symmetry import SU2, SU2Sector

SITE = spin_half(SU2)  # one spin-1/2 multiplet, dense dim 2
PHYS = SITE.phys


def bond_spaces(n_sites: int) -> list[GradedSpace]:
    """A small full-rank-enough seed: singlet boundaries, {j=0,1/2,1} in the middle."""
    # SU2Sector holds 2j, so 0 is the singlet: a one-dimensional trivial space at each
    # end is the open boundary, and it pins the whole chain to a total spin zero state.
    tri = GradedSpace.new(SU2, {SU2Sector(0): 1})
    # Interior bonds offer j = 0, 1/2, 1 with a couple of copies each. Only the *set* of
    # irreps has to be right -- multiplicities the ground state needs grow under the
    # sweeps, and ones it does not are truncated away -- so this stays deliberately small.
    mid = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2, SU2Sector(2): 1})
    return [tri] + [mid] * (n_sites - 1) + [tri]


def main(n_sites: int = 20, chi: int = 64):
    """Run SU(2) and U(1) side by side; returns both DMRG_outs and the SU(2) mid bond."""
    # A random seed already fills every offered sector, so no noise is needed to reach
    # them; the fixed seed only makes the printed sweep count reproducible.
    psi = MPS.random(PHYS, bond_spaces(n_sites), seed=0)
    # Flat chi, no schedule: chi counts dense states here too, so an SU(2) bond keeps
    # (2j+1)-fold more physical states per unit of cost than the U(1) run below.
    # ``models.heisenberg(n, SU2)`` is one invariant two-site term per bond: the site pair
    # (i, i+1) enters as a tuple, not two one-site factors, because S.S is irreducible
    # under SU(2) and from_terms splits it across the bond itself by fusing the two
    # sites and cutting the result -- the j=0 and j=1 channels of that split are the
    # MPO bond, and they are what the SU(2) grading is made of.
    su2 = dmrg_(psi, heisenberg_mpo(n_sites, SU2), chi=chi)
    u1, _ = heisenberg.main(n_sites, chi)

    # Same cut, both runs. dim counts dense states, reduced_dim counts multiplets: the
    # gap between them is the (2j+1) degeneracy SU(2) never has to store.
    mid = su2.psi[n_sites // 2].legs[0].space
    mid_u1 = u1.psi[n_sites // 2].legs[0].space
    print(f"U(1) : ~{u1.sweeps} sweeps  E = {u1.energy:.12f}  mid bond {mid_u1.dim} states")
    print(
        f"SU(2): ~{su2.sweeps} sweeps  E = {su2.energy:.12f}  "
        f"mid bond {mid.reduced_dim} multiplets, {mid.dim} dense"
    )
    print(f"|E_su2 - E_u1| = {abs(su2.energy - u1.energy):.1e}")
    return su2, u1, mid


if __name__ == "__main__":
    main()
