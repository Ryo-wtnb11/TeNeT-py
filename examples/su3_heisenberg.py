"""The SU(3) Heisenberg chain: the exchange term written as its own two blocks.

Run it standalone::

    uv run python examples/su3_heisenberg.py

The site is one fundamental ``3`` of SU(3) and the Hamiltonian is
``H = sum_i P_{i,i+1}``, the exchange (permutation) operator. On ``3 x 3 = 6 + 3bar``
the exchange is ``+1`` on the symmetric ``6`` and ``-1`` on the antisymmetric ``3bar``,
and that sentence *is* the construction: the rank-4 term tensor has exactly one block
per coupled sector, so ``SymmetricTensor.from_blocks`` writes ``+1`` and ``-1`` into
them and no Clebsch-Gordan array is spelled out. That is
:func:`tenet.models.sun_exchange`, and :func:`tenet.models.sun_heisenberg` hands it to
``MPO.from_terms`` whole, exactly as ``examples/su2_heisenberg.py`` passes ``S.S``.

Two oracles are printed beside the run: a numpy-only dense ED of the same permutation
chain, and Sutherland's Bethe-ansatz energy per site for the infinite chain.
"""

import numpy as np

from tenet import GradedSpace, TensorStructure
from tenet.models import sun_exchange, sun_heisenberg
from tenet.network import MPS, dmrg_, expectation_2site
from tenet.symmetry import SUNProvider, SUNSector

SU3 = SUNProvider(3)
ONE, THREE, THREEBAR, SIX = (SUNSector(d) for d in ((0, 0), (1, 0), (0, 1), (2, 0)))
PHYS = GradedSpace.new(SU3, {THREE: 1})  # one fundamental multiplet, dense dim 3

# Sutherland, Phys. Rev. B 12, 3795 (1975), the nested Bethe ansatz for the SU(N)
# fundamental chain: with H = sum_i P_{i,i+1} the infinite chain's energy per site is
# 1 + (2/N)(gamma + psi(1/N)), which for N = 3 is 1 - ln 3 - pi/(3 sqrt 3) = -0.703212...
SUTHERLAND = 1.0 - np.log(3.0) - np.pi / (3.0 * np.sqrt(3.0))


# The dense exchange, as the oracle needs it: P[a, b, c, d] = delta_ad delta_bc.
SWAP = np.eye(9).reshape(3, 3, 3, 3).transpose(0, 1, 3, 2)


# ``sun_exchange(3)`` is P on two fundamental sites: one block per coupled sector of
# 3 x 3, +1 on the symmetric 6 and -1 on the antisymmetric 3bar, written with
# SymmetricTensor.from_blocks -- degeneracy 1 on both sides, so the Clebsch-Gordan
# coefficients stay implicit. Its legs are two OUT (ket) then two IN (bra), the ordering
# expectation_2site and MPO.from_terms both read a term through.
P = sun_exchange(3)


def bond_spaces(n_sites: int) -> list[GradedSpace]:
    """Singlet boundaries -- the target sector -- and the low irreps in between."""
    # The trivial irrep at both ends: an open chain's outermost bonds are one-dimensional,
    # and choosing the singlet there is what selects an SU(3)-invariant ground state.
    end = GradedSpace.new(SU3, {ONE: 1})
    # Every irrep reachable from a few fundamentals -- (1,1) is the adjoint 8, (0,2) the
    # 6bar. The seed offers all of them, including the triality-nonzero 3 and 3bar an even
    # cut cannot use; the sweeps drop those, which is the symmetry doing the bookkeeping.
    mid = GradedSpace.new(
        SU3, {ONE: 2, THREE: 2, THREEBAR: 2, SUNSector((1, 1)): 2, SIX: 1, SUNSector((0, 2)): 1}
    )
    return [end] + [mid] * (n_sites - 1) + [end]


def ed_energy(n_sites: int) -> float:
    """Dense ED of the same permutation chain on ``(C^3)^n``, numpy only, no tenet."""
    # The rank-4 exchange flattened to the 9x9 matrix acting on one neighbouring pair.
    swap = SWAP.reshape(9, 9)
    # Identity on everything left of the bond, the swap on the pair, identity on the
    # rest: the Kronecker embedding of a two-site term into the full 3^n Hilbert space.
    h = sum(
        np.kron(np.kron(np.eye(3**i), swap), np.eye(3 ** (n_sites - i - 2)))
        for i in range(n_sites - 1)
    )
    # H is real symmetric, so eigvalsh returns the spectrum sorted -- [0] is the ground
    # state of the full space, with no symmetry sector assumed anywhere in this route.
    return float(np.linalg.eigvalsh(h)[0])


def run(n_sites: int, chi: int):
    # Random seed rather than a product state: a single fundamental is not an SU(3)
    # singlet, so there is no product state in the target sector to start from.
    psi = MPS.random(PHYS, bond_spaces(n_sites), seed=0)
    # H = sum_i P_{i,i+1}: one invariant two-site term per bond, the site pair passed
    # as a tuple, and from_terms splits it across the bond itself.
    return dmrg_(psi, sun_heisenberg(n_sites, 3), chi=chi)


def main(n_sites: int = 24, chi: int = 96, n_ed: int = 6):
    """DMRG at the defaults CI runs, against ED at ``n_ed`` and against Sutherland."""
    # The Dynkin labels of P's two blocks, read straight off the fusion structure:
    # (2,0) is the 6 and (0,1) the 3bar, which is 3 x 3 decomposed.
    coupled = [key.output_tree.coupled for key in TensorStructure(P.legs).block_order]
    print(f"P: one block per coupled sector of 3 x 3, {[c.dynkin for c in coupled]}")
    # Two numbers written into two blocks reproduce the dense permutation matrix to
    # round-off: the Clebsch-Gordan coefficients the library supplies are the rest of it.
    print(f"      max |P_dense - permutation matrix| = {abs(P.to_dense() - SWAP).max():.1e}")
    # Small chain first, where dense ED is still affordable and can pin DMRG exactly.
    short = run(n_ed, chi)
    exact = ed_energy(n_ed)
    print(f"N={n_ed:2d}  ~{short.sweeps} sweeps  E = {short.energy:.12f}  ED = {exact:.12f}")
    print(f"      |E_dmrg - E_ed| = {abs(short.energy - exact):.1e}")

    # Long chain: too large for ED, so it is judged against Sutherland's infinite-chain
    # value instead, which the finite open chain approaches from a bracketed pair below.
    long = run(n_sites, chi)
    mid = long.psi[n_sites // 2].legs[0].space
    # <P> on a middle bond: away from the ends, this is the bulk energy density, and it
    # sits between E/N and E/(N-1) for exactly the reason those two bracket Sutherland.
    bulk = expectation_2site(long.psi, P, n_sites // 2)
    print(
        f"N={n_sites:2d}  ~{long.sweeps} sweeps  E = {long.energy:.12f}  "
        f"mid bond {mid.reduced_dim} multiplets, {mid.dim} dense"
    )
    print("      mid bond:", " ".join(f"{a.dynkin}x{m}" for a, m in mid.sectors))
    # E/N counts N sites but the open chain has only N-1 bonds, so E/N overshoots the
    # infinite-chain value and E/(N-1) undershoots it: the two bracket Sutherland.
    print(f"      E/N = {long.energy / n_sites:.6f}  bulk bond <P> = {bulk:.6f}")
    print(f"      Sutherland (infinite chain) = {SUTHERLAND:.6f}")
    return short, long, exact


if __name__ == "__main__":
    main()
