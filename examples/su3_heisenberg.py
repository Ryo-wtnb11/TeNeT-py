"""The SU(3) Heisenberg chain: the exchange term written as its own two blocks.

Run it standalone::

    uv run python examples/su3_heisenberg.py

The site is one fundamental ``3`` of SU(3) and the Hamiltonian is
``H = sum_i P_{i,i+1}``, the exchange (permutation) operator. On ``3 x 3 = 6 + 3bar``
the exchange is ``+1`` on the symmetric ``6`` and ``-1`` on the antisymmetric ``3bar``,
and that sentence *is* the construction: the rank-4 term tensor has exactly one block
per coupled sector, so ``SymmetricTensor.from_blocks`` writes ``+1`` and ``-1`` into
them and no Clebsch-Gordan array is spelled out. ``MPO.from_terms`` takes the whole
two-site term, exactly as ``examples/su2_heisenberg.py`` passes ``S.S``.

Two oracles are printed beside the run: a numpy-only dense ED of the same permutation
chain, and Sutherland's Bethe-ansatz energy per site for the infinite chain.
"""

import numpy as np

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.network import MPO, MPS, dmrg_, expectation_2site
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


def exchange() -> SymmetricTensor:
    """``P`` on two fundamental sites: ``+1`` on the ``6``, ``-1`` on the ``3bar``."""
    legs = (Leg(PHYS, OUT), Leg(PHYS, OUT), Leg(PHYS, IN), Leg(PHYS, IN))
    eigenvalue = {SIX: 1.0, THREEBAR: -1.0}  # symmetric, antisymmetric
    structure = TensorStructure(legs)
    return SymmetricTensor.from_blocks(
        legs,
        {
            key: np.full(structure.block_shape(key), eigenvalue[key.output_tree.coupled])
            for key in structure.block_order
        },
    )


P = exchange()


def su3_mpo(n_sites: int) -> MPO:
    """``H = sum_i P_{i,i+1}`` as one invariant two-site term per bond."""
    return MPO.from_terms(n_sites, [(1.0, [(P, (i, i + 1))]) for i in range(n_sites - 1)])


def bond_spaces(n_sites: int) -> list[GradedSpace]:
    """Singlet boundaries -- the target sector -- and the low irreps in between."""
    end = GradedSpace.new(SU3, {ONE: 1})
    mid = GradedSpace.new(
        SU3, {ONE: 2, THREE: 2, THREEBAR: 2, SUNSector((1, 1)): 2, SIX: 1, SUNSector((0, 2)): 1}
    )
    return [end] + [mid] * (n_sites - 1) + [end]


def ed_energy(n_sites: int) -> float:
    """Dense ED of the same permutation chain on ``(C^3)^n``, numpy only, no tenet."""
    swap = SWAP.reshape(9, 9)
    h = sum(
        np.kron(np.kron(np.eye(3**i), swap), np.eye(3 ** (n_sites - i - 2)))
        for i in range(n_sites - 1)
    )
    return float(np.linalg.eigvalsh(h)[0])


def run(n_sites: int, chi: int):
    psi = MPS.random(PHYS, bond_spaces(n_sites), seed=0)
    return dmrg_(psi, su3_mpo(n_sites), chi=chi)


def main(n_sites: int = 24, chi: int = 96, n_ed: int = 6):
    """DMRG at the defaults CI runs, against ED at ``n_ed`` and against Sutherland."""
    coupled = [key.output_tree.coupled for key in TensorStructure(P.legs).block_order]
    print(f"P: one block per coupled sector of 3 x 3, {[c.dynkin for c in coupled]}")
    print(f"      max |P_dense - permutation matrix| = {abs(P.to_dense() - SWAP).max():.1e}")
    short = run(n_ed, chi)
    exact = ed_energy(n_ed)
    print(f"N={n_ed:2d}  {short.sweeps} sweeps  E = {short.energy:.12f}  ED = {exact:.12f}")
    print(f"      |E_dmrg - E_ed| = {abs(short.energy - exact):.1e}")

    long = run(n_sites, chi)
    mid = long.psi[n_sites // 2].legs[0].space
    bulk = expectation_2site(long.psi, P, n_sites // 2)
    print(
        f"N={n_sites:2d}  {long.sweeps} sweeps  E = {long.energy:.12f}  "
        f"mid bond {mid.reduced_dim} multiplets, {mid.dim} dense"
    )
    print("      mid bond:", " ".join(f"{a.dynkin}x{m}" for a, m in mid.sectors))
    print(f"      E/N = {long.energy / n_sites:.6f}  bulk bond <P> = {bulk:.6f}")
    print(f"      Sutherland (infinite chain) = {SUTHERLAND:.6f}")
    return short, long, exact


if __name__ == "__main__":
    main()
