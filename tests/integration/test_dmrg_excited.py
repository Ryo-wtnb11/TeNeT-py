"""M61 Stage D (#232, absorbing #216): excited states, against exact diagonalization.

``dmrg_(psi2, h, orthogonal_to=[psi1])`` targets the state above ``psi1``, and every
number below is compared to an ED value computed in this file rather than to a recorded
literal. Three symmetry regimes, because the projection vector rides the same mixed
transfer the whole stage is about: U(1) (the spin chain, where the ED block is small
enough to diagonalize whole), fZ2 (free fermions at N=12, so the Koszul string is live on
every bond) and SU(2) (one invariant ``S.S``, so recoupling is exercised).

Runtime is a few seconds; the fZ2 chain at N=12 is the slow one and is still under ten.
"""

import itertools

import numpy as np
import pytest

from tenet import GradedSpace
from tenet.network import MPO, MPS, Env, dmrg_, local_op
from tenet.symmetry import SU2, U1, FZ2Sector, SU2Sector, U1Sector, fZ2

# --- the oracles ---------------------------------------------------------------------


def heisenberg_block(n_sites: int, n_up: int) -> np.ndarray:
    """The open Heisenberg chain restricted to a fixed number of up spins, dense.

    ``test_dmrg.py::heisenberg_sz0``'s body with the sector left free, which is what the
    degeneracy check below needs: a triplet shows the same energy in ``S^z_tot = 0`` and
    ``S^z_tot = 1``, and that pair is the cross-sector statement.
    """
    states = [s for s in range(1 << n_sites) if bin(s).count("1") == n_up]
    index = {s: i for i, s in enumerate(states)}
    h = np.zeros((len(states), len(states)))
    for i, s in enumerate(states):
        for b in range(n_sites - 1):
            up, right = (s >> b) & 1, (s >> (b + 1)) & 1
            if up == right:
                h[i, i] += 0.25
            else:
                h[i, i] -= 0.25
                h[index[s ^ (1 << b) ^ (1 << (b + 1))], i] += 0.5
    return h


def singlet_spectrum(n_sites: int) -> list[float]:
    """The total-spin-0 energies: ``spec(S^z = 0)`` with ``spec(S^z = 1)`` taken out.

    Every multiplet of spin ``S >= 1`` puts one copy of its energy in both sectors, so
    what survives the multiset difference is exactly the singlets -- which is the sector
    an SU(2) MPS with a trivial boundary leg lives in. No ``S^2`` matrix needed.
    """
    rest = list(np.linalg.eigvalsh(heisenberg_block(n_sites, n_sites // 2 + 1)))
    out = []
    for e in np.linalg.eigvalsh(heisenberg_block(n_sites, n_sites // 2)):
        near = min(range(len(rest)), key=lambda k: abs(rest[k] - e)) if rest else None
        if near is not None and abs(rest[near] - e) < 1e-9:
            rest.pop(near)
        else:
            out.append(float(e))
    return out


def free_fermion_parity_spectrum(n_sites: int, parity: int) -> list[float]:
    """Many-body energies of the open hopping chain at fixed particle-number parity.

    ``H = -sum (c^dag_i c_{i+1} + h.c.)`` is quadratic, so a many-body energy is a subset
    sum of the single-particle spectrum and the enumeration is exact. An MPS with unit
    boundary legs on both ends lives in the even-parity sector.
    """
    single = np.zeros((n_sites, n_sites))
    for i in range(n_sites - 1):
        single[i, i + 1] = single[i + 1, i] = -1.0
    eps = np.linalg.eigvalsh(single)
    return sorted(
        float(sum(eps[list(occ)]))
        for k in range(n_sites + 1)
        if k % 2 == parity
        for occ in itertools.combinations(range(n_sites), k)
    )


# --- the models ----------------------------------------------------------------------


def _u1_heisenberg(n_sites):
    phys = GradedSpace.new(U1, {U1Sector(1): 1, U1Sector(-1): 1})
    sz, sp = np.diag([0.5, -0.5]), np.array([[0.0, 1.0], [0.0, 0.0]])
    op = {
        0: local_op(sz, phys=phys, charge=U1Sector(0)),
        2: local_op(sp, phys=phys, charge=U1Sector(2)),
        -2: local_op(sp.T, phys=phys, charge=U1Sector(-2)),
    }
    terms = []
    for i in range(n_sites - 1):
        terms += [
            (1.0, [(op[0], i), (op[0], i + 1)]),
            (0.5, [(op[2], i), (op[-2], i + 1)]),
            (0.5, [(op[-2], i), (op[2], i + 1)]),
        ]
    bonds = [
        GradedSpace.new(U1, {U1Sector(q): 1 for q in range(-w, w + 1, 2)})
        for w in (min(i, n_sites - i) for i in range(n_sites + 1))
    ]
    return phys, MPO.from_terms(n_sites, terms), bonds


def _fz2_chain(n_sites):
    phys = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = local_op(a.T, phys=phys, charge=FZ2Sector(1))
    c = local_op(a, phys=phys, charge=FZ2Sector(1))
    terms = []
    for i in range(n_sites - 1):
        terms += [(-1.0, [(cd, i), (c, i + 1)]), (-1.0, [(cd, i + 1), (c, i)])]
    unit = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
    both = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})
    return phys, MPO.from_terms(n_sites, terms), [unit] + [both] * (n_sites - 1) + [unit]


def _su2_heisenberg(n_sites):
    phys = GradedSpace.new(SU2, {SU2Sector(1): 1})
    sz, sp = np.diag([0.5, -0.5]), np.array([[0.0, 1.0], [0.0, 0.0]])
    ss = local_op(np.kron(sz, sz) + (np.kron(sp, sp.T) + np.kron(sp.T, sp)) / 2, phys=phys)
    h = MPO.from_terms(n_sites, [(1.0, [(ss, (i, i + 1))]) for i in range(n_sites - 1)])
    bonds = [
        GradedSpace.new(SU2, {SU2Sector(j): 1 for j in range(i % 2, min(i, n_sites - i) + 1, 2)})
        for i in range(n_sites + 1)
    ]
    return phys, h, bonds


def _two_lowest(phys, h, bonds, chi, seeds=(1, 2), **kwargs):
    """Ground state, then the state above it, and the overlap between the two."""
    psi1 = MPS.random(phys, bonds, seed=seeds[0])
    out1 = dmrg_(psi1, h, chi=chi, **kwargs)
    psi2 = MPS.random(phys, bonds, seed=seeds[1])
    out2 = dmrg_(psi2, h, chi=chi, orthogonal_to=[psi1], **kwargs)
    overlap = Env(psi2, MPO.identity(len(psi1), phys), bra=psi1).measure()
    return out1.energy, out2.energy, overlap


# --- the tests -----------------------------------------------------------------------


def test_u1_first_excited_matches_ed_in_the_target_sector():
    """N=8 Heisenberg: the two lowest ``S^z_tot = 0`` eigenvalues, and the residual overlap."""
    exact = np.linalg.eigvalsh(heisenberg_block(8, 4))
    e0, e1, overlap = _two_lowest(*_u1_heisenberg(8), chi=32)
    assert e0 == pytest.approx(exact[0], abs=1e-10)
    assert e1 == pytest.approx(exact[1], abs=1e-10)
    assert abs(overlap) < 1e-10


def test_a_degenerate_triplet_shows_the_same_energy_in_two_sectors():
    """Sector targeting and orthogonality agree on the same number, reached two ways.

    The N=8 open chain's first excited state is a triplet, so its energy is both the
    *second* eigenvalue of the ``S^z_tot = 0`` block -- which orthogonality against the
    ground state reaches -- and the *first* of the ``S^z_tot = 1`` block, which a charged
    ``D=1`` boundary leg reaches with no projection at all. Two mechanisms, one number.
    """
    sz0 = np.linalg.eigvalsh(heisenberg_block(8, 4))
    sz1 = np.linalg.eigvalsh(heisenberg_block(8, 5))
    assert sz0[1] == pytest.approx(sz1[0], abs=1e-10)  # the oracle says it is a triplet
    phys, h, _bonds = _u1_heisenberg(8)
    up, down = U1Sector(1), U1Sector(-1)
    charged = MPS.product(phys, [up, down, up, down, up, down, up, up])  # S^z_tot = 1
    out = dmrg_(charged, h, chi=32, max_sweeps=60)
    assert charged[0].legs[0].space == GradedSpace.new(U1, {U1Sector(2): 1})
    assert out.energy == pytest.approx(sz1[0], abs=1e-10)
    _e0, e1, _ov = _two_lowest(phys, h, _bonds, chi=32)
    assert out.energy == pytest.approx(e1, abs=1e-10)


def test_a_converged_state_from_another_sector_is_skipped_rather_than_projected():
    """It is orthogonal by the symmetry, so the run is the plain ground-state run."""
    phys, h, bonds = _u1_heisenberg(8)
    up, down = U1Sector(1), U1Sector(-1)
    other = MPS.product(phys, [up, down, up, down, up, down, up, up])  # S^z_tot = 1
    plain = dmrg_(MPS.random(phys, bonds, seed=1), h, chi=32)
    projected = dmrg_(MPS.random(phys, bonds, seed=1), h, chi=32, orthogonal_to=[other])
    assert projected.energy == pytest.approx(plain.energy, abs=0.0)


def test_fz2_first_excited_matches_the_free_fermion_enumeration_at_n12():
    """N=12 spinless fermions: the two lowest even-parity many-body energies."""
    even = free_fermion_parity_spectrum(12, 0)
    e0, e1, overlap = _two_lowest(*_fz2_chain(12), chi=64, seeds=(1, 5))
    assert e0 == pytest.approx(even[0], abs=1e-9)
    assert e1 == pytest.approx(even[1], abs=1e-9)
    assert abs(overlap) < 1e-9


def test_su2_first_excited_is_the_second_singlet():
    """N=8 SU(2): a trivial boundary leg targets the singlets, so the state above the
    ground state is the second one of them."""
    singlets = singlet_spectrum(8)
    e0, e1, overlap = _two_lowest(*_su2_heisenberg(8), chi=32, seeds=(0, 4))
    assert e0 == pytest.approx(singlets[0], abs=1e-10)
    assert e1 == pytest.approx(singlets[1], abs=1e-10)
    assert abs(overlap) < 1e-10
