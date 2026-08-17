"""Spinful Hubbard on fZ2 legs -- #147's Gate 4, the model the refusal lift is for.

The local ``d=4`` site is ``{|0>, |ud>, |u>, |d>}`` in *graded* order -- the even
sector ``{|0>, |ud>}`` first, then the odd ``{|u>, |d>}`` -- because a dense array
over a ``GradedSpace`` is laid out sector by sector in canonical order.

**The intra-site convention, stated once**: modes are ordered up before down,
``|ud> = c+_up c+_dn |0>``. So ``c_up = a (x) 1`` carries no intra-site sign
(the up mode is first), and ``c_dn = Z (x) a`` pays the Jordan-Wigner ``Z`` on the
up mode -- ``c_dn |ud> = -|u>``. Between sites the string is the local parity
``P = (-1)^(n_up + n_dn)``; intra-site is entirely inside the on-site matrices.
This is where the pre-M23 Hubbard probe went wrong, so every single-site matrix is
verified by hand before any chain is built.
"""

import json
import pathlib

import numpy as np
import pytest

from tenet import GradedSpace
from tenet.network import MPO, MPS, Env, dmrg_, local_op
from tenet.symmetry import FZ2Sector, fZ2

PHYS4 = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})

# Hand-written on-site operators in the graded basis (|0>, |ud>, |u>, |d>).
C_UP = np.zeros((4, 4))
C_UP[0, 2] = 1.0  # c_up |u> = |0>
C_UP[3, 1] = 1.0  # c_up |ud> = +|d>   (up is the first mode: no sign)
C_DN = np.zeros((4, 4))
C_DN[0, 3] = 1.0  # c_dn |d> = |0>
C_DN[2, 1] = -1.0  # c_dn |ud> = -|u>  (the intra-site JW sign on the up mode)
N_UP = C_UP.T @ C_UP
N_DN = C_DN.T @ C_DN
P_SITE = np.diag([1.0, 1.0, -1.0, -1.0])  # the inter-site JW string


def test_the_on_site_operators_are_the_documented_convention():
    """Each 4x4 against the two-mode kron construction, and the algebra by hand."""
    a, z, eye = np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([1.0, -1.0]), np.eye(2)
    perm = [0, 3, 2, 1]  # graded (|0>,|ud>,|u>,|d>) -> mode basis |n_up n_dn>
    for hand, mode in ((C_UP, np.kron(a, eye)), (C_DN, np.kron(z, a))):
        assert np.array_equal(hand, mode[np.ix_(perm, perm)])
    assert np.array_equal(N_UP, np.diag([0.0, 1.0, 1.0, 0.0]))
    assert np.array_equal(N_DN, np.diag([0.0, 1.0, 0.0, 1.0]))
    assert np.array_equal(P_SITE, np.diag([1, 1, -1, -1]) * 1.0)
    for x in (C_UP, C_DN):
        assert np.array_equal(x @ x, np.zeros((4, 4)))
        assert np.allclose(x @ x.T + x.T @ x, np.eye(4))  # {c, c+} = 1 per flavour
    assert np.array_equal(C_UP @ C_DN + C_DN @ C_UP, np.zeros((4, 4)))
    assert np.array_equal(C_UP @ C_DN.T + C_DN.T @ C_UP, np.zeros((4, 4)))


def _dense_c(n_sites, site, local):
    """The chain operator: ``P`` on every site left of ``site`` (inter-site JW)."""
    full = np.array([[1.0]])
    for f in [P_SITE] * site + [local] + [np.eye(4)] * (n_sites - site - 1):
        full = np.kron(full, f)
    return full


def _dense_h(n_sites, u, t=1.0):
    """``-t sum_(m,s) (c+_ms c_m+1,s + h.c.) + u sum_m n_up n_dn``, dense."""
    h = np.zeros((4**n_sites, 4**n_sites))
    for m in range(n_sites - 1):
        for local in (C_UP, C_DN):
            cd, c = _dense_c(n_sites, m, local.T), _dense_c(n_sites, m + 1, local)
            h += -t * (cd @ c + c.T @ cd.T)
    for m in range(n_sites):  # even on-site operator: no string
        h += u * np.kron(np.kron(np.eye(4**m), N_UP @ N_DN), np.eye(4 ** (n_sites - m - 1)))
    return h


def _even_indices(n_sites):
    par = np.array([0, 0, 1, 1])
    idx = np.arange(4**n_sites)
    total = np.zeros_like(idx)
    for _ in range(n_sites):
        total += par[idx % 4]
        idx //= 4
    return np.flatnonzero(total % 2 == 0)


def _ed_even(n_sites, u):
    h = _dense_h(n_sites, u)
    even = _even_indices(n_sites)
    return float(np.linalg.eigvalsh(h[np.ix_(even, even)]).min())


def test_the_dense_oracle_is_hermitian_and_free_at_u0():
    """The self-check that catches a wrong intra-site convention.

    At U=0 the 256 many-body eigenvalues are exactly the subset sums of the 2N=8
    single-particle energies (the open-chain hopping band, twice for spin). A sign
    slip in ``c_dn`` breaks this within the first few eigenvalues.
    """
    h = _dense_h(4, 0.0)
    assert np.abs(h - h.T).max() == 0.0
    single = np.linalg.eigvalsh(-(np.eye(4, k=1) + np.eye(4, k=-1)))
    modes = np.concatenate([single, single])  # spin degeneracy
    subset_sums = np.sort([modes[[b for b in range(8) if k >> b & 1]].sum() for k in range(256)])
    assert np.abs(np.sort(np.linalg.eigvalsh(h)) - subset_sums).max() < 1e-10
    assert np.abs(_dense_h(4, 3.0) - _dense_h(4, 3.0).T).max() == 0.0


def _rank3_terms(n_sites, u):
    cd_up = local_op(C_UP.T, phys=PHYS4, charge=FZ2Sector(1))
    c_up = local_op(C_UP, phys=PHYS4, charge=FZ2Sector(1))
    cd_dn = local_op(C_DN.T, phys=PHYS4, charge=FZ2Sector(1))
    c_dn = local_op(C_DN, phys=PHYS4, charge=FZ2Sector(1))
    op_nn = local_op(N_UP @ N_DN, phys=PHYS4, charge=FZ2Sector(0))
    terms = []
    for m in range(n_sites - 1):
        for cd, c in ((cd_up, c_up), (cd_dn, c_dn)):
            terms += [(-1.0, [(cd, m), (c, m + 1)]), (-1.0, [(cd, m + 1), (c, m)])]
    if u:
        terms += [(u, [(op_nn, m)]) for m in range(n_sites)]
    return terms


def _ksite_terms(n_sites, u):
    hop2 = sum(
        -(_dense_c(2, 0, m.T) @ _dense_c(2, 1, m) + (_dense_c(2, 0, m.T) @ _dense_c(2, 1, m)).T)
        for m in (C_UP, C_DN)
    )
    block = local_op(hop2, phys=PHYS4)
    op_nn = local_op(N_UP @ N_DN, phys=PHYS4, charge=FZ2Sector(0))
    terms = [(1.0, [(block, (m, m + 1))]) for m in range(n_sites - 1)]
    if u:
        terms += [(u, [(op_nn, m)]) for m in range(n_sites)]
    return terms


def _state(n_sites, chi_even, chi_odd, seed):
    unit = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
    mid = GradedSpace.new(fZ2, {FZ2Sector(0): chi_even, FZ2Sector(1): chi_odd})
    return MPS.random(PHYS4, [unit] + [mid] * (n_sites - 1) + [unit], seed=seed)


def test_hubbard_dmrg_matches_ed_across_the_u_sweep():
    """N=4, U/t in {0, 2, 4, 8}, both ``from_terms`` routes, against even-parity ED."""
    for u in (0.0, 2.0, 4.0, 8.0):
        ed = _ed_even(4, u)
        for terms in (_rank3_terms(4, u), _ksite_terms(4, u)):
            h = MPO.from_terms(4, terms)
            out = dmrg_(_state(4, 8, 8, seed=1), h, chi=16, cutoff=1e-14)
            assert out.energy == pytest.approx(ed, abs=1e-10), f"U={u}"


FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "mpskit_hubbard.json"


def test_mpskit_agrees_with_ed_and_dmrg_lands_on_both_at_n4():
    """The dual oracle: MPSKit.jl (independent library) and tenet's own ED.

    The fixture's validation block records MPSKit-vs-ED agreement measured at
    generation time; this test re-derives the ED side and asserts all three meet.
    """
    fix = json.loads(FIXTURE.read_text())
    for u_key, entry in fix["N4"].items():
        u = float(u_key[1:])
        ed = _ed_even(4, u)
        assert entry["energy"] == pytest.approx(ed, abs=2e-10)
        h = MPO.from_terms(4, _rank3_terms(4, u))
        out = dmrg_(_state(4, 8, 8, seed=2), h, chi=16, cutoff=1e-14)
        assert out.energy == pytest.approx(entry["energy"], abs=2e-10)


def test_hubbard_n6_is_self_consistent_and_meets_mpskit():
    """N=6 at two chi values: monotone, ``Env.measure`` == the variational energy,
    and the U/t=4 energy against the MPSKit fixture.

    N=6 rather than the issue's N=8: an N=8 spinful run at two chi values measured
    several times the whole new-test budget, and the self-consistency statement is
    size-independent. The fixture still carries N=8 entries for the day the budget
    grows.
    """
    fix = json.loads(FIXTURE.read_text())
    h = MPO.from_terms(6, _rank3_terms(6, 4.0))
    energies = []
    for chi in (12, 24):
        out = dmrg_(_state(6, 6, 6, seed=3), h, chi=chi, cutoff=1e-14)
        energies.append(out.energy)
        assert Env(out.psi, h).measure() == pytest.approx(out.energy, abs=1e-10)
    assert energies[1] <= energies[0] + 1e-12  # monotone in chi
    assert energies[1] == pytest.approx(fix["N6"]["U4"]["energy"], abs=1e-8)
