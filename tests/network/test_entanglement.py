"""M50 (#215): the per-bond Schmidt spectrum and entanglement entropy of a state.

The datum was computed at every bond of every sweep and returned nowhere. These tests pin
that it is now readable *from the state*, that it agrees with a dense oracle, that the
sector resolution is real, and -- the one that says the ``sqrt(qdim)`` weight is right
rather than merely self-consistent -- that an SU(2) state and the same state under U(1)
report the same entropy.
"""

import ast
import inspect
import math

import heisenberg_walkthrough as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

from tenet.network import MPS, common, dmrg_, entropy, spectrum, spectrum_sectors
from tenet.symmetry import U1Sector

from . import test_dmrg as dmrg_test
from . import test_mpo as mpo_test


def _u1_ground(n_sites, chi=32):
    psi = MPS.random(example.PHYS, example.bond_spaces(n_sites), seed=0)
    dmrg_(psi, example.mpo_from_terms(n_sites), chi=chi, max_sweeps=12)
    return psi


def _su2_ground(n_sites, chi=32):
    psi = MPS.random(mpo_test.SU2_PHYS, dmrg_test.su2_bond_spaces(n_sites), seed=0)
    dmrg_(psi, mpo_test.su2_heisenberg(n_sites), chi=chi, max_sweeps=12)
    return psi


def _dense_entropy(psi, alpha=1.0):
    """The oracle: ``psi.to_dense()`` reshaped at each cut, entropy of its singular values."""
    amps = np.asarray(psi.to_dense()).reshape(-1)
    amps = amps / np.linalg.norm(amps)
    out = {}
    for n in range(len(psi) - 1):
        p = np.linalg.svd(amps.reshape(2 ** (n + 1), -1), compute_uv=False) ** 2
        p = p[p > 1e-24]
        out[n] = (
            float(-np.sum(p * np.log(p)))
            if alpha == 1.0
            else float(np.log(np.sum(p**alpha)) / (1.0 - alpha))
        )
    return out


# --- the entropy is the state's, and it is the right number ---------------------------


@pytest.mark.parametrize("alpha", (1.0, 0.5, 2.0))
def test_the_entropy_profile_matches_a_dense_oracle(alpha):
    """Every internal cut of a converged 6-site chain against ``numpy.linalg.svd``.

    The oracle knows nothing about bonds, gauges or sectors: it reshapes the ``2**6``
    amplitude vector at each cut and takes singular values. Agreement therefore pins the
    normalization, the log base and the fact that the values read off the state are the
    Schmidt values of that cut and not of some other gauge.
    """
    psi = _u1_ground(6)
    got = psi.entanglement_entropy(alpha=alpha)
    want = _dense_entropy(psi, alpha)
    assert set(got) == set(want) == set(range(5))
    for n in want:
        assert got[n] == pytest.approx(want[n], abs=1e-8)


def test_the_schmidt_values_match_the_dense_singular_values():
    """The spectrum itself, not only the number derived from it."""
    psi = _u1_ground(6)
    amps = np.asarray(psi.to_dense()).reshape(-1)
    amps = amps / np.linalg.norm(amps)
    for n, vals in psi.schmidt_values().items():
        want = np.linalg.svd(amps.reshape(2 ** (n + 1), -1), compute_uv=False)
        got = np.array(sorted(vals, reverse=True))
        m = min(len(got), len(want))
        assert np.allclose(got[:m], want[:m], atol=1e-8)
        assert np.allclose(got[m:], 0.0, atol=1e-8)
        assert np.allclose(want[m:], 0.0, atol=1e-8)


def test_a_maximally_entangled_pair_is_one_nat():
    """The number a reader can check by hand: the two-site singlet cuts at ``log 2``."""
    psi = _u1_ground(2)
    assert psi.entanglement_entropy()[0] == pytest.approx(math.log(2.0), abs=1e-9)
    assert psi.entanglement_entropy(alpha=2.0)[0] == pytest.approx(math.log(2.0), abs=1e-9)


# --- the non-Abelian weight -----------------------------------------------------------


@pytest.mark.parametrize("n_sites", (2, 6))
@pytest.mark.parametrize("alpha", (1.0, 2.0))
def test_an_su2_state_has_the_entropy_of_the_same_state_under_u1(n_sites, alpha):
    """#215's criterion for the ``sqrt(qdim)`` weight, and the reason it is not obvious.

    The Heisenberg ground state of an even open chain is the same state whether the bond
    is graded by ``S^z`` or by total spin; only the *labelling* differs. Under SU(2) a
    ``j`` multiplet is one reduced value standing for ``2j + 1`` dense Schmidt values, so
    ``-sum p log p`` over the flattened spectrum reports the wrong number -- for the
    two-site singlet it reports ``0`` for a state whose entropy is ``log 2``. The
    multiplet weight is what closes that, and this equality is what says the weight is
    right rather than merely consistent with itself.
    """
    u1 = _u1_ground(n_sites).entanglement_entropy(alpha=alpha)
    su2 = _su2_ground(n_sites).entanglement_entropy(alpha=alpha)
    assert set(u1) == set(su2)
    for n in u1:
        assert su2[n] == pytest.approx(u1[n], abs=1e-7)


def test_the_flattened_spectrum_of_an_su2_bond_is_not_its_entropy():
    """The trap the test above guards, stated directly so a regression names itself."""
    psi = _su2_ground(2)
    flat = psi.schmidt_values()[0]
    naive = -sum(v * v * math.log(v * v) for v in flat if v > 1e-12)
    assert naive == pytest.approx(0.0, abs=1e-9)  # one multiplet, weight 1: no entropy at all
    assert psi.entanglement_entropy()[0] == pytest.approx(math.log(2.0), abs=1e-9)


# --- the sector resolution ------------------------------------------------------------


@pytest.mark.parametrize("ground", (_u1_ground, _su2_ground), ids=("u1", "su2"))
def test_the_sector_resolved_spectrum_flattens_to_the_flat_one(ground):
    """The two readers are one computation: ``spectrum`` *is* the flatten of the other."""
    psi = ground(6)
    flat, by_sector = psi.schmidt_values(), psi.schmidt_sectors()
    assert set(flat) == set(by_sector)
    for n, sectors in by_sector.items():
        merged = sorted((v for vals in sectors.values() for v in vals), reverse=True)
        assert merged == pytest.approx(flat[n], abs=1e-12)
        for vals in sectors.values():
            assert vals == sorted(vals, reverse=True)


def test_the_su2_bond_carries_its_entanglement_in_labelled_sectors():
    """What a flat list cannot say: which total-spin sector the entanglement sits in."""
    sectors = _su2_ground(6).schmidt_sectors()[2]
    live = {s: vals for s, vals in sectors.items() if max(vals, default=0.0) > 1e-8}
    assert len(live) > 1  # a mid-chain cut of a singlet ground state is not one multiplet


# --- what the readers promise about the state they read -------------------------------


def test_reading_a_non_canonical_state_neither_mutates_it_nor_reports_its_gauge():
    """``center is None`` canonizes -- on a **copy**, so the caller's state is untouched.

    ``compress_`` (``mps.py``:417-419) made the canonize-first choice for the same reason
    and is the precedent; the difference is that a reader must not re-gauge what it reads,
    so the copy is where the canonization happens.
    """
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=4)
    assert psi.center is None
    before = list(psi)
    got = psi.entanglement_entropy()
    assert psi.center is None
    assert list(psi) == before  # the same frozen tensors, in the same slots
    reference = psi.copy().canonize_(0)
    assert got == pytest.approx(reference.entanglement_entropy(), abs=1e-10)


def test_a_product_state_is_unentangled_at_every_cut():
    psi = MPS.product(example.PHYS, [U1Sector(1), U1Sector(-1)] * 3)
    assert all(abs(s) < 1e-12 for s in psi.entanglement_entropy().values())


def test_a_one_site_state_has_no_internal_bond():
    psi = MPS.product(example.PHYS, [U1Sector(1)])
    assert psi.schmidt_values() == psi.schmidt_sectors() == psi.entanglement_entropy() == {}


def test_a_non_positive_renyi_index_is_refused():
    """A Renyi index of zero counts the rank and a negative one is not an entropy; both
    would come back as a plausible finite float from the formula, so both are refused."""
    psi = _u1_ground(4)
    with pytest.raises(ValueError, match="Renyi index must be positive"):
        psi.entanglement_entropy(alpha=0.0)


# --- the weight is applied in one place -----------------------------------------------


def test_the_qdim_weight_lives_in_spectrum_sectors_alone():
    """#215's structural criterion, read off the source: ``spectrum`` and ``entropy`` both
    reach the bond through ``spectrum_sectors``, so there is one place a multiplet weight
    is decided and no second copy to drift."""
    src = inspect.getsource(common)
    assert src.count("qdim(sector) ** 0.5") == 1  # the weight itself, written once
    readers = {
        node.name: any(
            isinstance(sub, ast.Attribute) and sub.attr == "qdim" for sub in ast.walk(node)
        )
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.FunctionDef)
    }
    # ``spectrum`` reaches the bond only through ``spectrum_sectors``; ``entropy`` asks for
    # the quantum dimension itself, because the multiplet *count* is what a Renyi sum needs
    # and it is not recoverable from a weighted value.
    assert readers["spectrum"] is False
    assert readers["spectrum_sectors"] is True
    assert readers["entropy"] is True


def test_spectrum_keeps_its_signature():
    """The flat reader is unchanged: same one positional parameter, same return."""
    assert list(inspect.signature(spectrum).parameters) == ["s"]
    assert list(inspect.signature(spectrum_sectors).parameters) == ["s"]
    assert list(inspect.signature(entropy).parameters) == ["s", "alpha"]
