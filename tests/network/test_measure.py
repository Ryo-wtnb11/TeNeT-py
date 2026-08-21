"""M48 (#213): the public measurement API on top of the two-state ``Env``.

The engine half shipped with M61 Stage D -- ``Env(psi, h, bra=phi)`` and ``Env.project2``.
What is pinned here is the surface a user calls: ``overlap``, ``measure_mpo``,
``correlation_function`` and the whole-chain ``expectation_profile``, each against the
route it has to agree with and, for the profile, against a contraction count rather than a
claim.
"""

import ast
import inspect
import pathlib

import heisenberg_walkthrough as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

import tenet
from tenet.network import (
    MPO,
    MPS,
    Env,
    correlation_function,
    dmrg_,
    expectation_1site,
    expectation_2site,
    expectation_profile,
    local_op,
    measure_mpo,
    overlap,
)
from tenet.network import mps as mps_module
from tenet.symmetry import U1Sector

from . import test_hubbard as hubbard

SZ = np.diag([-0.5, 0.5])


def _sz_op(charge=None):
    return local_op(SZ, phys=example.PHYS, charge=charge)


def _ground(n_sites, chi=16):
    psi = MPS.random(example.PHYS, example.bond_spaces(n_sites), seed=0)
    dmrg_(psi, example.mpo_from_terms(n_sites), chi=chi, max_sweeps=8)
    return psi


def _dense_at(local, site, n_sites, d=2):
    out = np.array([[1.0]])
    for k in range(n_sites):
        out = np.kron(out, local if k == site else np.eye(d))
    return out


# --- <phi|psi> ------------------------------------------------------------------------


def test_the_overlap_of_a_state_with_itself_is_its_norm_squared():
    psi = _ground(6)
    assert overlap(psi, psi) == pytest.approx(psi.norm() ** 2, rel=1e-12)


def test_mps_norm_is_expressed_through_overlap_rather_than_beside_it():
    """#213's criterion, read off the source: one transfer pass, one implementation."""
    tree = ast.parse(pathlib.Path(mps_module.__file__).read_text())
    body = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "norm"
    )
    calls = {ast.unparse(node.func) for node in ast.walk(body) if isinstance(node, ast.Call)}
    assert "overlap" in calls
    assert not any(c.endswith("einsum") for c in calls)


def test_the_overlap_of_two_different_states_matches_a_dense_oracle():
    phi, psi = _ground(6), _ground(6, chi=4)
    a = np.asarray(phi.to_dense()).reshape(-1)
    b = np.asarray(psi.to_dense()).reshape(-1)
    assert overlap(phi, psi) == pytest.approx(float(np.dot(a.conj(), b)), abs=1e-10)


def test_an_overlap_of_two_lengths_is_refused():
    with pytest.raises(ValueError, match="same length"):
        overlap(_ground(4), _ground(6))


def test_the_overlap_reads_the_states_without_modifying_them():
    phi, psi = _ground(4), _ground(4, chi=4)
    before = (list(phi), phi.center, list(psi), psi.center)
    overlap(phi, psi)
    assert (list(phi), phi.center, list(psi), psi.center) == before


# --- <phi|H|psi> ----------------------------------------------------------------------


def test_measure_mpo_on_one_state_is_env_measure():
    psi = _ground(6)
    h = example.mpo_from_terms(6)
    assert measure_mpo(psi, h, psi) == pytest.approx(Env(psi, h).measure(), rel=1e-12)


def test_measure_mpo_with_the_identity_is_the_plain_overlap():
    """The two routes to ``<phi|psi>`` -- one transfer pass and one environment sweep."""
    phi, psi = _ground(6), _ground(6, chi=4)
    ident = MPO.identity(6, example.PHYS)
    assert measure_mpo(phi, ident, psi) == pytest.approx(overlap(phi, psi), abs=1e-10)
    assert measure_mpo(psi, ident, psi) == pytest.approx(psi.norm() ** 2, abs=1e-10)


def test_measure_mpo_between_two_different_states_matches_a_dense_oracle():
    phi, psi = _ground(6), _ground(6, chi=4)
    h = example.mpo_from_terms(6)
    a = np.asarray(phi.to_dense()).reshape(-1)
    b = np.asarray(psi.to_dense()).reshape(-1)
    want = float(a.conj() @ np.asarray(h.to_dense()).reshape(len(a), len(a)) @ b)
    assert measure_mpo(phi, h, psi) == pytest.approx(want, abs=1e-9)


def test_a_measurement_never_writes_into_a_sweep_cache():
    """``Env.measure`` builds its own pass; the environments a caller holds do not move."""
    psi = _ground(6)
    h = example.mpo_from_terms(6)
    env = Env(psi, h).setup_()
    held = dict(env.F)
    assert measure_mpo(psi, h, psi) == pytest.approx(env.measure(), rel=1e-12)
    assert dict(env.F) == held
    assert all(env.F[k] is v for k, v in held.items())


# --- correlation at a distance --------------------------------------------------------


def test_the_correlation_at_distance_one_is_expectation_2site():
    """The two adjacent-pair routes agree, and ``expectation_2site`` is untouched."""
    psi = _ground(6)
    sz3 = _sz_op(U1Sector(0))
    szsz = local_op(np.kron(SZ, SZ), phys=example.PHYS)
    got = correlation_function(psi, sz3, sz3, pairs=[(n, n + 1) for n in range(5)])
    for n in range(5):
        assert got[n, n + 1] == pytest.approx(expectation_2site(psi, szsz, n), abs=1e-10)


def test_the_whole_correlation_matrix_matches_a_dense_oracle():
    psi = _ground(6)
    amps = np.asarray(psi.to_dense()).reshape(-1)
    amps = amps / np.linalg.norm(amps)
    got = correlation_function(psi, _sz_op(U1Sector(0)), _sz_op(U1Sector(0)))
    assert set(got) == {(i, j) for i in range(6) for j in range(i + 1, 6)}
    for (i, j), value in got.items():
        op = _dense_at(SZ, i, 6) @ _dense_at(SZ, j, 6)
        assert value == pytest.approx(float(amps @ op @ amps), abs=1e-9)


def test_pairs_selects_and_a_bad_pair_is_refused():
    psi = _ground(4)
    sz3 = _sz_op(U1Sector(0))
    assert set(correlation_function(psi, sz3, sz3, pairs=[(0, 3)])) == {(0, 3)}
    for bad in ((1, 1), (2, 0), (0, 4), (-1, 2)):
        with pytest.raises(ValueError, match="0 <= i < j"):
            correlation_function(psi, sz3, sz3, pairs=[bad])


# --- fermions: the string is right, and #147's oracle is what says so -----------------


def test_a_fermionic_correlator_at_a_distance_carries_its_jordan_wigner_string():
    """``<c+_up,i c_up,j>`` on a converged Hubbard state, against the explicit-JW oracle.

    ``_dense_c`` (``tests/network/test_hubbard.py``) writes the string out -- the local
    parity ``P`` on every site left of the operator -- so agreement at ``j - i >= 2``,
    where the string is a nontrivial product and a missing sign is a *different number*
    rather than a rounding, is the statement that the distance-``r`` correlator is right.
    """
    n = 4
    h = MPO.from_terms(n, hubbard._rank3_terms(n, 4.0))
    psi = hubbard._state(n, 8, 8, seed=1)
    dmrg_(psi, h, chi=16, cutoff=1e-14, max_sweeps=60)
    amps = np.asarray(psi.to_dense()).reshape(-1)
    amps = amps / np.linalg.norm(amps)

    cd_up = local_op(hubbard.C_UP.T, phys=hubbard.PHYS4, charge=hubbard.FZ2Sector(1))
    c_up = local_op(hubbard.C_UP, phys=hubbard.PHYS4, charge=hubbard.FZ2Sector(1))
    got = correlation_function(psi, cd_up, c_up)
    assert any(j - i >= 2 for i, j in got)
    for (i, j), value in got.items():
        op = hubbard._dense_c(n, i, hubbard.C_UP.T) @ hubbard._dense_c(n, j, hubbard.C_UP)
        assert value == pytest.approx(float(amps @ op @ amps), abs=1e-8), (i, j)


# --- the whole-chain profile ----------------------------------------------------------


def test_the_profile_is_the_per_site_values():
    psi = _ground(8)
    sz = _sz_op()
    got = expectation_profile(psi, sz)
    assert got == pytest.approx([expectation_1site(psi, sz, n) for n in range(8)], abs=1e-10)


def test_the_profile_matches_a_dense_oracle_and_leaves_the_state_alone():
    psi = _ground(6)
    amps = np.asarray(psi.to_dense()).reshape(-1)
    amps = amps / np.linalg.norm(amps)
    before = (list(psi), psi.center)
    got = expectation_profile(psi, _sz_op())
    assert (list(psi), psi.center) == before
    for n, value in enumerate(got):
        assert value == pytest.approx(float(amps @ _dense_at(SZ, n, 6) @ amps), abs=1e-9)


def test_the_profile_costs_one_pass_and_the_per_site_loop_costs_one_per_site(monkeypatch):
    """#213's ``O(N**2) -> O(N)`` criterion, counted rather than claimed.

    ``expectation_1site`` ends in two *full-chain* transfer passes, so the profile every
    DMRG user writes is ``O(N)`` passes over an ``O(N)`` chain. The centre walk is one
    pass. At ``N = 24`` the difference is a fact: the loop spends more ``einsum`` calls
    than the chain has sites squared, and the profile spends fewer than a small multiple
    of the sites.
    """
    n = 24
    psi = MPS.random(example.PHYS, example.bond_spaces(n), seed=1).canonize_()
    sz = _sz_op()
    counts = []
    real = tenet.einsum

    def counting(*args, **kwargs):
        counts.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(tenet, "einsum", counting)
    expectation_profile(psi, sz)
    profile = len(counts)
    counts.clear()
    [expectation_1site(psi, sz, k) for k in range(n)]
    per_site = len(counts)

    assert profile < 8 * n
    assert per_site > n * n
    assert per_site > 10 * profile


def test_the_profile_refuses_an_operator_of_the_wrong_rank():
    psi = _ground(4)
    szsz = local_op(np.kron(SZ, SZ), phys=example.PHYS)
    with pytest.raises(ValueError, match="rank 2"):
        expectation_profile(psi, szsz)


# --- the surface ----------------------------------------------------------------------


def test_the_adjacent_pair_readers_keep_their_signatures():
    """Additions only: nothing #213 adds changes a name that was already public."""
    assert list(inspect.signature(expectation_1site).parameters) == ["psi", "o", "n"]
    assert list(inspect.signature(expectation_2site).parameters) == ["psi", "o", "n"]
    assert list(inspect.signature(overlap).parameters) == ["bra", "ket"]
    assert list(inspect.signature(measure_mpo).parameters) == ["bra", "h", "ket"]
    assert list(inspect.signature(correlation_function).parameters) == [
        "psi",
        "a",
        "b",
        "pairs",
    ]
    assert list(inspect.signature(expectation_profile).parameters) == ["psi", "o"]
