"""M49 (#214): ``H|psi>`` as a state, and the energy variance it unlocks.

The apply is checked against the dense oracle on every provider the suite carries and on
**both** of ``from_terms``' representations, because the compressed and deferred tables
write the MPO bond's ``dual`` flag differently and a rule that reads the flag rather than
the leg would be right for one and silently wrong for the other -- only for fermions. The
variance is checked two ways #214 names: it falls towards zero as ``chi`` grows, and
``<psi|H|psi>`` read through the apply agrees with ``Env.measure()``.
"""

import inspect

import heisenberg_walkthrough as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

from tenet.network import MPO, MPS, Env, dmrg_, overlap
from tenet.network import mps as mps_module

from . import test_dmrg as dmrg_test
from . import test_hubbard as hub
from . import test_mpo as mpo_test


def _u1(n, cutoff=1e-13):
    psi = MPS.random(example.PHYS, example.bond_spaces(n), seed=0).canonize_()
    return MPO.from_terms(n, mpo_test._heisenberg_terms(n), cutoff=cutoff), psi, 2


def _su2(n, cutoff=1e-13):
    psi = MPS.random(mpo_test.SU2_PHYS, dmrg_test.su2_bond_spaces(n), seed=0).canonize_()
    return mpo_test.su2_heisenberg(n, cutoff=cutoff), psi, 2


def _fz2(n, cutoff=1e-13):
    psi = hub._state(n, 6, 6, seed=1).canonize_()
    return MPO.from_terms(n, hub._rank3_terms(n, 4.0), cutoff=cutoff), psi, 4


def _dense_gap(h, psi, d, n):
    got = np.asarray(h.apply(psi).to_dense()).reshape(-1)
    want = np.asarray(h.to_dense()).reshape(d**n, d**n) @ np.asarray(psi.to_dense()).reshape(-1)
    return float(np.abs(got - want).max()), float(np.abs(want).max())


# --- the apply is the operator it claims to be ----------------------------------------


@pytest.mark.parametrize("case", (_u1, _su2, _fz2), ids=("u1", "su2", "fz2"))
@pytest.mark.parametrize("cutoff", (1e-13, None), ids=("compressed", "deferred"))
@pytest.mark.parametrize("n", (2, 4, 6))
def test_the_product_matches_the_dense_oracle(case, cutoff, n):
    """Every amplitude of ``H|psi>``, against ``h.to_dense() @ psi.to_dense()``.

    Both ``cutoff`` values are run because they are two *representations* of the same
    operator and they disagree about the MPO bond's ``dual`` flag -- the compressed table
    writes it dual, the deferred one plain. The turn-around that lets the two bonds fuse is
    charged by the leg (``inv=not dual``) rather than by the flag for exactly that reason,
    and under fZ2 the wrong choice is a wrong number rather than an error.
    """
    gap, scale = _dense_gap(*case(n, cutoff), n)
    assert gap < 1e-11 * max(scale, 1.0)


def test_the_identity_operator_returns_the_state():
    psi = hub._state(4, 6, 6, seed=2).canonize_()
    got = MPO.identity(4, hub.PHYS4).apply(psi)
    assert overlap(psi, got) == pytest.approx(overlap(psi, psi), rel=1e-12)
    gap, _ = _dense_gap(MPO.identity(4, hub.PHYS4), psi, 4, 4)
    assert gap < 1e-14


def test_a_numeric_w_mpo_applies_too():
    """``from_w`` carries no edge description at all; the apply reads site tensors."""
    n = 6
    psi = MPS.random(example.PHYS, example.bond_spaces(n), seed=0).canonize_()
    gap, scale = _dense_gap(example.mpo(n), psi, 2, n)
    assert gap < 1e-12 * max(scale, 1.0)


def test_the_two_representations_give_the_same_state():
    """The compressed and deferred tables are one operator, so they are one product."""
    n = 6
    psi = hub._state(n, 6, 6, seed=1).canonize_()
    terms = hub._rank3_terms(n, 4.0)
    a = MPO.from_terms(n, terms).apply(psi)
    b = MPO.from_terms(n, terms, cutoff=None).apply(psi)
    left = np.asarray(a.to_dense()).reshape(-1)
    right = np.asarray(b.to_dense()).reshape(-1)
    assert np.abs(left - right).max() < 1e-11


def test_the_input_state_is_not_modified_and_the_bond_is_the_fusion():
    n = 6
    h, psi, _ = _u1(n)
    before = (list(psi), psi.center)
    out = h.apply(psi)
    assert (list(psi), psi.center) == before
    assert out is not psi and out.center is None
    for k in range(n - 1):
        mpo_bond = h[k].legs[3].space.dim
        assert out[k].legs[2].space.dim == psi[k].legs[2].space.dim * mpo_bond
    # The two boundary legs are the *state's* own, not a fusion: the operator's boundary is
    # D=1 and is capped, which is what keeps <psi|H|psi> contractible against psi.
    assert out[0].legs[0] == psi[0].legs[0]
    assert out[n - 1].legs[2] == psi[n - 1].legs[2]


def test_an_operator_and_a_state_of_two_lengths_are_refused():
    h, psi, _ = _u1(6)
    with pytest.raises(ValueError, match="same length"):
        h.apply(MPS.random(example.PHYS, example.bond_spaces(4), seed=0))


# --- truncation is compress_, by name -------------------------------------------------


def test_the_apply_takes_no_truncation_keyword_and_compress_is_the_form():
    """#214's convention criterion: the *total* discarded weight, in ``compress_``'s name.

    ``apply`` has no ``chi``/``cutoff``, so there is one place the total-versus-maximum
    conventions could blur and it is not this one. ``compress_`` already takes the pair the
    sweep takes and already returns ``sqrt(sum_bond dw)``.
    """
    assert list(inspect.signature(MPO.apply).parameters) == ["self", "psi"]
    n = 8
    h, psi, _ = _u1(n)
    exact = h.apply(psi)
    wide = max(exact[k].legs[2].space.dim for k in range(n - 1))
    truncated = h.apply(psi)
    discarded = truncated.compress_(chi=8, cutoff=1e-12)
    assert isinstance(discarded, float) and 0.0 <= discarded < 1.0
    assert max(truncated[k].legs[2].space.dim for k in range(n - 1)) <= 8 < wide
    # Pythagoras, in compress_'s own convention: the kept fraction of the exact product.
    kept = overlap(exact, truncated) / (
        overlap(exact, exact) ** 0.5 * overlap(truncated, truncated) ** 0.5
    )
    assert 1.0 - kept < discarded**2 + 1e-9


# --- the variance ---------------------------------------------------------------------


def test_the_energy_read_through_the_apply_is_env_measure():
    """#214's first numerical criterion, and the statement that the apply is exact."""
    for case in (_u1, _su2, _fz2):
        h, psi, _ = case(6)
        got = overlap(psi, h.apply(psi)) / overlap(psi, psi)
        want = Env(psi, h).measure() / overlap(psi, psi)
        assert got == pytest.approx(want, abs=1e-10)


def test_the_variance_falls_towards_zero_as_chi_grows():
    """#214's second criterion. A change test can be satisfied by a run stuck on a wrong
    bond structure; the variance is the check that is not one, and on a chain whose exact
    ground state fits in the largest ``chi`` it goes to solver precision."""
    n = 8
    h = MPO.from_terms(n, mpo_test._heisenberg_terms(n))
    seen = []
    for chi in (2, 4, 8, 32):
        psi = MPS.random(example.PHYS, example.bond_spaces(n), seed=0)
        dmrg_(psi, h, chi=chi, cutoff=1e-14, max_sweeps=20)
        seen.append(h.variance(psi))
    assert all(v >= -1e-9 for v in seen)
    assert seen[-1] < 1e-8
    assert seen[-1] < seen[0]


def test_an_exact_eigenstate_has_zero_variance():
    """A product state is an exact eigenstate of the identity, and of a diagonal operator."""
    from tenet.symmetry import U1Sector

    psi = MPS.product(example.PHYS, [U1Sector(1), U1Sector(-1)] * 3)
    assert MPO.identity(6, example.PHYS).variance(psi) == pytest.approx(0.0, abs=1e-12)


def test_the_variance_is_the_same_on_both_representations():
    n = 6
    psi = hub._state(n, 6, 6, seed=1).canonize_()
    terms = hub._rank3_terms(n, 4.0)
    a = MPO.from_terms(n, terms).variance(psi)
    b = MPO.from_terms(n, terms, cutoff=None).variance(psi)
    assert a == pytest.approx(b, rel=1e-9)


@pytest.mark.parametrize("case", (_su2, _fz2), ids=("su2", "fz2"))
def test_the_variance_matches_a_dense_oracle(case):
    """``<H^2> - <H>^2`` on the dense operator, which needs no apply at all."""
    n = 4
    h, psi, d = case(n)
    amps = np.asarray(psi.to_dense()).reshape(-1)
    amps = amps / np.linalg.norm(amps)
    dense = np.asarray(h.to_dense()).reshape(d**n, d**n)
    e = float(amps @ dense @ amps)
    want = float(amps @ dense @ dense @ amps) - e * e
    assert h.variance(psi) == pytest.approx(want, abs=1e-9)


# --- the surface ----------------------------------------------------------------------


def test_no_container_arithmetic_shipped_without_a_caller():
    """#214's last criterion, stated as a test: what the variance did not need is absent.

    No ``MPO.__matmul__`` (the variance needs ``H|psi>``, not ``H H``), no ``MPS.__add__``
    or ``__mul__``, no ``MPO.dagger``/``plus_identity``. Each is real in the references and
    each has no caller here; adding them for symmetry is what this asserts against.
    """
    for name in ("__matmul__", "dagger", "plus_identity", "is_hermitian", "overlap"):
        assert not hasattr(MPO, name), name
    for name in ("__add__", "__mul__", "add"):
        assert not hasattr(MPS, name), name
    assert {"apply", "variance"} <= set(vars(MPO))
    assert "MPO" in mps_module.__all__
