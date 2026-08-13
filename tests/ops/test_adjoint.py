"""Tests for the categorical adjoint (dagger) — issue #31."""

import dataclasses

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import to_matrices
from tenet.ops.map import adjoint_plan
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 2})
W = GradedSpace.new(SU2, {HALF: 2, ONE: 1})
U = GradedSpace.new(SU2, {ZERO: 1, HALF: 2})
X = GradedSpace.new(SU2, {ZERO: 2, ONE: 1})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
P = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2})

# interleaved sides (OUT, IN, OUT, IN); several coupled sectors of different qdim
SU2_LEGS = (Leg(V, OUT, name="a"), Leg(W, IN), Leg(U, OUT), Leg(X, IN))
U1_LEGS = (Leg(Q, OUT), Leg(P, IN), Leg(Q, IN))
# mixes dual=True and dual=False: structural criteria only, dense is Milestone 4
DUAL_LEGS = (Leg(V, OUT, dual=True), Leg(W, IN), Leg(U, OUT), Leg(X, IN, dual=True))

ALL_LEGS = [
    pytest.param(SU2_LEGS, id="su2"),
    pytest.param(U1_LEGS, id="u1"),
    pytest.param(DUAL_LEGS, id="su2-dual"),
]


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def cx(legs, seed=0):
    """A genuinely complex random tensor (``SymmetricTensor.random`` draws real)."""
    re = SymmetricTensor.random(legs, seed=seed)
    im = SymmetricTensor.random(legs, seed=seed + 100)
    return SymmetricTensor(
        re.structure, tuple(a + 1j * b for a, b in zip(re.blocks, im.blocks, strict=True))
    )


def mat(t):
    """Dense matrix form: OUT axes to rows, IN axes to columns."""
    dense = t.to_dense()
    order = (*t.structure.out_axes, *t.structure.in_axes)
    rows = int(np.prod([dense.shape[a] for a in t.structure.out_axes], dtype=int))
    return dense.transpose(*order).reshape(rows, -1)


# --- surfaces -------------------------------------------------------------------


@pytest.mark.parametrize("legs", ALL_LEGS)
def test_three_spellings_agree(legs):
    t = cx(legs)
    assert t.adjoint() == tenet.adjoint(t)
    assert t.adjoint() == t.as_map().adjoint()


# --- legs and shapes ------------------------------------------------------------


@pytest.mark.parametrize("legs", ALL_LEGS)
def test_legs_flip_side_only(legs):
    t = cx(legs)
    d = t.adjoint()
    for i, (old, new) in enumerate(zip(t.legs, d.legs, strict=True)):
        assert new == dataclasses.replace(old, side=IN if old.side is OUT else OUT), i
        assert (new.space, new.dual, new.name) == (old.space, old.dual, old.name)
    assert d.reduced_shape == t.reduced_shape
    assert d.codomain == tuple(dataclasses.replace(x, side=OUT) for x in t.domain)
    assert d.domain == tuple(dataclasses.replace(x, side=IN) for x in t.codomain)


@pytest.mark.parametrize("legs", ALL_LEGS)
def test_block_shapes_are_a_permutation_and_no_transpose_is_called(legs, monkeypatch):
    t = cx(legs)
    real_do = ar.do

    def guard(fn, *args, **kw):
        assert fn != "transpose", "adjoint must not transpose blocks"
        return real_do(fn, *args, **kw)

    monkeypatch.setattr(ar, "do", guard)
    d = t.adjoint()
    assert sorted(b.shape for b in d.blocks) == sorted(b.shape for b in t.blocks)


# --- the plan -------------------------------------------------------------------


@pytest.mark.parametrize("legs", ALL_LEGS)
def test_plan_sources_are_a_bijection(legs):
    plan = adjoint_plan(tenet.TensorStructure(legs))
    assert sorted(plan.sources) == list(range(len(plan.sources)))
    assert len(plan.sources) == plan.new_structure.num_blocks


def test_plan_is_not_the_identity_permutation_for_su2():
    plan = adjoint_plan(tenet.TensorStructure(SU2_LEGS))
    assert plan.sources != tuple(range(len(plan.sources)))


def test_plan_is_cached_frozen_hashable_and_array_free():
    s = tenet.TensorStructure(SU2_LEGS)
    plan = adjoint_plan(s)
    assert adjoint_plan(s) is plan
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.sources = ()
    assert hash(plan.new_structure)
    for field in dataclasses.fields(plan):
        assert not hasattr(getattr(plan, field.name), "shape")


# --- algebra --------------------------------------------------------------------


@pytest.mark.parametrize("legs", ALL_LEGS)
def test_involution_is_exact(legs):
    t = cx(legs)
    assert t.adjoint().adjoint() == t


@pytest.mark.parametrize("legs", ALL_LEGS)
def test_anti_linearity(legs):
    a, b = 2.0 - 3.0j, -0.5 + 1.25j
    t, s = cx(legs, seed=0), cx(legs, seed=1)
    lhs = (a * t + b * s).adjoint()
    rhs = np.conj(a) * t.adjoint() + np.conj(b) * s.adjoint()
    assert tenet.allclose(lhs, rhs, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("legs", ALL_LEGS)
def test_norm_is_invariant(legs):
    t = cx(legs)
    assert t.adjoint().norm() == pytest.approx(t.norm(), abs=1e-12)


def test_adjoint_is_not_conj():
    t = cx(SU2_LEGS)
    assert t.adjoint() != t.conj()
    assert t.adjoint().structure != t.conj().structure
    assert [leg.side for leg in t.conj().legs] == [leg.side for leg in t.legs]
    assert [leg.side for leg in t.adjoint().legs] == [IN, OUT, IN, OUT]


def test_adjoint_is_not_dualized():
    t = cx(DUAL_LEGS)
    assert [leg.dual for leg in t.legs] == [True, False, False, True]
    assert [leg.dual for leg in t.adjoint().legs] == [leg.dual for leg in t.legs]
    assert any(leg.dualized() != leg for leg in t.legs)


def test_identity_is_self_adjoint():
    idt = tenet.identity((Leg(V, OUT), Leg(W, OUT)))
    dag = idt.adjoint()
    # identity's public layout is (OUT..., IN...) and the dagger preserves public
    # axis order, so id† lists the same legs as (IN..., OUT...). Regrouping is the
    # free case A of transpose (see test_compose_reverses); after it, id† is id
    # exactly, block for block.
    assert dag.transpose(*dag.structure.out_axes, *dag.structure.in_axes) == idt


def test_compose_reverses():
    # a ∘ b needs a.domain == b.codomain, so b's OUT legs are a's IN legs (W, X)
    a = cx((Leg(V, OUT), Leg(W, IN), Leg(U, OUT), Leg(X, IN)), seed=3)
    b = cx((Leg(W, OUT), Leg(X, OUT), Leg(U, IN)), seed=4)
    lhs = tenet.compose(a, b).adjoint()
    # compose puts OUT legs first, the dagger keeps public order: the two sides
    # hold the same legs in a different interleaving. Regrouping them is case A of
    # transpose (both within-side orders are the identity), so it is free even for
    # SU(2) and carries no coefficient.
    lhs = lhs.transpose(*lhs.structure.out_axes, *lhs.structure.in_axes)
    assert tenet.allclose(lhs, tenet.compose(b.adjoint(), a.adjoint()))


# --- matrix and dense oracles ---------------------------------------------------


def test_matrix_level_dagger():
    t = cx(SU2_LEGS)
    got, want = to_matrices(t.adjoint()), to_matrices(t)
    assert set(got) == set(want)
    for c in got:
        np.testing.assert_allclose(got[c], want[c].conj().T, rtol=0.0, atol=1e-12)


def test_dense_oracle_is_elementwise_conjugation():
    t = cx(SU2_LEGS)
    d = t.adjoint()
    np.testing.assert_allclose(d.to_dense(), np.conj(t.to_dense()), rtol=0.0, atol=1e-12)
    # different axis groupings: T†'s rows are T's columns, so this is not the above
    np.testing.assert_allclose(mat(d), mat(t).conj().T, rtol=0.0, atol=1e-10)


def test_dense_is_skipped_for_dual_legs():
    # Milestone 4: to_dense needs the Z-isomorphism for dual=True legs, so the
    # dual tensor above is covered by the structural/algebraic criteria only.
    with pytest.raises(NotImplementedError):
        cx(DUAL_LEGS).adjoint().to_dense()


# --- backend agnosticism --------------------------------------------------------


@pytest.mark.parametrize("legs", ALL_LEGS)
def test_jax_involution_and_anti_linearity(legs):
    use_jax()
    a, b = 2.0 - 3.0j, -0.5 + 1.25j
    t, s = cx(legs, seed=0).to_backend("jax"), cx(legs, seed=1).to_backend("jax")
    assert t.adjoint().backend == "jax"
    assert t.adjoint().adjoint() == t
    lhs = (a * t + b * s).adjoint()
    rhs = np.conj(a) * t.adjoint() + np.conj(b) * s.adjoint()
    assert tenet.allclose(lhs, rhs, rtol=0.0, atol=1e-12)
