"""The truncation decision as a returned object — issue #209 (M44).

Every criterion of the issue gets one test that fails when it is not met. The
load-bearing one is :func:`test_two_call_form_is_the_one_call_form`: ``select_bond``
plus ``svd(..., bond=)`` must be the *same operator* as ``svd_truncated``, factor for
factor, or the split has bought a second keep rule instead of exposing the one.

The SU(2) multiplet-boundary case is constructed rather than hoped for
(:func:`test_su2_max_bond_landing_inside_a_multiplet_is_reported`): the fixture spaces
carry only ``j=0`` (``qdim 1``) and ``j=1`` (``qdim 3``) so an odd ``max_bond`` cannot
be met exactly by any set of triplets, and the undershoot is arithmetic, not luck.
"""

import inspect

import numpy as np
import pytest
from test_linalg import PROVIDERS, SPLIT, singular_values, tensor, to_jax, use_jax

import tenet
from tenet import IN, OUT, GradedSpace, Leg, StructureChangingError, SymmetricTensor
from tenet.ops.linalg import BondSelection, _keep_counts, _lower
from tenet.symmetry import SU2, CapabilityError, SU2Sector

CASES = [(name, dict(max_bond=6)) for name in PROVIDERS] + [
    ("su2", dict(cutoff=1e-2, cutoff_mode="rsum2")),
    ("u1", dict(cutoff=0.1, cutoff_mode="rel")),
    ("fz2", dict(max_bond=4, cutoff=1e-8, cutoff_mode="abs")),
]


# --- the object -------------------------------------------------------------------


def test_select_bond_is_public_and_documented():
    assert "select_bond" in tenet.linalg.__all__
    assert "BondSelection" in tenet.linalg.__all__
    assert tenet.linalg.select_bond.__doc__ and tenet.linalg.BondSelection.__doc__


def test_the_selection_is_immutable_and_is_not_a_pytree():
    """Frozen beside ``MapLayout``, and ``tenet.pytree`` registers nothing for it —
    the record stays outside the trace and only ``.bond`` crosses in."""
    jax = use_jax()
    selection = tenet.linalg.select_bond(tensor("su2"), SPLIT, max_bond=6)
    with pytest.raises(Exception, match="assign|immutable|frozen"):
        selection.bond = None  # ty: ignore[invalid-assignment]
    # a registered container would flatten into its Python floats; a leaf is the whole
    # object. Either way it must not become a *container* by accident.
    leaves = jax.tree_util.tree_leaves(selection)
    assert leaves == [selection]


@pytest.mark.parametrize(("name", "kw"), CASES)
def test_the_selection_reports_its_own_dimensions(name, kw):
    t = tensor(name)
    selection = tenet.linalg.select_bond(t, SPLIT, **kw)
    qdim = t.provider.qdim
    assert isinstance(selection, BondSelection)
    assert selection.reduced_dim == selection.bond.reduced_dim == len(selection.kept)
    assert selection.dense_dim == pytest.approx(sum(qdim(c) * m for c, m in selection.bond.sectors))
    if kw.get("max_bond") is not None:
        assert selection.dense_dim <= kw["max_bond"]
        assert selection.undershoot == pytest.approx(kw["max_bond"] - selection.dense_dim)
    else:
        assert selection.undershoot is None


# --- the two-call form is the one-call form ---------------------------------------


@pytest.mark.parametrize(("name", "kw"), CASES)
def test_two_call_form_is_the_one_call_form(name, kw):
    """``svd(t, axes, bond=selection.bond)`` reproduces ``svd_truncated`` exactly."""
    t = tensor(name)
    selection = tenet.linalg.select_bond(t, SPLIT, **kw)
    u0, s0, vh0 = tenet.linalg.svd_truncated(t, SPLIT, **kw)
    u1, s1, vh1 = tenet.linalg.svd(t, SPLIT, bond=selection.bond)

    assert s1.structure == s0.structure
    for a, b in ((u0, u1), (s0, s1), (vh0, vh1)):
        assert a.structure == b.structure
        for x, y in zip(a.blocks, b.blocks, strict=True):
            assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("name", PROVIDERS)
def test_renorm_is_reported_rather_than_applied(name):
    """The record is bare; ``scale`` is the one place the rescaling lives, and it is
    exactly the factor ``svd_truncated(renorm=True)`` multiplies in."""
    t = tensor(name)
    selection = tenet.linalg.select_bond(t, SPLIT, max_bond=6, renorm=True)
    plain = tenet.linalg.select_bond(t, SPLIT, max_bond=6)
    assert plain.scale == 1.0
    assert selection.kept == plain.kept  # bare either way
    _, s, _ = tenet.linalg.svd_truncated(t, SPLIT, max_bond=6, renorm=True)
    scaled = sorted(float(v) for sigma in singular_values(s).values() for v in np.atleast_1d(sigma))
    expected = sorted(sigma * selection.scale for sigma, _, _ in selection.kept)
    assert scaled == pytest.approx(expected)


# --- the discarded half -----------------------------------------------------------


@pytest.mark.parametrize(("name", "kw"), CASES)
def test_kept_and_discarded_partition_the_spectrum(name, kw):
    t = tensor(name)
    selection = tenet.linalg.select_bond(t, SPLIT, **kw)
    mats = tenet.to_matrices(tenet.repartition(t, *SPLIT))
    everything = sorted(
        float(sigma)
        for b in mats.values()
        for sigma in np.linalg.svd(np.asarray(b), compute_uv=False)
    )
    got = sorted(sigma for sigma, _, _ in (*selection.kept, *selection.discarded))
    assert got == pytest.approx(everything)
    # kept is above discarded, entry by entry: one global spectrum, descending
    if selection.discarded:
        assert min(s for s, _, _ in selection.kept) >= max(s for s, _, _ in selection.discarded)


@pytest.mark.parametrize(("name", "kw"), CASES)
def test_discarded_weight_matches_an_independent_sum(name, kw):
    t = tensor(name)
    selection = tenet.linalg.select_bond(t, SPLIT, **kw)
    qdim = t.provider.qdim
    independent = sum(qdim(c) * sigma**2 for sigma, c, _ in selection.discarded)
    assert selection.discarded_weight == pytest.approx(independent)
    # and it is Pythagoras against the factorization it decided
    u, s, vh = tenet.linalg.svd(t, SPLIT, bond=selection.bond)
    pythagoras = float(tenet.norm(t)) ** 2 - float(tenet.norm(u @ s @ vh)) ** 2
    assert selection.discarded_weight == pytest.approx(pythagoras, abs=1e-10)


@pytest.mark.parametrize(("name", "kw"), CASES)
def test_kept_multiplicity_per_sector_agrees_with_keep_counts(name, kw):
    t = tensor(name)
    selection = tenet.linalg.select_bond(t, SPLIT, **kw)
    _, untruncated, _ = _lower(t, SPLIT)
    counts: dict = {}
    for _, c, _ in selection.kept:
        counts[c] = counts.get(c, 0) + 1
    assert _keep_counts(selection.bond, untruncated) == counts


def test_the_next_multiplet_is_the_largest_discarded_one():
    t = tensor("su2")
    selection = tenet.linalg.select_bond(t, SPLIT, max_bond=6)
    assert selection.next_multiplet == selection.discarded[0]
    assert selection.next_dense_cost == t.provider.qdim(selection.discarded[0][1])


def test_nothing_discarded_means_no_next_multiplet():
    t = tensor("u1")
    _, untruncated, _ = _lower(t, SPLIT)
    selection = tenet.linalg.select_bond(t, SPLIT, max_bond=10**6)
    assert selection.discarded == ()
    assert selection.next_multiplet is None
    assert selection.next_dense_cost == 0.0
    assert selection.bond == untruncated


# --- the non-Abelian multiplet boundary, constructed ------------------------------


def _su2_triplets_only(seed: int = 2) -> SymmetricTensor:
    """A square SU(2) map whose bond carries only ``j=1`` (``qdim 3``) and one ``j=0``.

    Every ``j=1`` multiplet costs 3 dense dimensions and the single ``j=0`` costs 1, so
    ``max_bond=5`` can be met only as ``1 + 3`` — the walk must undershoot by 1 unless
    the singlet happens to be admitted first.
    """
    space = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(2): 3})
    return SymmetricTensor.random((Leg(space, OUT), Leg(space, IN)), seed=seed)


def test_su2_max_bond_landing_inside_a_multiplet_is_reported():
    t = _su2_triplets_only()
    selection = tenet.linalg.select_bond(t, max_bond=5)

    # the budget cannot be spent exactly: 5 = 1 + 3 + (1 short of the next triplet)
    assert selection.dense_dim < 5
    assert selection.undershoot == pytest.approx(5 - selection.dense_dim)
    assert selection.undershoot > 0

    # and the record names what the leftover budget did not buy
    assert selection.next_multiplet is not None
    _, sector, _ = selection.next_multiplet
    assert selection.next_dense_cost == t.provider.qdim(sector)
    assert selection.dense_dim + selection.next_dense_cost > 5  # why it stopped

    # the undershoot is bounded by max qdim - 1, as svd_truncated documents
    assert selection.undershoot <= max(t.provider.qdim(c) for c in selection.bond) - 1


def test_the_su2_undershoot_is_the_number_svd_truncated_produced_all_along():
    """The report is new; the behaviour is not. Same bond, both routes."""
    t = _su2_triplets_only()
    selection = tenet.linalg.select_bond(t, max_bond=5)
    _, s, _ = tenet.linalg.svd_truncated(t, max_bond=5)
    assert s.structure.legs[0].space == selection.bond


# --- refusals ---------------------------------------------------------------------


def test_jit_refuses_with_svd_truncateds_message_discipline():
    jax = use_jax()
    t = to_jax(tensor("su2", seed=13))
    with pytest.raises(StructureChangingError) as excinfo:
        jax.jit(lambda x: tenet.linalg.select_bond(x, SPLIT, max_bond=8).dense_dim)(t)
    message = str(excinfo.value)
    assert "select_bond" in message
    assert "depend on the block values" in message
    assert "tenet.linalg.svd" in message
    assert "outside the traced region" in message


def test_grad_refuses_too():
    jax = use_jax()
    t = to_jax(tensor("su2", seed=13))

    def scalar(x):
        bond = tenet.linalg.select_bond(x, SPLIT, max_bond=8).bond
        u, s, vh = tenet.linalg.svd(x, SPLIT, bond=bond)
        return tenet.norm(u @ s @ vh)

    with pytest.raises(StructureChangingError, match="select_bond"):
        jax.grad(scalar)(t)


def test_argument_refusals_name_select_bond():
    t = tensor("u1")
    with pytest.raises(ValueError, match="select_bond needs at least one"):
        tenet.linalg.select_bond(t, SPLIT)
    with pytest.raises(ValueError, match="max_bond must be a positive"):
        tenet.linalg.select_bond(t, SPLIT, max_bond=0)
    with pytest.raises(ValueError, match="cutoff must be non-negative"):
        tenet.linalg.select_bond(t, SPLIT, cutoff=-1.0)
    with pytest.raises(ValueError, match="unknown cutoff_mode"):
        tenet.linalg.select_bond(t, SPLIT, cutoff=1e-9, cutoff_mode="nope")
    with pytest.raises(TypeError, match="renorm must be a bool"):
        tenet.linalg.select_bond(t, SPLIT, max_bond=2, renorm=1)


def test_keeping_nothing_is_refused_naming_select_bond():
    t = tensor("u1")
    with pytest.raises(ValueError, match="select_bond: cutoff="):
        tenet.linalg.select_bond(t, SPLIT, cutoff=10.0, cutoff_mode="abs")


def test_the_lowerings_capability_refusal_is_inherited_whole():
    """``select_bond`` lowers exactly as ``svd_truncated`` does, so it refuses exactly
    where ``svd_truncated`` refuses — same message, same axis."""
    from test_linalg import capability_less  # noqa: PLC0415

    with pytest.raises(CapabilityError) as excinfo:
        tenet.linalg.select_bond(capability_less(), axes=SPLIT, max_bond=2)
    message = str(excinfo.value)
    assert "BendingCoefficients" in message
    assert "axis 1" in message


# --- svd_truncated's surface is unchanged -----------------------------------------


def _shape(fn):
    """``[(name, kind, default)]`` — the part of a signature callers can break."""
    return [(p.name, p.kind.name, p.default) for p in inspect.signature(fn).parameters.values()]


def test_svd_truncated_keeps_its_signature():
    """#209 reimplements the body on the shared rule and touches nothing public."""
    assert _shape(tenet.linalg.svd_truncated) == [
        ("t", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("axes", "POSITIONAL_OR_KEYWORD", None),
        ("max_bond", "KEYWORD_ONLY", None),
        ("cutoff", "KEYWORD_ONLY", None),
        ("cutoff_mode", "KEYWORD_ONLY", "rsum2"),
        ("renorm", "KEYWORD_ONLY", False),
    ]


def test_svd_keeps_its_signature_too():
    assert _shape(tenet.linalg.svd) == [
        ("t", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("axes", "POSITIONAL_OR_KEYWORD", None),
        ("bond", "KEYWORD_ONLY", None),
    ]
