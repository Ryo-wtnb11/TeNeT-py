"""Tests for the truncated SVD and graded bond-space reconstruction — issue #64.

The discriminating criterion in this module is the ``qdim`` weight: a wrong weight
is invisible for U(1) and fZ2 (``qdim == 1``) and produces a perfectly valid
symmetric tensor under SU(2) that is simply not the best approximation. Every
selection assertion is therefore checked against an *independently written*
reference (:func:`reference_selection`, a direct transcription of the issue's eight
steps over ``numpy.linalg.svd``) and against the dense Eckart-Young error.
"""

import re

import numpy as np
import pytest
from test_linalg import (
    PROVIDERS,
    REPARTITION_MODULE,
    SPLIT,
    dense_matrix,
    tensor,
    to_jax,
    use_jax,
)

import tenet
from tenet import IN, OUT, GradedSpace, Leg, StructureChangingError, SymmetricTensor
from tenet.map_view import to_matrices
from tenet.symmetry import SU2, CapabilityError, SU2Sector

# --- an independent re-derivation of the specification -----------------------------


def raw_spectrum(t, axes=SPLIT):
    """``[(sigma, c, i)]`` descending by bare sigma — from NumPy, not from tenet."""
    mats = to_matrices(tenet.repartition(t, *axes))
    entries = [
        (float(s), c, i)
        for c, b in mats.items()
        for i, s in enumerate(np.linalg.svd(np.asarray(b), compute_uv=False))
    ]
    entries.sort(key=lambda e: (-e[0], e[1], e[2]))
    return entries


def reference_selection(t, axes=SPLIT, *, max_bond=None, cutoff=None, cutoff_mode="rsum2"):
    """The issue's steps 1-5, written out longhand. Returns the kept ``(c, i)`` list."""
    qdim = t.provider.qdim
    entries = raw_spectrum(t, axes)

    if cutoff is None:
        admissible = len(entries)
    elif cutoff_mode == "abs":
        admissible = sum(1 for s, _, _ in entries if s > cutoff)
    elif cutoff_mode == "rel":
        admissible = sum(1 for s, _, _ in entries if s > cutoff * entries[0][0])
    else:
        power = 2 if cutoff_mode in ("sum2", "rsum2") else 1
        weights = [qdim(c) * s**power for s, c, _ in entries]
        threshold = cutoff * sum(weights) if cutoff_mode.startswith("r") else cutoff
        admissible, dropped = len(entries), 0.0
        for w in reversed(weights):
            if dropped + w >= threshold:
                break
            dropped += w
            admissible -= 1

    kept, budget = [], 0.0
    for _s, c, i in entries[:admissible]:
        if max_bond is not None:
            budget += qdim(c)
            if budget > max_bond:
                break
        kept.append((c, i))
    return kept


def kept_pairs(bond):
    """``{(c, i)}`` for a truncated bond space: the kept indices are a prefix per sector."""
    return {(c, i) for c, k in bond.sectors for i in range(k)}


def dropped_weight(t, bond, axes=SPLIT):
    """``Sum_dropped qdim(c) sigma^2`` from the untruncated spectrum."""
    kept = kept_pairs(bond)
    return sum(t.provider.qdim(c) * s**2 for s, c, i in raw_spectrum(t, axes) if (c, i) not in kept)


def full_bond(name, axes=SPLIT):
    return tenet.linalg.svd(tensor(name), axes=axes)[0].legs[-1].space


# --- legs, the bond space and sector dropping ---------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_factor_legs_are_svds_apart_from_the_bond(name):
    t = tensor(name, seed=1)
    m = tenet.repartition(t, *SPLIT)
    u, s, vh = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=full_bond(name).dim // 2)
    bond = u.legs[-1].space
    assert u.legs == (*m.codomain, Leg(bond, IN))
    assert s.legs == (Leg(bond, OUT), Leg(bond, IN))
    assert vh.legs == (Leg(bond, OUT), *m.domain)
    assert u.legs[-1] == Leg(s.legs[0].space, IN)  # the identity mirror convention
    for got, want in zip(u.legs[:-1], m.codomain, strict=True):
        assert (got.space, got.dual, got.name) == (want.space, want.dual, want.name)
    for got, want in zip(vh.legs[1:], m.domain, strict=True):
        assert (got.space, got.dual, got.name) == (want.space, want.dual, want.name)


def test_the_bond_space_holds_exactly_the_sectors_with_a_kept_value():
    t = tensor("su2", seed=2)
    u, _, _ = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=12)
    bond = u.legs[-1].space
    kept = reference_selection(t, max_bond=12)
    want = {}
    for c, _ in kept:
        want[c] = want.get(c, 0) + 1
    assert bond.sectors == GradedSpace.new(t.provider, want).sectors
    assert all(m > 0 for _, m in bond.sectors)


def test_a_sector_disappears_from_the_truncated_bond_entirely():
    """The graded bond-space reconstruction bullet: fewer sectors, not zeroed ones."""
    t = tensor("su2", seed=2)
    untruncated = full_bond("su2")
    bond = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=6)[0].legs[-1].space
    gone = [c for c in untruncated if c not in bond]
    assert gone, "pick a tighter max_bond: no sector was dropped"
    for c in gone:
        assert c not in bond
        assert bond.degeneracy(c) == 0
    assert len(bond) < len(untruncated)


# --- max_bond is a dense budget ------------------------------------------------------


def test_max_bond_is_the_dense_dimension_not_the_reduced_one():
    t = tensor("su2", seed=3)
    max_bond = 11
    u, _, _ = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=max_bond)
    bond = u.legs[-1].space
    assert bond.dim <= max_bond
    # the two numbers are demonstrably different
    assert bond.reduced_dim < max_bond
    assert bond.reduced_dim < bond.dim

    # stop-on-overflow: the first rejected value could not have fitted
    entries = raw_spectrum(t)
    kept = kept_pairs(bond)
    rejected = [(s, c, i) for s, c, i in entries if (c, i) not in kept]
    assert bond.dim + SU2.qdim(rejected[0][1]) > max_bond


def test_max_bond_undershoots_by_less_than_the_largest_qdim():
    t = tensor("su2", seed=3)
    for max_bond in range(2, 30):
        bond = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=max_bond)[0].legs[-1].space
        if bond.dim < full_bond("su2").dim:
            assert max_bond - bond.dim < max(SU2.qdim(c) for c in full_bond("su2"))


def test_stop_on_overflow_rejects_a_qdim_3_value_a_later_qdim_1_value_would_have_fitted():
    """The consequence of the binding stop-on-overflow decision, made visible.

    The issue's own wording for this criterion ("the greedy keeps a qdim=1 sigma that
    is *smaller* than a rejected qdim=3 one") describes the keep-scanning rule the
    Key decisions explicitly reject: under stop-on-overflow the kept set is always a
    prefix of the bare-descending order, so nothing kept can be smaller than anything
    rejected. What *is* observable, and what discriminates this rule from unweighted
    counting, is asserted here: the walk stops at a ``qdim=3`` value whose cost
    overflows the budget while a *smaller*, cheaper ``qdim=1`` value further down the
    list would have fitted and is deliberately not taken.
    """
    t = tensor("su2", seed=3)
    entries = raw_spectrum(t)
    found = None
    for max_bond in range(2, 40):
        bond = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=max_bond)[0].legs[-1].space
        kept = kept_pairs(bond)
        rejected = [(s, c, i) for s, c, i in entries if (c, i) not in kept]
        if not rejected:
            continue
        blocker = rejected[0]
        room = max_bond - bond.dim
        if SU2.qdim(blocker[1]) == 3 and any(SU2.qdim(c) == 1 for _, c, _ in rejected[1:]):
            cheaper = next((s, c, i) for s, c, i in rejected[1:] if SU2.qdim(c) == 1)
            if room >= 1:
                assert cheaper[0] < blocker[0]  # smaller, cheap, and still not taken
                assert (cheaper[1], cheaper[2]) not in kept
                found = max_bond
                break
    assert found is not None, "no budget on this fixture exercises the overflow stop"


def test_the_kept_set_equals_the_independent_greedy_reference():
    t = tensor("su2", seed=3)
    for max_bond in (3, 5, 8, 13, 21):
        bond = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=max_bond)[0].legs[-1].space
        assert kept_pairs(bond) == set(reference_selection(t, max_bond=max_bond))


def test_nesting_across_budgets():
    t = tensor("su2", seed=3)
    sets = []
    for max_bond in range(2, 30):
        bond = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=max_bond)[0].legs[-1].space
        sets.append(kept_pairs(bond))
    assert len(set(map(len, sets))) > 1  # the sweep is not vacuous
    for small, large in zip(sets, sets[1:], strict=False):
        assert small <= large


# --- the dense oracle: Eckart-Young ---------------------------------------------------


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_dense_oracle_eckart_young(name):
    """``axes=None``: no bend, so no dual leg enters the dense expansion (#57's rule)."""
    t = tensor(name, seed=4)
    axes = (t.structure.out_axes, t.structure.in_axes)
    full = tenet.linalg.svd(t)[0].legs[-1].space
    u, s, vh = tenet.linalg.svd_truncated(t, max_bond=full.dim // 2)
    bond = u.legs[-1].space

    error = float(np.linalg.norm(dense_matrix(t) - dense_matrix(u @ s @ vh)) ** 2)
    assert error == pytest.approx(dropped_weight(t, bond, axes), abs=1e-12)

    dense_sigma = np.sort(np.linalg.svd(dense_matrix(t), compute_uv=False))[::-1]
    eckart_young = float(np.sum(dense_sigma[bond.dim :] ** 2))
    assert error == pytest.approx(eckart_young, abs=1e-12)


def test_an_unweighted_selection_fails_eckart_young_on_su2():
    """The criterion is discriminating: a qdim-weighted *sort key* is not optimal.

    Any prefix of the bare-descending order is Eckart-Young optimal at its own dense
    rank, so the selection that can fail this test is one whose ordering is wrong —
    here the plausible mistake of ranking by the retained weight ``qdim(c)*sigma^2``
    instead of by the bare ``sigma``. It keeps a different set at the same dense rank
    and loses more.
    """
    t = tensor("su2", seed=4)
    entries = raw_spectrum(t, (t.structure.out_axes, t.structure.in_axes))
    u, _, _ = tenet.linalg.svd_truncated(t, max_bond=tenet.linalg.svd(t)[0].legs[-1].space.dim // 2)
    bond = u.legs[-1].space

    wrong_order = sorted(entries, key=lambda e: (-SU2.qdim(e[1]) * e[0] ** 2, e[1], e[2]))
    wrong, budget = [], 0.0
    for _s, c, i in wrong_order:
        budget += SU2.qdim(c)
        if budget > bond.dim:
            break
        wrong.append((c, i))
    assert set(wrong) != kept_pairs(bond)

    dropped = sum(SU2.qdim(c) * s**2 for s, c, i in entries if (c, i) not in set(wrong))
    dense_sigma = np.sort(np.linalg.svd(dense_matrix(t), compute_uv=False))[::-1]
    eckart_young = float(np.sum(dense_sigma[bond.dim :] ** 2))
    assert dropped > eckart_young + 1e-9


@pytest.mark.parametrize("name", PROVIDERS)
def test_pythagoras_is_the_truncation_error(name):
    t = tensor(name, seed=5)
    u, s, vh = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=full_bond(name).dim // 2)
    bond = u.legs[-1].space
    error = float(tenet.norm(t)) ** 2 - float(tenet.norm(u @ s @ vh)) ** 2
    assert error == pytest.approx(dropped_weight(t, bond), abs=1e-12)


# --- cutoff ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["abs", "rel"])
def test_bare_cutoff_keeps_exactly_the_values_above_the_threshold(mode):
    t = tensor("su2", seed=6)
    entries = raw_spectrum(t)
    cutoff = 0.4 if mode == "rel" else 0.4 * entries[0][0]
    bond = (
        tenet.linalg.svd_truncated(t, axes=SPLIT, cutoff=cutoff, cutoff_mode=mode)[0].legs[-1].space
    )
    threshold = cutoff * entries[0][0] if mode == "rel" else cutoff
    assert kept_pairs(bond) == {(c, i) for s, c, i in entries if s > threshold}
    assert 0 < len(kept_pairs(bond)) < len(entries)  # the threshold bites, both ways


@pytest.mark.parametrize("mode", ["sum2", "rsum2", "sum1", "rsum1"])
def test_cumulative_cutoff_is_admissible_and_maximal(mode):
    t = tensor("su2", seed=6)
    entries = raw_spectrum(t)
    power = 2 if mode in ("sum2", "rsum2") else 1
    weights = {(c, i): SU2.qdim(c) * s**power for s, c, i in entries}
    if mode.startswith("r"):
        cutoff, threshold = 0.05, 0.05 * sum(weights.values())
    else:
        cutoff = threshold = 0.05 * sum(weights.values())

    bond = (
        tenet.linalg.svd_truncated(t, axes=SPLIT, cutoff=cutoff, cutoff_mode=mode)[0].legs[-1].space
    )
    kept = kept_pairs(bond)
    assert kept == set(reference_selection(t, cutoff=cutoff, cutoff_mode=mode))
    dropped = sum(w for k, w in weights.items() if k not in kept)
    assert dropped < threshold  # admissible
    smallest_kept = min((s, c, i) for s, c, i in entries if (c, i) in kept)
    assert dropped + weights[(smallest_kept[1], smallest_kept[2])] >= threshold  # maximal
    assert 0 < len(kept) < len(entries)


@pytest.mark.parametrize(("mode", "cutoff"), [("rsum2", 0.02), ("rel", 0.4), ("rsum1", 0.05)])
def test_relative_modes_are_scale_invariant(mode, cutoff):
    """The dimensional-consistency check: threshold and accumulator share a power."""
    t = tensor("su2", seed=6)
    one = (
        tenet.linalg.svd_truncated(t, axes=SPLIT, cutoff=cutoff, cutoff_mode=mode)[0].legs[-1].space
    )
    two = (
        tenet.linalg.svd_truncated(
            tenet.multiply(t, 2.0), axes=SPLIT, cutoff=cutoff, cutoff_mode=mode
        )[0]
        .legs[-1]
        .space
    )
    assert one == two
    assert len(kept_pairs(one)) < len(raw_spectrum(t))  # the cutoff actually truncated


@pytest.mark.parametrize("max_bond", [4, 9, 16, 40])
@pytest.mark.parametrize("cutoff", [1e-8, 0.02, 0.2])
def test_max_bond_and_cutoff_take_the_intersection(max_bond, cutoff):
    t = tensor("su2", seed=6)
    both = (
        tenet.linalg.svd_truncated(
            t, axes=SPLIT, max_bond=max_bond, cutoff=cutoff, cutoff_mode="rsum2"
        )[0]
        .legs[-1]
        .space
    )
    by_bond = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=max_bond)[0].legs[-1].space
    by_cutoff = (
        tenet.linalg.svd_truncated(t, axes=SPLIT, cutoff=cutoff, cutoff_mode="rsum2")[0]
        .legs[-1]
        .space
    )
    stricter = by_bond if by_bond.reduced_dim <= by_cutoff.reduced_dim else by_cutoff
    assert both == stricter
    assert kept_pairs(both) == kept_pairs(by_bond) & kept_pairs(by_cutoff)


# --- renorm ----------------------------------------------------------------------------


def test_renorm_preserves_the_norm_and_nothing_else():
    t = tensor("su2", seed=7)
    plain = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=9)
    scaled = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=9, renorm=True)
    assert float(tenet.norm(t)) ** 2 > float(tenet.norm(plain[0] @ plain[1] @ plain[2])) ** 2
    assert float(tenet.norm(scaled[0] @ scaled[1] @ scaled[2])) == pytest.approx(
        float(tenet.norm(t)), abs=1e-12
    )
    assert plain[0].legs[-1].space == scaled[0].legs[-1].space
    assert kept_pairs(plain[0].legs[-1].space) == kept_pairs(scaled[0].legs[-1].space)
    assert tenet.allclose(plain[0], scaled[0], rtol=0, atol=1e-12)  # only S is rescaled
    assert tenet.allclose(plain[2], scaled[2], rtol=0, atol=1e-12)


def test_renorm_must_be_a_bool():
    with pytest.raises(TypeError, match="p-norm"):
        tenet.linalg.svd_truncated(tensor("u1"), axes=SPLIT, max_bond=4, renorm=2)


# --- structure preserved through truncation ----------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_per_sector_isometry_survives_truncation(name):
    t = tensor(name, seed=8)
    u, _, vh = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=full_bond(name).dim // 2)
    bond = tenet.identity((Leg(u.legs[-1].space, OUT),))
    assert tenet.allclose(tenet.adjoint(u) @ u, bond, rtol=0, atol=1e-12)
    bond = tenet.identity((Leg(vh.legs[0].space, OUT),))
    assert tenet.allclose(vh @ tenet.adjoint(vh), bond, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", PROVIDERS)
def test_s_is_still_diagonal_real_non_negative_and_descending(name):
    t = tensor(name, seed=8)
    _, s, _ = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=full_bond(name).dim // 2)
    assert not np.issubdtype(np.asarray(s.blocks[0]).dtype, np.complexfloating)
    for c, mat in to_matrices(s).items():
        mat = np.asarray(mat)
        sigma = np.diagonal(mat)
        np.testing.assert_allclose(mat - np.diag(sigma), 0.0, atol=1e-14, err_msg=repr(c))
        assert np.all(sigma >= 0.0)
        assert np.all(np.diff(sigma) <= 1e-14)


def test_fz2_truncated_round_trip_across_odd_legs():
    t = tensor("fz2", seed=9)
    full = full_bond("fz2")
    u, s, vh = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=full.dim // 2)
    bond = u.legs[-1].space
    assert bond.dim < full.dim

    identity = tenet.identity((Leg(bond, OUT),))
    assert tenet.allclose(tenet.adjoint(u) @ u, identity, rtol=0, atol=1e-12)
    assert tenet.allclose(vh @ tenet.adjoint(vh), identity, rtol=0, atol=1e-12)

    back = tenet.repartition(u @ s @ vh, (0, 2), (1, 3)).transpose((0, 2, 1, 3))
    assert back.legs == t.legs
    error = float(tenet.norm(t)) ** 2 - float(tenet.norm(back)) ** 2
    assert error == pytest.approx(dropped_weight(t, bond), abs=1e-12)

    # and with a budget that truncates nothing, the whole round trip is exact
    u, s, vh = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=full.dim)
    back = tenet.repartition(u @ s @ vh, (0, 2), (1, 3)).transpose((0, 2, 1, 3))
    assert tenet.allclose(back, t, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", PROVIDERS)
def test_untruncated_equivalence(name):
    t = tensor(name, seed=10)
    want_u, want_s, want_vh = tenet.linalg.svd(t, axes=SPLIT)
    bond = want_u.legs[-1].space
    for kwargs in ({"max_bond": bond.dim}, {"cutoff": 0.0, "cutoff_mode": "abs"}):
        u, s, vh = tenet.linalg.svd_truncated(t, axes=SPLIT, **kwargs)
        assert u.legs[-1].space == bond, kwargs
        for got, want in zip((u, s, vh), (want_u, want_s, want_vh), strict=True):
            assert tenet.allclose(got, want, rtol=0, atol=1e-12)


# --- the map-style spelling ---------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_as_map_svd_truncated_matches_the_axes_spelling_and_bends_nothing(name, monkeypatch):
    t = tensor(name, seed=11)
    axes = (t.structure.out_axes, t.structure.in_axes)
    max_bond = tenet.linalg.svd(t)[0].legs[-1].space.dim // 2

    def no_bends(*args, **kwargs):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("as_map().svd_truncated() must perform zero bends")

    monkeypatch.setattr(REPARTITION_MODULE, "bend", no_bends)
    view = t.as_map().svd_truncated(max_bond=max_bond)
    want = tenet.linalg.svd_truncated(t, axes=axes, max_bond=max_bond)
    for got, expected in zip(view, want, strict=True):
        assert got.structure == expected.structure
        assert tenet.allclose(got, expected, rtol=0, atol=1e-14)


# --- refusals -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "tenet.linalg.svd"),
        ({"max_bond": 0}, "positive"),
        ({"max_bond": -3}, "positive"),
        ({"cutoff": -1e-9}, "non-negative"),
        ({"max_bond": 4, "cutoff_mode": "nope"}, "rsum2"),
        ({"max_bond": 4, "cutoff_mode": 4}, "strings"),
    ],
    ids=["neither", "zero-bond", "negative-bond", "negative-cutoff", "bad-mode", "int-mode"],
)
def test_argument_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        tenet.linalg.svd_truncated(tensor("u1"), axes=SPLIT, **kwargs)


def test_a_cutoff_that_drops_everything_is_a_value_error():
    t = tensor("u1", seed=12)
    largest = raw_spectrum(t)[0][0]
    with pytest.raises(ValueError) as excinfo:
        tenet.linalg.svd_truncated(t, axes=SPLIT, cutoff=10 * largest, cutoff_mode="abs")
    message = str(excinfo.value)
    assert "abs" in message
    assert repr(10 * largest) in message
    reported = re.search(r"largest available is ([0-9.e+-]+)", message)
    assert reported is not None
    assert float(reported.group(1)) == pytest.approx(largest, rel=1e-12)


def test_a_crossing_split_is_refused_without_bending_coefficients():
    from test_linalg import capability_less

    with pytest.raises(CapabilityError) as excinfo:
        tenet.linalg.svd_truncated(capability_less(), axes=SPLIT, max_bond=2)
    message = str(excinfo.value)
    assert "BendingCoefficients" in message
    assert "axis 1" in message


def test_ar_do_svd_truncated_still_raises():
    import autoray as ar

    with pytest.raises(Exception):  # noqa: B017  the exception type is autoray's business
        ar.do("svd_truncated", tensor("su2"))
    assert not hasattr(tenet, "svd_truncated")


# --- the structure-changing boundary ------------------------------------------------------------


def test_structure_changing_error_is_exported_and_is_a_type_error():
    assert issubclass(StructureChangingError, TypeError)
    assert tenet.StructureChangingError is StructureChangingError


def test_jit_refuses_and_says_why():
    jax = use_jax()
    t = to_jax(tensor("su2", seed=13))
    with pytest.raises(StructureChangingError) as excinfo:
        jax.jit(lambda x: tenet.linalg.svd_truncated(x, axes=SPLIT, max_bond=8)[0])(t)
    message = str(excinfo.value)
    assert "svd_truncated" in message
    assert "depend on the block values" in message
    assert "tenet.linalg.svd" in message
    assert "outside the traced region" in message


def test_grad_refuses_too():
    jax = use_jax()
    t = to_jax(tensor("su2", seed=13))

    def scalar(x):
        u, s, vh = tenet.linalg.svd_truncated(x, axes=SPLIT, max_bond=8)
        return tenet.norm(u @ s @ vh)

    with pytest.raises(StructureChangingError, match="svd_truncated"):
        jax.grad(scalar)(t)


def test_svd_is_still_jittable_right_next_to_the_refusal():
    """The boundary, both sides of it, in one screen (the issue's literal ask)."""
    jax = use_jax()
    t = to_jax(tensor("su2", seed=13))
    u = jax.jit(lambda x: tenet.linalg.svd(x, axes=SPLIT)[0])(t)
    assert isinstance(u, SymmetricTensor)
    assert u.backend == "jax"


def test_eager_jax_matches_the_numpy_backend():
    """The refusal is about *tracing*, not about JAX."""
    use_jax()
    t = tensor("su2", seed=13)
    got = tenet.linalg.svd_truncated(to_jax(t), axes=SPLIT, max_bond=9)
    want = tenet.linalg.svd_truncated(t, axes=SPLIT, max_bond=9)
    assert got[0].backend == "jax"
    assert got[0].legs[-1].space == want[0].legs[-1].space
    for a, b in zip(got, want, strict=True):
        assert a.structure == b.structure
    assert tenet.allclose(got[1], want[1], rtol=0, atol=1e-10)  # gauge-invariant
    assert tenet.allclose(got[0] @ got[1] @ got[2], want[0] @ want[1] @ want[2], rtol=0, atol=1e-10)


def test_sector_labels_are_su2_in_this_module():
    """Guards the fixture: the qdim weight is only non-vacuous with distinct qdims."""
    assert len({SU2.qdim(c) for c in full_bond("su2")}) > 1
    assert any(isinstance(c, SU2Sector) and SU2.qdim(c) == 3 for c in full_bond("su2"))
