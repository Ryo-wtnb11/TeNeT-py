"""Tests for explicit leg fusion and unfusion — issue #22.

The load-bearing test is the **dense isometry oracle**: fused dense data is
*not* reshaped dense data. The oracle builds the map explicitly from
``provider.cgc`` and the ``FusionLayout`` (a CG isometry for SU(2), a
permutation for U(1)), and the plain-reshape comparison is asserted to *fail*
so the check cannot degenerate into a tautology.
"""

import dataclasses
import enum
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.ops import fusion
from tenet.ops.fusion import FusionLayout, FusionPlan, FusionStep, fuse_spaces, fusion_plan
from tenet.symmetry import SU2, U1, SU2Sector, Trivial, TrivialSector, U1Sector

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
A = GradedSpace.new(SU2, {HALF: 2, ONE: 1})
B = GradedSpace.new(SU2, {HALF: 1, ONE: 2})
C = GradedSpace.new(SU2, {ZERO: 1, HALF: 2})

SU2_LEGS = (Leg(A, OUT), Leg(B, OUT), Leg(C, IN))
SU2_INTERLEAVED = (Leg(A, OUT), Leg(C, IN), Leg(B, OUT))
SU2_WIDE = (Leg(A, OUT), Leg(B, OUT), Leg(A, OUT), Leg(C, IN))

Q1 = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 1, U1Sector(1): 2})
Q2 = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(0): 3, U1Sector(1): 1})
U1_LEGS = (Leg(Q1, OUT), Leg(Q2, OUT), Leg(Q1, IN))

T3 = GradedSpace.new(Trivial, {TrivialSector(): 3})
T2 = GradedSpace.new(Trivial, {TrivialSector(): 2})
TRIV_LEGS = (Leg(T3, OUT), Leg(T2, OUT), Leg(T3, IN))

ALL_LEGS = {"su2": SU2_LEGS, "u1": U1_LEGS, "trivial": TRIV_LEGS}


def use_jax():
    jax = pytest.importorskip("jax")
    # ponytail: global x64 so float64 survives the move and the 1e-12 tolerances
    # in these tests mean something. Drop it if a test ever needs f32 semantics.
    jax.config.update("jax_enable_x64", True)
    return jax


# --- the dense oracle -----------------------------------------------------------


def isometry(a: Leg, b: Leg) -> np.ndarray:
    """``M[I, i, j]``: the explicit dense fusion map, built from ``cgc`` + layout.

    Deliberately written out index by index rather than derived from anything in
    ``tenet.ops.fusion`` — it is the independent check, not a mirror.
    """
    leg, layout = fuse_spaces(a, b)
    provider = a.provider
    m = np.zeros((leg.space.dim, a.space.dim, b.space.dim))
    for c, u_a, u_b, mu, offset, _ in layout.triples:
        sa, sb = a.space_sector(u_a), b.space_sector(u_b)
        m_a, m_b = a.degeneracy(sa), b.degeneracy(sb)
        d_a, d_b, d_c = (provider.irrep_dim(x) for x in (u_a, u_b, c))
        cg = provider.cgc(u_a, u_b, c)[..., mu]  # (d_a, d_b, d_c)
        oa, ob = a.space.sector_offset(sa), b.space.sector_offset(sb)
        oc = leg.space.sector_offset(c)
        for alpha_a in range(m_a):
            for alpha_b in range(m_b):
                # the fused degeneracy index: chunk offset + C-order pair index
                i = oc + (offset + alpha_a * m_b + alpha_b) * d_c
                m[
                    i : i + d_c,
                    oa + alpha_a * d_a : oa + (alpha_a + 1) * d_a,
                    ob + alpha_b * d_b : ob + (alpha_b + 1) * d_b,
                ] = cg.transpose(2, 0, 1)
    return m


def oracle(t: SymmetricTensor, axes) -> np.ndarray:
    """``to_dense(t)`` pushed through the isometry, one binary step at a time."""
    dense = t.to_dense()
    legs = list(t.legs)
    axes = sorted(axes)
    p = axes[0]
    for i, ax in enumerate(axes[1:]):
        q = ax - i  # every earlier step deleted one axis between p and ax
        n = len(legs)
        d = np.tensordot(isometry(legs[p], legs[q]), dense, axes=([1, 2], [p, q]))
        dense = d.transpose([*range(1, p + 1), 0, *range(p + 1, n - 1)])
        legs = [
            fuse_spaces(legs[p], legs[q])[0] if j == p else leg
            for j, leg in enumerate(legs)
            if j != q
        ]
    return dense


# --- API shape ------------------------------------------------------------------


def test_spellings_agree_and_the_fused_leg_lands_at_min_axes():
    t = SymmetricTensor.random(SU2_LEGS, seed=0)
    got = t.fuse((0, 1))
    assert got == t.fuse(0, 1)
    assert got == tenet.fuse(t, (0, 1))
    assert got == tenet.fuse(t, [0, 1])
    assert got.ndim == t.ndim - 1
    assert got.legs[0] == fuse_spaces(t.legs[0], t.legs[1])[0]
    assert got.legs[1] == t.legs[2]  # survivors keep their relative order


def test_fused_leg_is_non_dual_and_keeps_the_common_side():
    t = SymmetricTensor.random(SU2_LEGS, seed=1)
    leg = t.fuse((0, 1)).legs[0]
    assert leg.dual is False
    assert leg.side is OUT
    assert leg.name is None
    u = SymmetricTensor.random((Leg(A, IN), Leg(B, IN), Leg(C, OUT)), seed=1).fuse((0, 1))
    assert u.legs[0].side is IN and u.legs[0].dual is False


def test_non_adjacent_public_axes_are_allowed_and_land_at_min():
    t = SymmetricTensor.random(SU2_INTERLEAVED, seed=2)
    got = t.fuse((0, 2))
    assert got.ndim == 2
    assert got.legs[0] == fuse_spaces(t.legs[0], t.legs[2])[0]
    assert got.legs[1] == t.legs[1]
    np.testing.assert_allclose(got.to_dense(), oracle(t, (0, 2)), atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_single_axis_fuse_is_the_identity(axis):
    t = SymmetricTensor.random(SU2_LEGS, seed=3)
    assert t.fuse(axis) == t
    assert t.fuse((axis,)).legs == t.legs


# --- validation -----------------------------------------------------------------


def test_empty_axes_raises():
    t = SymmetricTensor.random(SU2_LEGS, seed=4)
    with pytest.raises(ValueError, match="empty"):
        t.fuse(())


def test_repeated_axes_names_them():
    t = SymmetricTensor.random(SU2_LEGS, seed=4)
    with pytest.raises(ValueError, match=r"\(1,\).*repeated"):
        t.fuse((1, 1, 0))


@pytest.mark.parametrize("axes", [(0, 3), (5,), (-1, 0)])
def test_out_of_range_axes_name_them(axes):
    t = SymmetricTensor.random(SU2_LEGS, seed=4)
    with pytest.raises(ValueError, match="out of range"):
        t.fuse(axes)


def test_mixed_sides_raises_naming_the_axes():
    t = SymmetricTensor.random(SU2_LEGS, seed=4)
    with pytest.raises(ValueError, match=r"axes \(0, 2\) mix sides"):
        t.fuse((0, 2))


def test_non_prefix_group_names_milestone_4_and_f_moves():
    t = SymmetricTensor.random(SU2_WIDE, seed=5)
    with pytest.raises(NotImplementedError) as e:
        t.fuse((1, 2))
    message = str(e.value)
    assert "(1, 2)" in message and "prefix" in message
    assert "Milestone 4" in message and "F-move" in message


def test_fusing_across_sides_at_the_space_level_raises():
    with pytest.raises(ValueError, match="no well-defined side"):
        fuse_spaces(Leg(A, OUT), Leg(B, IN))


# --- the fused space ------------------------------------------------------------


def test_su2_fused_degeneracies_match_the_hand_computation():
    """``m^f_c = sum_ab m_a m_b N^c_ab``, with at least two output sectors."""
    leg, layout = fuse_spaces(Leg(A, OUT), Leg(B, OUT))
    want: dict = {}
    for u_a, m_a in A.sectors:
        for u_b, m_b in B.sectors:
            for c in SU2.fusion(u_a, u_b):
                want[c] = want.get(c, 0) + m_a * m_b * SU2.n_symbol(u_a, u_b, c)
    # A = {1/2: 2, 1: 1}, B = {1/2: 1, 1: 2} -> 0, 1/2, 1, 3/2, 2
    assert dict(leg.space.sectors) == want
    assert len(want) >= 2
    assert dict(leg.space.sectors) == {ZERO: 4, HALF: 5, ONE: 4, SU2Sector(3): 5, SU2Sector(4): 2}
    assert leg.space.dim == A.dim * B.dim  # 7 * 8 == 56
    assert sum(extent for *_, extent in layout.triples) == leg.space.reduced_dim


def test_layout_chunks_tile_each_output_sector_exactly():
    leg, layout = fuse_spaces(Leg(A, OUT), Leg(B, OUT))
    for c, m in leg.space.sectors:
        chunks = layout.chunks(c)
        offset = 0
        for _, _, _, _, off, extent in chunks:
            assert off == offset  # contiguous, in layout order, no gaps
            offset += extent
        assert offset == m


# --- dense oracles --------------------------------------------------------------


def test_su2_dense_isometry_oracle_and_that_reshape_fails_it():
    t = SymmetricTensor.random(SU2_LEGS, seed=6)
    got = t.fuse((0, 1)).to_dense()
    np.testing.assert_allclose(got, oracle(t, (0, 1)), atol=1e-12)

    dense = t.to_dense()
    plain = dense.reshape(dense.shape[0] * dense.shape[1], dense.shape[2])
    assert plain.shape == got.shape
    assert not np.allclose(plain, got), "fused dense must not be a plain reshape for SU(2)"


def test_u1_dense_permutation_oracle_and_that_reshape_fails_it():
    t = SymmetricTensor.random(U1_LEGS, seed=7)
    m = isometry(t.legs[0], t.legs[1])
    # abelian: the map is a genuine permutation, not merely an isometry
    assert set(np.unique(m)) <= {0.0, 1.0}
    assert np.all(m.reshape(m.shape[0], -1).sum(axis=1) == 1)

    got = t.fuse((0, 1)).to_dense()
    np.testing.assert_allclose(got, oracle(t, (0, 1)), atol=1e-12)

    dense = t.to_dense()
    plain = dense.reshape(dense.shape[0] * dense.shape[1], dense.shape[2])
    assert plain.shape == got.shape
    assert not np.allclose(plain, got), "fused dense is not a reshape even for U(1)"


def test_trivial_fuse_is_exactly_a_plain_reshape():
    t = SymmetricTensor.random(TRIV_LEGS, seed=8)
    dense = t.to_dense()
    got = t.fuse((0, 1)).to_dense()
    np.testing.assert_array_equal(got, dense.reshape(6, 3))


# --- whole-side fusion ----------------------------------------------------------


def test_fusing_a_whole_side_collapses_its_trees_to_rank_one():
    t = SymmetricTensor.random(SU2_WIDE, seed=9)
    got = t.fuse(t.structure.out_axes)
    assert got.structure.out_axes == (0,)
    assert len(got.codomain) == 1
    for key in got.structure.block_order:
        assert key.output_tree.rank == 1
        assert key.output_tree.coupled == key.output_tree.uncoupled[0]
    np.testing.assert_allclose(got.to_dense(), oracle(t, (0, 1, 2)), atol=1e-12)


def test_fusing_the_in_side_works_too():
    legs = (Leg(A, IN), Leg(B, IN), Leg(C, OUT))
    t = SymmetricTensor.random(legs, seed=10)
    got = t.fuse(t.structure.in_axes)
    assert len(got.domain) == 1
    for key in got.structure.block_order:
        assert key.input_tree.rank == 1
    np.testing.assert_allclose(got.to_dense(), oracle(t, (0, 1)), atol=1e-12)


def test_iterated_equals_one_shot():
    t = SymmetricTensor.random(SU2_WIDE, seed=11)
    assert tenet.allclose(t.fuse((0, 1, 2)), t.fuse((0, 1)).fuse((0, 1)), rtol=0, atol=0)


# --- round trip -----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALL_LEGS))
def test_round_trip_leading_pair(name):
    axes = (0, 1)
    t = SymmetricTensor.random(ALL_LEGS[name], seed=12)
    back = t.fuse(axes).unfuse(min(axes), tuple(t.legs[i] for i in axes))
    assert back.legs == t.legs
    assert tenet.allclose(back, t, rtol=0, atol=0)


@pytest.mark.parametrize("name", sorted(ALL_LEGS))
def test_round_trip_whole_side(name):
    legs = ALL_LEGS[name]
    wide = (*legs[:2], legs[0], legs[2])  # three OUT legs, one IN
    t = SymmetricTensor.random(wide, seed=13)
    axes = t.structure.out_axes
    assert len(axes) == 3
    back = t.fuse(axes).unfuse(0, tuple(t.legs[i] for i in axes))
    assert back.legs == t.legs
    assert tenet.allclose(back, t, rtol=0, atol=0)


def test_round_trip_via_the_module_functions():
    t = SymmetricTensor.random(SU2_LEGS, seed=14)
    f = tenet.fuse(t, (0, 1))
    assert tenet.allclose(tenet.unfuse(f, 0, t.legs[:2]), t, rtol=0, atol=0)


def test_unfuse_with_a_single_leg_is_the_identity():
    t = SymmetricTensor.random(SU2_LEGS, seed=15)
    assert t.unfuse(0, (t.legs[0],)) == t


def test_unfuse_rejects_legs_that_do_not_reproduce_the_space():
    t = SymmetricTensor.random(SU2_LEGS, seed=16).fuse((0, 1))
    with pytest.raises(ValueError, match="carries"):
        t.unfuse(0, (Leg(A, OUT), Leg(A, OUT)))
    with pytest.raises(ValueError, match="side"):
        t.unfuse(0, (Leg(A, IN), Leg(B, IN)))
    with pytest.raises(ValueError, match="empty"):
        t.unfuse(0, ())


def test_unfuse_on_a_non_leading_axis_names_milestone_4():
    t = SymmetricTensor.random(SU2_WIDE, seed=17)
    with pytest.raises(NotImplementedError) as e:
        t.unfuse(1, (Leg(A, OUT), Leg(B, OUT)))
    assert "Milestone 4" in str(e.value) and "F-move" in str(e.value)


def test_unfuse_out_of_range_axis():
    t = SymmetricTensor.random(SU2_LEGS, seed=18)
    with pytest.raises(ValueError, match="out of range"):
        t.unfuse(9, (Leg(A, OUT),))


# --- isometry / exhaustiveness --------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALL_LEGS))
def test_norm_is_preserved(name):
    t = SymmetricTensor.random(ALL_LEGS[name], seed=19)
    assert tenet.norm(t.fuse((0, 1))) == pytest.approx(tenet.norm(t), abs=1e-12)


def test_chunk_extents_tile_every_target_block():
    """Every fused entry is written exactly once — structurally, not by hope."""
    t = SymmetricTensor.random(SU2_WIDE, seed=20)
    plan = fusion_plan(t.structure, (0, 1, 2))
    assert len(plan.steps) == 2
    for step in plan.steps:
        for key, chunks in zip(step.structure.block_order, step.targets, strict=True):
            total = sum(shape[step.axis] * shape[step.axis + 1] for _, shape in chunks)
            assert total == step.structure.block_shape(key)[step.axis]
            assert len({i for i, _ in chunks}) == len(chunks)  # no source used twice


def test_every_source_block_is_consumed_exactly_once():
    t = SymmetricTensor.random(SU2_LEGS, seed=21)
    step = fusion_plan(t.structure, (0, 1)).steps[0]
    used = [i for chunks in step.targets for i, _ in chunks]
    assert sorted(used) == list(range(len(t.blocks)))


# --- linearity and conj ---------------------------------------------------------


def test_fuse_is_linear():
    a = SymmetricTensor.random(SU2_LEGS, seed=22)
    b = SymmetricTensor.random(SU2_LEGS, seed=23)
    got = tenet.fuse(2.5 * a + b * (-0.75), (0, 1))
    want = 2.5 * tenet.fuse(a, (0, 1)) - 0.75 * tenet.fuse(b, (0, 1))
    assert tenet.allclose(got, want, rtol=0, atol=1e-12)


def test_fuse_commutes_with_conj():
    a = SymmetricTensor.random(SU2_LEGS, seed=24) + SymmetricTensor.random(SU2_LEGS, seed=25) * 1j
    assert np.iscomplexobj(a.blocks[0])
    assert tenet.allclose(a.fuse((0, 1)).conj(), a.conj().fuse((0, 1)), rtol=0, atol=0)


# --- plans: frozen, array-free, cached, deterministic ----------------------------


def _reachable(obj, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if isinstance(obj, tuple):
        for item in obj:
            yield from _reachable(item, seen)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _reachable(getattr(obj, f.name), seen)


def test_plan_and_layout_are_frozen_hashable_and_array_free():
    plan = fusion_plan(TensorStructure(SU2_WIDE), (0, 1, 2))
    hash(plan.steps[0].layout)
    for cls in (FusionLayout, FusionPlan, FusionStep):
        assert cls.__dataclass_params__.frozen
    for o in _reachable(plan):
        assert not isinstance(o, np.ndarray), f"array reachable from the plan: {o!r}"
        assert not isinstance(o, list | dict | set | bytearray), f"mutable container: {o!r}"
        assert isinstance(o, tuple | int | str | bool | float | enum.Enum | type(None)) or (
            dataclasses.is_dataclass(o) and o.__dataclass_params__.frozen
        ), f"neither frozen nor a scalar: {o!r}"


def test_plans_and_layouts_are_cached_by_identity():
    s = TensorStructure(SU2_LEGS)
    assert fusion_plan(s, (0, 1)) is fusion_plan(s, (0, 1))
    assert fuse_spaces(Leg(A, OUT), Leg(B, OUT)) is fuse_spaces(Leg(A, OUT), Leg(B, OUT))


def test_fuse_spaces_is_identical_across_python_hash_seeds():
    script = (
        "from tenet import OUT, GradedSpace, Leg\n"
        "from tenet.ops.fusion import fuse_spaces\n"
        "from tenet.symmetry import SU2, SU2Sector\n"
        "a = GradedSpace.new(SU2, {SU2Sector(1): 2, SU2Sector(2): 1})\n"
        "b = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 3})\n"
        "print(fuse_spaces(Leg(a, OUT), Leg(b, OUT)))\n"
    )
    outs = set()
    for seed in ("0", "1", "12345", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(
            subprocess.run(
                [sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True
            ).stdout
        )
    assert len(outs) == 1


# --- backend agnosticism --------------------------------------------------------


def test_jax_fuse_matches_numpy():
    use_jax()
    t = SymmetricTensor.random(SU2_WIDE, seed=26)
    j = t.to_backend("jax").fuse((0, 1, 2))
    assert j.backend == "jax"
    assert j.structure == t.fuse((0, 1, 2)).structure
    assert tenet.allclose(j.to_backend("numpy"), t.fuse((0, 1, 2)), rtol=0, atol=1e-12)


def test_jax_round_trip_and_norm():
    use_jax()
    t = SymmetricTensor.random(SU2_LEGS, seed=27).to_backend("jax")
    back = t.fuse((0, 1)).unfuse(0, t.legs[:2])
    assert back.backend == "jax"
    assert tenet.allclose(back, t, rtol=0, atol=1e-12)
    assert tenet.norm(t.fuse((0, 1))) == pytest.approx(tenet.norm(t), abs=1e-12)


def test_jax_non_adjacent_fuse_matches_numpy():
    use_jax()
    t = SymmetricTensor.random(SU2_INTERLEAVED, seed=28)
    j = t.to_backend("jax").fuse((0, 2))
    assert tenet.allclose(j.to_backend("numpy"), t.fuse((0, 2)), rtol=0, atol=1e-12)


# --- the module's own hygiene ---------------------------------------------------


def test_fusion_module_touches_no_numpy_no_cgc_no_to_dense():
    src = pathlib.Path(fusion.__file__).read_text()
    body = src.split('"""', 2)[2]  # drop the module docstring, which discusses them
    assert "np." not in body
    assert "import numpy" not in body
    assert ".cgc" not in body
    assert "to_dense" not in body
    assert "if provider ==" not in body
    assert "ar.do(" in body
    allowed = {'"transpose"', '"reshape"', '"concatenate"'}
    for chunk in body.split("ar.do(")[1:]:
        assert chunk.split(",")[0].strip() in allowed
