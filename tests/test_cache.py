"""Tests for ``tenet.cache`` — the cost-bounded plan caches.

The eviction mechanics are tested on a toy function, so that a budget of a few
terms is expressible; the transparency and the class-2/class-3 distinction are
tested on real plans, because that distinction is the design and a toy cannot
pin it.
"""

import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.cache import DEFAULT_BUDGET, _PlanCache, budget, plan_cache
from tenet.map_view import map_layout
from tenet.ops.contraction import _pattern_restore_plan, _restore_plan
from tenet.ops.map import _pattern_adjoint_plan, adjoint_plan
from tenet.ops.repartition import _pattern_repartition_plan, repartition_plan
from tenet.structure import _block_shape_table
from tenet.symmetry import SU2, SU2Sector


def chain(n: int) -> _PlanCache[tuple[int, ...]]:
    """A cache of ``n``-term values, budgeted at ``n`` terms, counting its own calls."""
    calls: list[int] = []

    @plan_cache(cost=len, limit=n)
    def sized(k: int) -> tuple[int, ...]:
        calls.append(k)
        return (k,) * k

    sized.calls = calls  # type: ignore[attr-defined]
    return sized


# --- eviction ----------------------------------------------------------------------


def test_eviction_drops_the_least_recently_used_and_stays_within_budget():
    sized = chain(10)
    for k in (1, 2, 3, 4):  # 10 terms exactly: nothing evicted yet
        sized(k)
    assert sized.cache_info().cost == 10
    assert sized.cache_info().evictions == 0

    sized(2)  # 2 is now the most recent; 1 is the least
    sized(5)  # +5 terms: over budget, so evict from the oldest end

    info = sized.cache_info()
    assert info.cost <= info.budget
    assert info.evictions > 0
    # 1 and 3 are the two oldest, and dropping both brings 15 back to 11 -- still over,
    # so 4 goes too; 2 and 5 are what fits and what was touched last
    sized(2)
    sized(5)
    assert sized.calls == [1, 2, 3, 4, 5], "an entry inside the budget was evicted"
    sized(1)
    assert sized.calls == [1, 2, 3, 4, 5, 1], "the least recently used entry survived"


def test_retained_cost_never_exceeds_the_budget():
    sized = chain(10)
    for k in range(1, 8):
        sized(k)
        info = sized.cache_info()
        assert info.cost <= info.budget, k


def test_an_oversized_entry_is_returned_but_not_retained():
    sized = chain(10)
    assert sized(25) == (25,) * 25
    info = sized.cache_info()
    assert info.currsize == 0
    assert info.cost == 0
    assert info.evictions == 0
    # and it is recomputed every time rather than held
    assert sized(25) == (25,) * 25
    assert sized.calls == [25, 25]


def test_an_oversized_entry_does_not_evict_what_fits():
    sized = chain(10)
    sized(4)
    sized(25)
    assert sized.cache_info().cost == 4
    sized(4)
    assert sized.calls == [4, 25], "caching the oversized value evicted the small one"


# --- counters ----------------------------------------------------------------------


def test_the_counters_move_as_expected():
    sized = chain(10)
    assert sized.cache_info() == (0, 0, None, 0, 0, 0, 10)

    sized(3)
    assert sized.cache_info() == (0, 1, None, 1, 0, 3, 10)
    sized(3)
    assert sized.cache_info() == (1, 1, None, 1, 0, 3, 10)
    sized(7)
    assert sized.cache_info() == (1, 2, None, 2, 0, 10, 10)
    sized(1)  # 11 > 10: evicts the least recent, which is 3
    assert sized.cache_info() == (1, 3, None, 2, 1, 8, 10)

    sized.cache_clear()
    assert sized.cache_info() == (0, 0, None, 0, 0, 0, 10)


# --- the environment override ------------------------------------------------------


def test_the_budget_default_is_the_module_default():
    assert budget() == DEFAULT_BUDGET


def test_the_environment_variable_overrides_the_budget(monkeypatch):
    monkeypatch.setenv("TENET_PLAN_CACHE_BUDGET", "37")
    assert budget() == 37

    @plan_cache(cost=len)
    def sized(k: int) -> tuple[int, ...]:
        return (k,) * k

    assert sized.budget == 37, "the budget is read at decoration time"


@pytest.mark.parametrize("bad", ["", "lots", "1e6", "-1"])
def test_a_malformed_environment_budget_is_loud(monkeypatch, bad):
    monkeypatch.setenv("TENET_PLAN_CACHE_BUDGET", bad)
    with pytest.raises(ValueError, match="TENET_PLAN_CACHE_BUDGET"):
        budget()


# --- transparency, on real plans ---------------------------------------------------


def structure(chi: int) -> TensorStructure:
    V = GradedSpace.new(SU2, {SU2Sector(0): chi, SU2Sector(1): chi, SU2Sector(2): chi})
    return TensorStructure((Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(V, IN)))


def test_a_hit_a_miss_and_a_post_eviction_recompute_agree():
    """On ``_block_shape_table``: the smallest cache that genuinely reads degeneracies."""
    args = (structure(2),)
    fresh = _block_shape_table.__wrapped__(*args)
    miss = _block_shape_table(*args)
    hit = _block_shape_table(*args)
    assert hit is miss
    assert miss == fresh

    kept = _block_shape_table.budget
    try:
        _block_shape_table.cache_clear()
        _block_shape_table.budget = 1  # every table here is oversized: nothing is retained
        again = _block_shape_table(*args)
    finally:
        _block_shape_table.budget = kept
        _block_shape_table.cache_clear()
    assert again is not miss
    assert again == miss, "a recompute after eviction returned a different table"


def test_map_layout_survives_a_round_trip_through_eviction():
    s2, s3 = structure(2), structure(3)
    first = map_layout(s2)
    kept = map_layout.budget
    try:
        map_layout.budget = 1  # oversized: neither structure is retained
        map_layout.cache_clear()
        assert map_layout(s2) == first
        assert map_layout(s2) is not first
        assert map_layout(s3) != first
    finally:
        map_layout.budget = kept
        map_layout.cache_clear()


def test_bounding_does_not_change_what_a_tensordot_computes():
    a = SymmetricTensor.random(structure(2).legs, seed=0)
    b = SymmetricTensor.random(structure(2).legs, seed=1)
    reference = tenet.tensordot(a, b, ((2, 3), (0, 1)))

    kept = {c: c.budget for c in (map_layout, _block_shape_table)}
    try:
        for c in kept:
            c.budget = 0  # nothing whatever is retained
            c.cache_clear()
        assert bool(tenet.allclose(tenet.tensordot(a, b, ((2, 3), (0, 1))), reference))
    finally:
        for c, value in kept.items():
            c.budget = value
            c.cache_clear()


# --- the class-2 / class-3 distinction ---------------------------------------------


def test_a_pattern_cache_is_not_bounded_and_a_growing_bond_adds_no_entry_to_it():
    """The design: block indices read sectors, not degeneracies (``tenet.cache``)."""
    assert not isinstance(_pattern_repartition_plan, _PlanCache)
    assert isinstance(map_layout, _PlanCache), "offsets and extents read the degeneracies"

    _pattern_repartition_plan.cache_clear()
    repartition_plan.cache_clear()
    for chi in range(1, 6):
        repartition_plan(structure(chi), (0, 1, 2), (3,))

    assert repartition_plan.cache_info().currsize == 5, "one outer entry per bond dimension"
    assert _pattern_repartition_plan.cache_info().currsize == 1, (
        "five bond dimensions, one sector pattern: the pattern cache must not grow"
    )


def test_the_outer_plan_shells_share_the_pattern_entry_terms():
    """Why the outer caches stay unbounded: their entries own no terms."""
    plans = [repartition_plan(structure(chi), (0, 1, 2), (3,)) for chi in (1, 2, 3)]
    assert len({id(p.terms) for p in plans}) == 1
    assert len({p.new_structure for p in plans}) == 3


def test_the_restore_plan_shells_share_the_pattern_entry_terms():
    """``_restore_plan`` composes terms, and a composed term reads no degeneracy."""
    args = ((0, 1, 2), (3,), ((1, 0, 2, 3),))
    _pattern_restore_plan.cache_clear()
    plans = [_restore_plan(structure(chi), *args) for chi in (1, 2, 3)]
    assert len({id(terms) for _, _, terms in plans}) == 1
    assert len({id(perm) for _, perm, _ in plans}) == 1
    assert len({s for s, _, _ in plans}) == 3
    assert _pattern_restore_plan.cache_info().currsize == 1


def test_the_restore_plan_shares_its_terms_through_a_bend_too():
    """The bending case: a leg crossing sides is where a degeneracy could enter, and does not."""
    plans = [_restore_plan(structure(chi), (0,), (1, 2, 3), ()) for chi in (1, 2, 3)]
    assert len({id(terms) for _, _, terms in plans}) == 1
    assert len({s.legs[1].side for s, _, _ in plans}) == 1
    assert plans[0][0].legs[1].side is IN, "this case must actually bend a leg"


def test_the_adjoint_plan_shells_share_the_pattern_entry_sources():
    """``sources`` is a permutation of ``block_order``, which reads no degeneracy."""
    _pattern_adjoint_plan.cache_clear()
    plans = [adjoint_plan(structure(chi)) for chi in (1, 2, 3)]
    assert len({id(p.sources) for p in plans}) == 1
    assert len({p.new_structure for p in plans}) == 3
    assert _pattern_adjoint_plan.cache_info().currsize == 1
