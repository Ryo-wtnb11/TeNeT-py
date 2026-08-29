"""Cost-bounded memoization for the plans keyed on a whole ``TensorStructure``.

Three kinds of cache live in this package, and only the third one is bounded here.

**Bounded by the program.** ``lib_fn(backend, name)``, ``_irrep(Dynkin)``,
``_diagonal_subscripts`` -- keyed on names a program spells out. Finite, tiny,
plain ``functools.cache``.

**Keyed on a sector pattern.** ``_pattern`` and everything reached through it
(``_pattern_bend_plan``, ``_pattern_repartition_plan``, ``_pattern_plan``,
``_pattern_braid_plan``, ``_pattern_restore_plan``, ``_pattern_adjoint_plan``,
``_crossing_signs``, ``_twist_signs``, ``_artin_braid``, ``permute_braided_tree``,
``symmetric_braiding``, ``properties``, ``_fusion_trees``, ``_coupled_sectors``,
``_all_trees``, ``_flat``). A block index is a function of the
legs' sectors, sides and duals -- never of their degeneracies -- so these are keyed on
the structure with every degeneracy set to 1. Growing a bond adds no entry, and a model
has finitely many sector patterns. They stay unbounded, and so do the thin outer caches
(``repartition_plan``, ``bend_plan``, ``permutation_plan``, ``braid_plan``,
``_restore_plan``, ``adjoint_plan``, ``_block_order``, ``_index_map``,
``_axis_sectors_table``) whose entries are ``replace(...)`` shells *sharing* the pattern
entry's term tuple: bounding a shell would evict the shell and leave the terms behind.
So does ``contraction_plan``, whose entry is axis tuples and one ``TensorStructure``,
and so do ``_axes`` and ``fuse_spaces``, whose values are counted in sectors rather than
in blocks.

**Keyed on a whole structure, with a large value.** ``map_layout``, ``_tables``,
``_rects``, ``_slots``, ``_block_shape_table``, ``dense_plan``, ``fusion_plan``,
``batch_plan``. Degeneracies are in the key *and* in the value -- these hold the offsets,
extents and block shapes a growing bond dimension is *defined* to move -- so a loop over
growing bond dimension adds one full-size entry per bond dimension and never drops one.
These are what ``plan_cache`` bounds.

Which class a cache belongs to is a question about the value, not about its size.
``_restore_plan`` and ``adjoint_plan`` were bounded here and are not any more. Composing
a restore with its transposes builds a term tuple no other plan owns -- one measured SU(2)
three-sector rank-8 intermediate holds 59,696 terms, and wider sector content reaches
613,468, which at 156 bytes per term (measured, ``(int, int, complex)`` tuples) are 9.3
MB and 95.7 MB in a single entry -- but every bond dimension composes the *same* tuple,
so keying the body on the pattern shares one entry and removes the growth instead of
capping it. ``dense_plan`` is the near miss: its Clebsch-Gordan arrays, which are what
its cost counts, are equally degeneracy-independent, but they sit in a ``Cell`` beside
the sector offsets and degeneracies of the dense grid, which are not, so it stays here
until that dataclass is split.

Notes
-----
``functools.lru_cache(maxsize=N)`` is the wrong bound here. Entry sizes span four
orders of magnitude -- 46 terms for an SU(2) rank-4 contraction, 613,468 for the widest
rank-8 intermediate measured -- so a cap on the *number* of entries caps nothing that
matters: ``maxsize=128`` still admits about 12 GB. The bound has to be on
accumulated *cost*, cost being a property of the value, which is why the cost function
is supplied at decoration rather than duck-typed in the hot path: plan types differ and
the hot path is a dict lookup.

An entry whose own cost exceeds the whole budget is returned but not retained.
Caching it would evict every other entry to hold one item, which is worse than
recomputing it; symmray takes the same "too many sectors, do not cache" bail-out.

Thread safety is the ``functools.cache`` standard and no better: concurrent misses on
the same key may both compute (the wrapped function runs outside the lock), and the
hit-path counters may undercount under threads. What is guaranteed is that the entry
table and the retained-cost total stay consistent -- every mutation holds a lock --
and that a hit never sees a half-inserted entry.
"""

from __future__ import annotations

import functools
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, NamedTuple

__all__ = ["DEFAULT_BUDGET", "CacheInfo", "budget", "plan_cache"]


DEFAULT_BUDGET = 1_000_000
"""Retained cost one bounded cache may hold, in *terms*, before it evicts.

A term is the unit each cost function counts: one ``(source, target, coefficient)``
tuple of a block map, one band or grid cell of a layout, one block row of a table. A
plan term measures 156 bytes as a Python tuple, which is also the unit ``dense_plan``
converts its Clebsch-Gordan bytes into, so this default is about 156 MB per cache. The
bound is per cache and this package decorates eight of them, though only one or two hold
large values in any given workload.

It is sized for the working set of a CTMRG or DMRG sweep: on the order of ten distinct
plans of 10^4 to 10^5 terms each, i.e. up to ~10^6 terms, which fits whole. A sweep
therefore evicts nothing and recomputes nothing. What the bound catches is the loop over
*growing* bond dimension, where every earlier bond's layout is dead and was being kept
forever. Override with the ``TENET_PLAN_CACHE_BUDGET`` environment variable.
"""


class CacheInfo(NamedTuple):
    """Counters of one bounded cache. The first four fields match ``functools``."""

    hits: int
    misses: int
    maxsize: None
    currsize: int
    evictions: int
    cost: int
    budget: int


def budget() -> int:
    """``DEFAULT_BUDGET``, or ``TENET_PLAN_CACHE_BUDGET`` when that is set."""
    raw = os.environ.get("TENET_PLAN_CACHE_BUDGET")
    if raw is None:
        return DEFAULT_BUDGET
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"TENET_PLAN_CACHE_BUDGET={raw!r} is not an integer; it is a plan-cache "
            "budget in terms, e.g. TENET_PLAN_CACHE_BUDGET=1000000"
        ) from None
    if value < 0:
        raise ValueError(f"TENET_PLAN_CACHE_BUDGET={value} is negative")
    return value


class _PlanCache[R]:
    """A least-recently-used memo bounded by accumulated value cost, not entry count."""

    def __init__(self, fn: Callable[..., R], cost: Callable[[Any], int], limit: int) -> None:
        self._fn = fn
        self._cost = cost
        self.budget = limit
        self._entries: OrderedDict[Any, tuple[R, int]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.retained = 0
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any) -> R:
        entries = self._entries
        try:
            found = entries[args]
        except KeyError:
            pass
        else:
            entries.move_to_end(args)
            self.hits += 1
            return found[0]

        self.misses += 1
        value = self._fn(*args)
        cost = self._cost(value)
        if cost > self.budget:
            # Retaining it would evict everything else to hold one item.
            return value
        with self._lock:
            if args not in entries:
                entries[args] = (value, cost)
                self.retained += cost
            while self.retained > self.budget:
                _, (_, dropped) = entries.popitem(last=False)
                self.retained -= dropped
                self.evictions += 1
        return value

    def cache_info(self) -> CacheInfo:
        """Hits, misses, entries, evictions and retained cost, as a named tuple."""
        return CacheInfo(
            self.hits,
            self.misses,
            None,
            len(self._entries),
            self.evictions,
            self.retained,
            self.budget,
        )

    def cache_clear(self) -> None:
        """Drop every entry and zero every counter. ``budget`` is kept."""
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = self.evictions = self.retained = 0


def plan_cache[R](
    *, cost: Callable[[Any], int], limit: int | None = None
) -> Callable[[Callable[..., R]], _PlanCache[R]]:
    """Memoize on positional arguments, evicting least-recently-used past ``limit``.

    Parameters
    ----------
    cost : callable
        Maps a return value to its size in *terms*. Called once per miss, never on the
        hit path, which is why it is fixed at decoration instead of being discovered
        from the value. Typed as taking ``Any`` deliberately: the return type is
        inferred from the decorated function alone, so a caller of the decorated
        function keeps it.
    limit : int, optional
        Retained cost this cache may hold. Defaults to ``budget()``,
        read once here at decoration time; assign to the cache's ``budget`` attribute
        to change it afterwards.

    Returns
    -------
    callable
        A decorator producing a callable with ``cache_info()`` and ``cache_clear()``.
        It keys on positional arguments; a keyword call is a ``TypeError``.

    Notes
    -----
    The memo is transparent: a hit, a miss and a recompute after eviction all return
    an equal value, because the wrapped function is a pure function of its arguments.
    Only *when* it runs changes.
    """
    resolved = budget() if limit is None else limit

    def decorate(fn: Callable[..., R]) -> _PlanCache[R]:
        return _PlanCache(fn, cost, resolved)

    return decorate
