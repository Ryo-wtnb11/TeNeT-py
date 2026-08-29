"""Axis permutation — ``T.transpose(*axes)`` / [tenet.transpose][].

Public axis order enters the categorical data through exactly two channels: which
leg is which, and the *relative order within each side* (a ``FusionBlockKey``'s
two trees list their uncoupled sectors in public axis order restricted to their
side). That splits every permutation in two:

**case A — both side permutations are the identity.** Only the OUT/IN
interleaving changed. An OUT leg and an IN leg are never adjacent lines of one
fusion diagram, so no tree, no coupled sector and no coefficient moves:
``block_order`` is literally the same tuple and ``blocks[i] -> blocks[i]``. Free
for *every* provider, SU(2) included — this is invariant 3 in action.

**case B — some side permutation is non-trivial.** That is a braid. It needs the
provider to state its own coefficients through
[PermutationCoefficients][tenet.symmetry.PermutationCoefficients]; every provider shipped here
does, and one that does not is refused loudly. A "just permute the block axes"
fast path would return a numerically plausible tensor that is silently the wrong
element of the fusion basis, and only a dense round-trip would catch it. A
non-Abelian provider returns a genuine multi-term expansion, which the
accumulation loop below sums into shared target blocks — the case it was written
for, and the case no Abelian provider ever exercises.

Both trees of a key are braided with the *same* coefficients. That is correct
while the provider's gauge is real (SU(2)'s is); a complex-gauge provider must
conjugate on the domain (input-tree) side.

``transpose`` never changes a leg's ``side``: moving a leg between domain and
codomain is a bend ([tenet.repartition][]). What it does change is
``out_axes``/``in_axes``, which are *positions*.

[braid][tenet.braid] generalizes all of it. ``transpose`` crosses exactly the pairs
its permutation inverts; a planar network also needs the crossings the leg order does
*not* spell, so ``braid`` takes the lines' incoming order as ``levels`` and crosses a
pair when the incoming and outgoing orders disagree about it. Monotone levels are
``transpose``, plan object included; the extra crossings ride on the same plan as one
grading sign per block.

No NumPy and no ``to_dense`` here (invariants 8/9): the plan is array-free
metadata and blocks move only through ``ar.do("transpose", ...)``.
"""

import operator
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from itertools import combinations
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.map_view import from_matrices, lower_plan, scaled
from tenet.structure import FusionBlockKey, TensorStructure, _pattern
from tenet.symmetry.base import (
    BraidingData,
    CapabilityError,
    PermutationCoefficients,
    Sector,
    TwistData,
    requires,
    supports,
)
from tenet.symmetry.coherence import symmetric_braiding

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["PermutationPlan", "braid", "braid_plan", "permutation_plan", "transpose", "twist"]


@dataclass(frozen=True, slots=True)
class PermutationPlan:
    """The categorical half of a transpose: static, array-free, hashable.

    ``terms`` is ``((source block index, target block index, coefficient), ...)``;
    a target may appear more than once (a genuine expansion sums into it) and the
    coefficients are plain Python numbers, never arrays.
    """

    axes: tuple[int, ...]
    new_structure: TensorStructure
    terms: tuple[tuple[int, int, complex], ...]


def _validated_axes(ndim: int, axes: Sequence[int]) -> tuple[int, ...]:
    """A permutation of ``range(ndim)``, or ``ValueError`` naming the culprit."""
    try:
        # operator.index keeps NumPy/JAX integer scalars usable and makes the
        # tuple hashable-as-plain-ints, so the plan cache does not fragment
        axes = tuple(operator.index(a) for a in axes)
    except TypeError as exc:
        raise ValueError(f"transpose: axes {tuple(axes)} must all be integers") from exc
    if len(axes) != ndim:
        raise ValueError(f"transpose: axes {axes} has length {len(axes)}, expected {ndim}")
    seen: set[int] = set()
    for a in axes:
        if not 0 <= a < ndim:
            raise ValueError(f"transpose: axis {a} is out of range for a {ndim}-dimensional tensor")
        if a in seen:
            raise ValueError(f"transpose: axis {a} is repeated in {axes}")
        seen.add(a)
    return axes


def _side_perm(
    axes: tuple[int, ...], old_side_axes: tuple[int, ...], new_side_axes: tuple[int, ...]
) -> tuple[int, ...]:
    """The permutation induced on one side's legs among themselves.

    Entry ``j`` is the position, within the *old* side, of the leg that ends up at
    position ``j`` of the new side — the same "old index becoming new index j"
    convention as ``axes`` itself, so it feeds ``permute_tree`` directly.
    """
    position = {old: j for j, old in enumerate(old_side_axes)}
    return tuple(position[axes[i]] for i in new_side_axes)


def _refuse(provider: Any, axes: tuple[int, ...], offenders: str, sectors: tuple[Any, ...]) -> None:
    """Turn the bare capability failure into a message a user can act on.

    The chirality check comes first: a provider that *has* R-symbols but whose
    braiding is not symmetric must be refused with the real reason (``axes``
    underdetermine the braid) rather than with a missing-method message —
    whether or not it also defines ``permute_tree``.
    """
    if supports(provider, BraidingData) and not symmetric_braiding(provider, sectors):
        raise CapabilityError(
            f"transpose: axes {axes} reorders legs within a side ({offenders}), which is a "
            f"braid, and provider {provider.name}'s braiding is chiral (R is not its own "
            "inverse), so axes alone cannot say which line crosses over which. "
            "[tenet.braid][] carries leg levels but resolves them through the same "
            "symmetric-braiding coefficients, so it refuses this provider too. "
            "Permutations that only change the OUT/IN interleaving, leaving each "
            "side's internal order intact, work today for every provider."
        )
    try:
        requires(provider, PermutationCoefficients)
    except CapabilityError as exc:
        raise CapabilityError(
            f"transpose: axes {axes} reorders legs within a side ({offenders}), which is a "
            f"braid, and provider {provider.name} does not implement PermutationCoefficients. "
            "Within-side reordering needs the provider's own F- and R-moves; it is not a "
            "block transpose and will not be faked. "
            "Permutations that only change the OUT/IN interleaving, leaving each side's "
            "internal order intact, work today for every provider."
        ) from exc


@cache
def permutation_plan(structure: TensorStructure, axes: tuple[int, ...]) -> PermutationPlan:
    """Plan ``axes`` on ``structure``. Cached: repeat calls return the same object.

    ``axes`` must already be a validated permutation ([transpose][tenet.transpose] does that);
    the cache is keyed on the frozen structure and the tuple, nothing else.

    The **body** is cached one level down, on ``_pattern(structure)`` — the same legs with
    every degeneracy 1. ``terms`` is block indices and coefficients, which depend on
    the legs' sectors, sides and duals and never on their degeneracies, so two structures
    that differ only in degeneracy share one enumeration and this entry holds nothing but
    ``new_structure``. A DMRG sweep moves a bond's degeneracies at every bond of every
    sweep until the state settles, and without the split each move rebuilds a plan that is
    identical term for term.
    """
    plan = _pattern_plan(_pattern(structure), axes)
    legs = tuple(structure.legs[i] for i in axes)
    if legs == plan.new_structure.legs:
        return plan
    return replace(plan, new_structure=TensorStructure(legs))


@cache
def _pattern_plan(structure: TensorStructure, axes: tuple[int, ...]) -> PermutationPlan:
    """[permutation_plan][tenet.permutation_plan]'s body, on a degeneracy-free structure."""
    new_structure = TensorStructure(tuple(structure.legs[i] for i in axes))
    out_perm = _side_perm(axes, structure.out_axes, new_structure.out_axes)
    in_perm = _side_perm(axes, structure.in_axes, new_structure.in_axes)

    if out_perm == tuple(range(len(out_perm))) and in_perm == tuple(range(len(in_perm))):
        # case A: every key survives untouched, and block_order is a pure function
        # of the legs, so the sorted key tuple is identical element for element.
        terms = tuple((i, i, 1.0) for i in range(structure.num_blocks))
        return PermutationPlan(axes, new_structure, terms)

    provider = structure.provider
    offenders = ", ".join(
        f"{side} axes {tuple(old[j] for j in perm)} (was {old})"
        for side, old, perm in (
            ("OUT", structure.out_axes, out_perm),
            ("IN", structure.in_axes, in_perm),
        )
        if perm != tuple(range(len(perm)))
    )
    # the sector sample for the symmetric-braiding property: every sector a braid
    # here can touch appears on some leg (fusion channels are probed by the check)
    sectors = tuple(sorted({a for leg in structure.legs for a, _ in leg.space.sectors}))
    _refuse(provider, axes, offenders, sectors)

    built: list[tuple[int, int, complex]] = []
    for src, key in enumerate(structure.block_order):
        # _refuse() above ran requires(provider, PermutationCoefficients);
        # a raise-based capability check does not narrow
        for out_tree, c_out in provider.permute_tree(key.output_tree, out_perm):  # ty: ignore[unresolved-attribute]
            for in_tree, c_in in provider.permute_tree(key.input_tree, in_perm):  # ty: ignore[unresolved-attribute]
                coeff = c_out * c_in
                if coeff == 0:
                    continue
                dst = new_structure.index_of(FusionBlockKey(out_tree, in_tree))
                built.append((src, dst, coeff))
    return PermutationPlan(axes, new_structure, tuple(built))


def transpose(t: "SymmetricTensor", axes: Sequence[int] | None = None) -> "SymmetricTensor":
    """The same abstract tensor with public axes reordered as ``axes``.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose public axes are reordered.
    axes : sequence of int or None, optional
        ``axes[i]`` is the OLD axis that becomes new axis ``i`` (NumPy
        convention); a permutation of ``range(t.ndim)``, negative indices
        refused. ``None`` (the default) reverses the axes.

    Returns
    -------
    SymmetricTensor
        The transposed tensor; ``side``, ``dual`` and ``name`` travel with
        each leg and no leg ever changes side.

    Raises
    ------
    ValueError
        If ``axes`` is not a permutation of ``range(t.ndim)`` — a non-integer,
        a wrong length, an out-of-range axis or a repeat, each named.
    CapabilityError
        If ``axes`` reorders legs *within* a side — a braid — and the
        provider does not implement
        [PermutationCoefficients][tenet.symmetry.PermutationCoefficients],
        or its [BraidingData][tenet.symmetry.BraidingData] is chiral
        (``R != R**-1``), in which case ``axes`` alone underdetermine the
        braid and an explicit ``braid(t, i, over=...)`` API would be needed.
        Permutations that only change the OUT/IN interleaving work for every
        provider.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> W = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT), Leg(W, IN)), seed=0)
    >>> tenet.transpose(t, (2, 0, 1)).legs == (t.legs[2], t.legs[0], t.legs[1])
    True
    """
    axes = tuple(reversed(range(t.ndim))) if axes is None else _validated_axes(t.ndim, tuple(axes))
    return _apply(t, permutation_plan(t.structure, axes), "transpose")


def _apply(t: "SymmetricTensor", plan: PermutationPlan, what: str) -> "SymmetricTensor":
    """Move ``t``'s blocks through ``plan``. Shared by ``transpose`` and ``braid``.

    ``what`` names the caller in the one error message this can raise.

    A permutation is a plan like any other, so it goes through
    [lower_plan][tenet.map_view.lower_plan] where that route applies: the source cells
    are read out of ``t``'s coupled-sector matrices and written straight into the
    result's, and the blocks this loop would build -- along with the gather that would
    then copy every one of them back into a matrix -- never exist. The loop below stays
    as the route for what ``lower_plan`` declines: an immutable backend, a watched torch
    tensor, a genuinely complex coefficient.
    """
    from tenet.tensor import _unchecked

    mats = lower_plan(t, plan.new_structure, plan.axes, plan.terms)
    if mats is not None:
        return from_matrices(plan.new_structure, mats)

    # one transpose per *distinct source*, not per term (#123): ``plan.axes`` is per-plan
    # and only the coefficient is per-term, so every term sharing a source used to
    # recompute a byte-identical array. #74's batched alternative -- stack a shape bucket,
    # transpose once, slice back out -- was prototyped and measured slower on every axis
    # (see #123 for the table and the refusal).
    blocks = t.blocks  # read once: it is a property, not a field
    moved = {
        src: ar.do("transpose", blocks[src], plan.axes) for src in {s for s, _, _ in plan.terms}
    }

    blocks: dict[int, Any] = {}
    for src, dst, coeff in plan.terms:
        contrib = moved[src]
        if coeff != 1:
            # keep a real coefficient real, so a real tensor stays real
            contrib = scaled(contrib, coeff.real if getattr(coeff, "imag", 0) == 0 else coeff)
        blocks[dst] = contrib if dst not in blocks else blocks[dst] + contrib

    n = plan.new_structure.num_blocks
    if len(blocks) != n:
        raise ValueError(
            f"{what}: the plan fills {len(blocks)} of {n} target blocks — "
            f"{t.provider.name}.permute_tree dropped terms"
        )
    # every block was written under a key of ``plan.new_structure``, from a source
    # block of ``t`` transposed by the plan's own axes: the shapes are the plan's
    # statement, not a fact to rediscover (#328)
    return _unchecked(plan.new_structure, tuple(blocks[i] for i in range(n)))


# ---------------------------------------------------------------------------
# braid: transpose plus the crossings the leg order does not spell
# ---------------------------------------------------------------------------


def _validated_levels(ndim: int, levels: Sequence[int]) -> tuple[int, ...]:
    """One integer height per public axis, or ``ValueError`` naming the culprit."""
    try:
        levels = tuple(operator.index(v) for v in levels)
    except TypeError as exc:
        raise ValueError(f"braid: levels {tuple(levels)} must all be integers") from exc
    if len(levels) != ndim:
        raise ValueError(f"braid: levels {levels} has length {len(levels)}, expected {ndim}")
    return levels


def _level_crossings(levels: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """The axis pairs whose incoming order disagrees with their current order.

    ``levels`` is the *incoming* planar order of the lines; ``i < j`` with
    ``levels[i] > levels[j]`` means line ``i`` arrives to the right of line ``j``
    and has to cross it to sit where it sits. Monotone (non-decreasing) levels
    give the empty tuple — the incoming order is already the leg order.
    """
    return tuple((i, j) for i, j in combinations(range(len(levels)), 2) if levels[i] > levels[j])


def _parity(provider: Any, sector: Sector) -> int:
    """The Z2 grading of ``sector``: ``1`` for an odd (fermionic) sector, else ``0``.

    Read off the ribbon twist, which is ``(-1)^parity`` exactly on a fermionic
    provider and ``1`` on every bosonic one — the twist *is* the grading datum
    (invariant: a symmetric provider's twist is ``+-1``).
    """
    return int(provider.twist(sector) == -1)


@cache
def _crossing_signs(
    structure: TensorStructure, crossed: tuple[tuple[int, int], ...]
) -> tuple[int, ...]:
    """``+-1`` per block: the grading sign of the lines crossed at ``crossed``.

    ``(-1)^(sum over crossed pairs of p_i p_j)`` on the block's own sectors — the
    same number YASTN's ``swap_gate`` negates a block by, and ``1`` for every
    bosonic provider, where the crossing is the identity morphism.
    """
    provider = structure.provider
    signs = []
    for key in structure.block_order:
        sectors = structure.axis_sectors(key)
        odd = sum(_parity(provider, sectors[i]) * _parity(provider, sectors[j]) for i, j in crossed)
        signs.append(-1 if odd % 2 else 1)
    return tuple(signs)


def _refuse_crossing(
    provider: Any, crossed: tuple[tuple[int, int], ...], sectors: tuple[Any, ...]
) -> None:
    """Refuse a crossing whose coefficient the provider cannot state."""
    if supports(provider, BraidingData) and not symmetric_braiding(provider, sectors):
        raise CapabilityError(
            f"braid: levels cross the axis pairs {crossed}, which is a braid, and provider "
            f"{provider.name}'s braiding is chiral (R is not its own inverse), so a crossing "
            "is not its own inverse either and the pairwise symmetric-braiding coefficient "
            "does not apply. Anyonic braiding is out of scope; monotone levels (a plain "
            "transpose) work today for every provider."
        )
    requires(provider, TwistData)


@cache
def braid_plan(
    structure: TensorStructure, axes: tuple[int, ...], levels: tuple[int, ...]
) -> PermutationPlan:
    """Plan ``braid(axes, levels)`` on ``structure``. Cached, like ``permutation_plan``.

    Monotone ``levels`` return [permutation_plan][tenet.permutation_plan]'s object itself,
    so a braid that is a transpose *is* the transpose, plan for plan.
    """
    plan = _pattern_braid_plan(_pattern(structure), axes, levels)
    legs = tuple(structure.legs[i] for i in axes)
    if legs == plan.new_structure.legs:
        return plan
    return replace(plan, new_structure=TensorStructure(legs))


@cache
def _pattern_braid_plan(
    structure: TensorStructure, axes: tuple[int, ...], levels: tuple[int, ...]
) -> PermutationPlan:
    """[braid_plan][tenet.braid_plan]'s body, on a degeneracy-free structure.

    The extra crossings factorize off the transpose: a pair the permutation already
    inverts *and* the levels cross is not crossed at all, and ``R`` times the grading
    sign is ``1`` there, so multiplying the transpose plan's coefficients by the
    grading sign of the level-crossed pairs lands on the right braid word either way.
    """
    plan = _pattern_plan(structure, axes)
    crossed = _level_crossings(levels)
    if not crossed:
        return plan
    provider = structure.provider
    sectors = tuple(sorted({a for leg in structure.legs for a, _ in leg.space.sectors}))
    _refuse_crossing(provider, crossed, sectors)
    signs = _crossing_signs(structure, crossed)
    if all(s == 1 for s in signs):
        return plan
    return replace(plan, terms=tuple((src, dst, c * signs[src]) for src, dst, c in plan.terms))


@cache
def _twist_signs(structure: TensorStructure, axes: tuple[int, ...]) -> tuple[complex, ...]:
    """``theta`` on the named legs, per block; ``()`` when every factor is ``1``.

    ``()`` is the whole bosonic story -- ``theta`` is ``1`` on every sector of a
    symmetric bosonic category, so [twist][tenet.twist] hands the tensor straight back
    and those paths stay bit-identical.
    """
    theta = structure.provider.twist  # ty: ignore[unresolved-attribute]  # requires() at the call
    signs, trivial = [], True
    for key in structure.block_order:
        sectors = structure.axis_sectors(key)
        factor: complex = 1.0
        for i in axes:
            factor *= theta(sectors[i])
        trivial = trivial and factor == 1
        signs.append(factor)
    return () if trivial else tuple(signs)


def twist(t: "SymmetricTensor", axes: Sequence[int] | int) -> "SymmetricTensor":
    """The ribbon twist ``theta`` on each named leg -- TensorKit's ``twist!``.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor. Its legs, sides and structure are untouched: a twist is a scalar
        per block, not a permutation and not a bend.
    axes : sequence of int, or int
        The public axes whose lines are twisted. Repeats are not deduplicated --
        twisting a leg twice pays ``theta**2``, as the diagram says.

    Returns
    -------
    SymmetricTensor
        ``t`` with each block multiplied by the product of ``theta`` over the sectors
        the named legs carry in that block. ``t`` itself when every factor is ``1``.

    Raises
    ------
    CapabilityError
        If the provider does not implement
        [TwistData][tenet.symmetry.TwistData], which is where ``theta`` lives.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> tenet.twist(t, 0) is t  # theta = 1 here, so the tensor comes back untouched
    True

    Notes
    -----
    **The twist is what makes a closed line's value unique.** ``sVect`` is a symmetric
    ribbon category, so a closed diagram evaluates to one number whatever order it is
    contracted in -- but only once every closure is the categorical one. A closure whose
    duality pairing runs *against* the direction the composition takes differs from it by
    ``theta`` on that line, and paying that is this function. It is PEPSKit's
    ``twistdual``/``twistnondual`` (``utility/util.jl``) with the ``isdual`` test left to
    the caller, because tenet spells ``V`` versus ``V*`` as ``side`` xor ``dual``
    (``outward_dual``) and the caller already knows which end it holds.

    ``theta`` is ``(-1)^parity`` on a fermion-parity grading and ``1`` on every bosonic one,
    the same grading datum [braid][tenet.braid] reads its crossing sign from.
    """
    axes = (axes,) if isinstance(axes, int) else tuple(axes)
    if not axes:
        return t
    requires(t.provider, TwistData)
    signs = _twist_signs(t.structure, axes)
    if not signs:
        return t
    return type(t)(
        t.structure,
        tuple(
            b if c == 1 else scaled(b, c.real if isinstance(c, complex) and c.imag == 0 else c)
            for b, c in zip(t.blocks, signs, strict=True)
        ),
    )


def braid(t: "SymmetricTensor", axes: Sequence[int], levels: Sequence[int]) -> "SymmetricTensor":
    """The same abstract tensor after a braid: ``axes`` reordered, ``levels`` crossed.

    ``levels[i]`` is the position of axis ``i``'s line in the diagram's *incoming*
    planar order; ``axes`` is the outgoing order, as in [tenet.transpose][]. Two
    lines cross exactly when those two orders disagree about them, which makes
    ``braid`` the two operations a planar embedding needs and ``transpose`` alone
    cannot spell:

    * **monotone ``levels``** — the incoming order is the leg order, so the crossings
      are the inversions of ``axes`` and the result is
      [tenet.transpose][]``(t, axes)``, through the very same plan object.
    * **``axes`` the identity, two levels inverted** — the lines cross and come back
      to the leg order they started in: a crossing with no net permutation, the
      *swap gate*. Its coefficient is the grading sign ``(-1)^(p_i p_j)``, which is
      ``1`` for every bosonic provider (the crossing is then the identity morphism)
      and YASTN's ``swap_gate`` for fermion parity.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose lines are braided.
    axes : sequence of int
        ``axes[i]`` is the OLD axis that becomes new axis ``i``; a permutation of
        ``range(t.ndim)``, negative indices refused.
    levels : sequence of int
        One height per OLD axis. Ties never cross; only the order matters, so
        ``(0, 1, 2)`` and ``(3, 7, 9)`` are the same braid.

    Returns
    -------
    SymmetricTensor
        The braided tensor; ``side``, ``dual`` and ``name`` travel with each leg
        and no leg ever changes side.

    Raises
    ------
    ValueError
        If ``axes`` is not a permutation of ``range(t.ndim)``, or ``levels`` is not
        ``t.ndim`` integers.
    CapabilityError
        If the braid needs coefficients the provider does not state — a within-side
        reorder without
        [PermutationCoefficients][tenet.symmetry.PermutationCoefficients], a level
        crossing without [TwistData][tenet.symmetry.TwistData], or either on a
        provider whose braiding is chiral (``R != R**-1``), which is out of scope.

    Notes
    -----
    Differentiability and tracing: exactly [tenet.transpose][]'s status — the plan is
    frozen, array-free metadata and the blocks move through one ``transpose`` and one
    real scalar each, so ``braid`` is shape-static and traces under ``jax.jit``/
    ``jax.grad``. No custom VJP is registered for either, and none is needed.

    This is TensorKit's ``braid(t, p, levels)`` widened by one step. There ``levels``
    only choose each crossing's *sense*, so under a symmetric braiding they drop out
    and ``braid == permute``; the crossings are always the inversions of ``p``. Here
    they also decide *which* pairs cross, which is what a planar embedding needs and
    what a permutation cannot say. Sense stays irrelevant, as it must be for a
    symmetric braiding.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> tenet.braid(t, (1, 0), (0, 1)) == tenet.transpose(t, (1, 0))  # monotone: a transpose
    True
    >>> tenet.braid(t, (0, 1), (1, 0)).legs == t.legs  # a crossing moves no leg
    True
    """
    axes = _validated_axes(t.ndim, tuple(axes))
    levels = _validated_levels(t.ndim, tuple(levels))
    return _apply(t, braid_plan(t.structure, axes, levels), "braid")
