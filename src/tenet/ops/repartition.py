"""Line bending — ``T.repartition(outputs=..., inputs=...)``.

Moving a leg between domain and codomain is a categorical **bend**, not a
Boolean flip of ``side``.
Two structural facts outlive whatever coefficients a provider supplies, and both
are fixed here:

* **A bend flips ``side`` *and* ``dual`` on the moved leg.** The ``GradedSpace``
  is untouched, so every block shape is a permutation of the old one; what
  changes is ``Leg.fused_sector``, which now returns the dualized label the new
  tree needs (a U(1) charge ``q`` arrives on the other side as ``-q``). A model
  that identified IN with ``dual`` could not express this at all — invariant 2
  doing real work. Two separate operations sit behind that sentence:
  [flip_dual][tenet.flip_dual] toggles ``dual`` alone (relabelling the space and paying the
  Z-isomorphism's scalar), while a bend is the operation that moves ``side``.
* **Our two trees are independent, both in ascending public-axis order.**
  TensorKit reads ``Hom(b₁⊗…⊗b_{N₂}, a₁⊗…⊗a_{N₁}) ≅ Hom(1, a₁⊗…⊗a_{N₁}⊗
  b*_{N₂}⊗…⊗b*₁)`` — the domain **reversed** — and therefore builds its
  ``repartition`` out of a cyclic index tuple. symtenet deliberately does **not**
  adopt that planar reading: the pairing of the two trees lives in
  ``FusionBlockKey``, not in a cyclic order. The consequence is that a bend
  appends to the destination tree's *end*, i.e. the moved leg takes the largest
  public position on its new side — which is exactly what [bend][tenet.bend] enforces
  and what [repartition][tenet.SymmetricTensor.repartition]'s final ``transpose`` then corrects.

[bend][tenet.bend] is the only new mathematics, and it is deliberately minimal: it
bends the **last leg of its own side** and nothing else. Everything else is
reached by ``transpose``, so all reordering refusals come from that
already-tested capability gate:

```text
transpose  bring the leg to be moved to the end
bend       one primitive bend per moved leg
transpose  deliver (*outputs, *inputs)
```

The coefficient is ``sqrt(dim(c)/dim(a)) · B(a,b,c)`` times a Frobenius-Schur
phase, and for Trivial and U(1) it is provably exactly ``1``; SU(2) computes it
from its B-symbols. A provider that supplies none of that does not
implement ``BendingCoefficients`` and is refused loudly rather than handed a
plausible tensor with the wrong norm.

No ``to_dense`` here (invariant 9), and no NumPy in a ``TensorStructure`` (invariant 8):
the plans themselves stay array-free metadata. The index arrays that batch a plan's terms
into array operations are NumPy, but they are a cached side table keyed on the plan, never
a structural field, and they carry no numerical value — only which block goes where.
"""

import operator
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.backend import lib_fn
from tenet.cache import plan_cache
from tenet.fusion_tree import FusionTree
from tenet.leg import IN, OUT, Leg, Side
from tenet.map_view import from_matrices, is_identity_plan, lower_plan, map_layout, scaled
from tenet.ops.batch import band_scale, batch_plan, cast_coefficients
from tenet.ops.permutation import permutation_plan
from tenet.space import GradedSpace
from tenet.structure import TensorStructure, _pattern
from tenet.symmetry.base import (
    BendingCoefficients,
    CapabilityError,
    FSIndicatorData,
    TwistData,
    requires,
)

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "BendPlan",
    "RepartitionPlan",
    "bend",
    "bend_plan",
    "flip_dual",
    "repartition",
    "repartition_plan",
]


@dataclass(frozen=True, slots=True)
class BendPlan:
    """The categorical half of one bend: static, array-free, hashable.

    ``terms`` is ``((source block index, target block index, coefficient), ...)``
    with plain Python numbers, never arrays — the same shape of plan as
    ``PermutationPlan``.
    """

    new_structure: TensorStructure
    terms: tuple[tuple[int, int, complex], ...]


def _side_axes(structure: TensorStructure, axis: int) -> tuple[int, ...]:
    return structure.out_axes if structure.legs[axis].side is OUT else structure.in_axes


def _refuse(structure: TensorStructure, axis: int) -> None:
    """Turn the bare capability failure into a message a user can act on."""
    provider = structure.provider
    try:
        requires(provider, BendingCoefficients)
    except CapabilityError as exc:
        raise CapabilityError(
            f"repartition: moving axis {axis} between domain and codomain is a line bend, "
            f"and provider {provider.name} does not implement BendingCoefficients. The "
            "bending coefficient is sqrt(dim(c)/dim(a))·B(a,b,c) with an extra "
            "Frobenius-Schur phase on an already-dual line, so a provider must supply "
            "quantum dimensions, B-symbols and Frobenius-Schur signs (bend_unique for "
            "an Abelian symmetry where all three are 1, bend_braided otherwise — SU(2) "
            "takes the latter). Faking it would give correct shapes, correct sector "
            "bookkeeping and a wrong norm. A repartition that moves no leg across "
            "sides works today for every provider."
        ) from exc


@cache
def bend_plan(structure: TensorStructure, axis: int) -> BendPlan:
    """Plan bending ``axis`` of ``structure``. Cached: repeat calls return one object.

    ``axis`` must be the last leg of its own side; [bend][tenet.bend] validates that.

    The **body** is cached one level down, on ``_pattern(structure)``: block indices and
    bending coefficients read no degeneracy, so structures that differ only in degeneracy
    share one plan and this entry holds nothing but ``new_structure``.
    """
    plan = _pattern_bend_plan(_pattern(structure), axis)
    legs = _bent_legs(structure, axis)
    if legs == plan.new_structure.legs:
        return plan
    return replace(plan, new_structure=TensorStructure(legs))


def _on_side(leg: Leg, side: Side) -> Leg:
    """``leg`` on ``side``: itself, or bent — ``side`` and ``dual`` both flipped."""
    return leg if leg.side is side else replace(leg, side=side, dual=not leg.dual)


def _bent_legs(structure: TensorStructure, axis: int) -> tuple[Leg, ...]:
    """``structure``'s legs with ``axis`` flipped in ``side`` and ``dual``, moved last."""
    leg = structure.legs[axis]
    moved = _on_side(leg, IN if leg.side is OUT else OUT)
    return (*(other for i, other in enumerate(structure.legs) if i != axis), moved)


@cache
def _pattern_bend_plan(structure: TensorStructure, axis: int) -> BendPlan:
    """[bend_plan][tenet.bend_plan]'s body, on a degeneracy-free structure."""
    _refuse(structure, axis)
    provider = structure.provider
    leg = structure.legs[axis]
    right = leg.side is OUT
    # _refuse() above ran requires(provider, BendingCoefficients)
    bend_tree = provider.bend_right if right else provider.bend_left  # ty: ignore[unresolved-attribute]

    moved = replace(leg, side=IN if right else OUT, dual=not leg.dual)
    new_structure = TensorStructure(
        (*(other for i, other in enumerate(structure.legs) if i != axis), moved)
    )

    terms = []
    for src, key in enumerate(structure.block_order):
        for new_key, coeff in bend_tree(key, dual=leg.dual):
            if coeff == 0:
                continue
            terms.append((src, new_structure.index_of(new_key), coeff))
    return BendPlan(new_structure, tuple(terms))


def bend(t: "SymmetricTensor", axis: int) -> "SymmetricTensor":
    """Bend ``axis`` to the other side: ``side`` flipped, ``dual`` flipped, moved last.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose leg is bent.
    axis : int
        The public axis to bend. It must currently be the **last leg of its
        own side** (it need not be the last public axis).

    Returns
    -------
    SymmetricTensor
        The bent tensor: the moved leg takes the largest public position on
        its new side, with ``side`` and ``dual`` both flipped; ``space`` and
        ``name`` are preserved, so the block shapes are a permutation of the
        old ones.

    Raises
    ------
    ValueError
        If ``axis`` is out of range, or is not the last leg of its own side
        ([repartition][tenet.SymmetricTensor.repartition] transposes first and therefore never
        triggers this).
    CapabilityError
        If the provider does not implement ``BendingCoefficients`` — the
        coefficient is ``sqrt(dim(c)/dim(a))·B(a,b,c)`` with a
        Frobenius-Schur phase, and faking it would give correct shapes with a
        wrong norm.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> b = tenet.bend(a, 0)  # the OUT leg becomes a dual IN leg, moved last
    >>> b.legs[-1].side, b.legs[-1].dual
    (<Side.IN: 'in'>, True)
    """
    from tenet.tensor import _unchecked

    if not 0 <= axis < t.ndim:
        raise ValueError(f"bend: axis {axis} is out of range for a {t.ndim}-axis tensor")
    side_axes = _side_axes(t.structure, axis)
    if axis != side_axes[-1]:
        raise ValueError(
            f"bend: axis {axis} is not the last leg of its own "
            f"{t.legs[axis].side.value} side {side_axes}; a bend acts on the last "
            "uncoupled line of a tree. Transpose it into place first (repartition does)"
        )

    plan = bend_plan(t.structure, axis)
    perm = (*(i for i in range(t.ndim) if i != axis), axis)

    # a bend is a plan like any other, so it takes the same lowered route ``transpose``
    # and ``braid`` do: the source cells are read out of ``t``'s coupled-sector matrices
    # and written straight into the result's, and the blocks the loop below would build
    # -- along with the gather that would copy every one of them back into a matrix --
    # never exist (invariant 8). The loop stays for what ``lower_plan`` declines.
    mats = lower_plan(t, plan.new_structure, perm, plan.terms)
    if mats is not None:
        return from_matrices(plan.new_structure, mats)

    # one transpose per *distinct source*, not per term (#123): the permutation is
    # per-plan and only the coefficient is per-term, so every term sharing a source used
    # to recompute a byte-identical array. #74's batched alternative -- stack a shape
    # bucket, transpose once, slice back out -- was prototyped and measured slower on
    # every axis (see #123); this is the lever that was actually in the loop.
    sources = {s for s, _, _ in plan.terms}
    # a blockless tensor has no backend to resolve against, and no source to move either
    transpose = lib_fn(t.backend, "transpose") if sources else None
    blocks = t.blocks  # read once: it is a property, not a field
    moved = {src: transpose(blocks[src], perm) for src in sources} if transpose else {}

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
            f"bend: the plan fills {len(blocks)} of {n} target blocks — "
            f"{t.provider.name}'s bending coefficients dropped terms"
        )
    # one block per key of ``plan.new_structure``, each a transposed source block of
    # ``t`` under the plan's own permutation: the plan is the statement that the
    # shapes are right, so the ordinary constructor would only re-derive it (#328)
    return _unchecked(plan.new_structure, tuple(blocks[i] for i in range(n)))


def _validated(
    ndim: int, outputs: Sequence[int], inputs: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """``(outputs, inputs)`` as int tuples forming a permutation of ``range(ndim)``."""
    parts = []
    for name, axes in (("outputs", outputs), ("inputs", inputs)):
        try:
            parts.append(tuple(operator.index(a) for a in axes))
        except TypeError as exc:
            raise ValueError(f"repartition: {name} {tuple(axes)} must all be integers") from exc

    seen: set[int] = set()
    for name, axes in zip(("outputs", "inputs"), parts, strict=True):
        for a in axes:
            if not 0 <= a < ndim:
                raise ValueError(
                    f"repartition: axis {a} in {name} is out of range for a "
                    f"{ndim}-dimensional tensor"
                )
            if a in seen:
                raise ValueError(f"repartition: axis {a} appears twice in outputs/inputs")
            seen.add(a)
    missing = sorted(set(range(ndim)) - seen)
    if missing:
        raise ValueError(
            f"repartition: axis {missing[0]} is in neither outputs {parts[0]} nor "
            f"inputs {parts[1]}; together they must be a permutation of range({ndim})"
        )
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class RepartitionPlan:
    """The whole transpose→bend→transpose sandwich as one plan. Array-free, hashable.

    ``perm`` is the single composed axis permutation applied to every block;
    ``terms`` is ``((source block index, target block index, coefficient), ...)``
    with the whole chain's coefficients multiplied through. Same shape as
    ``BendPlan`` plus ``perm``, because a permutation is per-step, never
    per-term.
    """

    new_structure: TensorStructure
    perm: tuple[int, ...]
    terms: tuple[tuple[int, int, complex], ...]


def _compose(
    terms: tuple[tuple[int, int, complex], ...], following: tuple[tuple[int, int, complex], ...]
) -> tuple[tuple[int, int, complex], ...]:
    """Sparse product of two block maps, duplicate ``(src, dst)`` pairs summed.

    Zero results are kept: they are genuine cancellations, and the target block
    still has to exist for the fill check.
    """
    by_src: dict[int, list[tuple[int, complex]]] = {}
    for src, dst, coeff in following:
        by_src.setdefault(src, []).append((dst, coeff))

    merged: dict[tuple[int, int], complex] = {}
    for src, mid, coeff in terms:
        for dst, other in by_src.get(mid, ()):
            key = (src, dst)
            product = coeff * other
            merged[key] = product if key not in merged else merged[key] + product
    return tuple((src, dst, coeff) for (src, dst), coeff in merged.items())


@cache
def repartition_plan(
    structure: TensorStructure, outputs: tuple[int, ...], inputs: tuple[int, ...]
) -> RepartitionPlan:
    """Plan the whole ``repartition``. Cached: repeat calls return one object.

    The **body** is cached one level down, on ``_pattern(structure)``: the composed
    permutation and the composed block map are functions of the legs' sectors, sides and
    duals alone, and this entry holds only ``new_structure`` — ``structure``'s own legs in
    the requested order, with ``side`` and ``dual`` flipped on each leg that crossed.
    Without that split a growing U(1) bond rebuilds the whole transpose-bend-transpose
    chain at every bond whose degeneracies moved, which dominates the early sweeps of a
    long chain.
    """
    plan = _pattern_repartition_plan(_pattern(structure), outputs, inputs)
    want = {ax: OUT for ax in outputs} | {ax: IN for ax in inputs}
    legs = tuple(_on_side(structure.legs[ax], want[ax]) for ax in (*outputs, *inputs))
    if legs == plan.new_structure.legs:
        return plan
    return replace(plan, new_structure=TensorStructure(legs))


@cache
def _pattern_repartition_plan(
    structure: TensorStructure, outputs: tuple[int, ...], inputs: tuple[int, ...]
) -> RepartitionPlan:
    """[repartition_plan][tenet.repartition_plan]'s body, on a degeneracy-free structure.

    Walks exactly the chain [repartition][tenet.SymmetricTensor.repartition] executes --
    transpose the crossing leg to the end, bend it, and one final transpose -- but over
    structures instead of tensors, composing the sparse block maps and the
    permutations as it goes. ``bend``'s own permutation is the identity here (the
    leg has just been transposed to the end), so only the transposes' axes
    compose into ``perm``.
    """
    ndim = len(structure.legs)
    want = {ax: OUT for ax in outputs} | {ax: IN for ax in inputs}
    crossing = [ax for ax in range(ndim) if structure.legs[ax].side is not want[ax]]
    for ax in crossing:
        # refuse before any composition, naming the axis in the caller's numbering
        _refuse(structure, ax)

    perm = tuple(range(ndim))
    terms = tuple((i, i, 1.0) for i in range(structure.num_blocks))

    labels = list(range(ndim))  # labels[p] is the original axis now at position p
    for ax in crossing:
        p = labels.index(ax)
        axes = tuple(i for i in range(ndim) if i != p) + (p,)
        plan = permutation_plan(structure, axes)
        # applying ``perm`` then ``axes`` to a block is applying their composite
        perm = tuple(perm[i] for i in axes)
        terms = _compose(terms, plan.terms)
        structure = plan.new_structure
        labels.append(labels.pop(p))

        bplan = bend_plan(structure, ndim - 1)
        terms = _compose(terms, bplan.terms)
        structure = bplan.new_structure

    position = {a: p for p, a in enumerate(labels)}
    axes = tuple(position[ax] for ax in (*outputs, *inputs))
    plan = permutation_plan(structure, axes)
    return RepartitionPlan(
        plan.new_structure, tuple(perm[i] for i in axes), _compose(terms, plan.terms)
    )


def sides_plan(
    structure: TensorStructure, outputs: tuple[int, ...], inputs: tuple[int, ...]
) -> tuple[TensorStructure, tuple[int, ...], tuple[tuple[int, int, complex], ...]]:
    """The plan ``repartition(t, outputs, inputs)`` applies, as ``(structure, perm, terms)``.

    Parameters
    ----------
    structure : TensorStructure
        The structure the plan reads.
    outputs, inputs : tuple of int
        The public axes that end up OUT and IN; together a permutation of the axes.

    Returns
    -------
    new_structure : TensorStructure
        The repartitioned structure.
    perm : tuple of int
        The one per-block axis permutation.
    terms : tuple of (int, int, complex)
        ``(source block, target block, coefficient)``.

    Notes
    -----
    A repartition that moves no leg across sides is a plain transpose, and its plan is
    the plain permutation plan; asking ``repartition_plan`` for it would build the same
    thing through the bend chain's composition. Callers that want the *plan* rather than
    the tensor -- the operand lowering, the factorizations' ``_lower`` -- share this one
    branch instead of repeating it.
    """
    want = {ax: OUT for ax in outputs} | {ax: IN for ax in inputs}
    if any(structure.legs[ax].side is not want[ax] for ax in range(len(structure.legs))):
        plan = repartition_plan(structure, outputs, inputs)
        return plan.new_structure, plan.perm, plan.terms
    step = permutation_plan(structure, (*outputs, *inputs))
    return step.new_structure, step.axes, step.terms


def repartition(
    t: "SymmetricTensor", outputs: Sequence[int], inputs: Sequence[int]
) -> "SymmetricTensor":
    """Public axes ``outputs`` become OUT and ``inputs`` become IN.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor to repartition.
    outputs : sequence of int
        The public axes (in ``t``'s *original* numbering) that end up OUT.
    inputs : sequence of int
        The public axes that end up IN. Together with ``outputs`` they must
        be a permutation of ``range(t.ndim)``; negatives are refused.

    Returns
    -------
    SymmetricTensor
        The repartitioned tensor: public axis order exactly
        ``(*outputs, *inputs)``, and its legs are ``t``'s legs with ``side``
        (and, for every axis that actually crossed, ``dual``) adjusted.

    Raises
    ------
    ValueError
        If ``outputs``/``inputs`` contain a non-integer, an out-of-range or
        repeated axis, or miss an axis — together they must be a permutation
        of ``range(t.ndim)``.
    CapabilityError
        If a leg must cross between domain and codomain — a line bend — and
        the provider does not implement ``BendingCoefficients``. A
        repartition that moves no leg across sides works for every provider.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> W = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT), Leg(W, IN)), seed=0)
    >>> r = tenet.repartition(t, (0,), (1, 2))  # axis 1 crosses to the domain
    >>> r.structure.out_axes, r.structure.in_axes, r.legs[1].dual
    ((0,), (1, 2), True)

    Notes
    -----
    Owns no mathematics of its own: it transposes each crossing leg to the end,
    bends it, and transposes once more to the requested order — the whole chain
    composed once by ``repartition_plan`` and executed in a single pass, so
    every block is copied once instead of once per step.
    """

    outputs, inputs = _validated(t.ndim, outputs, inputs)

    want = {ax: OUT for ax in outputs} | {ax: IN for ax in inputs}
    if not any(t.legs[ax].side is not want[ax] for ax in range(t.ndim)):
        # no leg crosses: this is a plain transpose and must not pay for a plan
        return t.transpose((*outputs, *inputs))

    plan = repartition_plan(t.structure, outputs, inputs)
    return apply_plan(t, plan.new_structure, plan.perm, plan.terms, "repartition")


def _looped(
    t: "SymmetricTensor",
    perm: tuple[int, ...],
    terms: tuple[tuple[int, int, complex], ...],
    blocks: dict[int, Any],
) -> dict[int, Any]:
    """Accumulate ``terms`` into ``blocks``, one term at a time.

    The plan's original execution, kept whole: it runs the buckets batching would lose
    money on, and it is what [apply_plan][tenet.ops.repartition.apply_plan]'s batched path
    is checked against.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor the plan reads.
    perm : tuple of int
        The plan's single per-block axis permutation.
    terms : tuple of (int, int, complex)
        ``(source block, target block, coefficient)``.
    blocks : dict of int to array
        Destination index to block; accumulated into, and returned.

    Returns
    -------
    dict of int to array
        ``blocks``, with every term of ``terms`` added in.

    Notes
    -----
    One transpose per *distinct source*, not per term: ``perm`` is per-plan and only the
    coefficient is per-term, so terms sharing a source would otherwise recompute a
    byte-identical array — 2.87 terms per source at SU(2) ``chi=6`` against exactly 1.00
    at U(1), the multi-term expansion being what a non-Abelian provider's coefficients
    produce and an Abelian one never does.
    """
    if not terms:  # a blockless tensor has no backend to resolve against either
        return blocks
    transpose = lib_fn(t.backend, "transpose")
    source = t.blocks  # read once: it is a property, not a field
    moved = {src: transpose(source[src], perm) for src in {s for s, _, _ in terms}}
    for src, dst, coeff in terms:
        contrib = moved[src]
        if coeff != 1:
            # keep a real coefficient real, so a real tensor stays real
            contrib = scaled(contrib, coeff.real if getattr(coeff, "imag", 0) == 0 else coeff)
        blocks[dst] = contrib if dst not in blocks else blocks[dst] + contrib
    return blocks


def _batches(t: "SymmetricTensor") -> bool:
    """Whether this tensor's backend is one the batched path has been measured to help.

    Batching trades Python iterations for array operations, and whether that trade pays
    is a property of the *backend*, not of the plan: NumPy's per-call overhead is small
    enough that a few hundred terms already lose to it, while an array library's is an
    order of magnitude larger, so the loop it replaces was never the cost and the stack
    and the gather it adds are graph nodes with backward passes of their own. Measured on
    an SU(2) bending plan, batched over looped:

    ======  ======  ==============  ============
    terms   NumPy   JAX eager       JAX traced
    ======  ======  ==============  ============
    270     0.72    2.16            1.69
    467     0.42    1.68            2.02
    4076    0.41    1.09            1.33
    4041    --      1.11            0.87
    74272   0.30    --              --
    ======  ======  ==============  ============

    So the gate is the backend and not a term count: JAX's loss is not monotone in the
    term count -- the two four-thousand-term plans above disagree in *direction* -- so a
    threshold would be a number with nothing behind it. Block shapes, bucket
    multiplicities and the backward pass of the gather all enter, and none of them is
    read off the term count.

    PyTorch takes the loop because it has not been measured, not because it has been
    measured to lose; ``tests/backends/test_torch.py`` is where that measurement would go.

    A tensor with no blocks has no backend to ask about, and no terms to batch either.
    """
    return bool(t.blocks) and t.backend == "numpy"


def apply_plan(
    t: "SymmetricTensor",
    structure: TensorStructure,
    perm: tuple[int, ...],
    terms: tuple[tuple[int, int, complex], ...],
    caller: str,
) -> "SymmetricTensor":
    """Move ``t``'s blocks by ``(perm, terms)`` and build the tensor they fill.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor the plan reads.
    structure : TensorStructure
        The plan's ``new_structure``.
    perm : tuple of int
        The plan's single per-block axis permutation.
    terms : tuple of (int, int, complex)
        ``(source block, target block, coefficient)``.
    caller : str
        The name the fill-check message opens with.

    Returns
    -------
    SymmetricTensor
        The tensor the plan builds.

    Raises
    ------
    ValueError
        If the plan does not fill every target block — the provider's coefficients
        dropped terms.

    Notes
    -----
    Every term shares the plan's single ``perm``, so the whole plan is one transpose per
    source followed by a scatter-add, and terms whose destination has the same shape,
    hence whose source has the same shape, stack and run as array operations instead of
    Python iterations. The buckets are few: a rank-8 SU(2) intermediate with 74,800
    blocks and 447,752 terms has 1 bucket at uniform degeneracies and ~4,000 at the most
    ragged ones. [batch_plan][tenet.ops.batch.batch_plan] holds the index arrays, keyed on
    the plan, so the grouping is paid once per distinct plan rather than once per call.

    **The grouping runs through [lower_plan][tenet.map_view.lower_plan] where it can**,
    which writes each bucket's rows straight into the result's coupled-sector matrices.
    The tensor holds those matrices, so the blocks this would otherwise build -- and the
    gather that would then copy every one of them into a matrix -- never exist: one pass
    where there were two. It declines on an immutable backend and on a genuinely complex
    coefficient, and there the loop below builds blocks and the constructor gathers them.

    A bucket too small to repay the array-operation overhead — an Abelian plan is one term
    per destination, so nothing to fuse and pure loss — stays on
    [_looped][tenet.ops.repartition._looped], which is also the reference the batched path
    is tested against, term for term and bit for bit, and which runs the whole plan on
    every other backend (see [_batches][tenet.ops.repartition._batches]).
    """
    from tenet.tensor import _unchecked

    if structure == t.structure and is_identity_plan(structure, perm, terms):
        return t  # the plan rebuilds what it reads; tensors are immutable, so hand it back

    mats = lower_plan(t, structure, perm, terms)
    if mats is not None:
        return from_matrices(structure, mats)

    groups, loose = batch_plan(structure, perm, terms) if _batches(t) else ((), terms)

    source = t.blocks if groups else ()  # read once: it is a property, not a field
    blocks: dict[int, Any] = {}
    for srcs, buckets in groups:
        stacked = ar.do("stack", tuple(source[s] for s in srcs))
        stacked = ar.do("transpose", stacked, (0, *(ax + 1 for ax in perm)))
        for take, coeff, width, dsts in buckets:
            rows = ar.do("multiply", stacked[take], cast_coefficients(coeff, stacked))
            rows = ar.do("reshape", rows, (len(dsts), width, *ar.shape(rows)[1:]))
            acc = rows[:, 0]
            for i in range(1, width):
                acc = ar.do("add", acc, rows[:, i])
            for p, dst in enumerate(dsts):
                blocks[dst] = acc[p]
    _looped(t, perm, loose, blocks)

    n = structure.num_blocks
    if len(blocks) != n:
        raise ValueError(
            f"{caller}: the plan fills {len(blocks)} of {n} target blocks — "
            f"{t.provider.name}'s coefficients dropped terms"
        )
    # ``structure`` is the structure the plan was built against and every ``dst`` is
    # an index into its ``block_order``; the batched and looped walks both write the
    # shape that structure's own tables dictate (#328)
    return _unchecked(structure, tuple(blocks[i] for i in range(n)))


def _flip_refuse(structure: TensorStructure) -> None:
    """Turn the bare capability failure into a message a user can act on."""
    provider = structure.provider
    for capability in (FSIndicatorData, TwistData):
        try:
            requires(provider, capability)
        except CapabilityError as exc:
            raise CapabilityError(
                f"flip_dual: toggling a leg's dual flag re-expresses the leg through the "
                f"V_a -> V_a^* isomorphism, and provider {provider.name} does not implement "
                f"{capability.__name__}. The scalar is chi_a * theta_a — the Frobenius-Schur "
                "indicator (FSIndicatorData) times the twist (TwistData) — per flipped leg "
                "per fusion tree (both 1 for a "
                "bosonic Abelian symmetry, twist (-1)^parity for fermion parity, chi the FS "
                "phase for SU(2)/SU(N)). Faking it would give correct shapes, correct "
                "sector bookkeeping and a wrong sign."
            ) from exc


def _flip_axes(structure: TensorStructure, axes: Any) -> tuple[int, ...]:
    """``axes`` — an int, a leg name, or a sequence of either — as axis indices."""
    ndim = structure.ndim
    single = isinstance(axes, str) or not isinstance(axes, Sequence)
    resolved = []
    for item in (axes,) if single else tuple(axes):
        try:
            ax = operator.index(item)
        except TypeError:
            names = [i for i, leg in enumerate(structure.legs) if leg.name == item]
            if not names:
                raise ValueError(
                    f"flip_dual: no leg is named {item!r}; axes are ints or leg names"
                ) from None
            if len(names) > 1:
                raise ValueError(
                    f"flip_dual: leg name {item!r} is ambiguous — axes {tuple(names)} all "
                    "carry it; use the axis index instead"
                ) from None
            ax = names[0]
        else:
            if not 0 <= ax < ndim:
                raise ValueError(
                    f"flip_dual: axis {ax} is out of range for a {ndim}-dimensional tensor"
                )
        if ax in resolved:
            raise ValueError(f"flip_dual: axis {ax} is repeated in {axes!r}")
        resolved.append(ax)
    return tuple(resolved)


Runs = tuple[tuple[complex, int], ...]
"""``(coefficient, extent)`` per band of one side of one coupled-sector matrix."""


@plan_cache(cost=lambda plan: 1 + sum(len(v) for pair in plan[1] for v in pair if v is not None))
def _flip_plan(
    structure: TensorStructure, picked: tuple[int, ...], inv: bool
) -> tuple[TensorStructure, tuple[tuple[Runs | None, Runs | None], ...]]:
    """``flip_dual``'s structure and its row/column scalings, one pair per coupled sector.

    Parameters
    ----------
    structure : TensorStructure
        The structure to flip.
    picked : tuple of int
        The axes whose ``dual`` flag toggles, already resolved and deduplicated.
    inv : bool
        Whether the exact inverse is wanted instead.

    Returns
    -------
    new_structure : TensorStructure
        The flipped structure -- each picked leg's space relabelled through
        ``provider.dual`` and its ``dual`` flag toggled.
    scales : tuple of (Runs or None, Runs or None)
        Per coupled sector, in ``map_layout(structure).sectors`` order, the coefficient
        to scale each row band of ``B_c`` by and the coefficient to scale each column
        band by, as ``(coefficient, extent)`` runs; ``None`` where that side's
        coefficients are all 1. Plain Python numbers, never arrays -- the array is
        [band_scale][tenet.ops.batch.band_scale]'s, cached beside the plan.

    Raises
    ------
    RuntimeError
        If the flipped structure does not lay out identically to the original -- see the
        Notes, which is the claim this checks rather than assumes.

    Notes
    -----
    The flip is a *scaling of the stored matrices*, not a rebuild of them, and that is a
    derivation rather than an observation. The relabel and the flag toggle cancel inside
    ``Leg.fused_sector``, so every fusion-tree leaf survives the flip untouched; that
    fixes ``block_order``, and it fixes the degeneracy each leaf carries, because
    ``Leg.degeneracy`` reads the relabelled space through the same cancelling pair. Both
    inputs to [map_layout][tenet.map_layout] -- which configurations exist, and what each
    one's degeneracies are -- are therefore unchanged, and the flipped structure gets not
    a permutation of the source's bands but the *same* bands, offsets and extents. The
    relabel that a U(1) leg's sectors undergo (``{q}`` arriving as ``{-q}``, which
    ``GradedSpace.new`` then re-sorts) reaches only ``leg.space``, which no part of the
    layout reads without going back through ``space_sector``. The check above is the
    proof standing rather than the assumption resting: it costs one tuple comparison per
    distinct flip and is cached with the plan.

    What is left is one scalar per block, ``chi_a * theta_a`` per flipped leg per tree,
    and that scalar **factorizes**: a flipped OUT leg reads the block's output tree only
    and a flipped IN leg its input tree only, so the coefficient of the block at
    ``(row band, column band)`` is a row coefficient times a column coefficient. A
    per-block grid of scalars that factorizes is a pair of diagonal scalings, and a
    diagonal scaling of a matrix is one array operation -- so ``flip_dual`` reads
    ``data`` and hands back a tensor that still holds matrices, instead of cutting every
    block out to multiply it by a number (invariant 8).
    """
    provider = structure.provider
    legs = list(structure.legs)
    for ax in picked:
        leg = legs[ax]
        relabelled = GradedSpace.new(
            provider, tuple((provider.dual(a), m) for a, m in leg.space.sectors)
        )
        legs[ax] = replace(leg, space=relabelled, dual=not leg.dual)
    new_structure = TensorStructure(tuple(legs))

    layout, flipped = map_layout(structure), map_layout(new_structure)
    if (
        new_structure.block_order != structure.block_order
        or flipped.sectors != layout.sectors
        or flipped.rows != layout.rows
        or flipped.cols != layout.cols
    ):
        raise RuntimeError(
            f"flip_dual: relabelling axes {picked} moves {provider.name}'s layout, so the "
            "flip is not a scaling of the coupled-sector matrices; the block route is what "
            "such a grading would need"
        )

    # Position of each flipped axis inside its own side's tree.
    tree_pos = {ax: k for k, ax in enumerate(structure.out_axes)}
    tree_pos |= {ax: k for k, ax in enumerate(structure.in_axes)}
    out_picked = tuple(ax for ax in picked if structure.legs[ax].side is OUT)
    in_picked = tuple(ax for ax in picked if structure.legs[ax].side is IN)

    def coefficient(tree: FusionTree, axes: tuple[int, ...]) -> complex:
        """One band's scalar: chi * theta per flipped leg of this side, on this tree."""
        factor: complex = 1.0
        for ax in axes:
            leg = structure.legs[ax]
            # the leaf is invariant under the flip, so old tree and new tree agree.
            # flip_dual() ran requires(provider, FSIndicatorData/TwistData) before this
            # plan; chi * theta is the flip scalar, per flipped leg per tree.
            a = tree.uncoupled[tree_pos[ax]]
            base = complex(provider.frobenius_schur(a) * provider.twist(a))  # ty: ignore[unresolved-attribute]
            if leg.side is IN:  # the input tree enters the pairing conjugated
                base = base.conjugate()
            if not inv:
                factor *= base if leg.dual else 1.0
            else:
                factor *= 1.0 if leg.dual else base.conjugate()
        return factor

    def side_runs(axes: tuple[int, ...], bands: tuple[tuple[Any, int, int], ...]) -> Runs | None:
        """One side's runs, or ``None`` when every coefficient on it is 1."""
        runs = tuple((coefficient(tree, axes), extent) for tree, _, extent in bands)
        return None if all(coeff == 1 for coeff, _ in runs) else runs

    scales = tuple(
        (
            side_runs(out_picked, layout.row_bands(c)),
            side_runs(in_picked, layout.col_bands(c)),
        )
        for c in layout.sectors
    )
    return new_structure, scales


def flip_dual(
    t: "SymmetricTensor",
    axes: int | Hashable | Sequence[int | Hashable],
    *,
    inv: bool = False,
) -> "SymmetricTensor":
    """Toggle the ``dual`` flag of ``axes``, keeping the tensor the same morphism.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose legs are flipped.
    axes : int, leg name, or sequence of either
        The legs to flip; ``flip_dual(t, ())`` is ``t``. A name must be carried by
        exactly one leg.
    inv : bool, optional
        ``flip_dual`` is **not** an involution; ``inv=True`` applies the exact
        inverse instead. Default ``False``.

    Returns
    -------
    SymmetricTensor
        The same morphism with each named leg's ``dual`` toggled and its
        space relabelled through ``provider.dual``; ``side`` and ``name``
        are unchanged, and so are the block set, order and shapes.

    Raises
    ------
    ValueError
        If an axis is out of range or repeated, if a leg name matches no leg,
        or if it matches more than one (use the axis index instead).
    CapabilityError
        If the provider does not implement
        [FSIndicatorData][tenet.symmetry.FSIndicatorData] and
        [TwistData][tenet.symmetry.TwistData] — the scalar is
        ``chi_a * theta_a`` per flipped leg per fusion tree, and faking it
        would give correct shapes with a wrong sign.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> f = tenet.flip_dual(a, 0)  # charge q relabelled as -q on a dual leg
    >>> f.legs[0].dual, f.legs[0].space.sectors
    (True, ((U1Sector(charge=-1), 1), (U1Sector(charge=0), 1)))
    >>> bool(tenet.allclose(tenet.flip_dual(f, 0, inv=True), a))
    True

    Notes
    -----
    **Not** ``numpy.flip``: no axis is reversed and no element moves. Each named
    leg's ``dual`` flag is toggled and its space is relabelled through
    ``provider.dual`` (so a U(1) leg over charges ``{q}`` comes back over
    ``{-q}``), which is the ``V_a -> V_a^*`` isomorphism made explicit — the
    operation TensorKit spells ``flip``. The name is qualified here and not there
    because Python has ``numpy.flip``, which reverses element order along an axis
    of the *tensor* while this toggles a flag on a *leg* -- a different operand and
    a different operation under one name, reachable through autoray's module lookup.
    YASTN, the Python API reference, qualifies the same operation the same
    way (``flip_signature`` / ``flip_charges``); ``dual`` is this package's noun for
    the flag. ``side`` and ``name`` are unchanged:
    moving a leg between domain and codomain stays
    [repartition][tenet.SymmetricTensor.repartition]'s job.

    Two contracts, both TensorKit's: flipping the two legs of a
    contractible pair leaves the contraction result unchanged, and ``flip_dual`` is
    **not** an involution — flipping the same leg twice multiplies each tree by
    ``chi_a * theta_a`` once (``-1`` on an SU(2) half-integer or odd
    fermion-parity line), and ``inv=True`` is the exact inverse instead.

    Because the relabel and the flag toggle cancel inside ``Leg.fused_sector``,
    every fusion-tree leaf -- and with it the block set, order, shapes and coupled-sector
    layout -- is unchanged, so the whole operation is one scalar per block. That scalar
    factorizes over the two trees, which makes it a diagonal scaling of the rows and of
    the columns of each stored matrix: no block is ever cut out to be multiplied by a
    number.
    """
    from tenet.tensor import SymmetricTensor, _relabelled

    picked = _flip_axes(t.structure, axes)
    if not picked:
        return t
    _flip_refuse(t.structure)

    new_structure, scales = _flip_plan(t.structure, picked, inv)
    if all(rows is None and cols is None for rows, cols in scales):
        # a bosonic grading has chi * theta == 1 everywhere, so the flip is a pure
        # relabel: keep the storage in whichever form it is already held in
        return _relabelled(t, new_structure)

    data = []
    for mat, (rows, cols) in zip(t.data, scales, strict=True):
        if rows is not None:
            mat = ar.do("multiply", mat, band_scale(rows, mat, 0))
        if cols is not None:
            mat = ar.do("multiply", mat, band_scale(cols, mat, 1))
        data.append(mat)
    return SymmetricTensor.from_data(new_structure, tuple(data))
