"""Line bending — ``T.repartition(outputs=..., inputs=...)`` — Milestone 3.

Moving a leg between domain and codomain is a categorical **bend**, not a
Boolean flip of ``side`` (docs/design.md "Repartitioning is different from viewing").
Two structural facts outlive whatever coefficients a provider supplies, and both
are fixed here:

* **A bend flips ``side`` *and* ``dual`` on the moved leg.** The ``GradedSpace``
  is untouched, so every block shape is a permutation of the old one; what
  changes is ``Leg.fused_sector``, which now returns the dualized label the new
  tree needs (a U(1) charge ``q`` arrives on the other side as ``-q``). A model
  that identified IN with ``dual`` could not express this at all — invariant 2
  doing real work. Since #142 that sentence names two separate operations:
  [flip][tenet.flip] toggles ``dual`` alone (relabelling the space and paying the
  Z-isomorphism's scalar), while a bend is the operation that moves ``side``.
* **Our two trees are independent, both in ascending public-axis order.**
  TensorKit reads ``Hom(b₁⊗…⊗b_{N₂}, a₁⊗…⊗a_{N₁}) ≅ Hom(1, a₁⊗…⊗a_{N₁}⊗
  b*_{N₂}⊗…⊗b*₁)`` — the domain **reversed** — and therefore builds its
  ``repartition`` out of a cyclic index tuple. TeNeT-py deliberately does **not**
  adopt that planar reading: the pairing of the two trees lives in
  ``FusionBlockKey``, not in a cyclic order. The consequence is that a bend
  appends to the destination tree's *end*, i.e. the moved leg takes the largest
  public position on its new side — which is exactly what [bend][tenet.bend] enforces
  and what [repartition][tenet.SymmetricTensor.repartition]'s final ``transpose`` then corrects.

[bend][tenet.bend] is the only new mathematics, and it is deliberately minimal: it
bends the **last leg of its own side** and nothing else. Everything else is
reached by ``transpose`` (#21), so all reordering refusals come from that
already-tested capability gate:

```text
transpose  bring the leg to be moved to the end
bend       one primitive bend per moved leg
transpose  deliver (*outputs, *inputs)
```

The coefficient is ``sqrt(dim(c)/dim(a)) · B(a,b,c)`` times a Frobenius-Schur
phase, and for Trivial and U(1) it is provably exactly ``1``; SU(2) computes it
from its B-symbols (#38). A provider that supplies none of that does not
implement ``BendingCoefficients`` and is refused loudly rather than handed a
plausible tensor with the wrong norm.

No NumPy and no ``to_dense`` here (invariants 8/9): plans are array-free
metadata and blocks move only through ``ar.do("transpose", ...)``.
"""

import operator
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.leg import IN, OUT
from tenet.ops.permutation import permutation_plan
from tenet.space import GradedSpace
from tenet.structure import TensorStructure
from tenet.symmetry.base import BendingCoefficients, CapabilityError, FlipPhase, requires

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "BendPlan",
    "RepartitionPlan",
    "bend",
    "bend_plan",
    "flip",
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
    """
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
    from tenet.tensor import SymmetricTensor

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

    # one transpose per *distinct source*, not per term (#123): the permutation is
    # per-plan and only the coefficient is per-term, so every term sharing a source used
    # to recompute a byte-identical array. #74's batched alternative -- stack a shape
    # bucket, transpose once, slice back out -- was prototyped and measured slower on
    # every axis (see #123); this is the lever that was actually in the loop.
    moved = {src: ar.do("transpose", t.blocks[src], perm) for src in {s for s, _, _ in plan.terms}}

    blocks: dict[int, Any] = {}
    for src, dst, coeff in plan.terms:
        contrib = moved[src]
        if coeff != 1:
            # keep a real coefficient real, so a real tensor stays real
            contrib = contrib * (coeff.real if getattr(coeff, "imag", 0) == 0 else coeff)
        blocks[dst] = contrib if dst not in blocks else blocks[dst] + contrib

    n = plan.new_structure.num_blocks
    if len(blocks) != n:
        raise ValueError(
            f"bend: the plan fills {len(blocks)} of {n} target blocks — "
            f"{t.provider.name}'s bending coefficients dropped terms"
        )
    return SymmetricTensor(plan.new_structure, tuple(blocks[i] for i in range(n)))


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

    Walks exactly the chain [repartition][tenet.SymmetricTensor.repartition] used to
    execute — transpose the
    crossing leg to the end, bend it, and one final transpose — but over
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
    from tenet.tensor import SymmetricTensor

    outputs, inputs = _validated(t.ndim, outputs, inputs)

    want = {ax: OUT for ax in outputs} | {ax: IN for ax in inputs}
    if not any(t.legs[ax].side is not want[ax] for ax in range(t.ndim)):
        # no leg crosses: this is a plain transpose and must not pay for a plan
        return t.transpose((*outputs, *inputs))

    plan = repartition_plan(t.structure, outputs, inputs)

    # one transpose per *distinct source*, not per term (#123): ``plan.perm`` is per-plan
    # and only the coefficient is per-term, so every term sharing a source used to
    # recompute a byte-identical array -- 2.87 terms per source at SU(2) chi=6 against
    # exactly 1.00 at U(1), the multi-term expansion being what a non-Abelian provider's
    # coefficients produce and an Abelian one never does. #74's batched alternative
    # (stack a shape bucket, transpose once, slice back out) was prototyped and measured
    # slower on every axis -- see #123 for the table and the refusal.
    moved = {
        src: ar.do("transpose", t.blocks[src], plan.perm) for src in {s for s, _, _ in plan.terms}
    }

    blocks: dict[int, Any] = {}
    for src, dst, coeff in plan.terms:
        contrib = moved[src]
        if coeff != 1:
            # keep a real coefficient real, so a real tensor stays real
            contrib = contrib * (coeff.real if getattr(coeff, "imag", 0) == 0 else coeff)
        blocks[dst] = contrib if dst not in blocks else blocks[dst] + contrib

    n = plan.new_structure.num_blocks
    if len(blocks) != n:
        raise ValueError(
            f"repartition: the plan fills {len(blocks)} of {n} target blocks — "
            f"{t.provider.name}'s coefficients dropped terms"
        )
    return SymmetricTensor(plan.new_structure, tuple(blocks[i] for i in range(n)))


def _flip_refuse(structure: TensorStructure) -> None:
    """Turn the bare capability failure into a message a user can act on."""
    provider = structure.provider
    try:
        requires(provider, FlipPhase)
    except CapabilityError as exc:
        raise CapabilityError(
            f"flip: toggling a leg's dual flag re-expresses the leg through the "
            f"V_a -> V_a^* isomorphism, and provider {provider.name} does not implement "
            "FlipPhase. The scalar is chi_a * theta_a — the Frobenius-Schur phase times "
            "the twist — per flipped leg per fusion tree, so a provider must supply "
            "flip_phase (1 for a bosonic Abelian symmetry, (-1)^parity for fermion "
            "parity, the FS phase for SU(2)/SU(N)). Faking it would give correct "
            "shapes, correct sector bookkeeping and a wrong sign."
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
                    f"flip: no leg is named {item!r}; axes are ints or leg names"
                ) from None
            if len(names) > 1:
                raise ValueError(
                    f"flip: leg name {item!r} is ambiguous — axes {tuple(names)} all "
                    "carry it; use the axis index instead"
                ) from None
            ax = names[0]
        else:
            if not 0 <= ax < ndim:
                raise ValueError(f"flip: axis {ax} is out of range for a {ndim}-dimensional tensor")
        if ax in resolved:
            raise ValueError(f"flip: axis {ax} is repeated in {axes!r}")
        resolved.append(ax)
    return tuple(resolved)


def flip(
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
        The legs to flip; ``flip(t, ())`` is ``t``. A name must be carried by
        exactly one leg.
    inv : bool, optional
        ``flip`` is **not** an involution; ``inv=True`` applies the exact
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
        [FlipPhase][tenet.symmetry.FlipPhase] — the scalar is
        ``chi_a * theta_a`` per flipped leg per fusion tree, and faking it
        would give correct shapes with a wrong sign.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> f = tenet.flip(a, 0)  # charge q relabelled as -q on a dual leg
    >>> f.legs[0].dual, f.legs[0].space.sectors
    (True, ((U1Sector(charge=-1), 1), (U1Sector(charge=0), 1)))
    >>> bool(tenet.allclose(tenet.flip(f, 0, inv=True), a))
    True

    Notes
    -----
    **Not** ``numpy.flip``: no axis is reversed and no element moves. Each named
    leg's ``dual`` flag is toggled and its space is relabelled through
    ``provider.dual`` (so a U(1) leg over charges ``{q}`` comes back over
    ``{-q}``), which is the ``V_a -> V_a^*`` isomorphism made explicit — the
    operation TensorKit calls ``flip``. ``side`` and ``name`` are unchanged:
    moving a leg between domain and codomain stays
    [repartition][tenet.SymmetricTensor.repartition]'s job.

    Two contracts, both TensorKit's: flipping the two legs of a
    contractible pair leaves the contraction result unchanged, and ``flip`` is
    **not** an involution — flipping the same leg twice multiplies each tree by
    ``chi_a * theta_a`` once (``-1`` on an SU(2) half-integer or odd
    fermion-parity line), and ``inv=True`` is the exact inverse instead.

    Because the relabel and the flag toggle cancel inside ``Leg.fused_sector``,
    every fusion-tree leaf — and with it the block set, order and shapes — is
    unchanged: the whole operation is one scalar per block.
    """
    from tenet.tensor import SymmetricTensor

    picked = _flip_axes(t.structure, axes)
    if not picked:
        return t
    _flip_refuse(t.structure)

    structure = t.structure
    provider = structure.provider
    legs = list(structure.legs)
    for ax in picked:
        leg = legs[ax]
        relabelled = GradedSpace.new(
            provider, tuple((provider.dual(a), m) for a, m in leg.space.sectors)
        )
        legs[ax] = replace(leg, space=relabelled, dual=not leg.dual)
    new_structure = TensorStructure(tuple(legs))

    # Position of each flipped axis inside its own side's tree.
    tree_pos = {ax: k for k, ax in enumerate(structure.out_axes)}
    tree_pos |= {ax: k for k, ax in enumerate(structure.in_axes)}

    blocks = []
    for key, block in zip(structure.block_order, t.blocks, strict=True):
        factor: complex = 1.0
        for ax in picked:
            leg = structure.legs[ax]
            out = leg.side is OUT
            tree = key.output_tree if out else key.input_tree
            # the leaf is invariant under the flip, so old key and new key agree
            # flip() ran requires(provider, FlipPhase) before building this plan
            base = complex(provider.flip_phase(tree.uncoupled[tree_pos[ax]]))  # ty: ignore[unresolved-attribute]
            if not out:  # the input tree enters the pairing conjugated
                base = base.conjugate()
            if not inv:
                factor *= base if leg.dual else 1.0
            else:
                factor *= 1.0 if leg.dual else base.conjugate()
        if factor != 1:
            # keep a real coefficient real, so a real tensor stays real
            block = block * (factor.real if factor.imag == 0 else factor)
        blocks.append(block)
    # same key set, same sorted order: blocks stay aligned position for position
    return SymmetricTensor(new_structure, tuple(blocks))
