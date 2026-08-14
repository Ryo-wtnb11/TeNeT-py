"""Line bending — ``T.repartition(outputs=..., inputs=...)`` — Milestone 3.

Moving a leg between domain and codomain is a categorical **bend**, not a
Boolean flip of ``side`` (README "Repartitioning is different from viewing").
Two structural facts outlive whatever coefficients a provider supplies, and both
are fixed here:

* **A bend flips ``side`` *and* ``dual`` on the moved leg.** The ``GradedSpace``
  is untouched, so every block shape is a permutation of the old one; what
  changes is ``Leg.fused_sector``, which now returns the dualized label the new
  tree needs (a U(1) charge ``q`` arrives on the other side as ``-q``). A model
  that identified IN with ``dual`` could not express this at all — invariant 2
  doing real work.
* **Our two trees are independent, both in ascending public-axis order.**
  TensorKit reads ``Hom(b₁⊗…⊗b_{N₂}, a₁⊗…⊗a_{N₁}) ≅ Hom(1, a₁⊗…⊗a_{N₁}⊗
  b*_{N₂}⊗…⊗b*₁)`` — the domain **reversed** — and therefore builds its
  ``repartition`` out of a cyclic index tuple. TeNeT-py deliberately does **not**
  adopt that planar reading: the pairing of the two trees lives in
  ``FusionBlockKey``, not in a cyclic order. The consequence is that a bend
  appends to the destination tree's *end*, i.e. the moved leg takes the largest
  public position on its new side — which is exactly what :func:`bend` enforces
  and what :func:`repartition`'s final ``transpose`` then corrects.

:func:`bend` is the only new mathematics, and it is deliberately minimal: it
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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.leg import IN, OUT
from tenet.ops.permutation import permutation_plan
from tenet.structure import TensorStructure
from tenet.symmetry.base import BendingCoefficients, CapabilityError, requires

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "BendPlan",
    "RepartitionPlan",
    "bend",
    "bend_plan",
    "repartition",
    "repartition_plan",
]


@dataclass(frozen=True, slots=True)
class BendPlan:
    """The categorical half of one bend: static, array-free, hashable.

    ``terms`` is ``((source block index, target block index, coefficient), ...)``
    with plain Python numbers, never arrays — the same shape of plan as
    :class:`~tenet.ops.permutation.PermutationPlan`.
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

    ``axis`` must be the last leg of its own side; :func:`bend` validates that.
    """
    _refuse(structure, axis)
    provider = structure.provider
    leg = structure.legs[axis]
    right = leg.side is OUT
    bend_tree = provider.bend_right if right else provider.bend_left

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

    ``axis`` must currently be the **last leg of its own side** (it need not be
    the last public axis). ``space`` and ``name`` are preserved, so the block
    shapes are a permutation of the old ones. :func:`repartition` transposes
    first and therefore never triggers the ``ValueError``.
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

    blocks: dict[int, Any] = {}
    for src, dst, coeff in plan.terms:
        contrib = ar.do("transpose", t.blocks[src], perm)
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
    :class:`BendPlan` plus ``perm``, because a permutation is per-step, never
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

    Walks exactly the chain :func:`repartition` used to execute — transpose the
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

    The result's public axis order is exactly ``(*outputs, *inputs)``, and its
    legs are ``t``'s legs with ``side`` (and, for every axis that actually
    crossed, ``dual``) adjusted. Axes are named in ``t``'s *original* numbering
    throughout.

    Owns no mathematics of its own: it transposes each crossing leg to the end,
    bends it, and transposes once more to the requested order — the whole chain
    composed once by :func:`repartition_plan` and executed in a single pass, so
    every block is copied once instead of once per step.
    """
    from tenet.tensor import SymmetricTensor

    outputs, inputs = _validated(t.ndim, outputs, inputs)

    want = {ax: OUT for ax in outputs} | {ax: IN for ax in inputs}
    if not any(t.legs[ax].side is not want[ax] for ax in range(t.ndim)):
        # no leg crosses: this is a plain transpose and must not pay for a plan
        return t.transpose((*outputs, *inputs))

    plan = repartition_plan(t.structure, outputs, inputs)

    blocks: dict[int, Any] = {}
    for src, dst, coeff in plan.terms:
        contrib = ar.do("transpose", t.blocks[src], plan.perm)
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
