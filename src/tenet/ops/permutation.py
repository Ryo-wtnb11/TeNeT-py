"""Axis permutation — ``T.transpose(*axes)`` / [tenet.transpose][] — Milestone 2.

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
codomain is a bend (``repartition``, Milestone 3). What it does change is
``out_axes``/``in_axes``, which are *positions*.

No NumPy and no ``to_dense`` here (invariants 8/9): the plan is array-free
metadata and blocks move only through ``ar.do("transpose", ...)``.
"""

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry.base import CapabilityError, PermutationCoefficients, requires

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["PermutationPlan", "permutation_plan", "transpose"]


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


def _refuse(provider: Any, axes: tuple[int, ...], offenders: str) -> None:
    """Turn the bare capability failure into a message a user can act on."""
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
    """
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
    _refuse(provider, axes, offenders)

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
        [PermutationCoefficients][tenet.symmetry.PermutationCoefficients].
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
    from tenet.tensor import SymmetricTensor

    axes = tuple(reversed(range(t.ndim))) if axes is None else _validated_axes(t.ndim, tuple(axes))
    plan = permutation_plan(t.structure, axes)

    # one transpose per *distinct source*, not per term (#123): ``plan.axes`` is per-plan
    # and only the coefficient is per-term, so every term sharing a source used to
    # recompute a byte-identical array. #74's batched alternative -- stack a shape bucket,
    # transpose once, slice back out -- was prototyped and measured slower on every axis
    # (see #123 for the table and the refusal).
    moved = {
        src: ar.do("transpose", t.blocks[src], plan.axes) for src in {s for s, _, _ in plan.terms}
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
            f"transpose: the plan fills {len(blocks)} of {n} target blocks — "
            f"{t.provider.name}.permute_tree dropped terms"
        )
    return SymmetricTensor(plan.new_structure, tuple(blocks[i] for i in range(n)))
