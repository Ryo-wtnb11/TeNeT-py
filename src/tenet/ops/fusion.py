"""Explicit leg fusion and its inverse — the primitive docs/design.md puts in place of ``reshape``.

The whole operation rests on one observation: a left-associated fusion tree
``uncoupled = (u_0, u_1, u_2, ...)`` *already contains* the map that fuses its
two leading lines — that is its first vertex. Dropping that vertex gives the
tree ``uncoupled = (e_1, u_2, ...)`` with coefficient exactly ``1``, for any
provider, multiplicity included. So there is one code path, uniform across
Trivial / U(1) / SU(2), it is exact, and it never touches a Clebsch-Gordan
array. What is *not* reachable this way is fusing a group that is not a prefix
of its side's order — that needs an F-move (Milestone 4).

Conventions fixed here once and depended on downstream:

* **Side prefix, not public adjacency.** The fused axes, ordered by public
  position, must be the first ``k`` legs of their side. They need *not* be
  publicly adjacent: an interleaved leg from the other side carries no
  categorical content. The fused leg lands at ``min(axes)`` and the survivors
  keep their relative order.
* **The binary layout** (``k``-way fusion is ``k-1`` iterated binary fuses, so
  only this needs specifying). Triples ``(u_a, u_b, mu)`` are enumerated with
  ``u_a`` outermost in leg ``A``'s canonical sector order, ``u_b`` next in
  ``B``'s, ``mu`` innermost. Each contributes a contiguous chunk of extent
  ``m_a * m_b`` at the accumulated offset within its output sector ``c``, and
  inside a chunk the index is ``alpha_a * m_b + alpha_b`` (C order) — so the
  block operation is exactly a backend reshape.
* **The fused leg is ``dual=False``**, ``side`` is the inputs' common side, and
  ``name`` is ``None``: the tree labels are already dual-resolved, so ``c`` is
  reproduced by ``fused_sector`` identically.
* **No fusion history is stored anywhere** (invariant 4). [unfuse][tenet.unfuse] takes
  the target legs explicitly and validates them by recomputing the layout.

Blocks move only through ``ar.do("transpose"/"reshape"/"concatenate")`` plus
basic slicing; there is no NumPy call and no ``cgc``/``to_dense`` here. The
Clebsch-Gordan tensors appear only in the tests' dense oracle.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.fusion_tree import FusionTree
from tenet.leg import OUT, Leg, Side
from tenet.space import GradedSpace
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry.base import Sector

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "FusionLayout",
    "FusionPlan",
    "FusionStep",
    "fuse",
    "fuse_spaces",
    "fused_leg",
    "fusion_plan",
    "unfuse",
]


# --- the layout -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FusionLayout:
    """Where each ``(u_a, u_b, mu)`` triple sits in the fused degeneracy axis.

    Array-free and hashable (invariant 8). ``triples`` holds
    ``(c, u_a, u_b, mu, offset, extent)`` with ``extent == m_a * m_b``, in
    enumeration order; the chunks of one ``c`` tile ``[0, m^f_c)`` exactly.
    """

    triples: tuple[tuple[Sector, Sector, Sector, int, int, int], ...]

    def chunks(self, c: Sector) -> tuple[tuple[Sector, Sector, Sector, int, int, int], ...]:
        """The triples feeding output sector ``c``, in layout (offset) order."""
        return tuple(t for t in self.triples if t[0] == c)


@cache
def fuse_spaces(a: Leg, b: Leg) -> tuple[Leg, FusionLayout]:
    """The fused leg of ``a`` and ``b``, and the layout that defines it.

    Pure metadata: no tensor, no blocks, no arrays. Deterministic across
    processes, because every enumeration order it rests on is canonical.
    """
    if a.side is not b.side:
        raise ValueError(
            f"fuse: cannot fuse a {a.side.value} leg with a {b.side.value} leg; the result "
            "would have no well-defined side, i.e. no position in Hom(domain, codomain)"
        )
    if a.provider != b.provider:
        raise ValueError(
            f"fuse: legs come from different providers, {a.provider.name} and {b.provider.name}"
        )
    provider = a.provider
    triples: list[tuple[Sector, Sector, Sector, int, int, int]] = []
    degeneracies: dict[Sector, int] = {}
    for sa in a.sectors:
        u_a, m_a = a.fused_sector(sa), a.degeneracy(sa)
        for sb in b.sectors:
            u_b, m_b = b.fused_sector(sb), b.degeneracy(sb)
            for c in provider.fusion(u_a, u_b):
                for mu in range(provider.n_symbol(u_a, u_b, c)):
                    offset = degeneracies.get(c, 0)
                    triples.append((c, u_a, u_b, mu, offset, m_a * m_b))
                    degeneracies[c] = offset + m_a * m_b
    return Leg(GradedSpace.new(provider, degeneracies), a.side), FusionLayout(tuple(triples))


def fused_leg(legs: Sequence[Leg]) -> Leg:
    """The leg that fusing ``legs`` (left to right) produces. ``legs`` must be non-empty."""
    if not legs:
        raise ValueError("fused_leg: need at least one leg")
    leg = legs[0]
    for other in legs[1:]:
        leg, _ = fuse_spaces(leg, other)
    return leg


# --- the tree action: drop / restore the leading vertex --------------------------


def _drop_vertex(tree: FusionTree) -> FusionTree:
    """``(u_0, u_1, u_2, ...) -> (e_1, u_2, ...)``. Coefficient 1, exactly."""
    return FusionTree(
        (tree.lines()[1], *tree.uncoupled[2:]),
        tree.inner[1:],
        tree.multiplicities[1:],
        tree.coupled,
    )


def _add_vertex(tree: FusionTree, u_a: Sector, u_b: Sector, mu: int) -> FusionTree:
    """Inverse of ``_drop_vertex``; ``tree.uncoupled[0]`` is the fused sector ``c``."""
    n = tree.rank + 1
    return FusionTree(
        (u_a, u_b, *tree.uncoupled[1:]),
        (tree.uncoupled[0], *tree.inner)[: max(n - 2, 0)],
        (mu, *tree.multiplicities),
        tree.coupled,
    )


def _fuse_key(key: FusionBlockKey, side: Side) -> FusionBlockKey:
    if side is OUT:
        return FusionBlockKey(_drop_vertex(key.output_tree), key.input_tree)
    return FusionBlockKey(key.output_tree, _drop_vertex(key.input_tree))


def _side_tree(key: FusionBlockKey, side: Side) -> FusionTree:
    return key.output_tree if side is OUT else key.input_tree


# --- the plan -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FusionStep:
    """One binary fuse, lowered. Frozen, hashable, array-free.

    ``targets[j]`` lists ``(source block index, split shape)`` for target block
    ``j``, in layout order. The *split* shape is the source block's shape after
    ``perm``, i.e. with the two merging degeneracy axes already adjacent at
    ``axis`` and ``axis + 1`` — merging them is the reshape, and keeping them
    separate here is what makes the step invertible without a second plan.
    """

    structure: TensorStructure
    axis: int
    perm: tuple[int, ...]
    targets: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]
    layout: FusionLayout


@dataclass(frozen=True, slots=True)
class FusionPlan:
    """A ``k``-way fusion as ``k-1`` binary steps. ``structure`` is the final one."""

    structure: TensorStructure
    steps: tuple[FusionStep, ...]


@cache
def fusion_plan(s: TensorStructure, axes: tuple[int, ...]) -> FusionPlan:
    """Plan fusing ``axes`` of ``s``. Cached on the (frozen, array-free) structure."""
    axes, _ = _check_axes(s, axes)
    steps = []
    current = s
    for i, ax in enumerate(axes[1:]):
        # every earlier step deleted one axis strictly between axes[0] and ax
        step = _binary_step(current, axes[0], ax - i)
        steps.append(step)
        current = step.structure
    return FusionPlan(current, tuple(steps))


def _check_axes(s: TensorStructure, axes: tuple[int, ...], op: str = "fuse") -> tuple:
    """Validate and return ``(sorted axes, side)``. Loud on every failure mode."""
    if not axes:
        raise ValueError(f"{op}: axes is empty; fusion needs at least one axis")
    bad = tuple(ax for ax in axes if not isinstance(ax, int) or isinstance(ax, bool))
    if bad:
        raise TypeError(f"{op}: axes must be ints, got {bad!r}")
    # Simplification: no negative-axis normalization. `fuse` places the result at
    # min(axes), and mixing -1 with 0 makes "min" mean two things.
    out_of_range = tuple(ax for ax in axes if not 0 <= ax < s.ndim)
    if out_of_range:
        raise ValueError(f"{op}: axes {out_of_range} are out of range for a {s.ndim}-axis tensor")
    repeated = tuple(sorted({ax for ax in axes if axes.count(ax) > 1}))
    if repeated:
        raise ValueError(f"{op}: axes {repeated} are repeated")

    ordered = tuple(sorted(axes))
    sides = {s.legs[ax].side for ax in ordered}
    if len(sides) > 1:
        raise ValueError(
            f"{op}: axes {ordered} mix sides "
            f"({', '.join(f'{ax}:{s.legs[ax].side.value}' for ax in ordered)}); the result "
            "would have no well-defined side"
        )
    side = sides.pop()
    side_axes = s.out_axes if side is OUT else s.in_axes
    if len(ordered) > 1 and ordered != side_axes[: len(ordered)]:
        raise NotImplementedError(
            f"{op}: axes {ordered} are not a prefix of the {side.value} side's order "
            f"{side_axes}; fusing a non-prefix group needs an F-move to bring the legs "
            "to the front of the fusion tree, which is Milestone 4"
        )
    return ordered, side


def _binary_step(s: TensorStructure, p: int, q: int) -> FusionStep:
    """Fuse public axes ``p < q`` (both first on their side) into one at ``p``."""
    side = s.legs[p].side
    leg, layout = fuse_spaces(s.legs[p], s.legs[q])
    new_legs = tuple(leg if i == p else other for i, other in enumerate(s.legs) if i != q)
    target = TensorStructure(new_legs)

    # bring the merging degeneracy axes together: q lands directly after p
    perm = [i for i in range(s.ndim) if i != q]
    perm.insert(p + 1, q)
    perm_t = tuple(perm)

    offsets = {(c, u_a, u_b, mu): off for c, u_a, u_b, mu, off, _ in layout.triples}
    index = {key: j for j, key in enumerate(target.block_order)}
    buckets: list[list[tuple[int, int, tuple[int, ...]]]] = [[] for _ in target.block_order]
    for i, key in enumerate(s.block_order):
        tree = _side_tree(key, side)
        u_a, u_b = tree.uncoupled[0], tree.uncoupled[1]
        mu, c = tree.multiplicities[0], tree.lines()[1]
        shape = s.block_shape(key)
        buckets[index[_fuse_key(key, side)]].append(
            (offsets[c, u_a, u_b, mu], i, tuple(shape[j] for j in perm_t))
        )

    targets = []
    for bucket, key in zip(buckets, target.block_order, strict=True):
        chunks = sorted(bucket)
        total = sum(shape[p] * shape[p + 1] for _, _, shape in chunks)
        # every entry written exactly once: the chunks tile the fused axis
        if total != target.block_shape(key)[p]:
            raise RuntimeError(
                f"fuse: layout does not tile block {key}: chunks cover {total} of "
                f"{target.block_shape(key)[p]} fused-axis entries"
            )
        targets.append(tuple((i, shape) for _, i, shape in chunks))
    return FusionStep(target, p, perm_t, tuple(targets), layout)


# --- execution ------------------------------------------------------------------


def _merged(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    return (*shape[:axis], shape[axis] * shape[axis + 1], *shape[axis + 2 :])


def _apply(step: FusionStep, blocks: tuple[Any, ...]) -> tuple[Any, ...]:
    identity = tuple(range(len(step.perm)))
    out = []
    for chunks in step.targets:
        parts = []
        for i, shape in chunks:
            block = blocks[i]
            if step.perm != identity:
                block = ar.do("transpose", block, step.perm)
            parts.append(ar.do("reshape", block, _merged(shape, step.axis)))
        out.append(parts[0] if len(parts) == 1 else ar.do("concatenate", parts, axis=step.axis))
    return tuple(out)


def _unapply(step: FusionStep, blocks: tuple[Any, ...]) -> tuple[Any, ...]:
    """Invert ``_apply``: slice each chunk back out and undo the merge."""
    identity = tuple(range(len(step.perm)))
    inverse = tuple(sorted(range(len(step.perm)), key=step.perm.__getitem__))
    out: list[Any] = [None] * sum(len(chunks) for chunks in step.targets)
    for block, chunks in zip(blocks, step.targets, strict=True):
        offset = 0
        for i, shape in chunks:
            extent = shape[step.axis] * shape[step.axis + 1]
            # Simplification: basic slicing, not an ar.do — every backend spells a
            # contiguous slice the same way, and autoray has no split wrapper.
            piece = block[(slice(None),) * step.axis + (slice(offset, offset + extent),)]
            piece = ar.do("reshape", piece, shape)
            out[i] = piece if step.perm == identity else ar.do("transpose", piece, inverse)
            offset += extent
    return tuple(out)


# --- public API -----------------------------------------------------------------


def fuse(t: "SymmetricTensor", axes: Sequence[int]) -> "SymmetricTensor":
    """Fuse the given public axes into one leg, placed at ``min(axes)``.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose axes are fused.
    axes : sequence of int
        The public axes to fuse. They must share one side and, ordered by
        public position, be exactly the first ``k`` legs of that side; public
        adjacency is *not* required. A single-axis ``axes`` is the identity.

    Returns
    -------
    SymmetricTensor
        The fused tensor: the new leg sits at ``min(axes)`` with
        ``dual=False``, the inputs' common ``side`` and ``name=None``; the
        survivors keep their relative order.

    Raises
    ------
    ValueError
        If ``axes`` is empty, out of range, or repeated, or if the axes mix
        sides (the result would have no well-defined side).
    TypeError
        If an axis is not an ``int``.
    NotImplementedError
        If the axes are not a prefix of their side's order; fusing a
        non-prefix group needs an F-move, which is Milestone 4.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> f = tenet.fuse(t, (0, 1))
    >>> f.ndim, f.shape
    (2, (4, 2))
    >>> bool(tenet.allclose(tenet.unfuse(f, 0, t.legs[:2]), t))
    True
    """
    from tenet.tensor import SymmetricTensor

    plan = fusion_plan(t.structure, tuple(axes))
    blocks = t.blocks
    for step in plan.steps:
        blocks = _apply(step, blocks)
    return SymmetricTensor(plan.structure, blocks)


def unfuse(t: "SymmetricTensor", axis: int, legs: Sequence[Leg]) -> "SymmetricTensor":
    """Split ``axis`` back into ``legs``, inserted **consecutively** at ``axis``.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor whose fused axis is split.
    axis : int
        The public axis to split. Must be the first axis of its side.
    legs : sequence of Leg
        The legs the axis should split into. They must share the axis's side
        and reproduce ``t.legs[axis]`` exactly (up to ``name``) under
        iterated ``fuse_spaces`` — no fusion history is stored, so the caller
        supplies the target legs and a mismatch fails loudly rather than
        producing a garbage layout.

    Returns
    -------
    SymmetricTensor
        The split tensor, ``legs`` inserted consecutively at ``axis``.

    Raises
    ------
    ValueError
        If ``axis`` is out of range; if ``legs`` is empty or a leg's side
        differs from the axis's; or if the given legs do not fuse back to
        ``t.legs[axis]`` (a ``dual`` or space mismatch, named in the message).
    NotImplementedError
        If ``axis`` is not the first axis of its side; splitting a
        non-leading leg needs an F-move, which is Milestone 4.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> f = tenet.fuse(t, (0, 1))
    >>> bool(tenet.allclose(tenet.unfuse(f, 0, t.legs[:2]), t))
    True

    Notes
    -----
    Because the split legs are inserted consecutively, ``fuse`` followed by
    ``unfuse`` reproduces the original tensor exactly when the fused axes were
    publicly consecutive; fusing publicly non-adjacent axes is not invertible
    without the history this deliberately does not keep.
    """
    from tenet.tensor import SymmetricTensor

    legs = tuple(legs)
    if not 0 <= axis < t.ndim:
        raise ValueError(f"unfuse: axis {axis} is out of range for a {t.ndim}-axis tensor")
    if not legs:
        raise ValueError("unfuse: legs is empty; give the legs the axis should split into")
    side = t.legs[axis].side
    side_axes = t.structure.out_axes if side is OUT else t.structure.in_axes
    if axis != side_axes[0]:
        raise NotImplementedError(
            f"unfuse: axis {axis} is not the first axis of the {side.value} side "
            f"{side_axes}; splitting a non-leading leg needs an F-move, which is Milestone 4"
        )
    for i, leg in enumerate(legs):
        if leg.side is not side:
            raise ValueError(
                f"unfuse: legs[{i}] has side {leg.side.value}, but axis {axis} is on the "
                f"{side.value} side"
            )
    _check_reproduces(fused_leg(legs), t.legs[axis], axis)
    if len(legs) == 1:
        return SymmetricTensor(t.structure, t.blocks)

    unfused = TensorStructure((*t.legs[:axis], *legs, *t.legs[axis + 1 :]))
    plan = fusion_plan(unfused, tuple(range(axis, axis + len(legs))))
    blocks = t.blocks
    for step in reversed(plan.steps):
        blocks = _unapply(step, blocks)
    return SymmetricTensor(unfused, blocks)


def _check_reproduces(got: Leg, want: Leg, axis: int) -> None:
    """Raise unless ``got`` matches ``want`` up to ``name``, naming the mismatch."""
    if got.dual != want.dual:
        raise ValueError(
            f"unfuse: axis {axis} has dual={want.dual}, but a fused leg is always dual=False"
        )
    if got.space != want.space:
        raise ValueError(
            f"unfuse: the given legs fuse to the space {got.space.sectors}, but axis {axis} "
            f"carries {want.space.sectors}"
        )
