"""Symmetry restriction — ``cast`` (issue #92).

Let a tensor forget part of its symmetry: ``cast(t, U1)`` takes an SU(2)-symmetric
tensor and returns the U(1)-symmetric tensor describing the same physical object,
each SU(2) irrep ``j`` splitting into the weights ``2 S_z = 2j, 2j-2, ..., -2j``.

The basis transformation is a **permutation and nothing else** — the magnetic
basis of ``V_j`` *is* the ``S_z`` weight basis, so nothing is rotated, only
reordered, because our target layout is canonical (``GradedSpace`` order,
contiguous slabs) while the source's is irrep-major. There are no coefficients
and no signs, not even on a ``dual`` leg: ``to_dense`` applies the provider's
``V_a -> V_a^*`` isomorphism there, and SU(2)'s is an antidiagonal ``±1`` signed
permutation, so reversing the magnetic order and negating the weight cancel.
That convention is pinned by a refusal in the tests, not by this comment.

Which target sectors a provider's dense basis vectors carry is a provider fact,
asked through the :class:`~tenet.symmetry.base.SymmetryCast` capability — never
an ``isinstance`` branch here (invariant 5).

Simplification: the whole cast is ``to_dense``, one ``take`` per axis, ``from_dense``.
The ceiling is ``Π legs[i].space.dim`` floats live at once; the upgrade path is
per-cell gathering over ``dense_plan(structure).cells``, the decomposition #82
already computes and caches. Trigger: a cast whose dense array does not fit.
"""

from collections import Counter
from typing import TYPE_CHECKING

import autoray as ar
import numpy as np

from tenet.leg import Leg
from tenet.ops.dense import from_dense, to_dense
from tenet.space import GradedSpace
from tenet.symmetry.base import (
    CapabilityError,
    ClebschGordan,
    FusionProvider,
    Sector,
    SymmetryCast,
    requires,
)

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["cast"]


def cast(
    t: "SymmetricTensor", target: FusionProvider, *, atol: float | None = None
) -> "SymmetricTensor":
    """Restrict ``t`` to the smaller symmetry ``target``, in the dense basis.

    Requires :class:`~tenet.symmetry.base.SymmetryCast` on ``t``'s provider (and
    ``ClebschGordan``, plus ``DualBasis`` for a ``dual`` leg, through
    :func:`~tenet.ops.dense.to_dense`). ``side``, ``dual`` and ``name`` are
    carried through per leg unchanged; only ``space`` changes.

    ``atol`` is forwarded to :func:`~tenet.ops.dense.from_dense`, whose default
    symmetry check is the free correctness oracle here — a wrong branching or a
    wrong sort produces an array that is not ``target``-symmetric and is refused.
    ``atol=math.inf`` skips it, which is what makes ``cast`` traceable.

    Forgetting is not invertible: the result has strictly more free parameters
    than its source, so there is no inverse cast in this direction.
    """
    requires(t.structure.provider, SymmetryCast)
    requires(target, ClebschGordan)
    legs, takes = zip(*(_cast_leg(leg, target) for leg in t.legs), strict=True)
    dense = to_dense(t)
    for axis, take in enumerate(takes):
        # a concrete NumPy index array: every backend's `take` accepts one, and
        # under `jit` it constant-folds rather than becoming a traced operand
        dense = ar.do("take", dense, np.asarray(take), axis=axis)
    return from_dense(dense, legs, atol=atol)


def _cast_leg(leg: Leg, target: FusionProvider) -> tuple[Leg, list[int]]:
    """The new leg, and the per-axis gather that reorders the dense basis.

    The source dense index enumerates ``(a, alpha, k)`` in that nesting
    (``GradedSpace``'s canonical sector order, then degeneracy, then magnetic
    index — ``sector_offset`` plus ``alpha * d_a + k``). Labelling each with
    ``branch(target, a)[k]`` and stably sorting by that label gives the target
    layout, whose degeneracy index is therefore ``(a ascending, alpha ascending,
    k ascending)`` within each target sector. ``sorted`` is stable, so this is
    reproducible across processes without a tie-break rule — and it is what makes
    two independently cast tensors contractable against each other.
    """
    p = leg.provider
    labels: list[Sector] = []
    for a, m in leg.space.sectors:
        charges = p.branch(target, a)
        for q in charges:
            if target.irrep_dim(q) != 1:
                raise CapabilityError(
                    f"cast: target {target.name} sector {q!r} has irrep_dim "
                    f"{target.irrep_dim(q)} > 1; one target label per dense basis vector "
                    "is only well-defined when the target's irreps are one-dimensional, "
                    "i.e. when the target is abelian"
                )
        labels.extend(charges * m)  # (alpha, k): d_a labels, repeated m times
    take = sorted(range(len(labels)), key=labels.__getitem__)
    space = GradedSpace.new(target, Counter(labels))
    return Leg(space, leg.side, leg.dual, leg.name), take
