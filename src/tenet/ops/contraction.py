"""Pairwise contraction — ``tenet.tensordot``, ``tenet.trace``, ``tenet.einsum``.

There is no new mathematics here, and that is the point: composition stays
first-class and a contraction is lowered to
operations that already exist and are already dense-oracle tested::

    a ──repartition(outputs=free, inputs=contracted)──▶  a' : Hom(K, F_a)
    b ──repartition(outputs=contracted, inputs=free)──▶  b' : Hom(F_b, K)

    compose(a', b') : Hom(F_b, F_a)   # legs = a' codomain ++ b' domain, which is
                                      # already the documented output order
      ──repartition(each free leg back to its original side)──▶
      ──transpose──▶ (a free..., b free...)

``repartition`` is itself the transpose/bend/transpose sandwich, so the
whole module is axis bookkeeping over frozen metadata plus four already-cached
plans. TensorKit's ``tensorcontract!`` ends in ``permute(compose(sA, sB), pAB)``
for the same reason.

Two design decisions carry the module:

* **Contractibility is ``same space`` + opposite ``outward_dual``**, and it
  is phrased that way because it is *bend-invariant*: a bend flips ``side`` and
  ``dual`` together, so ``dual xor (side is IN)`` flips twice and is unchanged.
  A rule phrased on ``(side, dual)`` would have to be restated after every bend
  the lowering performs. Dimensions are never compared (invariant 2): a
  charge-reversed U(1) partner has the same dimension and the wrong space.
* **Free legs come back exactly as they went in** — same ``space``, ``side``,
  ``dual`` and ``name`` — at the cost of a second repartition. Letting a free IN
  leg return as an OUT leg because that is where the composition left it would
  make the output legs depend on the lowering. The round trip is *exact*, not
  approximately so, because ``bend_left`` is the conjugate of ``bend_right``,
  i.e. its genuine inverse.

Everything is refused before any block moves: axis validation, contractibility
and the ``BendingCoefficients`` requirement of every leg that will cross — on
both operands, *including free legs*, which is the non-obvious half.
``compose``'s own check stays as a redundant guard and can no longer fire on
user error; if it does, this module's rewriting is wrong.

``einsum`` sits on top and owns no mathematics at all: it parses the equation
into label→axis maps, hands the shared labels to [tensordot][tenet.tensordot] as one axis
pair, and hands the requested output order to [transpose][tenet.transpose]. The
parser is hand-rolled — ``opt_einsum`` parses ellipsis, unicode labels, the
interleaved format and shape-driven broadcasting, none of which is meaningful
for symmetric tensors. ``opt_einsum`` enters one level up instead, at
the *path* level: with three or more operands ``contract_path`` orders the
pairwise contractions and each step is the two-operand call above, so the
scheduler adds a loop and no mathematics. It is imported lazily, inside that
branch only; ``cotengra``'s optimizers arrive through ``optimize=`` as ordinary
``PathOptimizer`` objects and ``cotengra`` is imported nowhere in ``tenet``.

``einsum_chain`` sits beside ``einsum`` and owns a *run* of pair-contractions. Between two
steps there is nothing to materialize: step k's restore-repartition and step k+1's operand
lowering are both plans of ``(source, target, coefficient)`` over blocks that are views into
step k's sector matrices, so composing them gives one plan from matrices to matrices. The
steps are the caller's rather than a path finder's, because the composition rule fixes the
intermediate leg order and the bent wires at every pair and a single multi-operand equation
would leave both to ``opt_einsum``.

No ``to_dense``, no NumPy and no provider branching here (invariants 8/9).
"""

import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.cache import plan_cache
from tenet.leg import IN, OUT, Leg
from tenet.map_view import check_square, lower_plan, to_matrices
from tenet.ops.basic import _check_same_structure
from tenet.ops.map import block_ref, compose_lowered, identity
from tenet.ops.permutation import permutation_plan, transpose, twist
from tenet.ops.repartition import _compose as compose_terms
from tenet.ops.repartition import apply_plan, repartition, repartition_plan, sides_plan
from tenet.structure import TensorStructure
from tenet.symmetry.base import (
    BendingCoefficients,
    CapabilityError,
    PivotalData,
    QuantumDimensionData,
    requires,
)

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "ContractionPlan",
    "contractible",
    "contraction_plan",
    "einsum",
    "einsum_chain",
    "full_trace",
    "inner",
    "outward_dual",
    "tensordot",
    "trace",
]

Axes = Any
"""``((i, ...), (j, ...))`` or the NumPy integer form ``n``."""


def outward_dual(leg: Leg) -> bool:
    """The leg's object once every line is read as outgoing: ``False`` = V, ``True`` = V*.

    ``Hom(D, C) ≃ Hom(1, C ⊗ D*)``: an OUT leg contributes ``X``, an IN leg
    contributes ``X*``. So the all-out object is dualized exactly when
    ``leg.side is IN`` xor ``leg.dual``.
    """
    return leg.dual != (leg.side is IN)


def contractible(x: Leg, y: Leg) -> bool:
    """A leg of ``a`` and a leg of ``b`` contract iff their all-out objects are duals.

    Symmetric, bend-invariant, and never a dimension comparison.
    """
    return x.space == y.space and outward_dual(x) != outward_dual(y)


def _validated(a: TensorStructure, b: TensorStructure, axes: Axes) -> tuple[tuple[int, ...], ...]:
    """``(a axes, b axes)`` as plain-``int`` tuples, checked in the CALLER's numbering.

    ``operator.index`` normalization is what keeps the plan cache from fragmenting
    on NumPy/JAX integer scalars, exactly as in ``permutation_plan``'s
    ``_validated_axes``; it also collapses NumPy's integer and bare-integer forms
    onto the one canonical key.
    """
    if hasattr(axes, "__index__"):
        n = operator.index(axes)
        if n < 0:
            raise ValueError(f"tensordot: axes={n} is negative; the integer form counts axes")
        if n > a.ndim or n > b.ndim:
            raise ValueError(
                f"tensordot: axes={n} contracts the last {n} axes of a ({a.ndim}-dimensional) "
                f"against the first {n} of b ({b.ndim}-dimensional); there are not that many"
            )
        return tuple(range(a.ndim - n, a.ndim)), tuple(range(n))

    try:
        raw_a, raw_b = axes
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"tensordot: axes {axes!r} must be an integer or a pair (a_axes, b_axes)"
        ) from exc

    parts = []
    for name, t, raw in (("a", a, raw_a), ("b", b, raw_b)):
        # NumPy's ergonomics: a bare integer means a one-element axis list
        raw = (raw,) if hasattr(raw, "__index__") else raw
        try:
            got = tuple(operator.index(x) for x in raw)
        except TypeError as exc:
            raise ValueError(
                f"tensordot: axes for {name}, {tuple(raw)}, must all be integers"
            ) from exc
        seen: set[int] = set()
        for x in got:
            if not 0 <= x < t.ndim:
                raise ValueError(
                    f"tensordot: axis {x} of {name} is out of range for a {t.ndim}-dimensional "
                    "tensor (negative axes are not accepted, as in transpose and repartition)"
                )
            if x in seen:
                raise ValueError(
                    f"tensordot: axis {x} appears twice in {name}'s axes {got}; a repeated "
                    "axis inside one operand is a diagonal, not a contraction"
                )
            seen.add(x)
        parts.append(got)

    if len(parts[0]) != len(parts[1]):
        raise ValueError(
            f"tensordot: axes {parts[0]} of a and {parts[1]} of b have different lengths "
            f"({len(parts[0])} vs {len(parts[1])}); contracted axes are paired in order"
        )
    return tuple(parts)


def _crossing(t: TensorStructure, contracted: tuple[int, ...], to_out: bool) -> tuple[int, ...]:
    """Axes of ``t`` that must change side, in ``t``'s own numbering.

    ``to_out`` says where the *contracted* axes are headed: ``False`` for ``a``
    (whose contracted legs become its domain) and ``True`` for ``b``.
    """
    want = {ax: (OUT if to_out else IN) for ax in contracted}
    return tuple(
        ax for ax, leg in enumerate(t.legs) if leg.side is not want.get(ax, IN if to_out else OUT)
    )


def _refuse_bends(
    a: TensorStructure, b: TensorStructure, ca: tuple[int, ...], cb: tuple[int, ...]
) -> None:
    """Refuse before any block moves, naming every axis that would have to bend."""
    offenders = [("a", ax) for ax in _crossing(a, ca, False)]
    offenders += [("b", ax) for ax in _crossing(b, cb, True)]
    if not offenders:
        return
    try:
        requires(a.provider, BendingCoefficients)
    except CapabilityError as exc:
        named = ", ".join(f"axis {ax} of {which}" for which, ax in offenders)
        shaped = (a.in_axes, b.out_axes)
        raise CapabilityError(
            f"tensordot: this axis pattern moves legs between domain and codomain "
            f"({named}), which is a line bend, and provider {a.provider.name} does not "
            "implement BendingCoefficients. A composition-shaped pattern needs no bending "
            "capability at all: axes=(a's IN axes, b's OUT axes) in matching order, here "
            f"axes={shaped!r}, contracts the whole of a's domain against the whole of b's "
            "codomain with zero bends and works today for every provider."
        ) from exc


@dataclass(frozen=True, slots=True)
class ContractionPlan:
    """The categorical half of a pairwise contraction: static, array-free, hashable.

    Holds no block indices and no coefficients — those belong to the sub-plans
    (``permutation_plan``, ``bend_plan``, ``map_layout``), which are cached in
    their own right and already shared between every tensor of the same
    structure. Caching them a second time here would duplicate the same
    coefficients with a second chance to go stale. What this object owns is the
    small pure-Python derivation that decides *which* sub-plans run, plus
    ``new_structure``, the output legs known without contracting anything.

    """

    # Simplification: no ``src/tenet/planning/`` package (docs/design.md's proposed tree).
    # Today it would hold re-exports of plans that already live next to their ops
    # (``PermutationPlan``/``ops.permutation``, ``BendPlan``/``ops.repartition``,
    # ``FusionPlan``/``ops.fusion``, ``AdjointPlan``/``ops.map``,
    # ``MapLayout``/``map_view``) — churn against exhaustively tested modules for
    # zero behaviour change. Create it when plans are shared *across* operations:
    # M8's path-level planning, or M9's shape bucketing.

    a_outputs: tuple[int, ...]
    a_inputs: tuple[int, ...]
    b_outputs: tuple[int, ...]
    b_inputs: tuple[int, ...]
    restore_outputs: tuple[int, ...]
    restore_inputs: tuple[int, ...]
    final_transpose: tuple[int, ...]
    new_structure: TensorStructure


# Simplification: unbounded `cache`, and deliberately so where `_restore_plan` below is
# bounded — this plan holds axis tuples and `new_structure`, no block indices and no
# coefficients (see the class docstring), so an entry is a hundred bytes whatever the
# bond dimension. `tenet.cache` says which caches that argument does not cover.
@cache
def contraction_plan(a: TensorStructure, b: TensorStructure, axes: Axes) -> ContractionPlan:
    """Plan ``tensordot(a, b, axes)``. Cached: repeat calls return one object.

    Raises here, before any block is touched and before one sub-plan is built:
    invalid axes, mismatched providers, non-contractible pairs, a contraction
    leaving no free leg, and a missing ``BendingCoefficients`` for a leg that
    must cross. ``axes`` accepts everything [tensordot][tenet.tensordot] accepts; it is
    normalized to plain ``int`` tuples before anything else, so the integer form
    and NumPy integer scalars land on the same cache entry.
    """
    ca, cb = _validated(a, b, axes)
    if a.provider != b.provider:
        raise ValueError(
            f"tensordot: a has provider {a.provider.name} and b has {b.provider.name}; "
            "two tensors of different symmetries have no common fusion category"
        )
    for k, (i, j) in enumerate(zip(ca, cb, strict=True)):
        x, y = a.legs[i], b.legs[j]
        if not contractible(x, y):
            reason = (
                "their spaces differ"
                if x.space != y.space
                else "both ends of the wire point the same way (equal outward duals: "
                f"{outward_dual(x)}); one leg must be the dual object of the other"
            )
            raise ValueError(
                f"tensordot: pair {k}, axis {i} of a ({x!r}) and axis {j} of b ({y!r}), "
                f"do not contract: {reason}. Contractibility is the same space plus "
                "opposite `dual xor (side is IN)`; dimensions are never compared"
            )

    fa = tuple(i for i in range(a.ndim) if i not in set(ca))
    fb = tuple(j for j in range(b.ndim) if j not in set(cb))
    if not fa and not fb:
        raise ValueError(
            "tensordot: this contraction leaves no free leg, and TensorStructure needs at "
            "least one leg (a rank-0 SymmetricTensor does not exist; see the simplification note "
            "on an explicit provider field). Returning a bare backend scalar would make the "
            "return type depend on the arguments — tenet.norm is the precedent for scalars "
            "leaving the tensor world explicitly"
        )
    _refuse_bends(a, b, ca, cb)

    # every free leg comes back exactly as it went in, so the output legs are known
    # here — no block, and no sub-plan, has moved
    free = (*(a.legs[i] for i in fa), *(b.legs[j] for j in fb))
    outputs = tuple(k for k, leg in enumerate(free) if leg.side is OUT)
    inputs = tuple(k for k, leg in enumerate(free) if leg.side is IN)
    position = {k: p for p, k in enumerate((*outputs, *inputs))}
    return ContractionPlan(
        a_outputs=fa,
        a_inputs=ca,
        b_outputs=cb,
        b_inputs=fb,
        restore_outputs=outputs,
        restore_inputs=inputs,
        final_transpose=tuple(position[k] for k in range(len(free))),
        new_structure=TensorStructure(free),
    )


def tensordot(a: "SymmetricTensor", b: "SymmetricTensor", axes: Axes) -> "SymmetricTensor":
    """Contract ``axes[0]`` of ``a`` against ``axes[1]`` of ``b``, pairwise in order.

    Parameters
    ----------
    a : SymmetricTensor
        The left operand; its free legs lead in the output.
    b : SymmetricTensor
        The right operand; must share ``a``'s provider.
    axes : tuple of two axis sequences, or int
        ``((i, ...), (j, ...))`` pairs axis ``i`` of ``a`` with axis ``j`` of
        ``b``, in order; NumPy's integer form ``axes=n`` contracts the last
        ``n`` axes of ``a`` against the first ``n`` of ``b``. Negative indices
        are refused, as in [transpose][tenet.transpose] and
        [repartition][tenet.SymmetricTensor.repartition]. ``axes=((), ())`` is the outer
        product.

    Returns
    -------
    SymmetricTensor
        The contraction. Public axis order is ``a``'s free axes (in ``a``'s
        order) followed by ``b``'s free axes (in ``b``'s order), matching
        ``np.tensordot``; every free leg is returned **unchanged** — same
        ``space``, ``side``, ``dual`` and ``name``.

    Raises
    ------
    ValueError
        If ``axes`` is malformed (negative, repeated, out of range, or
        mismatched pair lengths); if the operands' providers differ; if a
        paired pair of legs is not contractible (same space plus opposite
        ``dual xor (side is IN)`` — dimensions are never compared); or if the
        contraction would leave no free leg (a scalar leaves the tensor world
        through [norm][tenet.norm], [inner][tenet.inner] or
        [full_trace][tenet.full_trace] instead).
    CapabilityError
        If the axis pattern moves a leg between domain and codomain (a line
        bend) and the provider does not implement ``BendingCoefficients``.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> b = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)
    >>> c = tenet.tensordot(a, b, axes=((1,), (0,)))
    >>> c.legs == (a.legs[0], b.legs[1])
    True

    Notes
    -----
    ``axes=((), ())`` is the outer product and falls out of the
    lowering rather than being special-cased.

    A **block-less operand** — legs that cannot couple to any total charge — is legal
    and needs no special case either. It lowers to no coupled-sector matrices, so every
    coupled sector of the output is one the operands do not both carry, and the
    composition's own missing-sector rule writes it as zeros: the result is the
    structurally implied tensor, itself block-less whenever the free legs cannot couple
    and explicit zeros wherever they can. Those zeros take their backend and dtype from
    the other operand, and NumPy ``float64`` when both operands are block-less.

    All refusals, and all axis bookkeeping, live in ``contraction_plan``;
    what is left here is the execution of four already-tested operations.
    """
    # normalize first so the integer form and NumPy integer scalars share one entry
    return _tensordot(a, b, _validated(a.structure, b.structure, axes), ())


def _tensordot(
    a: "SymmetricTensor",
    b: "SymmetricTensor",
    axes: tuple[tuple[int, ...], ...],
    after: tuple[tuple[int, ...], ...],
) -> "SymmetricTensor":
    """[tensordot][tenet.tensordot] on validated axes, with ``after`` folded into the restore.

    Parameters
    ----------
    a, b : SymmetricTensor
        The operands.
    axes : tuple of tuple of int
        The contracted axes, already through ``_validated``.
    after : tuple of tuple of int
        Further transposes the caller would apply to the result, outermost last.
        [einsum][tenet.einsum] passes its output reordering here.

    Returns
    -------
    SymmetricTensor
        The contraction, transposed by ``after``.

    Notes
    -----
    ``after`` exists so that the restore-repartition and every transpose that follows it
    are **one** plan: each of them re-applies its own coefficients, and materialising a
    block twice for a movement worth one pass is what composing them avoids.
    Composing plans is ``repartition_plan``'s own
    idiom — the transpose/bend/transpose sandwich is composed exactly this way — so this
    is the same rewriting one level up, not a new rule. The terms and their coefficients
    are unchanged; only the number of passes over them is.
    """
    return _contracted(a, b, axes, after).realized("tensordot")


@dataclass(frozen=True, slots=True)
class _Pending:
    """A tensor and a plan that has not been applied to it yet -- the chain's carrier.

    ``structure`` is what the tensor *will* have once ``(perm, terms)`` is applied;
    ``source`` still holds the blocks the plan reads. Nothing is moved until a lowering
    composes onto it or [realized][tenet.ops.contraction._Pending.realized] is asked
    for the tensor.
    """

    source: "SymmetricTensor"
    structure: TensorStructure
    perm: tuple[int, ...]
    terms: tuple[tuple[int, int, complex], ...]

    @property
    def legs(self) -> tuple[Leg, ...]:
        """The legs the plan's result carries."""
        return self.structure.legs

    @property
    def ndim(self) -> int:
        """The rank the plan's result carries."""
        return len(self.structure.legs)

    def then(
        self,
        structure: TensorStructure,
        perm: tuple[int, ...],
        terms: tuple[tuple[int, int, complex], ...],
    ) -> "_Pending":
        """This plan followed by ``(perm, terms)``, composed into one plan."""
        return _Pending(
            self.source,
            structure,
            tuple(self.perm[i] for i in perm),
            compose_terms(self.terms, terms),
        )

    def realized(self, caller: str) -> "SymmetricTensor":
        """Apply the plan and hand back the tensor."""
        return apply_plan(self.source, self.structure, self.perm, self.terms, caller)


Operand = Any
"""A chain step's operand: a ``SymmetricTensor`` or a ``_Pending`` standing for one."""


def _ref(*xs: Operand) -> Any:
    """A block to take a backend and a dtype from, first block-carrying operand wins.

    An operand whose legs cannot couple to any total charge carries no block, and so
    neither a backend nor a dtype: the zeros standing for it take both from the other
    operand, and NumPy ``float64`` when neither operand has a block. See
    [block_ref][tenet.ops.map.block_ref], which is where that rule lives.
    """
    return block_ref(*(x.source if isinstance(x, _Pending) else x for x in xs))


def _lower_operand(
    x: Operand, outputs: tuple[int, ...], inputs: tuple[int, ...]
) -> tuple[TensorStructure, Mapping[Any, Any]]:
    """``_lowered`` for an operand that may still be carrying an unapplied plan.

    Parameters
    ----------
    x : SymmetricTensor or _Pending
        The operand.
    outputs, inputs : tuple of int
        The repartition the contraction needs, in ``x``'s public numbering.

    Returns
    -------
    structure : TensorStructure
        The lowered operand's structure.
    mats : Mapping
        Its coupled-sector matrices.

    Notes
    -----
    The whole point of the chain: a ``_Pending``'s plan and this lowering compose into
    **one** matrix-to-matrix plan, so the tensor between two lowerings is never written.
    Where ``lower_plan`` declines -- an immutable backend -- the pending plan is applied
    and the ordinary route taken, which is the same values by a longer road.
    """
    if isinstance(x, _Pending):
        fused = x.then(*sides_plan(x.structure, outputs, inputs))
        mats = lower_plan(fused.source, fused.structure, fused.perm, fused.terms)
        if mats is not None:
            return fused.structure, mats
        x = x.realized("einsum_chain")
    return _lowered(x, outputs, inputs)


def _contracted(
    a: Operand, b: Operand, axes: tuple[tuple[int, ...], ...], after: tuple[tuple[int, ...], ...]
) -> _Pending:
    """``_tensordot``'s body, stopping one step short: the restore is left unapplied."""
    plan = contraction_plan(a.structure, b.structure, axes)
    sa, ma = _lower_operand(a, plan.a_outputs, plan.a_inputs)
    sb, mb = _lower_operand(b, plan.b_outputs, plan.b_inputs)
    joined = TensorStructure(
        (*(sa.legs[i] for i in sa.out_axes), *(sb.legs[i] for i in sb.in_axes))
    )
    c = compose_lowered(joined, ma, mb, _ref(a, b))
    return _Pending(
        c,
        *_restore_plan(
            c.structure, plan.restore_outputs, plan.restore_inputs, (plan.final_transpose, *after)
        ),
    )


def _bent(x: Operand, term: str, bend: str) -> tuple[Operand, str]:
    """``x`` with every wire named in ``bend`` moved to the other side, plan deferred.

    The bend the composition rule demands: both ends of a wire that turns around in the
    intended planar diagram are repartitioned before
    the composition, which pays the categorical bending coefficient by construction. In
    a chain the repartition is a *plan*, composed onto whatever the operand is already
    carrying, so the bent tensor is never written.
    """
    flip = set(bend)
    outs = tuple(i for i, label in enumerate(term) if (x.legs[i].side is OUT) != (label in flip))
    ins = tuple(i for i in range(len(term)) if i not in outs)
    plan = sides_plan(x.structure, outs, ins)
    pending = x.then(*plan) if isinstance(x, _Pending) else _Pending(x, *plan)
    return pending, "".join(term[i] for i in (*outs, *ins))


def _step(equation: str, a: Operand, b: Operand, bend: str) -> _Pending:
    """One pair-contraction of a chain, its result left as an unapplied plan."""
    (ta, tb), out = _parse(equation, (a, b))
    if bend:
        a, ta = _bent(a, ta, bend)
        b, tb = _bent(b, tb, bend)
    shared = [label for label in ta if label in tb]
    free = [label for label in ta if label not in shared]
    free += [label for label in tb if label not in shared]
    axes = _validated(
        a.structure,
        b.structure,
        (
            tuple(ta.index(label) for label in shared),
            tuple(tb.index(label) for label in shared),
        ),
    )
    return _contracted(a, b, axes, (tuple(free.index(label) for label in out),))


def einsum_chain(
    steps: Sequence[tuple[str, "SymmetricTensor | None", "SymmetricTensor | None", str]],
) -> "SymmetricTensor":
    """A run of pair-contractions with nothing materialized between them.

    Parameters
    ----------
    steps : sequence of (equation, a, b, bend)
        One entry per pair-contraction, in order. ``equation`` is a two-operand
        [einsum][tenet.einsum] equation. ``a`` and ``b`` are its operands, and exactly
        one of them is ``None`` in every step after the first, standing for the previous
        step's result -- which side it stands on is the operand order, and operand order
        is categorical data (a Koszul sign for a fermionic provider), so it is written
        out rather than assumed. ``bend`` names the wires whose two ends are moved to
        the other side before the composition, as
        [repartition][tenet.repartition] would; ``""`` is a straight composition.

    Returns
    -------
    SymmetricTensor
        The last step's result.

    Raises
    ------
    ValueError
        If ``steps`` is empty, if the first step names ``None``, or from the delegated
        parsing and contraction of any step.
    CapabilityError
        If a step needs a bend the provider cannot supply, as in
        [tensordot][tenet.tensordot].

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> b = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)
    >>> c = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=2)
    >>> chained = tenet.einsum_chain(
    ...     [("ab,bc->ac", a, b, ""), ("ab,bc->ac", None, c, "")]
    ... )
    >>> bool(tenet.allclose(chained, tenet.einsum("ab,bc->ac", tenet.einsum("ab,bc->ac", a, b), c)))
    True

    Notes
    -----
    Step ``k``'s restore -- the repartition that puts the product back on its public legs,
    with the final transpose already folded in -- and step ``k+1``'s operand lowering are
    both plans of ``(source, target, coefficient)`` over blocks that are *views* into
    step ``k``'s sector matrices. Composing them gives one plan from step ``k``'s
    matrices to step ``k+1``'s: one strided pass per term, the coefficients multiplied
    through, and no tensor written in between. The terms and their coefficients are the
    ones the separate calls apply -- only when they are applied changes.

    The steps are the caller's, not a path finder's: a chain states the intermediate leg
    order and the bends at each pair, which is what the composition rule fixes and what
    a single multi-operand equation would leave to ``opt_einsum``.
    """
    if not steps:
        raise ValueError(
            "einsum_chain: no steps were given; a chain is one or more (equation, a, b, "
            "bend) pair-contractions, as in einsum_chain([('ab,bc->ac', a, b, '')])"
        )
    acc: _Pending | None = None
    for k, (equation, a, b, bend) in enumerate(steps):
        if a is None or b is None:
            if acc is None:
                raise ValueError(
                    f"einsum_chain: step 0 of {equation!r} names None, which stands for the "
                    "previous step's result; the first step's two operands are both tensors"
                )
            if a is None and b is None:
                raise ValueError(
                    f"einsum_chain: step {k} of {equation!r} names None on both sides; one "
                    "operand is the previous step's result, the other is a tensor"
                )
        acc = _step(equation, acc if a is None else a, acc if b is None else b, bend)
    # steps is non-empty, so the loop assigned; ty sees only the None seed
    return acc.realized("einsum_chain")  # ty: ignore[unresolved-attribute]


@plan_cache(cost=lambda result: len(result[2]))
def _restore_plan(
    structure: TensorStructure,
    outputs: tuple[int, ...],
    inputs: tuple[int, ...],
    transposes: tuple[tuple[int, ...], ...],
) -> tuple[TensorStructure, tuple[int, ...], tuple[tuple[int, int, complex], ...]]:
    """The restore-repartition followed by ``transposes``, composed into one plan.

    Parameters
    ----------
    structure : TensorStructure
        The composition's structure, before the free legs are put back.
    outputs, inputs : tuple of int
        The restore-repartition's axes.
    transposes : tuple of tuple of int
        Applied in order after it; each a permutation of ``range(ndim)``.

    Returns
    -------
    new_structure : TensorStructure
        The result's structure.
    perm : tuple of int
        The one per-block axis permutation of the whole chain.
    terms : tuple of (int, int, complex)
        ``(source block, target block, coefficient)``, the coefficients multiplied
        through and duplicate ``(source, target)`` pairs summed.

    Notes
    -----
    Cached on the same key shape as every other plan, but *cost-bounded* (see
    ``tenet.cache``): unlike ``repartition_plan``, whose entry is a shell sharing the
    pattern cache's terms, composing the restore with ``transposes`` builds a term tuple
    this entry owns, keyed on degeneracies that tuple does not read. One measured entry
    for an SU(2) three-sector rank-8 intermediate holds 59,696 terms (9.3 MB), a wider
    one 613,468 (95.7 MB), and an unbounded cache kept one such entry per bond dimension
    for the life of the process.

    ``repartition_plan`` is asked for the restore even when nothing crosses -- with an
    empty crossing list its body is a single ``permutation_plan``, which is what the
    chain needs anyway.
    """
    plan = repartition_plan(structure, outputs, inputs)
    new_structure, perm, terms = plan.new_structure, plan.perm, plan.terms
    for axes in transposes:
        step = permutation_plan(new_structure, axes)
        perm = tuple(perm[i] for i in step.axes)
        terms = compose_terms(terms, step.terms)
        new_structure = step.new_structure
    return new_structure, perm, terms


def _lowered(
    t: "SymmetricTensor", outputs: tuple[int, ...], inputs: tuple[int, ...]
) -> tuple[TensorStructure, Mapping[Any, Any]]:
    """``repartition(t, outputs, inputs)``'s structure and its sector matrices.

    Parameters
    ----------
    t : SymmetricTensor
        The operand, as the caller passed it.
    outputs, inputs : tuple of int
        The repartition ``tensordot`` would apply; together a permutation of
        ``range(t.ndim)``, already validated by ``contraction_plan``.

    Returns
    -------
    structure : TensorStructure
        The repartitioned operand's structure.
    mats : Mapping
        Its coupled-sector matrices, as [to_matrices][tenet.to_matrices] returns them.

    Notes
    -----
    ``compose`` consumes an operand only as matrices, so the repartitioned *tensor*
    between the two is a temporary that exists to be copied: applying the plan writes
    every term once and lowering copies every block again. ``lower_plan`` composes the
    two and writes each term straight into its slot, one pass whether or not the term
    carries a coefficient. Which terms are applied, and with
    which coefficients, is untouched -- only when the writes happen. The ordinary route
    remains the fallback wherever ``lower_plan`` declines (an immutable backend, so JAX
    is unaffected).
    """
    structure, perm, terms = sides_plan(t.structure, outputs, inputs)
    mats = lower_plan(t, structure, perm, terms)
    return structure, to_matrices(repartition(t, outputs, inputs)) if mats is None else mats


def trace(t: "SymmetricTensor", axes: Sequence[int]) -> "SymmetricTensor":
    """Close axis ``i`` of ``t`` onto axis ``j`` — the **supertrace** on a graded provider.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor to close one pair of legs on. Must keep at least one free
        leg afterwards.
    axes : sequence of two ints
        ``(i, j)``, the two public axes to contract against each other. The
        two legs must be contractible (same space, opposite outward dual).

    Returns
    -------
    SymmetricTensor
        ``t`` with the pair closed; the free legs keep their order and are
        returned unchanged, as in [tensordot][tenet.tensordot].

    Raises
    ------
    ValueError
        From the delegated [tensordot][tenet.tensordot] call: a non-contractible
        pair, out-of-range axes, or a trace that would leave no free leg.
    CapabilityError
        If closing a same-side pair needs a bend and the provider does not
        implement ``BendingCoefficients``, or if the provider does not implement
        [TwistData][tenet.symmetry.TwistData] (the closure's ``theta``).

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> W = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT), Leg(W, IN)), seed=0)
    >>> tenet.trace(t, (1, 2)).ndim
    1

    Notes
    -----
    **The closed wire pays the ribbon twist, and that is what makes a loop's value unique.** Of
    the two wires this closure runs — one to each leg of the identity — exactly one has its
    duality pairing against the direction the composition takes, and the categorical closure
    differs from the naive one by ``theta`` there. Without it the same fermionic loop takes
    different values depending on which of its wires was chosen to close, measured at a spread
    of 2.0 on a 4-cycle; with it the choice does not matter to 1e-16
    (``tests/ops/test_twist.py``). It is PEPSKit's ``str`` — ``tr`` for a bosonic braiding,
    where ``theta`` is 1 and [tenet.twist][] hands the tensor straight back. The *map view*'s
    closure, [full_trace][tenet.full_trace], is the pivotal trace and deliberately not this.

    The identity's OUT leg meets
    whichever of the two carries the dual object it needs, so a same-side pair
    (which needs a bend) and an OUT/IN pair are the same code path — TensorKit
    keeps a separate ``trace_permute!`` as a *performance* special case, not as
    different mathematics.

    The identity is built on ``t``'s own backend and dtype — the
    ``dtype=ar.get_dtype_name(ref), like=ref`` spelling ``ops/map.py::compose``
    already uses for its zero-filled sectors. Without it a torch-backed
    ``t`` would meet NumPy blocks in ``matmul``. A block-less ``t`` carries neither,
    and the identity is then NumPy ``float64`` by the same rule the contraction's
    zeros follow.
    """
    i, j = axes
    p = 0 if t.legs[j].side is OUT else 1
    ref = _ref(t)
    t = twist(t, i if t.legs[i].side is OUT else j)
    return tensordot(
        t,
        identity((t.legs[j],), dtype=ar.get_dtype_name(ref), like=ref),
        axes=((i, j), (p, 1 - p)),
    )


def full_trace(t: "SymmetricTensor") -> Any:
    """``Σ_c qdim(c) · tr(M_c)`` — the categorical trace of an endomorphism, a scalar.

    Parameters
    ----------
    t : SymmetricTensor
        A square map: its codomain and domain must carry the same
        ``(space, dual)`` sequence, in order.

    Returns
    -------
    scalar
        The backend's own scalar — no ``float()``, which would make the
        function unusable under ``jit``/``grad``/``vmap``; callers needing a
        Python float say ``float(tenet.full_trace(t))``.

    Raises
    ------
    CapabilityError
        If ``t``'s provider does not implement
        [QuantumDimensionData][tenet.symmetry.QuantumDimensionData] and
        [PivotalData][tenet.symmetry.PivotalData], or is not *spherical*
        (``qdim(c) != qdim(dual(c))`` on some traced sector — the left and
        right traces would disagree and ``full_trace`` refuses to pick one).
    ValueError
        If the map is not square space-wise (``check_square``'s refusal).

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> int(tenet.full_trace(tenet.identity((Leg(V, OUT),))))
    2

    Notes
    -----
    **Open diagrams are tensors; closed diagrams exit to backend scalars, explicitly
    and by name.** [tensordot][tenet.tensordot], [einsum][tenet.einsum] and
    [trace][tenet.trace] never return a
    scalar — a contraction that closes a network is a ``ValueError``, and a
    ``SymmetricTensor`` still has no rank 0. Leaving the tensor world is a separate,
    named call — [norm][tenet.norm], ``full_trace``, [inner][tenet.inner] — which returns the
    backend's own scalar and is therefore traceable and differentiable.

    The pair closed is the **map view**: codomain against domain, in order, the same
    view ``eigh``, ``expm``, ``svd`` and ``to_matrices`` act through. Any rank with a
    square map, so a rank-4 ``(V OUT, W OUT | V IN, W IN)`` gives ``np.einsum("abab->")``
    and not an axis-adjacent pairing; [trace][tenet.trace] remains the way to close one *chosen*
    pair and to keep a tensor.

    The ``qdim`` weight is the same one [norm][tenet.norm] carries, and it is what makes
    ``full_trace(t) == np.trace(t.to_dense())`` hold for a rank-2 map; dropping it is
    wrong for any non-Abelian provider.
    """
    requires(t.provider, QuantumDimensionData)
    requires(t.provider, PivotalData)
    check_square(t, "full_trace")
    if not t.blocks:
        return 0.0
    # requires() above; raise-based check does not narrow
    qdim = t.provider.qdim  # ty: ignore[unresolved-attribute]
    dual = t.provider.dual
    mats = to_matrices(t)
    for c in mats:
        # the spherical property, checked on the sectors actually traced (exact:
        # quantum dimensions come from exact arithmetic on every provider)
        if qdim(c) != qdim(dual(c)):
            raise CapabilityError(
                f"full_trace: provider {t.provider.name} is not spherical at {c!r} "
                f"(qdim {qdim(c)!r} != dual's {qdim(dual(c))!r}); the left and right "
                "traces disagree, so the qdim-weighted spherical trace is undefined"
            )
    return sum(qdim(c) * ar.do("trace", m) for c, m in mats.items())


def inner(a: "SymmetricTensor", b: "SymmetricTensor") -> Any:
    """``<a|b> = Σ_τ qdim(c_τ) · <A_τ, B_τ>`` — [norm][tenet.norm]'s sesquilinear sibling.

    Parameters
    ----------
    a : SymmetricTensor
        The bra side; the pairing is sesquilinear (conjugate-linear) in ``a``.
    b : SymmetricTensor
        The ket side; must have the same structure as ``a``.

    Returns
    -------
    scalar
        The backend's own scalar ``<a|b>``, traceable and differentiable;
        ``inner(a, a)`` equals ``norm(a)**2``.

    Raises
    ------
    ValueError
        If the structures do not match — different providers, different ``ndim``,
        or a differing leg; the message names the first differing axis and both
        legs, exactly as [zip_blocks][tenet.zip_blocks] and [add][tenet.add] do,
        since the aligned-blocks precondition is the same one.
    CapabilityError
        If ``a``'s provider does not implement
        [QuantumDimensionData][tenet.symmetry.QuantumDimensionData] (the ``qdim``
        weight).

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> round(float(tenet.inner(a, a) - tenet.norm(a) ** 2), 6)
    0.0

    Notes
    -----
    **Coefficient space, per fusion-tree block, ``qdim``-weighted — no diagram.**
    This is literally [norm][tenet.norm]'s body with the square replaced by a
    conjugated pair, so ``inner(a, a) == norm(a) ** 2`` holds *identically* rather
    than numerically, and the dense ``Σ conj(a) · b`` over ``to_dense`` is its
    oracle. It is TensorKit's spelling as well
    (``src/tensors/vectorinterface.jl``: ``Σ_c dim(c) · inner(block(t1, c), block(t2, c))``
    in both fusion-style branches), i.e. the pairing MPSKit's Krylov machinery runs on.

    **Drawing the pairing as a diagram would be wrong here.** Contracting every axis but
    the first, then closing, makes the still-open axis-0 lines *cross* the contracted
    ones, and on a graded provider each crossing of two odd lines pays ``-1``: an
    invariant scalar has (axis-0 sector) = (sector of the rest), so exactly the
    odd-sector blocks would enter the sum with the wrong sign and ``inner(t, t)`` would
    differ from ``norm(t) ** 2``. No diagram, no crossing, no twist — and no rank cap
    either: this works at any rank.

    Returns the backend's own scalar, so the whole function stays traceable and
    differentiable, as [norm][tenet.norm] is.
    """
    provider = a.provider
    requires(provider, QuantumDimensionData)
    _check_same_structure(a, b, "inner")
    if not a.blocks:
        return 0.0
    # No float(): concretizing here makes `inner` unusable under jit/grad/vmap.
    return sum(
        # requires() above; raise-based check does not narrow
        provider.qdim(key.coupled)  # ty: ignore[unresolved-attribute]
        * ar.do("sum", ar.do("conj", x) * y)
        for (key, x), y in zip(a.items(), b.blocks, strict=True)
    )


def _parse(equation: str, operands: tuple["SymmetricTensor", ...]) -> tuple[list[str], str]:
    """``(terms, output)`` for an equation over any number of operands, or a loud refusal.

    The operands enter only through their ranks, so this is ``_parsed`` on
    ``(equation, ndims)`` with the terms handed back as a fresh mutable list.
    """
    terms, out = _parsed(equation, tuple(t.ndim for t in operands))
    return list(terms), out


# Simplification: unbounded `cache`, matching every other plan cache in the module. An
# equation is written into a caller's source, not generated per call, so the key space is
# bounded by the program; the upgrade path if one is ever generated is `lru_cache(maxsize=)`.
@cache
def _parsed(equation: str, ndims: tuple[int, ...]) -> tuple[tuple[str, ...], str]:
    """``_parse``'s body, keyed on everything it reads, returning only immutables.

    ``_contract_path`` re-parses once per pairwise step, so an n-operand ``einsum``
    parses n times; the parse is ~10% of an eight-operand call's cumulative time.

    Every refusal below is a categorical statement, not a parser limitation, and
    each one names what to write instead.
    """
    if not ndims:
        raise ValueError(
            f"einsum: equation {equation!r} was given no operands; einsum takes one or more "
            "tensors after the equation, as in tenet.einsum('abc,cde->abde', A, B)"
        )
    if "." in equation:
        raise ValueError(
            f"einsum: equation {equation!r} contains '...'; ellipsis means broadcasting over "
            "unlabelled axes, and symmetric tensors do not broadcast (docs/design.md 'Supported "
            "ndarray operations should be explicit'). Label every axis explicitly"
        )
    lhs, arrow, rhs = equation.replace(" ", "").partition("->")
    terms = lhs.split(",")
    for label in lhs.replace(",", "") + rhs:
        if not label.isalpha() or not label.isascii():
            raise ValueError(
                f"einsum: label {label!r} in equation {equation!r} is not an ASCII letter; "
                "labels are single letters a-z / A-Z, one per axis"
            )
    if len(terms) != len(ndims):
        raise ValueError(
            f"einsum: equation {equation!r} has {len(terms)} comma-separated term(s) but "
            f"{len(ndims)} operand(s) were given; there is exactly one term per operand"
        )
    for k, (term, ndim) in enumerate(zip(terms, ndims, strict=True)):
        if len(term) != ndim:
            raise ValueError(
                f"einsum: term {term!r} labels {len(term)} axes but operand {k} is "
                f"{ndim}-dimensional; every axis gets exactly one label"
            )
    counts = Counter(lhs.replace(",", ""))
    over = [label for label, n in counts.items() if n > 2]
    if over:
        raise ValueError(
            f"einsum: label {over[0]!r} occurs {counts[over[0]]} times in {equation!r}; a label "
            "names one wire and so occurs at most twice, once on each of its two ends"
        )
    for k, term in enumerate(terms):
        repeated = [label for label, n in Counter(term).items() if n > 1]
        if repeated:
            label = repeated[0]
            if arrow and label in rhs:
                raise ValueError(
                    f"einsum: label {label!r} is repeated inside term {term!r} and also appears "
                    "in the output, i.e. a diagonal. A diagonal is not defined for a symmetric "
                    "tensor: it is not equivariant unless the two legs' bases are identified, "
                    "which is extra data the tensor does not carry (invariant 11)"
                )
            # Simplification: refused, not lowered. `trace(t, (i, j))` would serve "ii->" in
            # one line; add it when a caller actually writes the equation that way.
            raise ValueError(
                f"einsum: label {label!r} is repeated inside term {term!r}, i.e. a trace over "
                f"two axes of operand {k}. einsum contracts *between* operands; use "
                f"tenet.trace(t, axes) for the trace, then einsum on the result"
            )

    # NumPy's implicit rule: every label occurring exactly once, alphabetically
    out = rhs if arrow else "".join(sorted(label for label, n in counts.items() if n == 1))
    seen: set[str] = set()
    for label in out:
        if label in seen:
            raise ValueError(
                f"einsum: label {label!r} is repeated in the output {out!r}; two output axes "
                "cannot carry the same label"
            )
        seen.add(label)
        if label not in counts:
            raise ValueError(
                f"einsum: output label {label!r} of {equation!r} appears in no input term; "
                f"the input labels are {''.join(sorted(counts))!r}"
            )
    for label, n in counts.items():
        if n == 1 and label not in seen:
            raise ValueError(
                f"einsum: input label {label!r} of {equation!r} is missing from the output, "
                "which would mean summing that axis away. Summing an axis is a contraction "
                "with the all-ones vector, which is not equivariant and has no categorical "
                "meaning (invariant 11); keep the label, or contract it against another tensor"
            )
    return tuple(terms), out


def _plan_shape(t: "SymmetricTensor") -> tuple[int, ...]:
    """What the path finder is allowed to see: physical dimensions, ``Σ_a m_a d_a``.

    A planner asked to minimize FLOPs must see the physical extent of an axis, not its
    degeneracy count. A provider without ``ClebschGordanData`` has no
    physical shape at all, and there ``reduced_shape`` is the only thing on
    offer; it can degrade *path quality*, never correctness, since the path only
    ever decides the order of contractions that are each individually checked.
    """
    try:
        return t.shape
    except CapabilityError:
        return t.reduced_shape


def einsum(
    equation: str, *operands: "SymmetricTensor", optimize: Any = "auto"
) -> "SymmetricTensor":
    """``tenet.einsum("abc,cde,ef->abdf", A, B, C)`` — any number of operands.

    Parameters
    ----------
    equation : str
        The label equation, one comma-separated term per operand. Labels are
        single ASCII letters, one per axis; a label occurs at most twice in
        the whole equation (a wire has two ends). ``->`` may be omitted, in
        which case the output is every label occurring exactly once, sorted
        (the NumPy rule).
    *operands : SymmetricTensor
        One or more tensors, in the equation's term order.
    optimize : str, path, or opt_einsum.paths.PathOptimizer, optional
        Consulted only with three or more operands, where it is handed to
        ``opt_einsum.contract_path`` unchanged; cotengra's optimizers are such
        objects and work here without ``cotengra`` being imported. Default
        ``"auto"``.

    Returns
    -------
    SymmetricTensor
        The contraction, its public axes in the output labels' order; free
        legs come back exactly as they went in.

    Raises
    ------
    ValueError
        The parser's refusals, each naming what to write instead: no
        operands; ellipsis (symmetric tensors do not broadcast); a non-ASCII
        or non-letter label; a term/operand count or length mismatch; a label
        occurring more than twice; a label repeated *within* one operand — a
        diagonal (not equivariant, invariant 11) or a single-operand trace
        (use [trace][tenet.trace]); a repeated output label; an output label
        appearing in no input; or an input label missing from the output,
        which would sum an axis away (not equivariant, invariant 11). Also
        [tensordot][tenet.tensordot]'s refusals for each pairwise step, e.g. a
        shared label whose two legs are not contractible.
    CapabilityError
        If a pairwise step needs a bend the provider cannot supply, as in
        [tensordot][tenet.tensordot].

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> b = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)
    >>> c = tenet.einsum("ab,bc->ac", a, b)
    >>> bool(tenet.allclose(c, tenet.tensordot(a, b, axes=((1,), (0,)))))
    True

    Notes
    -----
    Repeated labels
    within one operand (a trace or a diagonal) and ellipsis are refused; see the
    message on each.

    With one or two operands this is the pairwise lowering and ``optimize`` is
    not consulted (``opt_einsum`` is not even imported). With three or more the
    pairwise order is chosen by ``opt_einsum.contract_path`` from the operands'
    physical [shape][tenet.SymmetricTensor.shape]\\ s, and ``optimize`` is handed
    to it unchanged: a strategy name, an explicit path, or any
    ``opt_einsum.paths.PathOptimizer`` — cotengra's optimizers are such objects
    and work here without ``cotengra`` being imported. A strategy *name* asks for a
    search, so its path is cached on ``(equation, shapes, name)``; a path and a
    ``PathOptimizer`` are consulted on every call. Every step of the path is
    then this same two-operand call, so the mathematics is unchanged; the path is
    chosen from static structure only, and is therefore baked in at trace time
    under ``jax.jit`` like every other structural decision.

    Shared labels are contracted in order of first appearance in the **first**
    operand — any order gives the same tensor, but a nondeterministic one would
    fragment plan caches and make ``jit`` retrace. The final transpose is a
    public permutation and therefore fully categorical (a Koszul sign for a
    fermionic provider, a braid for SU(2)); it is never skipped, and
    ``permutation_plan``'s case A already makes the identity permutation free.
    """
    terms, out = _parse(equation, operands)
    if len(operands) == 1:
        return transpose(operands[0], tuple(terms[0].index(label) for label in out))
    if len(operands) > 2:
        return _contract_path(terms, out, operands, optimize)

    shared = [label for label in terms[0] if label in terms[1]]
    axes = (
        tuple(terms[0].index(label) for label in shared),
        tuple(terms[1].index(label) for label in shared),
    )
    # tensordot's output is a's free legs then b's free legs, in each operand's order
    free = [label for label in terms[0] if label not in shared]
    free += [label for label in terms[1] if label not in shared]
    # exactly two operands on this path (the len() checks above); a tuple
    # unpack of statically unknown length is what the checker refuses
    # The output reordering stays a separate ``transpose`` and is *not* folded into the
    # contraction's restore plan, though ``_tensordot`` would take it: the reordering is
    # applied to a value ``tensordot`` has already returned, and folding it would mean
    # ``einsum`` no longer calling ``tensordot``.
    return transpose(
        tensordot(*operands, axes),  # ty: ignore[too-many-positional-arguments]
        tuple(free.index(label) for label in out),
    )


# Simplification: unbounded `cache`, matching every other plan cache in the module; the
# upgrade path if a caller ever generates equations per call is `lru_cache(maxsize=)`.
@cache
def _path(
    equation: str, shapes: tuple[tuple[int, ...], ...], optimize: str
) -> tuple[tuple[int, ...], ...]:
    """The pairwise order ``opt_einsum`` chooses, for a named strategy.

    Path search is not free against a graded tensor's block work, which is what
    a dense estimate gets wrong: blocks are small and numerous, so the search
    measured 0.05 ms against 0.20 ms of block work and ``opt_einsum.contract_path``
    was 29% of a six-operand ``einsum`` by cumulative time.

    Only a strategy *name* is cached, which is what makes this invisible: an
    explicit path is already a path, and a ``PathOptimizer`` object may be stateful
    and deliberately non-deterministic (cotengra's are), so both keep going to
    ``opt_einsum`` on every call, as the docstring's "handed to ``opt_einsum``
    unchanged" promises. The key is complete — nothing but the equation, the
    operands' plan shapes and the strategy reaches the search.
    """
    import opt_einsum as oe

    # `opt_einsum` types `optimize` as a Literal union of strategy names; a plain `str`
    # is exactly what a caller passes, and validating it here would duplicate the
    # refusal `contract_path` already raises for an unknown name.
    strategy: Any = optimize
    path, _ = oe.contract_path(equation, *shapes, shapes=True, optimize=strategy)
    return tuple(path)


def _contract_path(
    terms: list[str], out: str, operands: tuple["SymmetricTensor", ...], optimize: Any
) -> "SymmetricTensor":
    """Execute a three-or-more-operand equation as a sequence of pairwise ones.

    Imported lazily so ``import tenet`` never pays for ``opt_einsum`` and the
    two-operand path never touches it.

    The parser's refusals do the reasoning here: every label occurs at most twice
    in the whole equation, so the labels shared by any two chosen terms appear
    nowhere else and are *always* fully contracted — no hyper-indices, no batch
    labels, no partial contractions. An intermediate's free legs are the input
    legs unchanged (``contraction_plan``'s contract), so no new categorical
    work happens between steps either.

    **Precondition for a braided provider** (fermion parity, or a product containing
    it): the answer is path-independent only for steps adjacent in the caller's
    order — which is every step of a chain, and every step of any path
    ``opt_einsum`` returns without an outer product. A step that reaches across an
    intervening operand, as one can in a network with a loop, drags the contracted
    wire past that operand's legs, and the Koszul sign of *that* crossing is not
    expressible in the pairwise API this loop is built from (a dense fold of the
    same network is ambiguous in exactly the same way).
    """
    # Simplification: the ceiling above is a fermionic loop network whose best path is
    # not adjacent; the upgrade path is a sign correction computed from the skipped
    # operands' parities, which is M9's categorical path planning and needs new
    # mathematics rather than a scheduler.
    equation = f"{','.join(terms)}->{out}"
    shapes = tuple(_plan_shape(t) for t in operands)
    if isinstance(optimize, str):
        # A hit skips `oe.contract_path` outright rather than re-entering it with the
        # cached path: re-entry only re-validates the path against the same equation and
        # the same shapes that produced it, and the loop below rejects a non-pair step
        # itself. Skipping it is 0.21 -> 0.18 ms at three operands and 0.80 -> 0.74 ms
        # at six: at the low end it is more of the win than the search is.
        path = _path(equation, shapes, optimize)
    else:
        import opt_einsum as oe

        path, _ = oe.contract_path(equation, *shapes, shapes=True, optimize=optimize)
    terms, tensors = list(terms), list(operands)
    # `rank` is each entry's position in the equation as the *caller* wrote it, which
    # `opt_einsum`'s bookkeeping (pop both, append the result) does not preserve. It
    # decides which of the two is the left operand of the pairwise call below, and it
    # matters: the two ends of a wire are not interchangeable for a fermionic provider
    # (the cap V*⊗V → 1 is not the cap V⊗V* → 1), so `einsum("ab,bc->ac", A, B)` and
    # `einsum("bc,ab->ac", B, A)` differ by a Koszul sign. Contracting always in the
    # caller's order is what makes the answer independent of the path.
    rank = list(range(len(operands)))
    for step in path:
        if len(step) != 2:
            raise ValueError(
                f"einsum: the contraction path {path!r} contains the step {step!r}, which "
                f"contracts {len(step)} operands at once; every step must be a pair, since "
                "each one is lowered to a two-operand contraction. A single-operand step "
                "would be a sum over an axis, which the parser already refuses"
            )
        i, j = sorted(step, reverse=True)  # pop the high index first
        hi = (rank.pop(i), terms.pop(i), tensors.pop(i))
        lo = (rank.pop(j), terms.pop(j), tensors.pop(j))
        (ra, ta, a), (rb, tb, b) = sorted((hi, lo), key=operator.itemgetter(0))
        keep = set(out).union(*map(set, terms))  # labels some later step still needs
        mid = "".join([x for x in ta if x in keep] + [x for x in tb if x in keep])
        tensors.append(einsum(f"{ta},{tb}->{mid}", a, b))
        terms.append(mid)
        rank.append(min(ra, rb))
    # the last intermediate carries `out`'s labels, but in the order the path left
    # them; one categorical transpose puts them in the order the caller asked for
    return transpose(tensors[0], tuple(terms[0].index(label) for label in out))
