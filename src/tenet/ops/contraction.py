"""Pairwise contraction — ``tenet.tensordot``, ``tenet.trace``, ``tenet.einsum`` — M5.

There is no new mathematics here, and that is the point (docs/design.md "Composition
remains first-class", "Bending and contraction"). A contraction is lowered to
operations that already exist and are already dense-oracle tested::

    a ──repartition(outputs=free, inputs=contracted)──▶  a' : Hom(K, F_a)
    b ──repartition(outputs=contracted, inputs=free)──▶  b' : Hom(F_b, K)

    compose(a', b') : Hom(F_b, F_a)   # legs = a' codomain ++ b' domain, which is
                                      # already the documented output order
      ──repartition(each free leg back to its original side)──▶
      ──transpose──▶ (a free..., b free...)

``repartition`` is itself the transpose/bend/transpose sandwich (#32), so the
whole module is axis bookkeeping over frozen metadata plus four already-cached
plans. TensorKit's ``tensorcontract!`` ends in ``permute(compose(sA, sB), pAB)``
for the same reason.

Two design decisions carry the module:

* **Contractibility is ``same space`` + opposite :func:`outward_dual`**, and it
  is phrased that way because it is *bend-invariant*: a bend flips ``side`` and
  ``dual`` together, so ``dual xor (side is IN)`` flips twice and is unchanged.
  A rule phrased on ``(side, dual)`` would have to be restated after every bend
  the lowering performs. Dimensions are never compared (invariant 2): a
  charge-reversed U(1) partner has the same dimension and the wrong space.
* **Free legs come back exactly as they went in** — same ``space``, ``side``,
  ``dual`` and ``name`` — at the cost of a second repartition. Letting a free IN
  leg return as an OUT leg because that is where the composition left it would
  make the output legs depend on the lowering. The round trip is *exact*, not
  approximately so, because ``bend_left`` is the conjugate of ``bend_right``
  (#38), i.e. its genuine inverse.

Everything is refused before any block moves: axis validation, contractibility
and the ``BendingCoefficients`` requirement of every leg that will cross — on
both operands, *including free legs*, which is the non-obvious half.
``compose``'s own check stays as a redundant guard and can no longer fire on
user error; if it does, this module's rewriting is wrong.

``einsum`` sits on top and owns no mathematics at all: it parses the equation
into label→axis maps, hands the shared labels to :func:`tensordot` as one axis
pair, and hands the requested output order to :func:`~tenet.transpose`. The
parser is hand-rolled — ``opt_einsum`` parses ellipsis, unicode labels, the
interleaved format and shape-driven broadcasting, none of which is meaningful
for symmetric tensors. ``opt_einsum`` enters one level up instead (M8, #67), at
the *path* level: with three or more operands ``contract_path`` orders the
pairwise contractions and each step is the two-operand call above, so the
scheduler adds a loop and no mathematics. It is imported lazily, inside that
branch only; ``cotengra``'s optimizers arrive through ``optimize=`` as ordinary
``PathOptimizer`` objects and ``cotengra`` is imported nowhere in ``tenet``.

No ``to_dense``, no NumPy and no provider branching here (invariants 8/9).
"""

import operator
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.leg import IN, OUT, Leg
from tenet.map_view import check_square, to_matrices
from tenet.ops.map import adjoint, compose, identity
from tenet.ops.permutation import transpose
from tenet.ops.repartition import repartition
from tenet.structure import TensorStructure
from tenet.symmetry.base import (
    BendingCoefficients,
    CapabilityError,
    QuantumDimension,
    requires,
)

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "ContractionPlan",
    "contractible",
    "contraction_plan",
    "einsum",
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
    :attr:`new_structure`, the output legs known without contracting anything.

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


# Simplification: unbounded `cache`, matching every other plan cache (see structure.py's
# note). The ceiling is a workload that generates genuinely new structures per call
# — data-dependent truncation inside a loop, which docs/design.md already places outside
# JIT; the upgrade path is `lru_cache(maxsize=...)`.
@cache
def contraction_plan(a: TensorStructure, b: TensorStructure, axes: Axes) -> ContractionPlan:
    """Plan ``tensordot(a, b, axes)``. Cached: repeat calls return one object.

    Raises here, before any block is touched and before one sub-plan is built:
    invalid axes, mismatched providers, non-contractible pairs, a contraction
    leaving no free leg, and a missing ``BendingCoefficients`` for a leg that
    must cross. ``axes`` accepts everything :func:`tensordot` accepts; it is
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

    ``axes`` is ``((i, ...), (j, ...))`` or NumPy's integer form (``axes=2`` means
    the last two axes of ``a`` against the first two of ``b``). Negative indices
    are refused, as in :func:`~tenet.transpose` and :func:`~tenet.repartition`.

    Output public axis order is ``a``'s free axes (in ``a``'s order) followed by
    ``b``'s free axes (in ``b``'s order), matching ``np.tensordot``; every free
    leg is returned **unchanged** — same ``space``, ``side``, ``dual`` and
    ``name``. ``axes=((), ())`` is the outer product and falls out of the
    lowering rather than being special-cased.

    All refusals, and all axis bookkeeping, live in :func:`contraction_plan`;
    what is left here is the execution of four already-tested operations.
    """
    # normalize first so the integer form and NumPy integer scalars share one entry
    plan = contraction_plan(a.structure, b.structure, _validated(a.structure, b.structure, axes))
    c = compose(
        repartition(a, plan.a_outputs, plan.a_inputs),
        repartition(b, plan.b_outputs, plan.b_inputs),
    )
    return transpose(
        repartition(c, plan.restore_outputs, plan.restore_inputs), plan.final_transpose
    )


def trace(t: "SymmetricTensor", axes: Sequence[int]) -> "SymmetricTensor":
    """Contract axis ``i`` of ``t`` against axis ``j`` of ``t``, through the identity.

    ``axes=(i, j)``; the two legs must be :func:`contractible`, which the
    delegated :func:`tensordot` call checks. The identity's OUT leg meets
    whichever of the two carries the dual object it needs, so a same-side pair
    (which needs a bend) and an OUT/IN pair are the same code path — TensorKit
    keeps a separate ``trace_permute!`` as a *performance* special case, not as
    different mathematics.

    The identity is built on ``t``'s own backend and dtype — the
    ``dtype=ar.get_dtype_name(ref), like=ref`` spelling ``ops/map.py::compose``
    already uses for its zero-filled sectors (#95). Without it a torch-backed
    ``t`` would meet NumPy blocks in ``matmul``.
    """
    i, j = axes
    p = 0 if t.legs[j].side is OUT else 1
    ref = t.blocks[0]
    return tensordot(
        t,
        identity((t.legs[j],), dtype=ar.get_dtype_name(ref), like=ref),
        axes=((i, j), (p, 1 - p)),
    )


def full_trace(t: "SymmetricTensor") -> Any:
    """``Σ_c qdim(c) · tr(M_c)`` — the categorical trace of an endomorphism, a scalar.

    **Open diagrams are tensors; closed diagrams exit to backend scalars, explicitly
    and by name.** :func:`tensordot`, :func:`einsum` and :func:`trace` never return a
    scalar — a contraction that closes a network is a ``ValueError``, and a
    ``SymmetricTensor`` still has no rank 0. Leaving the tensor world is a separate,
    named call — :func:`~tenet.norm`, ``full_trace``, :func:`inner` — which returns the
    backend's own scalar and is therefore traceable and differentiable.

    The pair closed is the **map view**: codomain against domain, in order, the same
    view ``eigh``, ``expm``, ``svd`` and ``to_matrices`` act through. Any rank with a
    square map, so a rank-4 ``(V OUT, W OUT | V IN, W IN)`` gives ``np.einsum("abab->")``
    and not an axis-adjacent pairing; :func:`trace` remains the way to close one *chosen*
    pair and to keep a tensor.

    The ``qdim`` weight is the same one :func:`~tenet.norm` carries, and it is what makes
    ``full_trace(t) == np.trace(t.to_dense())`` hold for a rank-2 map; dropping it is
    wrong for any non-Abelian provider. Returns the backend's own scalar — no ``float()``,
    which would make the function unusable under ``jit``/``grad``/``vmap``; callers
    needing a Python float say ``float(tenet.full_trace(t))``.
    """
    requires(t.provider, QuantumDimension)
    check_square(t, "full_trace")
    if not t.blocks:
        return 0.0
    qdim = t.provider.qdim
    return sum(qdim(c) * ar.do("trace", m) for c, m in to_matrices(t).items())


def inner(a: "SymmetricTensor", b: "SymmetricTensor") -> Any:
    """``<a|b>``: contract every axis but the first, then :func:`full_trace` the rest.

    Sesquilinear in ``a``, and :func:`~tenet.norm`'s sibling — ``inner(a, a)`` is
    ``norm(a)**2``. Works at any rank, which is what lets
    :func:`~tenet.network.lanczos` be a plain vector-space algorithm: the adjoint flips
    every leg, so axis 0 of ``adjoint(a)`` is IN and axis 0 of ``b`` is OUT, and the
    leftover rank-2 map is exactly what :func:`full_trace` closes.

    The ``string.ascii_lowercase`` labelling caps this at rank 26, above which
    :func:`einsum`'s own parser raises with a clear message. A rank-27 tensor has other
    problems.
    """
    rest = string.ascii_lowercase[1 : a.ndim]
    return full_trace(einsum(f"L{rest},l{rest}->lL", adjoint(a), b))


def _parse(equation: str, operands: tuple["SymmetricTensor", ...]) -> tuple[list[str], str]:
    """``(terms, output)`` for an equation over any number of operands, or a loud refusal.

    Every refusal below is a categorical statement, not a parser limitation, and
    each one names what to write instead.
    """
    if not operands:
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
    if len(terms) != len(operands):
        raise ValueError(
            f"einsum: equation {equation!r} has {len(terms)} comma-separated term(s) but "
            f"{len(operands)} operand(s) were given; there is exactly one term per operand"
        )
    for k, (term, t) in enumerate(zip(terms, operands, strict=True)):
        if len(term) != t.ndim:
            raise ValueError(
                f"einsum: term {term!r} labels {len(term)} axes but operand {k} is "
                f"{t.ndim}-dimensional; every axis gets exactly one label"
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
    return terms, out


def _plan_shape(t: "SymmetricTensor") -> tuple[int, ...]:
    """What the path finder is allowed to see: physical dimensions, ``Σ_a m_a d_a``.

    A planner asked to minimize FLOPs must see the physical extent of an axis
    (#19), not its degeneracy count. A provider without ``ClebschGordan`` has no
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

    Labels are single letters. ``->`` may be omitted, in which case the output is
    every label occurring exactly once, sorted (the NumPy rule). Repeated labels
    within one operand (a trace or a diagonal) and ellipsis are refused; see the
    message on each.

    With one or two operands this is the pairwise lowering and ``optimize`` is
    not consulted (``opt_einsum`` is not even imported). With three or more the
    pairwise order is chosen by ``opt_einsum.contract_path`` from the operands'
    physical :attr:`~tenet.SymmetricTensor.shape`\\ s, and ``optimize`` is handed
    to it unchanged: a strategy name, an explicit path, or any
    ``opt_einsum.paths.PathOptimizer`` — cotengra's optimizers are such objects
    and work here without ``cotengra`` being imported. Every step of the path is
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
    return transpose(tensordot(*operands, axes), tuple(free.index(label) for label in out))


# Simplification: no path cache. Path finding for a ten-tensor network is microseconds
# against block work measured in milliseconds, and the key would have to include
# `optimize`, which may be a stateful, deliberately non-deterministic optimizer.
# The ceiling is a hot loop over a large network re-planned every call; the upgrade
# path is a @cache keyed on (equation, shapes, optimize) restricted to `str`.
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
    legs unchanged (:func:`contraction_plan`'s contract), so no new categorical
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
    import opt_einsum as oe

    path, _ = oe.contract_path(
        f"{','.join(terms)}->{out}",
        *(_plan_shape(t) for t in operands),
        shapes=True,
        optimize=optimize,
    )
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
