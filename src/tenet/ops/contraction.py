"""Pairwise contraction — ``tenet.tensordot``, ``tenet.trace``, ``tenet.einsum`` — M5.

There is no new mathematics here, and that is the point (README "Composition
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
for symmetric tensors, and README places ``opt_einsum``/``cotengra`` at the
*path* level (M8, three or more operands) instead.

No ``to_dense``, no NumPy and no provider branching here (invariants 8/9).
"""

import operator
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tenet.leg import IN, OUT, Leg
from tenet.ops.map import compose, identity
from tenet.ops.permutation import transpose
from tenet.ops.repartition import repartition
from tenet.symmetry.base import BendingCoefficients, CapabilityError, requires

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["contractible", "einsum", "outward_dual", "tensordot", "trace"]

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


def _validated(
    a: "SymmetricTensor", b: "SymmetricTensor", axes: Axes
) -> tuple[tuple[int, ...], ...]:
    """``(a axes, b axes)`` as int tuples, checked in the CALLER's numbering."""
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


def _crossing(t: "SymmetricTensor", contracted: tuple[int, ...], to_out: bool) -> tuple[int, ...]:
    """Axes of ``t`` that must change side, in ``t``'s own numbering.

    ``to_out`` says where the *contracted* axes are headed: ``False`` for ``a``
    (whose contracted legs become its domain) and ``True`` for ``b``.
    """
    want = {ax: (OUT if to_out else IN) for ax in contracted}
    return tuple(
        ax for ax, leg in enumerate(t.legs) if leg.side is not want.get(ax, IN if to_out else OUT)
    )


def _refuse_bends(
    a: "SymmetricTensor", b: "SymmetricTensor", ca: tuple[int, ...], cb: tuple[int, ...]
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
        shaped = (a.structure.in_axes, b.structure.out_axes)
        raise CapabilityError(
            f"tensordot: this axis pattern moves legs between domain and codomain "
            f"({named}), which is a line bend, and provider {a.provider.name} does not "
            "implement BendingCoefficients. A composition-shaped pattern needs no bending "
            "capability at all: axes=(a's IN axes, b's OUT axes) in matching order, here "
            f"axes={shaped!r}, contracts the whole of a's domain against the whole of b's "
            "codomain with zero bends and works today for every provider."
        ) from exc


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
            "least one leg (a rank-0 SymmetricTensor does not exist; see its ponytail note "
            "on an explicit provider field). Returning a bare backend scalar would make the "
            "return type depend on the arguments — tenet.norm is the precedent for scalars "
            "leaving the tensor world explicitly"
        )
    _refuse_bends(a, b, ca, cb)

    c = compose(repartition(a, fa, ca), repartition(b, cb, fb))

    free = (*(a.legs[i] for i in fa), *(b.legs[j] for j in fb))
    outputs = tuple(k for k, leg in enumerate(free) if leg.side is OUT)
    inputs = tuple(k for k, leg in enumerate(free) if leg.side is IN)
    position = {k: p for p, k in enumerate((*outputs, *inputs))}
    return transpose(repartition(c, outputs, inputs), tuple(position[k] for k in range(len(free))))


def trace(t: "SymmetricTensor", axes: Sequence[int]) -> "SymmetricTensor":
    """Contract axis ``i`` of ``t`` against axis ``j`` of ``t``, through the identity.

    ``axes=(i, j)``; the two legs must be :func:`contractible`, which the
    delegated :func:`tensordot` call checks. The identity's OUT leg meets
    whichever of the two carries the dual object it needs, so a same-side pair
    (which needs a bend) and an OUT/IN pair are the same code path — TensorKit
    keeps a separate ``trace_permute!`` as a *performance* special case, not as
    different mathematics.
    """
    i, j = axes
    p = 0 if t.legs[j].side is OUT else 1
    return tensordot(t, identity((t.legs[j],)), axes=((i, j), (p, 1 - p)))


def _parse(equation: str, operands: tuple["SymmetricTensor", ...]) -> tuple[list[str], str]:
    """``(terms, output)`` for a one- or two-operand equation, or a loud refusal.

    Every refusal below is a categorical statement, not a parser limitation, and
    each one names what to write instead.
    """
    if not operands:
        raise ValueError(
            f"einsum: equation {equation!r} was given no operands; einsum takes one or two "
            "tensors after the equation, as in tenet.einsum('abc,cde->abde', A, B)"
        )
    if "." in equation:
        raise ValueError(
            f"einsum: equation {equation!r} contains '...'; ellipsis means broadcasting over "
            "unlabelled axes, and symmetric tensors do not broadcast (README 'Supported "
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
    if len(operands) > 2 or len(terms) > 2:
        raise ValueError(
            f"einsum: equation {equation!r} with {len(operands)} operands asks for a "
            "contraction over three or more tensors, which needs a contraction *path*; "
            "pairwise contraction must be correct before a path planner is added (README "
            "'Milestone 5'), and planner hand-off to opt_einsum/cotengra is Milestone 8. "
            "Chain two calls today, choosing the order yourself: "
            "tenet.einsum('ij,jk->ik', tenet.einsum('ab,bi->ai', A, B), C)"
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
            # ponytail: refused, not lowered. `trace(t, (i, j))` would serve "ii->" in
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


def einsum(equation: str, *operands: "SymmetricTensor") -> "SymmetricTensor":
    """``tenet.einsum("abc,cde->abde", A, B)`` — one or two operands.

    Labels are single letters. ``->`` may be omitted, in which case the output is
    every label occurring exactly once, sorted (the NumPy rule). Repeated labels
    within one operand (a trace or a diagonal), ellipsis, and three or more
    operands are refused; see the message on each.

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

    shared = [label for label in terms[0] if label in terms[1]]
    axes = (
        tuple(terms[0].index(label) for label in shared),
        tuple(terms[1].index(label) for label in shared),
    )
    # tensordot's output is a's free legs then b's free legs, in each operand's order
    free = [label for label in terms[0] if label not in shared]
    free += [label for label in terms[1] if label not in shared]
    return transpose(tensordot(*operands, axes), tuple(free.index(label) for label in out))
