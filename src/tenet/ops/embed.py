"""Graded-space plumbing: ``embed`` (#83), ``restrict`` (#90), ``direct_sum`` (#91).

The *growing* direction of a bond. ``svd_truncated`` (#64) shrinks a graded bond
space by deciding it from data and ``svd(..., bond=B)`` (#77) holds one fixed;
``embed`` is the only way to make one bigger, or to give a tensor a sector it did
not have. ``restrict`` is its adjoint — the slice back down to a contained space,
refusing rather than silently discarding whatever leaves the target — and
``direct_sum`` is the general case of which ``embed`` is "the second operand is
zero": ``t`` into the leading slots, ``u`` into the trailing ones, added.

All three are the *same* primitive, ``ar.do("pad")`` (``restrict`` runs it
backwards as basic slicing), which is why they share one module and one
containment checker.

Placement is a **prefix in the degeneracy index**: source degeneracy ``alpha``
lands at target ``alpha``, matching ``svd``'s "the largest ``k`` singular values
are a prefix" convention, so ``embed`` and a later ``svd(..., bond=)`` agree
about which slots are the old ones. It is *not* a prefix of the dense index: the
within-slab layout is ``alpha * d_a + m`` (``SymmetricTensor.to_dense``), so
whenever ``d_a > 1`` the embedded data is strided in the dense array and every
later sector's slab offset moves as well.

``embed`` changes the ``TensorStructure``, but it is **traceable**, and calling
it structure-changing would be a category error. What #64's
``StructureChangingError`` refuses is a structure decided *from block values*;
here the target comes from ``legs``, which is static metadata the caller chose —
the same argument #77 makes for ``bond=``. So it composes inside ``jit`` (which
retraces when ``legs`` changes, not when block values do) and inside ``grad``
(``pad`` is linear, and its adjoint is the slice back down, which every backend
derives on its own).

Nothing here touches a coefficient, so there is no capability to gate, no
Clebsch-Gordan, no dense detour and no provider branching — two ``ar.do`` calls
and a dict lookup.
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import autoray as ar
import numpy as np

from tenet.structure import TensorStructure
from tenet.symmetry.base import QuantumDimension, requires

if TYPE_CHECKING:
    from tenet.leg import Leg
    from tenet.tensor import SymmetricTensor

__all__ = ["direct_sum", "embed", "restrict"]

# Per-op wording for the shared containment refusals: (the verb, the side that
# may be missing a sector, the closing clause pointing at the other direction).
_WORDING = {
    "embed": (
        "embedding",
        "target space",
        "the target space must contain the source's, and embed never truncates — "
        "shrinking is a different operation with different semantics",
    ),
    "restrict": (
        "restriction",
        "tensor",
        "the tensor's space must contain the target's, and restrict never grows — "
        "growing is embed, and a target decided from the block values is svd_truncated",
    ),
}


def _check_containment(small: TensorStructure, large: TensorStructure, *, op: str) -> None:
    """Refuse anything that is not an inclusion, before a single block is read.

    Total and structural: leg count, provider, ``side``, ``dual`` and, per axis,
    every sector of ``small`` present in ``large`` with a degeneracy at least as
    large. ``name`` is deliberately *not* compared — it is user bookkeeping, the
    same stance ``ProductSpace.matches`` documents ("neither is ``name``").

    ``op`` is both the message prefix and the *direction*: ``embed`` grows the
    tensor into the target, so the tensor is ``small``; ``restrict`` cuts it down
    to the target, so the tensor is ``large``. Which of the two the messages call
    "the tensor" swaps with it, and nothing else does.
    """
    verb, missing, closing = _WORDING[op]
    tensor, target = (small, large) if op == "embed" else (large, small)
    if small.ndim != large.ndim:
        raise ValueError(
            f"{op}: target has {target.ndim} legs, but the tensor has {tensor.ndim}; "
            f"{verb} never adds or drops an axis"
        )
    for i, (a, b) in enumerate(zip(tensor.legs, target.legs, strict=True)):
        if a.provider != b.provider:
            raise ValueError(
                f"{op}: axis {i} has provider {a.provider.name}, but the target leg has "
                f"{b.provider.name}; {verb} never casts between symmetries"
            )
        if a.side is not b.side:
            raise ValueError(
                f"{op}: axis {i} has side {a.side.value!r}, but the target leg has "
                f"{b.side.value!r}; moving a leg between domain and codomain is "
                f"repartition, not {op}"
            )
        if a.dual != b.dual:
            raise ValueError(
                f"{op}: axis {i} has dual={a.dual}, but the target leg has dual={b.dual}; "
                f"dual is categorical (invariant 2) and {op} never flips it"
            )
    for i, (x, y) in enumerate(zip(small.legs, large.legs, strict=True)):
        for sector in x.space:
            if y.space.degeneracy(sector) < x.space.degeneracy(sector):
                raise ValueError(
                    f"{op}: axis {i} sector {sector!r} has degeneracy "
                    f"{tensor.legs[i].space.degeneracy(sector)} in the tensor but "
                    f"{target.legs[i].space.degeneracy(sector)} in the target space"
                    + (
                        f" (the sector is absent from the {missing} entirely)"
                        if y.space.degeneracy(sector) == 0
                        else ""
                    )
                    + "; "
                    + closing
                )


def _pad(x: Any, pad_width: tuple[tuple[int, int], ...]) -> Any:
    """``ar.do("pad", x, pad_width)``, spelled so every backend agrees (#95).

    autoray 0.10.1's torch translation is
    ``tuple(chain.from_iterable(pad_width))[::-1]``, which reverses each
    ``(before, after)`` pair as well as the axis order that torch wants — so a
    ``((0, 1), (0, 2))`` pad lands the data in the *trailing* slots instead of the
    leading ones. Right shape, wrong slots, no error: measured on autoray 0.10.1
    with torch 2.13.0. Concatenation against a zero slab needs no translation and
    is the same values on every backend, so it replaces ``pad`` here rather than a
    backend test being written around it.
    """
    for axis, (before, after) in enumerate(pad_width):
        for width, leading in ((before, True), (after, False)):
            if not width:
                continue
            shape = list(ar.do("shape", x))
            shape[axis] = width
            zeros = ar.do("zeros", tuple(shape), dtype=ar.get_dtype_name(x), like=x)
            x = ar.do("concatenate", (zeros, x) if leading else (x, zeros), axis=axis)
    return x


def embed(t: "SymmetricTensor", legs: Sequence["Leg"]) -> "SymmetricTensor":
    """``t`` re-expressed on larger legs: same data, leading slots, zeros elsewhere.

    ``legs[i]`` must match ``t.legs[i]`` in provider, ``side`` and ``dual``, and its
    space must **contain** ``t.legs[i].space``: every sector ``a`` of the source
    appears in the target with ``degeneracy(a) >= source.degeneracy(a)``. New
    sectors are allowed and arrive as zero blocks. ``name`` is taken from ``legs``
    and never compared.

    Since ``_block_order`` enumerates ``product(*(leg.sectors ...))`` and
    ``fusion_trees`` is a pure function of the sector tuple, containment makes the
    source's keys a *subset* of the target's — which is the fact the loop below
    rests on, and why no key can silently lose its data.
    """
    from tenet.tensor import SymmetricTensor

    new = TensorStructure(tuple(legs))
    _check_containment(t.structure, new, op="embed")

    index = {k: i for i, k in enumerate(t.structure.block_order)}
    ref = t.blocks[0]
    blocks = []
    for key in new.block_order:
        shape = new.block_shape(key)
        if key not in index:
            blocks.append(ar.do("zeros", shape, dtype=ar.get_dtype_name(ref), like=ref))
            continue
        src = t.blocks[index[key]]
        pad = tuple((0, n - m) for n, m in zip(shape, t.structure.block_shape(key), strict=True))
        # identity axes stay identity objects: no pad call, no copy
        blocks.append(src if not any(hi for _, hi in pad) else _pad(src, pad))
    return SymmetricTensor(new, tuple(blocks))


def restrict(
    t: "SymmetricTensor", legs: Sequence["Leg"], *, atol: float | None = None
) -> "SymmetricTensor":
    """``t`` re-expressed on smaller legs: the leading degeneracy slots, nothing else.

    The exact mirror of :func:`embed`: ``legs[i]`` must match ``t.legs[i]`` in
    provider, ``side`` and ``dual``, and its space must be **contained in**
    ``t.legs[i].space`` — every sector ``a`` of the target appears in the source
    with ``degeneracy(a) >= target.degeneracy(a)``. Sectors absent from ``legs``
    are dropped entirely. ``name`` is taken from ``legs`` and never compared.

    Placement is the same **prefix in the degeneracy index** ``embed`` uses, so
    the two agree about which slots are the "old" ones, and so does a subsequent
    ``svd(..., bond=B)`` (#77).

    Data outside the kept slots is **refused**, not discarded: the residual
    ``sqrt(‖t‖² - ‖restrict(t)‖²)`` — exact by Pythagoras, since kept and dropped
    occupy disjoint slots under the same ``qdim`` weight — is compared against
    ``atol``, which defaults to ``sqrt(eps(dtype)) * ‖t‖``, the relative spelling
    ``from_dense`` uses (#82) for the same units reason. ``atol=math.inf``
    projects without checking, and that is the form that goes inside ``jit``; the
    comparison is a concrete-value question and raises JAX's own
    ``ConcretizationTypeError`` under a trace otherwise.

    This is *not* a truncation. The target comes from ``legs``, static metadata
    the caller chose, so ``restrict`` is shape-static and traceable and sits with
    ``embed`` and ``svd(..., bond=)`` on the traceable side of #64's
    ``StructureChangingError``. A target decided *from the block values* ("drop
    whatever falls below 1e-8") is :func:`tenet.linalg.svd_truncated`'s job.
    """
    from tenet.tensor import SymmetricTensor

    new = TensorStructure(tuple(legs))
    _check_containment(new, t.structure, op="restrict")

    index = {k: i for i, k in enumerate(t.structure.block_order)}
    out = SymmetricTensor(
        new,
        tuple(
            t.blocks[index[key]][tuple(slice(0, m) for m in new.block_shape(key))]
            for key in new.block_order
        ),
    )
    if atol != math.inf:
        _refuse_discarded(t, out, atol)
    return out


def _sum_squares(x: Any) -> float:
    return float(ar.do("sum", ar.do("abs", x) ** 2))


def _refuse_discarded(t: "SymmetricTensor", kept: "SymmetricTensor", atol: float | None) -> None:
    """Refuse a restriction that threw away mass, naming the worst offending key.

    The per-key accumulation exists only for that message; it runs on the check
    path, which is untraceable anyway, so ``atol=math.inf`` pays nothing for it.
    """
    provider = t.provider
    requires(provider, QuantumDimension)
    index = {k: i for i, k in enumerate(kept.structure.block_order)}
    worst: tuple[float, Any] = (0.0, None)
    discarded_sq = total_sq = 0.0
    for key, block in t.items():
        # requires() above; raise-based check does not narrow
        weight = provider.qdim(key.coupled)  # ty: ignore[unresolved-attribute]
        whole = weight * _sum_squares(block)
        total_sq += whole
        i = index.get(key)
        # a dropped key loses all of its mass; a kept one loses the complement
        here = whole if i is None else max(whole - weight * _sum_squares(kept.blocks[i]), 0.0)
        discarded_sq += here
        if here > worst[0]:
            worst = (here, key)

    residual = math.sqrt(discarded_sq)
    if atol is None:
        dtype = np.promote_types(ar.get_dtype_name(t.blocks[0]), np.float32)  # see #95
        atol = math.sqrt(float(np.finfo(dtype).eps)) * math.sqrt(total_sq)
    if residual > atol:
        raise ValueError(
            f"restrict: discarded data — residual {residual:.6g} exceeds atol {atol:.6g}; "
            f"the worst block is {worst[1]} (mass {math.sqrt(worst[0]):.6g}). Pass a "
            "larger atol to accept it, or atol=math.inf to restrict without checking"
        )


def _check_summable(
    t: "SymmetricTensor", u: "SymmetricTensor", axes: int | Sequence[int]
) -> frozenset[int]:
    """Normalize ``axes`` and refuse every structural mismatch, before any block."""
    if t.ndim != u.ndim:
        raise ValueError(
            f"direct_sum: the operands have {t.ndim} and {u.ndim} legs; a direct sum "
            "never adds or drops an axis"
        )
    raw = (axes,) if isinstance(axes, int) else tuple(axes)
    if not raw:
        raise ValueError(
            "direct_sum: axes is empty; with nothing to sum along the operands would have "
            "to be added instead — that is tenet.add, and it needs identical structures"
        )
    normalized = []
    for ax in raw:
        if not -t.ndim <= ax < t.ndim:
            raise ValueError(
                f"direct_sum: axis {ax} is out of range for {t.ndim} legs (valid: "
                f"{-t.ndim} to {t.ndim - 1})"
            )
        normalized.append(ax % t.ndim)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"direct_sum: duplicate axis in axes={raw!r} (normalized {normalized})")

    summed = frozenset(normalized)
    for i, (a, b) in enumerate(zip(t.legs, u.legs, strict=True)):
        if a.provider != b.provider:
            raise ValueError(
                f"direct_sum: axis {i} has provider {a.provider.name} in t but "
                f"{b.provider.name} in u; a direct sum never casts between symmetries"
            )
        if a.side is not b.side:
            raise ValueError(
                f"direct_sum: axis {i} has side {a.side.value!r} in t but {b.side.value!r} "
                "in u; moving a leg between domain and codomain is repartition"
            )
        if a.dual != b.dual:
            raise ValueError(
                f"direct_sum: axis {i} has dual={a.dual} in t but dual={b.dual} in u; "
                "V ⊕ V* is not a graded space, and dual is categorical (invariant 2)"
            )
        if i not in summed and a.space != b.space:
            raise ValueError(
                f"direct_sum: axis {i} is not summed but the operands' spaces differ: "
                f"{a.space!r} vs {b.space!r}; every axis outside `axes` must agree exactly"
            )
    return summed


def _summed_legs(
    t: "SymmetricTensor", u: "SymmetricTensor", axes: frozenset[int]
) -> Iterator["Leg"]:
    """``t``'s legs, with the summed ones' degeneracies added sector-wise.

    ``name`` comes from ``t`` and the operands' need not agree — user bookkeeping,
    the stance ``ProductSpace.matches`` documents.
    """
    for i, (a, b) in enumerate(zip(t.legs, u.legs, strict=True)):
        if i not in axes:
            yield a
            continue
        yield replace(a, space=a.space.direct_sum(b.space))


def direct_sum(
    t: "SymmetricTensor", u: "SymmetricTensor", axes: int | Sequence[int]
) -> "SymmetricTensor":
    """``t ⊕ u`` along ``axes``: sector-wise degeneracy sums, ``t`` leading.

    Every axis **not** in ``axes`` must agree exactly between the operands in
    ``space``, ``side`` and ``dual`` (``name`` is user bookkeeping and is taken
    from ``t``). Every axis **in** ``axes`` must agree in provider, ``side`` and
    ``dual``; its spaces may differ freely, and the result's space has
    ``m_a = t.degeneracy(a) + u.degeneracy(a)`` for every sector of either.
    A ``dual`` mismatch is refused, never coerced — :func:`tenet.flip` is the
    way to normalise the operands' dual conventions before summing.

    Placement is ``t`` in the **leading** degeneracy slots and ``u`` in the
    trailing ones, on every summed axis — the same prefix convention ``embed``
    and ``svd(..., bond=)`` use, and the same order as TensorKit's ``catdomain``.

    With ONE summed axis every block splits into exactly two slabs, ``t``'s then
    ``u``'s. With MORE than one, the "mixed" regions — leading on one summed axis
    and trailing on another — are **zero**, because neither ``pad`` ever writes
    them, and that is precisely what makes a two-sided direct sum the
    block-diagonal map ``T ⊕ U``.

    Structure-static and traceable: the result structure comes from the operands'
    legs and ``axes``, which are metadata, so this composes inside ``jit`` and
    ``grad``. Linear in each operand separately.
    """
    from tenet.tensor import SymmetricTensor

    summed = _check_summable(t, u, axes)
    new = TensorStructure(tuple(_summed_legs(t, u, summed)))
    ref = t.blocks[0]
    # Promoted once, up front, and applied to every part: a key only one operand
    # contributes to must not come out at that operand's own dtype.
    dtype = np.promote_types(ar.get_dtype_name(ref), ar.get_dtype_name(u.blocks[0])).name  # #95
    if t.backend != u.backend:
        # `add` gets promotion for free because every one of its blocks meets
        # both operands; here a key only `t` touches would never see `u`'s
        # backend, so the result would carry blocks of *both*. Refused, not
        # half-promoted: `to_backend` is the explicit spelling.
        raise ValueError(
            f"direct_sum: the operands are on different backends, {t.backend} and "
            f"{u.backend}; call .to_backend(...) on one of them first"
        )
    tables = tuple({k: i for i, k in enumerate(o.structure.block_order)} for o in (t, u))

    blocks = []
    for key in new.block_order:
        sectors = new.axis_sectors(key)
        acc = None
        for operand, table, leading in ((t, tables[0], True), (u, tables[1], False)):
            i = table.get(key)
            if i is None:  # this operand lacks one of the key's sectors
                continue
            pad = tuple(
                (0, 0)
                if ax not in summed
                else (
                    (0, u.legs[ax].space.degeneracy(a))
                    if leading
                    else (t.legs[ax].space.degeneracy(a), 0)
                )
                for ax, a in enumerate(sectors)
            )
            part = ar.do("astype", _pad(operand.blocks[i], pad), dtype)
            acc = part if acc is None else acc + part
        if acc is None:
            # reachable only with two or more summed axes: the key takes one
            # sector from t's space and another from u's, so neither has it
            acc = ar.do("zeros", new.block_shape(key), dtype=dtype, like=ref)
        blocks.append(acc)
    return SymmetricTensor(new, tuple(blocks))


# Simplification: no plan object and no cache. embed does a dict lookup and a
# subtraction per block — there is no coefficient to compute, no tree to permute
# and no capability to gate, and the categorical work (TensorStructure) is
# already cached where it matters. Add a memoized boolean for
# _check_containment if a profile ever shows it on the hot path — a plan, never.
# Simplification: `_check_containment` is shared between embed and restrict, arguments
# swapped, with `op` naming the direction and the message prefix. Split it into
# two checkers only if the two refusals ever need to say genuinely different
# things — a third word of per-op wording is the trigger.
# Simplification: `axes` as a set, deliberately — one loop covers the one-axis bond
# merge and the two-sided block-diagonal assembly. Split into named helpers only
# if callers turn out to spell the two-sided case wrong often enough to matter.
# Simplification: pairwise only. `reduce(lambda a, b: direct_sum(a, b, axes), ts)` is
# one line at the call site and allocates exactly the intermediates a naive N-ary
# version would; add a genuinely fused N-way pad when a profile shows those
# intermediates dominating.
