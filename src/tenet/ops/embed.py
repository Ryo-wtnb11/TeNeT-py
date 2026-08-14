"""Zero-padding inclusion into larger graded spaces — ``tenet.embed`` (issue #83).

The *growing* direction of a bond. ``svd_truncated`` (#64) shrinks a graded bond
space by deciding it from data and ``svd(..., bond=B)`` (#77) holds one fixed;
``embed`` is the only way to make one bigger, or to give a tensor a sector it did
not have.

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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import autoray as ar

from tenet.structure import TensorStructure

if TYPE_CHECKING:
    from tenet.leg import Leg
    from tenet.tensor import SymmetricTensor

__all__ = ["embed"]


def _check_containment(old: TensorStructure, new: TensorStructure) -> None:
    """Refuse anything that is not an inclusion, before a single block is read.

    Total and structural: leg count, provider, ``side``, ``dual`` and, per axis,
    every source sector present in the target with a degeneracy at least as
    large. ``name`` is deliberately *not* compared — it is user bookkeeping, the
    same stance ``ProductSpace.matches`` documents ("neither is ``name``").
    """
    if old.ndim != new.ndim:
        raise ValueError(
            f"embed: target has {new.ndim} legs, but the tensor has {old.ndim}; "
            "embedding never adds or drops an axis"
        )
    for i, (a, b) in enumerate(zip(old.legs, new.legs, strict=True)):
        if a.provider != b.provider:
            raise ValueError(
                f"embed: axis {i} has provider {a.provider.name}, but the target leg has "
                f"{b.provider.name}; embedding never casts between symmetries"
            )
        if a.side is not b.side:
            raise ValueError(
                f"embed: axis {i} has side {a.side.value!r}, but the target leg has "
                f"{b.side.value!r}; moving a leg between domain and codomain is "
                "repartition, not embed"
            )
        if a.dual != b.dual:
            raise ValueError(
                f"embed: axis {i} has dual={a.dual}, but the target leg has dual={b.dual}; "
                "dual is categorical (invariant 2) and embed never flips it"
            )
        for sector in a.space:
            source, target = a.space.degeneracy(sector), b.space.degeneracy(sector)
            if target < source:
                raise ValueError(
                    f"embed: axis {i} sector {sector!r} has degeneracy {source} in the "
                    f"tensor but {target} in the target space"
                    + (
                        " (the sector is absent from the target space entirely)"
                        if target == 0
                        else ""
                    )
                    + "; the target space must contain the source's, and embed never "
                    "truncates — shrinking is a different operation with different "
                    "semantics"
                )


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
    _check_containment(t.structure, new)

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
        blocks.append(src if not any(hi for _, hi in pad) else ar.do("pad", src, pad))
    return SymmetricTensor(new, tuple(blocks))


# ponytail: no plan object and no cache. embed does a dict lookup and a
# subtraction per block — there is no coefficient to compute, no tree to permute
# and no capability to gate, and the categorical work (TensorStructure) is
# already cached where it matters. Add a memoized boolean for
# _check_containment if a profile ever shows it on the hot path — a plan, never.
# ponytail: no `restrict` (the adjoint, slicing back down). Mechanically five
# lines, but its refusals — what happens to non-zero data in a dropped slot —
# have no caller to settle them. tests/ops/test_embed.py performs it by hand.
