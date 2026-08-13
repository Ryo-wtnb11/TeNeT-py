"""Morphism composition ``a ∘ b`` and ``identity`` — Milestone 3.

Composition is one ``matmul`` per coupled sector and nothing else. Nothing
recouples: ``a``'s domain and ``b``'s codomain are the *same ordered legs*, so
``map_layout`` enumerates the same trees in the same order on both sides of the
join (band order is ``block_order`` restricted, a pure function of the legs), and
Clebsch-Gordan orthonormality ``Σ_u X_f^† X_{f'} = δ_{ff'} id_c`` collapses the
shared index. That is why SU(2) composition needs no F/R symbols at all — the
concrete payoff of keeping the coupled-sector matrix representation.

Compatibility is ``(space, dual, order)``, exactly, never dimension: comparing
only sizes would let a charge-reversed U(1) partner through and silently produce
a non-equivariant result. Order cannot be waived either — matching "up to a
reordering" would need a within-side transpose, which is a braid (#21).

No ``to_dense`` here, no provider branching, and NumPy appears only as
:func:`identity`'s default dtype.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import autoray as ar
import numpy as np

from tenet.leg import IN, OUT, Leg
from tenet.map_view import as_map, from_matrices, map_layout, to_matrices
from tenet.structure import TensorStructure

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["compose", "identity"]


def _check_composable(a: "SymmetricTensor", b: "SymmetricTensor") -> None:
    """Raise ``ValueError`` naming the offending axis on *both* tensors.

    Follows ``ops.basic._check_same_structure``'s per-axis style: the position in
    the domain/codomain is useless on its own when the public axis order
    interleaves the sides, so both public axis indices are printed.
    """
    domain, codomain = as_map(a).domain, as_map(b).codomain
    if len(domain.legs) != len(codomain.legs):
        raise ValueError(
            f"compose: a has {len(domain.legs)} IN legs but b has {len(codomain.legs)} "
            f"OUT legs; composition pairs them one for one "
            f"(a.domain={domain.legs!r}, b.codomain={codomain.legs!r})"
        )
    i = domain.matches(codomain)
    if i is None:
        return

    x, y = domain.legs[i], codomain.legs[i]
    message = (
        f"compose: a's domain and b's codomain differ at position {i} "
        f"(public axis {a.structure.in_axes[i]} of a, public axis {b.structure.out_axes[i]} "
        f"of b): {x!r} vs {y!r}. "
    )
    if x.provider != y.provider:
        message += f"The providers differ ({x.provider.name} vs {y.provider.name}). "
    raise ValueError(
        message + "Composition requires the same (space, dual) in the same order — side is "
        "not compared and name is ignored, dimensions alone are never enough. It never "
        "reorders legs within a side; use tenet.transpose for that, or repartition (#32) "
        "if a leg has to change side."
    )


def compose(a: "SymmetricTensor", b: "SymmetricTensor") -> "SymmetricTensor":
    """``a ∘ b``: ``b``'s codomain is consumed by ``a``'s domain. Spelled ``a @ b``.

    The result's public axis order is ``a``'s OUT legs followed by ``b``'s IN legs,
    each in its own public order.
    """
    _check_composable(a, b)
    ma, mb = to_matrices(a), to_matrices(b)
    structure = TensorStructure((*a.codomain, *b.domain))
    layout = map_layout(structure)

    mats: dict[Any, Any] = {
        c: ar.do("matmul", ma[c], mb[c]) for c in layout.sectors if c in ma and c in mb
    }
    if len(mats) < len(layout.sectors):
        # A coupled sector carried by only one operand contributes nothing: the
        # missing side has zero trees there, so the product is zero. Deliberate
        # (the result structure still declares the sector), not a KeyError.
        ref = next(iter(mats.values()), a.blocks[0])
        dtype = ar.get_dtype_name(ref)
        for c in layout.sectors:
            if c not in mats:
                mats[c] = ar.do("zeros", layout.shape(c), dtype=dtype, like=ref)
    return from_matrices(structure, mats)


def identity(legs: Sequence[Leg], *, dtype: Any = np.float64) -> "SymmetricTensor":
    """``id`` on ``ProductSpace(legs)``: the legs mirrored as ``(OUT..., IN...)``.

    ``space``, ``dual`` and ``name`` are kept and only ``side`` is set, so that
    ``identity(t.codomain) @ t == t``. Dualizing the mirror would build a cup, a
    different morphism (#32).

    ``B_c = eye`` for every coupled sector, and nothing else — which is also the
    sharpest test of ``MapLayout``: this is the identity morphism only because the
    row and column orderings are *derived* from ``block_order`` rather than
    invented, so mirrored legs give mirrored bands.
    """
    legs = tuple(legs)
    structure = TensorStructure(
        (*(replace(leg, side=OUT) for leg in legs), *(replace(leg, side=IN) for leg in legs))
    )
    layout = map_layout(structure)
    return from_matrices(
        structure,
        {c: ar.do("eye", layout.shape(c)[0], dtype=dtype, like="numpy") for c in layout.sectors},
    )
