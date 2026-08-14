"""Fixed-structure blockwise decompositions — ``svd`` and ``qr``, Milestone 7.

The lowering is the one README "Linear algebra" draws, and every arrow of it is
already built: :func:`~tenet.ops.repartition.repartition` puts the requested
``left`` into the codomain and ``right`` into the domain, :func:`to_matrices`
hands back one dense ``B_c`` per coupled sector, and the backend factorizes each
``B_c`` on its own. Clebsch-Gordan orthonormality has already collapsed the
coupled index into ``B_c`` (``ops.map`` module docstring), so a decomposition
needs no F/R symbols of its own and this module introduces **no new capability
protocol**: the refusals it can produce are ``transpose``'s and ``bend``'s,
inherited whole through ``repartition`` and raised before any block moves.

The only genuinely new object is the **bond space**: a fresh ``GradedSpace``
whose degeneracy at ``c`` is ``min(*layout.shape(c))``, read straight off
:class:`~tenet.map_view.MapLayout`. It is static metadata, so both functions are
shape-static and traceable.

Fixed-structure only (README "Structure-changing differentiation", invariant
10): this is the *compact* SVD/QR, with no truncation, no tolerance and no
zero-sector elimination. A sector whose ``B_c`` happens to be rank-deficient
keeps its full ``min`` bond degeneracy and carries zero singular values;
dropping them is structure-changing and belongs outside the jit boundary.

Conventions:

* The bond leg is non-dual on both sides and differs only in ``side`` — exactly
  ``identity``'s mirror convention. ``Leg.fused_sector`` is then the identity on
  it, so the one-leg tree on the bond side is the trivial ``c → c`` coupling and
  the coupled sectors of ``U``, ``S`` and ``Vh`` are literally
  ``layout.sectors``. It is also the only choice ``compose`` accepts, since
  ``_check_composable`` compares ``(space, dual, order)`` and ignores ``side``.
* ``S`` is a diagonal operator ``SymmetricTensor`` on the bond space, not a dict
  of vectors, so ``U @ S @ Vh`` is a plain :func:`tenet.compose` chain and no
  ``absorb`` enum is needed. The raw singular values are
  ``{c: ar.do("diagonal", m) for c, m in tenet.to_matrices(S).items()}``.
  ``S`` is real even when ``U`` and ``Vh`` are complex.
  ponytail: ``S`` stores a dense ``m_c × m_c`` block per sector where a vector
  would do; add a diagonal storage type when a bond dimension makes that memory
  actually hurt, i.e. never before truncation exists.
* Reconstruction is exact against ``repartition(t, left, right)``, not against
  ``t``: the factors' public axis order is ``(*left, bond)`` and ``(bond,
  *right)``, and no axis order could remember that ``t`` interleaved its sides.
  ``repartition``/``transpose`` take the user back, exactly.
* The gauge freedom (per-singular-value phases; the sign of ``R``'s diagonal) is
  never fixed here. Callers wanting a stabilized ``QR`` do it themselves.

No ``svd``/``qr`` registration in ``array/dispatch.py``: that list is closed and
``ar.do("svd", ...)`` must keep raising (invariant 11). No NumPy, no
``to_dense`` and no provider branching in this module.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import autoray as ar

from tenet.leg import IN, OUT, Leg
from tenet.map_view import from_matrices, map_layout, to_matrices
from tenet.ops.repartition import repartition
from tenet.space import GradedSpace
from tenet.structure import TensorStructure

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["qr", "svd"]

Axes = tuple[Sequence[int], Sequence[int]] | None


def _lower(t: "SymmetricTensor", axes: Axes) -> tuple["SymmetricTensor", GradedSpace, dict]:
    """``(repartitioned tensor, bond space, {c: B_c})``.

    Everything goes through ``repartition``, including the ``axes=None`` /
    ``as_map()`` path: when ``left``/``right`` already match the current sides it
    performs zero bends and one transpose, and when they match the current order
    too that transpose is the identity permutation. No fast path to maintain, and
    ``repartition`` owns the axis validation (original numbering, no negatives,
    no repeats, every axis exactly once across the two sides).
    """
    if axes is None:
        axes = (t.structure.out_axes, t.structure.in_axes)
    left, right = axes
    m = repartition(t, left, right)
    layout = map_layout(m.structure)
    # min(rows_c, cols_c) is metadata, never the numerical rank: taking the rank
    # would make the output structure depend on block values (invariant 10).
    bond = GradedSpace.new(m.provider, {c: min(layout.shape(c)) for c in layout.sectors})
    return m, bond, to_matrices(m)


def svd(
    t: "SymmetricTensor", axes: Axes = None
) -> tuple["SymmetricTensor", "SymmetricTensor", "SymmetricTensor"]:
    """``T = U ∘ S ∘ Vh``, exactly — the *compact* SVD. No truncation.

    ``axes=(left, right)`` names public axes in ``t``'s own numbering; ``left``
    becomes the codomain and ``right`` the domain. ``axes=None`` uses the current
    partition and is what ``t.as_map().svd()`` calls.

    Legs: ``U`` is ``(*left legs, bond IN)``, ``S`` is ``(bond OUT, bond IN)`` and
    ``Vh`` is ``(bond OUT, *right legs)``, so ``U @ S @ Vh`` equals
    ``repartition(t, left, right)`` block for block.
    """
    m, bond, mats = _lower(t, axes)
    parts = {c: ar.do("linalg.svd", b, full_matrices=False) for c, b in mats.items()}
    return (
        from_matrices(
            TensorStructure((*m.codomain, Leg(bond, IN))), {c: p[0] for c, p in parts.items()}
        ),
        from_matrices(
            TensorStructure((Leg(bond, OUT), Leg(bond, IN))),
            {c: ar.do("diag", p[1]) for c, p in parts.items()},
        ),
        from_matrices(
            TensorStructure((Leg(bond, OUT), *m.domain)), {c: p[2] for c, p in parts.items()}
        ),
    )


def qr(t: "SymmetricTensor", axes: Axes = None) -> tuple["SymmetricTensor", "SymmetricTensor"]:
    """``T = Q ∘ R``, the reduced/compact QR. Same skeleton and bond space as :func:`svd`.

    Legs: ``Q`` is ``(*left legs, bond IN)`` and ``R`` is ``(bond OUT, *right legs)``.
    The sign of ``R``'s diagonal is the backend's; no stabilization is applied.
    """
    m, bond, mats = _lower(t, axes)
    parts = {c: ar.do("linalg.qr", b) for c, b in mats.items()}
    return (
        from_matrices(
            TensorStructure((*m.codomain, Leg(bond, IN))), {c: p[0] for c, p in parts.items()}
        ),
        from_matrices(
            TensorStructure((Leg(bond, OUT), *m.domain)), {c: p[1] for c, p in parts.items()}
        ),
    )
