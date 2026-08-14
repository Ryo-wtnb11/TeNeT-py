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
from tenet.symmetry.base import QuantumDimension, StructureChangingError, requires

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["qr", "svd", "svd_truncated"]

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


# --- truncation, Milestone 7 (#64) -------------------------------------------------
# Everything below is structure-changing: the bond space is decided by the singular
# values, so it is NOT traceable. Kept self-contained at the end of the module so the
# shape-static half above reads on its own.

_CUTOFF_MODES = ("abs", "rel", "sum2", "rsum2", "sum1", "rsum1")

_NOT_TRACEABLE = (
    "svd_truncated decides its output structure from the singular values -- the bond "
    "GradedSpace's degeneracies, and which sectors survive at all, depend on the block "
    "values -- so it cannot run inside a traced region (jit, grad, vmap). Either run it "
    "outside the traced region, or use tenet.linalg.svd, which is exact, shape-static "
    "and traceable."
)


def _spectrum(parts: dict) -> list[tuple[float, object, int]]:
    """``[(sigma, c, i), ...]`` descending by **bare** sigma, ties by ``(c, i)``.

    ``float(sigma)`` is the tracer check, and it is the honest one: the selection
    genuinely needs Python floats to sort, and asking the value for its value is the
    only test that is about the actual requirement (never importing JAX, no guess at a
    backend's tracer type). JAX raises ``ConcretizationTypeError``, a ``TypeError``;
    a backend that raises something else joins the ``except`` tuple when it appears.
    """
    try:
        entries = [(float(sigma), c, i) for c, p in parts.items() for i, sigma in enumerate(p[1])]
    except TypeError as exc:
        raise StructureChangingError(_NOT_TRACEABLE) from exc
    entries.sort(key=lambda e: (-e[0], e[1], e[2]))
    return entries


def _admissible(spectrum: list, qdim, cutoff: float | None, mode: str) -> int:
    """How long a prefix of ``spectrum`` the cutoff admits.

    The bare sigma is what ``"abs"``/``"rel"`` compare; only the cumulative modes
    carry the ``qdim`` weight, and the relative ones are dimensionally consistent --
    threshold and accumulator are the same power of sigma, which is what makes
    ``"rel"``/``"rsum1"``/``"rsum2"`` scale-invariant.
    """
    if cutoff is None:
        return len(spectrum)
    if mode == "abs":
        return sum(1 for sigma, _, _ in spectrum if sigma > cutoff)
    if mode == "rel":
        threshold = cutoff * spectrum[0][0]
        return sum(1 for sigma, _, _ in spectrum if sigma > threshold)
    power = 2 if mode in ("sum2", "rsum2") else 1
    weights = [qdim(c) * sigma**power for sigma, c, _ in spectrum]
    threshold = cutoff * sum(weights) if mode.startswith("r") else cutoff
    kept, dropped = len(spectrum), 0.0
    for w in reversed(weights):
        if dropped + w >= threshold:
            break
        dropped += w
        kept -= 1
    return kept


def _validate(max_bond, cutoff, cutoff_mode, renorm) -> None:
    if max_bond is None and cutoff is None:
        raise ValueError(
            "svd_truncated needs at least one of max_bond or cutoff; for the untruncated "
            "factorization call tenet.linalg.svd, which is exact and jittable"
        )
    if max_bond is not None and max_bond <= 0:
        raise ValueError(f"max_bond must be a positive dense bond dimension, got {max_bond!r}")
    if cutoff is not None and cutoff < 0:
        raise ValueError(f"cutoff must be non-negative, got {cutoff!r}")
    if not isinstance(cutoff_mode, str):
        raise ValueError(
            f"cutoff_mode must be one of the strings {_CUTOFF_MODES}, got {cutoff_mode!r}; "
            "quimb's integer codes 1-6 are not accepted"
        )
    if cutoff_mode not in _CUTOFF_MODES:
        raise ValueError(f"unknown cutoff_mode {cutoff_mode!r}; expected one of {_CUTOFF_MODES}")
    if not isinstance(renorm, bool):
        raise TypeError(
            f"renorm must be a bool, got {renorm!r}: it means 'preserve tenet.norm(T)', not "
            "quimb's p-norm power. A `renorm: int` p-norm generalization is the follow-up "
            "if a caller ever needs p=1"
        )


def svd_truncated(
    t: "SymmetricTensor",
    axes: Axes = None,
    *,
    max_bond: int | None = None,
    cutoff: float | None = None,
    cutoff_mode: str = "rsum2",
    renorm: bool = False,
) -> tuple["SymmetricTensor", "SymmetricTensor", "SymmetricTensor"]:
    """``U, S, Vh`` on a *truncated* bond space. **NOT jittable.**

    Same factor legs, same conventions and the same capability refusals as
    :func:`svd`; the only difference is the bond :class:`~tenet.space.GradedSpace`,
    whose degeneracy at ``c`` is the number of kept singular values there and which
    **omits ``c`` entirely** when that number is zero. That is README Milestone 7's
    "graded bond-space reconstruction", and it is why this is a sibling of
    :func:`svd` rather than a keyword on it: a keyword would make one function
    traceable or not depending on the value of an argument, which is exactly the
    distinction README says the library must never hide. Under ``jax.jit`` or
    ``jax.grad`` it raises
    :class:`~tenet.symmetry.base.StructureChangingError`.

    Selection is over **one global spectrum**, always, in every mode:

    * the sort key is the **bare** ``sigma`` (descending; ties by sector order then
      index) -- "how large is this singular value" has nothing to do with
      multiplicity;
    * the **cost** and the **weight** are ``qdim(c)``-weighted, because the reduced
      index ``i`` in sector ``c`` stands for ``qdim(c)`` dense basis states. It is
      the same weight :func:`tenet.norm` carries. Greedy-descending under a dense
      budget is then optimal rather than a heuristic, so the result is the best
      approximation of its achieved dense rank (Eckart-Young, sector-blind).

    ``max_bond`` bounds the **dense** bond dimension ``Sum_c qdim(c)*m_c``, not the
    reduced ``Sum_c m_c``. For U(1) and fermionic parity these coincide; for SU(2) they do not, and
    that will surprise people. The walk **stops** at the first singular value that
    would overflow the budget rather than scanning on for a cheaper one that still
    fits, which is what keeps the kept set nested as ``max_bond`` grows; the
    documented consequence is that ``max_bond`` may be undershot by up to
    ``max qdim(c) - 1``.

    ``cutoff_mode`` (quimb's names and quimb's semantics; the integer codes 1-6 quimb
    also accepts are listed only so an M8 shim is a lookup -- **only the strings are
    accepted here**):

    ==== ========= ===========================================================
    code mode      keeps
    ==== ========= ===========================================================
    1    ``abs``   ``sigma > cutoff``
    2    ``rel``   ``sigma > cutoff * sigma_max`` (the bare global max)
    3    ``sum2``  drops the largest set with ``Sum qdim(c) sigma^2 < cutoff``
    4    ``rsum2`` as ``sum2``, threshold ``cutoff * tenet.norm(T)**2``
    5    ``sum1``  as ``sum2`` at power 1, weight ``qdim(c) sigma``
    6    ``rsum1`` as ``rsum2`` at power 1
    ==== ========= ===========================================================

    ``max_bond`` and ``cutoff`` together take the intersection. ``None`` means "no
    truncation" -- there are no ``-1`` sentinels, and passing neither is refused,
    naming :func:`svd`. ``renorm=True`` scales the kept singular values by
    ``sqrt(norm(T)**2 / Sum_kept qdim(c) sigma^2)`` so that
    ``tenet.norm(U @ S @ Vh) == tenet.norm(t)``; it is a bool, not quimb's p-norm
    power.

    No ``absorb`` enum and no fourth return value: ``S`` is a tensor, so absorbing is
    a one-line ``compose``, and the truncation error is exactly
    ``tenet.norm(t)**2 - tenet.norm(U @ S @ Vh)**2`` by Pythagoras.

    Raises ``ValueError`` if every singular value is dropped -- a bond space with no
    sectors is never returned.
    """
    _validate(max_bond, cutoff, cutoff_mode, renorm)
    m, _, mats = _lower(t, axes)
    provider = m.provider
    requires(provider, QuantumDimension)
    qdim = provider.qdim

    parts = {c: ar.do("linalg.svd", b, full_matrices=False) for c, b in mats.items()}
    spectrum = _spectrum(parts)
    if not spectrum:
        raise ValueError("svd_truncated: this tensor has no singular values at all")

    admissible = _admissible(spectrum, qdim, cutoff, cutoff_mode)
    keep_count: dict = {}
    kept_weight, budget = 0.0, 0.0
    for sigma, c, _ in spectrum[:admissible]:
        if max_bond is not None:
            budget += qdim(c)
            if budget > max_bond:
                break  # stop; never scan on for a cheaper sector that still fits
        keep_count[c] = keep_count.get(c, 0) + 1
        kept_weight += qdim(c) * sigma**2
    if not keep_count:
        raise ValueError(
            f"svd_truncated: cutoff={cutoff!r} in cutoff_mode={cutoff_mode!r} with "
            f"max_bond={max_bond!r} keeps no singular value at all (the largest available "
            f"is {spectrum[0][0]!r}); a bond space with no sectors is not a tensor"
        )

    bond = GradedSpace.new(provider, keep_count)
    scale = 1.0
    if renorm:
        total = sum(qdim(c) * sigma**2 for sigma, c, _ in spectrum)
        scale = (total / kept_weight) ** 0.5
    # sigma_c is descending within each sector, so the kept indices are a prefix.
    return (
        from_matrices(
            TensorStructure((*m.codomain, Leg(bond, IN))),
            {c: parts[c][0][:, :k] for c, k in keep_count.items()},
        ),
        from_matrices(
            TensorStructure((Leg(bond, OUT), Leg(bond, IN))),
            {c: ar.do("diag", parts[c][1][:k] * scale) for c, k in keep_count.items()},
        ),
        from_matrices(
            TensorStructure((Leg(bond, OUT), *m.domain)),
            {c: parts[c][2][:k, :] for c, k in keep_count.items()},
        ),
    )
