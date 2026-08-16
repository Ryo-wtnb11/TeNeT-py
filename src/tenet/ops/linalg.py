"""Fixed-structure blockwise decompositions — ``svd``, ``qr``, ``eigh``, ``polar``, ``lq``.

Milestone 7. Every function here is the same lowering with a different backend
call and a different set of output legs; ``_lower`` is shared verbatim.

The lowering is the one docs/design.md "Linear algebra" draws, and every arrow of it is
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

Fixed-structure only (docs/design.md "Structure-changing differentiation", invariant
10): this is the *compact* SVD/QR, with no truncation, no tolerance and no
zero-sector elimination. A sector whose ``B_c`` happens to be rank-deficient
keeps its full ``min`` bond degeneracy and carries zero singular values;
dropping them is structure-changing and belongs outside the jit boundary.
``svd(..., bond=B)`` is not an exception: ``B`` is a ``GradedSpace`` the caller
decided *outside* the traced region, so it is static metadata like every other
structure here, and the projection onto it is a per-sector prefix slice.

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
  Simplification: ``S`` stores a dense ``m_c × m_c`` block per sector where a vector
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
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import autoray as ar

from tenet.leg import IN, OUT, Leg
from tenet.map_view import check_square, from_matrices, map_layout, to_matrices
from tenet.ops.repartition import repartition
from tenet.space import GradedSpace
from tenet.structure import TensorStructure
from tenet.symmetry.base import QuantumDimension, StructureChangingError, requires

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "eig",
    "eigh",
    "eigvals",
    "expm",
    "left_null",
    "lq",
    "polar",
    "qr",
    "right_null",
    "svd",
    "svd_truncated",
]

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


def _keep_counts(bond: GradedSpace | None, untruncated: GradedSpace) -> dict:
    """``{c: k_c}``, the per-sector prefix length ``svd`` keeps.

    Pure metadata: a ``GradedSpace`` against a ``GradedSpace``, no block value
    anywhere, so the refusal is total and raises identically inside and outside a
    trace. ``bond=None`` keeps everything, which is the identity slice.
    """
    if bond is None:
        return dict(untruncated.sectors)
    for c, k in bond.sectors:
        available = untruncated.degeneracy(c)
        if k > available:
            raise ValueError(
                f"svd: bond asks for {k} singular values in sector {c!r}, but this "
                f"tensor's untruncated bond degeneracy there is {available}"
                + (
                    " (the sector does not appear in the map layout at all)"
                    if available == 0
                    else " = min(rows_c, cols_c)"
                )
                + ". bond= selects a prefix of an existing spectrum, so it must be a "
                "subspace of the untruncated bond; a missing or short sector is never "
                "padded with zeros, which would declare a bond the tensor does not have "
                "and divide by zero in the gradient. Take bond from "
                "tenet.linalg.svd_truncated on this tensor and this partition."
            )
    return dict(bond.sectors)


def svd(
    t: "SymmetricTensor", axes: Axes = None, *, bond: GradedSpace | None = None
) -> tuple["SymmetricTensor", "SymmetricTensor", "SymmetricTensor"]:
    """``T = U ∘ S ∘ Vh`` (``bond=None``) or ``T ≈ U ∘ S ∘ Vh`` on a **pre-decided** bond.

    ``axes=(left, right)`` names public axes in ``t``'s own numbering; ``left``
    becomes the codomain and ``right`` the domain. ``axes=None`` uses the current
    partition and is what ``t.as_map().svd()`` calls.

    ``bond=None`` is the *compact* SVD: exact, on the ``min(rows_c, cols_c)`` bond,
    no truncation. **``bond=B`` is the same factorization projected onto ``B``, and
    then the reconstruction is no longer exact** — ``U @ S @ Vh`` is the best
    approximation of ``repartition(t, left, right)`` at those per-sector ranks
    (Eckart-Young), not equal to it. In each sector ``c`` of ``B`` the largest
    ``B.degeneracy(c)`` singular values are kept — a prefix, since ``sigma_c`` comes
    back descending — and every sector absent from ``B`` is dropped. The truncation
    error stays one line by Pythagoras: ``norm(t)**2 - norm(U @ S @ Vh)**2``.

    ``B`` must be a **subspace** of the untruncated bond: every sector of ``B`` is a
    sector of the ``min`` bond, with no larger degeneracy. Refused with a
    ``ValueError`` naming the sector and both degeneracies, structurally and before a
    single block is read, so the refusal is as traceable as the rest of the function.
    Nothing is ever silently zero-padded.

    ``bond=`` does *not* make this function structure-changing, which is why it is a
    keyword here while truncation is a separate :func:`svd_truncated`: a
    :class:`~tenet.space.GradedSpace` is frozen, hashable, array-free metadata that
    the **caller** decided, so ``svd(t, bond=B)`` is exactly as shape-static, jittable
    and differentiable as ``svd(t)``. What #64 refused to make a keyword is the
    *decision*; what is a keyword here is the decision's *result*. The pairing::

        _, s, _ = tenet.linalg.svd_truncated(t0, axes, max_bond=D)   # outside jit/grad
        bond = s.structure.legs[0].space

        @jax.jit
        def step(t):
            u, s, vh = tenet.linalg.svd(t, axes, bond=bond)          # inside, fixed shape

    **The gradient under ``bond=`` is the exact truncated backward, and it needed no
    code.** Reverse mode differentiates the per-sector prefix slice generically,
    zero-padding the cotangent, and the *matrix* SVD underneath is the compact one
    from :mod:`tenet.ad`. That composition is not the usual approximation: because
    the bond degeneracy is ``min(rows_c, cols_c)`` and never the numerical rank, the
    discarded space never leaves the factorization, and the cross block of
    :mod:`tenet.ad`'s broadened ``F`` — rows ``i > k`` against columns ``j <= k``,
    weighted by ``1/(sigma_j - sigma_i)`` — is exactly the correction
    Francuz-Schuch-Vanhecke add in Eqs. (14)-(15) of Phys. Rev. Research 7, 013237
    (2025). Measured against central differences it is flat at finite-difference
    noise across ``sigma_perp/sigma_min`` from ``0.1`` to ``0.9``, while the
    zeroth-order rule (the same VJP formed against the *kept* factors only, which is
    what the CTMRG/iPEPS literature runs on) is off by 11% at a ratio of ``0.1``, i.e.
    ``O(sigma_perp/sigma_min)``. Both numbers are pinned in
    ``tests/backends/test_ad.py``. Degeneracy is the one remaining caveat, and it is
    :mod:`tenet.ad`'s: a multiplet *straddling* the cut makes the kept subspace
    gauge-dependent, so the gradient there is finite but meaningless.

    Legs, in both modes: ``U`` is ``(*left legs, bond IN)``, ``S`` is
    ``(bond OUT, bond IN)`` and ``Vh`` is ``(bond OUT, *right legs)``.
    """
    m, untruncated, mats = _lower(t, axes)
    keep = _keep_counts(bond, untruncated)  # structural; before any block is factorized
    space = untruncated if bond is None else bond
    parts = {c: ar.do("linalg.svd", mats[c], full_matrices=False) for c in keep}
    # sigma_c is descending, so the kept indices are a prefix -- and at full rank the
    # slice is the identity, which is why bond=<the full min bond> is not a code path.
    return (
        from_matrices(
            TensorStructure((*m.codomain, Leg(space, IN))),
            {c: parts[c][0][:, :k] for c, k in keep.items()},
        ),
        from_matrices(
            TensorStructure((Leg(space, OUT), Leg(space, IN))),
            {c: ar.do("diag", parts[c][1][:k]) for c, k in keep.items()},
        ),
        from_matrices(
            TensorStructure((Leg(space, OUT), *m.domain)),
            {c: parts[c][2][:k, :] for c, k in keep.items()},
        ),
    )


def qr(t: "SymmetricTensor", axes: Axes = None) -> tuple["SymmetricTensor", "SymmetricTensor"]:
    """``T = Q ∘ R``, the reduced/compact QR. Same skeleton and bond space as :func:`svd`.

    Legs: ``Q`` is ``(*left legs, bond IN)`` and ``R`` is ``(bond OUT, *right legs)``.
    The sign of ``R``'s diagonal is the backend's; no stabilization is applied.

    Differentiability: the gradient is JAX's own, and it is the standard rule
    (Liao-Liu-Wang-Xiang Eq. (5) for the square/tall case; Roberts-Roberts
    Eqs. (9)-(10) for the wide one, which is what JAX >= 0.10 implements). It is
    finite and correct for any sector whose ``R`` is nonsingular. A sector matrix
    that is **exactly** rank-deficient -- an exact zero on ``R``'s diagonal --
    gives ``NaN``, and that is not stabilized here: unlike ``svd``'s degeneracy,
    the QR of a rank-deficient matrix is itself non-unique, so there is no correct
    value to broaden towards. See :mod:`tenet.ad`.
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


# --- issue #63: eigh, polar and lq ------------------------------------------------


def _dagger(mat):
    """``B_c†`` for one dense coupled-sector matrix. Two backend calls, no capability."""
    return ar.do("conj", ar.do("transpose", mat))


def eigh(t: "SymmetricTensor", axes: Axes = None) -> tuple["SymmetricTensor", "SymmetricTensor"]:
    """``T = V ∘ W ∘ V†`` for a self-adjoint ``T``. Returns ``(W, V)``.

    Legs: ``W`` is ``(bond OUT, bond IN)``, diagonal and real; ``V`` is
    ``(*left legs, bond IN)``. The map must be square *space-wise* — see
    :func:`~tenet.map_view.check_square` — and for a square map ``_lower``'s ``min(rows, cols)``
    is a no-op, so the bond space is literally the fused domain.

    **Hermiticity of the numbers is the caller's responsibility and is deliberately
    not checked.** A numerical check needs a tolerance, and a tolerance comparison
    is a data-dependent branch, which cannot run inside a traced region (invariant
    9); ``eigh`` is fixed-structure and must stay jittable. A non-Hermitian input
    is *not* refused: the backend reads one triangle and you get whatever that
    gives. The user-side check needs no new API::

        max(norm(B - B.conj().T) for B in tenet.to_matrices(tenet.repartition(T, l, r)).values())

    run once, outside the hot loop.

    **Eigenvalues come back in the backend's order — ascending within each coupled
    sector** (LAPACK's), in deliberate contrast to :func:`svd`'s descending ``S``.
    Re-sorting would be a cosmetic permutation of ``W`` and of ``V``'s columns, and
    would still buy nothing across sectors, where no global order exists either way.
    ``W`` is real even for complex input.
    """
    m, bond, mats = _lower(t, axes)
    check_square(m, "eigh")
    parts = {c: ar.do("linalg.eigh", b) for c, b in mats.items()}
    return (
        from_matrices(
            TensorStructure((Leg(bond, OUT), Leg(bond, IN))),
            {c: ar.do("diag", p[0]) for c, p in parts.items()},
        ),
        from_matrices(
            TensorStructure((*m.codomain, Leg(bond, IN))), {c: p[1] for c, p in parts.items()}
        ),
    )


def polar(
    t: "SymmetricTensor", axes: Axes = None, side: str = "left"
) -> tuple["SymmetricTensor", "SymmetricTensor"]:
    """``T = W ∘ P`` (``side="left"``) or ``T = P ∘ W`` (``side="right"``).

    Always returns ``(W, P)`` — the isometry first, whichever side it sits on. The
    name says *which side the isometry sits on*, TensorKit's convention; a tuple
    whose order depended on a keyword would be a footgun.

    ``W`` carries exactly ``repartition(t, left, right)``'s structure and ``P`` is
    an endomorphism of the domain (``side="left"``) or of the codomain
    (``side="right"``), its legs mirrored from that side. No bond leg survives, so
    ``polar`` is the one M7 decomposition insensitive to the ``min``-rank bond
    convention — and the reason it is the gauge-fixing primitive for M8.

    ``W`` is only a *partial* isometry when some ``B_c`` is rank-deficient: then
    ``W†W`` is an orthogonal projector rather than the identity and ``P`` is
    singular. Same fact as ``svd``'s zero-rank sectors — structure is metadata,
    rank is data.

    Simplification: ``PolarViaSVD``, MatrixAlgebraKit's own default, because ``autoray``
    has no uniform ``linalg.polar`` and the alternative is a Newton-Schulz
    iteration, i.e. a convergence tolerance and a second numerical framework. Swap
    a backend ``polar`` into the sector loop below if one ever becomes uniform.
    """
    if side not in ("left", "right"):
        raise ValueError(f"polar: side must be 'left' or 'right', got {side!r}")
    m, _, mats = _lower(t, axes)

    isometry, positive = {}, {}
    for c, b in mats.items():
        u, s, vh = ar.do("linalg.svd", b, full_matrices=False)
        isometry[c] = ar.do("matmul", u, vh)
        d = ar.do("diag", s)
        positive[c] = (
            ar.do("matmul", _dagger(vh), ar.do("matmul", d, vh))
            if side == "left"
            else ar.do("matmul", u, ar.do("matmul", d, _dagger(u)))
        )

    mirrored = m.domain if side == "left" else m.codomain
    return (
        from_matrices(m.structure, isometry),
        from_matrices(
            TensorStructure(
                (
                    *(replace(leg, side=OUT) for leg in mirrored),
                    *(replace(leg, side=IN) for leg in mirrored),
                )
            ),
            positive,
        ),
    )


def lq(t: "SymmetricTensor", axes: Axes = None) -> tuple["SymmetricTensor", "SymmetricTensor"]:
    """``T = L ∘ Q``, the reduced LQ. Same bond space as :func:`qr`, by construction.

    Legs: ``L`` is ``(*left legs, bond IN)`` and ``Q`` is ``(bond OUT, *right legs)``.
    Per sector ``B† = q r`` gives ``B = r† q†``: two dense conjugate-transposes in
    the loop, no new backend primitive and no capability at all. The categorical
    spelling ``adjoint → qr → adjoint`` would give factors whose public axis order
    is ``(bond, *left)``, and straightening that needs ``tenet.transpose`` — i.e.
    ``PermutationCoefficients`` and the fermionic Koszul signs — for the same numbers.

    No stabilization: the sign of ``L``'s diagonal is the backend's, as in :func:`qr`.

    Differentiability: the gradient is JAX's own, inherited through ``qr(B†)`` --
    so a ``rows > cols`` sector, the ordinary case here, hands JAX a *wide* matrix
    and needs the wide-QR JVP of JAX >= 0.10 (Roberts-Roberts Eqs. (9)-(10);
    Liao-Liu-Wang-Xiang Eq. (5) for the other side). It is finite and correct for
    any sector whose ``L`` is nonsingular. A sector matrix that is **exactly**
    rank-deficient -- an exact zero on ``L``'s diagonal -- gives ``NaN``, and that
    is not stabilized here: unlike ``svd``'s degeneracy, the LQ of a rank-deficient
    matrix is itself non-unique, so there is no correct value to broaden towards.
    See :mod:`tenet.ad`.
    """
    m, bond, mats = _lower(t, axes)
    parts = {c: ar.do("linalg.qr", _dagger(b)) for c, b in mats.items()}
    return (
        from_matrices(
            TensorStructure((*m.codomain, Leg(bond, IN))),
            {c: _dagger(p[1]) for c, p in parts.items()},
        ),
        from_matrices(
            TensorStructure((Leg(bond, OUT), *m.domain)),
            {c: _dagger(p[0]) for c, p in parts.items()},
        ),
    )


# --- issue #88: left_null and right_null ----------------------------------------------


def _complement(layout, caller: str) -> dict:
    """``{c: extra_c}``, the structural complement's per-sector degeneracy.

    ``rows_c - cols_c`` for ``left_null`` and ``cols_c - rows_c`` for
    ``right_null``, kept only where positive. Metadata against metadata — a
    :class:`~tenet.map_view.MapLayout` and nothing else — so the refusal is total
    and raises identically inside and outside a trace, before any block is
    factorized. Sectors with no complement are simply absent: ``GradedSpace.new``
    refuses ``m <= 0``, so a zero-degeneracy sector cannot be carried anyway.
    """
    other = "right_null" if caller == "left_null" else "left_null"
    per_sector = tuple((c, layout.shape(c)) for c in layout.sectors)
    keep = {}
    for c, (rows, cols) in per_sector:
        extra = rows - cols if caller == "left_null" else cols - rows
        if extra > 0:
            keep[c] = extra
    if not keep:
        shapes = ", ".join(f"{c!r}: (rows={r}, cols={k})" for c, (r, k) in per_sector)
        raise ValueError(
            f"{caller}: no coupled sector has "
            + ("rows_c > cols_c" if caller == "left_null" else "cols_c > rows_c")
            + f" for this partition, so the orthogonal complement is empty and a space "
            f"with no sectors is not a tensor. Per-sector shapes: {shapes}. "
            f"The complement on the other side is tenet.linalg.{other}."
        )
    return keep


def left_null(t: "SymmetricTensor", axes: Axes = None) -> "SymmetricTensor":
    """The isometry onto the orthogonal complement of ``T``'s image: ``N† T = 0``.

    Legs: ``N`` is ``(*left legs, bond IN)`` — :func:`qr`'s ``Q`` legs exactly — so
    that ``adjoint(N) @ N == identity(bond)`` and ``adjoint(N) @ repartition(t,
    left, right)`` is zero. "Left" and "null" compose into the wrong intuition
    about half the time, so the identity rather than the word is the contract:
    this is the *cokernel*, the null space of ``T†``.

    The bond space is **structural**: degeneracy ``rows_c - cols_c`` in every
    coupled sector where ``rows_c > cols_c``, and the sector is **omitted** where
    it is not. It is read off :class:`~tenet.map_view.MapLayout` — metadata
    against metadata, no block value anywhere — so this is shape-static, jittable
    and differentiable, the same argument ``svd(..., bond=)`` and ``embed`` make.

    **This is the shape null space, not the numerical one.** A rank-deficient
    ``B_c`` has a *larger* true null space; what is returned is a subspace of it —
    always orthogonal to ``T``, never complete. Determining the numerical rank
    would make the output structure depend on block values, which is ``_lower``'s
    standing refusal ("min(rows_c, cols_c) is metadata, never the numerical rank")
    applied to the complement.
    Simplification: shape-null only. A numerical sibling taking a rank tolerance arrives
    as a **separate, non-traceable** function raising ``StructureChangingError``
    under a trace — ``svd``/``svd_truncated``'s split, reused — when a caller
    turns up with an opinion about the discarded directions.

    Complete QR, not a full SVD, and the reason is measured: JAX refuses to
    differentiate a full SVD (``_svd_jvp_rule``'s "not implemented for full
    matrices") for exactly the non-square shapes that have a null space, while the
    complete QR differentiates since JAX 0.10 ("and when ``full_matrices`` is
    ``True``"), which is the floor this library already declares.

    Raises ``ValueError`` if no coupled sector has ``rows_c > cols_c``: the
    complement is empty, and the message names every sector's ``(rows, cols)``.
    """
    m, _, mats = _lower(t, axes)
    layout = map_layout(m.structure)
    keep = _complement(layout, "left_null")  # structural; before any block is factorized
    bond = GradedSpace.new(m.provider, keep)
    parts = {c: ar.do("linalg.qr", mats[c], mode="complete") for c in keep}
    return from_matrices(
        TensorStructure((*m.codomain, Leg(bond, IN))),
        {c: parts[c][0][:, layout.shape(c)[1] :] for c in keep},
    )


def right_null(t: "SymmetricTensor", axes: Axes = None) -> "SymmetricTensor":
    """The mirror of :func:`left_null` on the domain side: ``T N† = 0``, ``N N† = id``.

    Legs: ``N`` is ``(bond OUT, *right legs)`` — :func:`lq`'s ``Q`` legs — and the
    bond degeneracy is ``cols_c - rows_c`` where positive. This is the *kernel*,
    where :func:`left_null` is the cokernel.

    Per sector ``B† = q r`` complete and ``N = (q[:, rows_c:])†``: the same two
    dense conjugate-transposes :func:`lq` already uses, no new backend primitive
    and no second implementation of the factorization. Every paragraph of
    :func:`left_null` — the structural bond, the shape-versus-numerical stance,
    the complete QR and the refusal — applies unchanged.
    """
    m, _, mats = _lower(t, axes)
    layout = map_layout(m.structure)
    keep = _complement(layout, "right_null")  # structural; before any block is factorized
    bond = GradedSpace.new(m.provider, keep)
    parts = {c: ar.do("linalg.qr", _dagger(mats[c]), mode="complete") for c in keep}
    return from_matrices(
        TensorStructure((Leg(bond, OUT), *m.domain)),
        {c: _dagger(parts[c][0][:, layout.shape(c)[0] :]) for c in keep},
    )


# --- issue #86: expm ----------------------------------------------------------------

_NO_SCIPY = (
    "tenet.linalg.expm on the NumPy backend needs SciPy, which is not a dependency of "
    "this library: autoray resolves linalg.expm to scipy.linalg.expm there, and NumPy "
    "itself ships no matrix exponential. Install it with `pip install scipy` (or "
    "`uv add scipy`), or move the tensor to the JAX backend, whose linalg.expm is "
    "jax.scipy.linalg.expm and needs nothing extra. SciPy is deliberately not declared: "
    "it is a large compiled, platform-specific wheel, which is exactly what the "
    "dependency rule in REPOSITORY_RULES.md admitted opt-einsum by *not* being."
)


def expm(t: "SymmetricTensor", axes: Axes = None, *, alpha: Any = 1.0) -> "SymmetricTensor":
    """``exp(alpha * T)`` for a square map, one dense exponential per coupled sector.

    ``axes=(left, right)`` names public axes in ``t``'s own numbering, as everywhere
    in this module; ``axes=None`` uses the current partition and is what
    ``t.as_map().expm()`` calls. The map must be square *space-wise* — the same
    ``(space, dual)``-in-order requirement, and the same refusal, :func:`eigh` has —
    because ``exp`` of a morphism is only defined for an endomorphism.

    The result carries ``repartition(t, left, right)``'s structure exactly: no bond
    leg is created and no leg changes, so ``expm`` is the second decomposition (with
    :func:`polar`'s ``W``) that is insensitive to the ``min``-rank bond convention.

    ``alpha`` is a scalar multiplying ``T`` before exponentiation, and it is where
    ``-i dt`` goes: ``expm(h, alpha=-1j * dt)`` is the Trotter gate. A complex
    ``alpha`` promotes real blocks to complex by the backend's own rule, one coupled
    sector at a time, which is why it is a multiplier here rather than the caller's
    ``expm(alpha * t)``.

    **Hermiticity is neither required nor assumed.** Unlike an ``eigh``-based
    exponential — ``V diag(exp(alpha w)) V†``, which reads one triangle and returns a
    plausible wrong answer off the Hermitian locus — this is correct for any square
    map, and its gradient carries no ``1/(w_i - w_j)``, so a degenerate spectrum is
    finite under stock JAX and :mod:`tenet.ad` is not needed here at all.

    **The ceiling is JAX's and it is silent.** ``jax.scipy.linalg.expm`` is
    scaling-and-squaring with Padé under a ``max_squarings=16`` limit enforced by
    ``lax.cond(n_squarings > max_squarings, _nan, _compute, ...)``
    (``jax/_src/scipy/linalg.py``): a block whose ``‖alpha·B_c‖`` needs more
    squarings comes back as ``NaN`` rather than raising. No guard is added here — a
    norm comparison is a data-dependent branch and could not run inside a trace
    (invariant 9) — and the caller's escape is the one physics already uses,
    ``exp(alpha H) = exp(alpha H / n)**n``.
    Simplification: no norm guard; add one only outside the traced region, in the caller.

    On the NumPy backend the exponential is SciPy's, and SciPy is *not* a dependency
    of this library; without it the call raises an ``ImportError`` naming
    ``pip install scipy``.
    """
    m, _, mats = _lower(t, axes)
    check_square(m, "expm")
    try:
        blocks = {c: ar.do("linalg.expm", alpha * b) for c, b in mats.items()}
    except ImportError as exc:  # loud and actionable, as tenet.ad's JAX import is
        raise ImportError(_NO_SCIPY) from exc
    return from_matrices(m.structure, blocks)


# --- issue #87: eig and eigvals ------------------------------------------------------
# Siblings of eigh, sharing its skeleton, its return order and its refusal. Two names
# rather than one keyword because they have *different contracts*: eigvals is
# differentiable and eig is not, and a keyword flipping that would hide exactly the
# distinction svd/svd_truncated exists to show.


def eig(t: "SymmetricTensor", axes: Axes = None) -> tuple["SymmetricTensor", "SymmetricTensor"]:
    """``T V = V W`` for a square, not necessarily Hermitian ``T``. Returns ``(W, V)``.

    Legs, and the return order, are :func:`eigh`'s exactly: ``W`` is
    ``(bond OUT, bond IN)`` and diagonal, ``V`` is ``(*left legs, bond IN)``, and for
    a square map ``_lower``'s ``min(rows, cols)`` is a no-op so the bond space is the
    fused domain. Same ``(space, dual)``-in-order square-map refusal.

    **Both outputs are complex, always, even for a real input** — a real matrix has
    complex eigenvalues in conjugate pairs, so there is no real answer to return.
    ``W`` is complex where :func:`eigh`'s is real. Without ``x64`` under JAX it is
    ``complex64``, the backend's own dtype policy, as ``to_backend`` documents.

    **``V`` is not an isometry.** Right eigenvectors of a non-normal matrix are not
    orthogonal: ``adjoint(V) @ V`` is not the identity, and the reconstruction is
    ``V W V^-1``, which this library cannot spell because it has no ``inv``. The
    checkable statement, and the one the tests use, is the residual ``T @ V - V @ W``.

    Eigenvalues come back in the backend's order, **unsorted**. :func:`eigh`'s
    argument, only stronger: complex numbers have no order at all, so "sorted" would
    have to mean "by ``|λ|`` descending", a choice — and the caller is the one who
    knows whether "dominant" means largest modulus, largest real part, or largest
    within a chosen sector. It is one ``max`` over one comprehension out there.

    **Not differentiable under JAX** — ``jax.grad`` raises JAX's own
    ``NotImplementedError`` naming ``enable_eigvec_derivs``, and that error is
    propagated unchanged, neither caught nor re-phrased and above all not opted into:
    the flag turns on an eigenvector derivative under assumptions on the input that
    JAX cannot check, and a library may not make an unverifiable numerical assumption
    on every caller's behalf. Apply it to your own blocks if you want it. Use
    :func:`eigvals` when the objective needs only the spectrum; it is differentiable.

    Platform, as measured rather than as upstream's stale docstring has it: CPU
    **and** NVIDIA GPU (cuSolver by default since JAX 0.8.0); TPU has no lowering.
    """
    m, bond, mats = _lower(t, axes)
    check_square(m, "eig")
    parts = {c: ar.do("linalg.eig", b) for c, b in mats.items()}
    return (
        from_matrices(
            TensorStructure((Leg(bond, OUT), Leg(bond, IN))),
            {c: ar.do("diag", p[0]) for c, p in parts.items()},
        ),
        from_matrices(
            TensorStructure((*m.codomain, Leg(bond, IN))), {c: p[1] for c, p in parts.items()}
        ),
    )


def eigvals(t: "SymmetricTensor", axes: Axes = None) -> "SymmetricTensor":
    """The eigenvalues of a square map, as a diagonal ``(bond OUT, bond IN)`` tensor.

    **Differentiable**, unlike :func:`eig`: JAX implements the eigenvalues-only JVP
    unconditionally, and it needs no assumption a library cannot check. Jittable on
    CPU and on NVIDIA GPU — the "CPU backend" line in ``jax.numpy``'s own ``eigvals``
    docstring is stale upstream text, not behaviour. Complex always, order is the
    backend's, exactly as :func:`eig`'s.

    Equal to ``eig(t)[0]`` — the same LAPACK/cuSolver driver with the eigenvector job
    switched off — and it is the values-only call that carries the gradient.
    """
    m, bond, mats = _lower(t, axes)
    check_square(m, "eigvals")
    return from_matrices(
        TensorStructure((Leg(bond, OUT), Leg(bond, IN))),
        {c: ar.do("diag", ar.do("linalg.eigvals", b)) for c, b in mats.items()},
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
    **omits ``c`` entirely** when that number is zero. That is docs/design.md Milestone 7's
    "graded bond-space reconstruction", and it is why this is a sibling of
    :func:`svd` rather than a keyword on it: a keyword would make one function
    traceable or not depending on the value of an argument, which is exactly the
    distinction docs/design.md says the library must never hide. Under ``jax.jit`` or
    ``jax.grad`` it raises
    :class:`~tenet.symmetry.base.StructureChangingError`.

    The bond space returned here is the input to ``svd(t, axes, bond=...)``, which is
    the traceable half of the pairing: decide the structure once, out here, then
    project onto it inside ``jit``/``grad``. ``bond=`` is a keyword on :func:`svd`
    because it carries the *result* of the decision, not the decision.

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
