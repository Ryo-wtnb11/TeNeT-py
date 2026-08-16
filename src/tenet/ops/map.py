"""Morphism composition ``a ∘ b``, ``identity`` and the adjoint ``T†`` — Milestone 3.

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

:func:`adjoint` is the dagger, and it is one of four operations that are easy to
confuse and are deliberately kept apart (invariant 2, docs/design.md "Conjugation,
duality, and adjoint are distinct"):

======================  ==========  ==========  =======================  ==============
operation               ``side``    ``dual``    blocks                   coefficients
======================  ==========  ==========  =======================  ==============
``conj`` (#20)          unchanged   unchanged   conjugated               none
``leg.dualized`` (#6)   unchanged   flipped     Z-isomorphism (M4)       FS signs
``repartition`` (#32)   some flip   same flip   bent                     B-symbols
``adjoint`` (#31)       all flip    unchanged   conjugated, key-swapped  none
======================  ==========  ==========  =======================  ==============

``adjoint`` needs no bending coefficient precisely because it flips *every* side
at once: ``⊕_c B_c ⊗ id_c`` is simply re-read as ``⊕_c B_c† ⊗ id_c``, so trees,
coupled sectors and multiplicities all survive and only their row/column roles
trade places. And it needs no block transpose because reduced axes travel with
their own legs (invariant 7) and the public axis order is untouched: the whole
transpose is absorbed into the key swap ``(ot, it) → (it, ot)``.

No ``to_dense`` here and no provider branching. NumPy appears as
:func:`identity`'s default dtype and as :func:`random_isometry`'s draw — a
constructor runs at setup time, outside any trace, and ``to_backend`` is the
documented route onto a device (#9's convention, unchanged).
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar
import numpy as np

from tenet.leg import IN, OUT, Leg
from tenet.map_view import as_map, from_matrices, map_layout, to_matrices
from tenet.ops.embed import embed
from tenet.structure import FusionBlockKey, TensorStructure

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "AdjointPlan",
    "adjoint",
    "adjoint_plan",
    "compose",
    "identity",
    "isometry",
    "random_isometry",
]


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

    For a long chain at a *fixed* partition in eager NumPy, the matrix form can
    be kept between steps by hand (measured ~1.1x asymptotically; a persisted
    layout in the library was evaluated and rejected — zero cache-hit rate in
    real ``tensordot`` chains, and ``from_matrices`` is already zero-copy)::

        acc = to_matrices(ts[0])
        for t in ts[1:]:
            mb = to_matrices(t)
            acc = {c: acc[c] @ mb[c] for c in acc}
        out = from_matrices(TensorStructure((*ts[0].codomain, *ts[-1].domain)), acc)
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


def identity(
    legs: Sequence[Leg], *, dtype: Any = np.float64, like: Any = "numpy"
) -> "SymmetricTensor":
    """``id`` on ``ProductSpace(legs)``: the legs mirrored as ``(OUT..., IN...)``.

    ``space``, ``dual`` and ``name`` are kept and only ``side`` is set, so that
    ``identity(t.codomain) @ t == t``. Dualizing the mirror would build a cup, a
    different morphism (#32).

    ``B_c = eye`` for every coupled sector, and nothing else — which is also the
    sharpest test of ``MapLayout``: this is the identity morphism only because the
    row and column orderings are *derived* from ``block_order`` rather than
    invented, so mirrored legs give mirrored bands.

    ``like`` is anything ``ar.do`` accepts — a backend name or a reference array —
    and defaults to today's ``"numpy"``: ``identity(legs)`` has no tensor to infer
    a backend from, so a caller that *does* have one (``ops/contraction.py::trace``)
    passes it. Before #95 this was hard-coded, which made ``trace`` on a torch
    tensor contract torch blocks against NumPy ones.
    """
    legs = tuple(legs)
    structure = TensorStructure(
        (*(replace(leg, side=OUT) for leg in legs), *(replace(leg, side=IN) for leg in legs))
    )
    layout = map_layout(structure)
    return from_matrices(
        structure,
        {c: ar.do("eye", layout.shape(c)[0], dtype=dtype, like=like) for c in layout.sectors},
    )


# --- issue #89: isometry and random_isometry ------------------------------------------


def _mirrored(codomain: Sequence[Leg], domain: Sequence[Leg]) -> TensorStructure:
    """``(*codomain OUT, *domain IN)`` — ``identity``'s stance for two spaces.

    ``space``, ``dual`` and ``name`` are kept from the arguments and only ``side``
    is set.
    """
    return TensorStructure(
        (
            *(replace(leg, side=OUT) for leg in codomain),
            *(replace(leg, side=IN) for leg in domain),
        )
    )


def isometry(
    codomain: Sequence[Leg], domain: Sequence[Leg], *, dtype: Any = np.float64
) -> "SymmetricTensor":
    """The inclusion ``domain -> codomain``: ``W† W = id(domain)``, ``(W W†)² = W W†``.

    ``codomain[i]`` must contain ``domain[i]`` sector-wise — same provider, same
    ``dual``, every sector of the domain present in the codomain with a degeneracy
    at least as large. ``side`` is *set*, not compared, exactly as
    :func:`identity` does, and the result's legs are ``(*codomain OUT, *domain
    IN)``.

    The whole body, and every refusal, is :func:`~tenet.embed` of
    :func:`identity`: the blocks come from the identity morphism and the placement
    — each degeneracy slot into the *same* slot of the larger leg — is ``embed``'s
    prefix convention, the one ``svd(..., bond=)`` and ``restrict`` already share.
    A per-coupled-sector rectangular ``eye`` would also produce an isometry, and it
    was rejected for naming a *different* map: it sends the ``j``-th column band to
    the ``j``-th row band, an arbitrary correspondence whenever the two sides' band
    orders do not line up.

    Containment is required **per leg**, which is stricter than the fused,
    sector-wise ``domain ≾ codomain`` TensorKit imposes: a target where no single
    leg contains its partner but the fusion does is refused here.
    """
    # Simplification: per-leg containment. The fused case is a coupled-sector ``eye`` and
    # about four lines, and it lands when a caller can say *which* basis correspondence
    # they meant — making that choice silently, inside a function called ``isometry``, is
    # what gets found six months later as a gauge bug.
    domain = tuple(domain)
    return embed(identity(domain, dtype=dtype), _mirrored(codomain, domain).legs)


def random_isometry(
    codomain: Sequence[Leg],
    domain: Sequence[Leg],
    *,
    seed: int | None = None,
    dtype: Any = np.float64,
) -> "SymmetricTensor":
    """A Haar-random isometry: ``W† W = id(domain)``, independent per coupled sector.

    Per coupled sector ``c``, a ``(rows_c, cols_c)`` Gaussian draw, a QR, and the
    sign fix that makes the result Haar-distributed. Requires ``rows_c >= cols_c``
    in every coupled sector — the *fused* containment condition, read structurally
    off :class:`~tenet.map_view.MapLayout`, which is weaker than
    :func:`isometry`'s per-leg one; a sector that fails it is a ``ValueError``
    naming the sector and both dimensions, raised before any draw.

    **"Haar per coupled sector" is the product of per-sector Haar measures, not
    Haar on the dense space.** A symmetric isometry lives in a product of unitary
    groups, one per coupled sector, so the block-diagonal ensemble is the correct
    one — but a moment computed against a *dense* Haar formula will not match, and
    that is the ensemble, not a bug.

    NumPy draws through ``np.random.default_rng(seed)``, exactly as
    ``SymmetricTensor.random`` and ``identity``'s NumPy fill already do: a
    constructor is not a traced operation, and ``t.to_backend("jax")`` is the route
    onto a device. ``seed=None`` is non-reproducible.

    A complex ``dtype`` gets a genuinely complex (Ginibre) draw, deliberately
    departing from ``SymmetricTensor.random``'s "real draws cast to dtype"
    shortcut: a real orthogonal matrix cast to ``complex128`` *is* an isometry and
    would pass every other criterion here, while being Haar on ``O(n)`` rather than
    on ``U(n)``. Three lines, and it is a correctness trap otherwise.
    """
    structure = _mirrored(codomain, domain)
    layout = map_layout(structure)
    shapes = tuple((c, layout.shape(c)) for c in layout.sectors)
    wide = [(c, s) for c, s in shapes if s[0] < s[1]]
    if wide:
        c, (rows, cols) = wide[0]
        raise ValueError(
            f"random_isometry: coupled sector {c!r} has rows={rows} < cols={cols}, so no "
            f"isometry exists there — W† W = id(domain) needs every coupled sector to be "
            f"tall or square. This is the *fused* condition, read off the map layout, and "
            f"it is weaker than tenet.isometry's per-leg containment: a codomain leg may "
            f"be smaller than its domain partner and still pass here. All sector shapes: "
            + ", ".join(f"{s!r}: (rows={r}, cols={k})" for s, (r, k) in shapes)
        )

    rng = np.random.default_rng(seed)
    complex_draw = np.issubdtype(np.dtype(dtype), np.complexfloating)
    mats = {}
    for c, shape in shapes:
        a = rng.standard_normal(shape)
        if complex_draw:
            a = a + 1j * rng.standard_normal(shape)
        q, r = np.linalg.qr(a)
        # Mezzadri, Notices of the AMS 54, 592 (2007): plain QR of a Gaussian is
        # NOT Haar — LAPACK leaves R's diagonal with arbitrary signs (phases),
        # which biases Q's columns. Q * diag(d/|d|) is the correction, and it is
        # the same normalization TensorKit's positive-R `randisometry` applies.
        # Do not delete this line as redundant; a test measures the bias without it.
        d = np.diagonal(r)
        mats[c] = (q * (d / np.abs(d))).astype(dtype)
    return from_matrices(structure, mats)


@dataclass(frozen=True, slots=True)
class AdjointPlan:
    """The categorical half of a dagger: static, array-free, hashable.

    ``sources[i]`` is the *old* block index feeding new block ``i`` — a pure
    permutation of ``range(num_blocks)``, because swapping the two trees of a key
    is a bijection on ``block_order`` (though not an order-preserving one, which
    is why it is computed through ``index_of`` and never assumed).
    """

    new_structure: TensorStructure
    sources: tuple[int, ...]


@cache
def adjoint_plan(structure: TensorStructure) -> AdjointPlan:
    """The dagger plan for ``structure``. Cached: repeat calls return one object."""
    flipped = tuple(replace(leg, side=IN if leg.side is OUT else OUT) for leg in structure.legs)
    new_structure = TensorStructure(flipped)
    sources = tuple(
        structure.index_of(FusionBlockKey(key.input_tree, key.output_tree))
        for key in new_structure.block_order
    )
    return AdjointPlan(new_structure, sources)


def adjoint(t: "SymmetricTensor") -> "SymmetricTensor":
    """``T†``: the Euclidean-adjoint morphism in ``Hom(codomain, domain)``.

    Every leg keeps its ``space``, ``dual`` and ``name`` and flips its ``side``;
    the public axis order is unchanged; the block for key ``(ot, it)`` becomes the
    conjugate of the block for key ``(it, ot)`` — no axis permutation is needed,
    because reduced axes travel with their own legs (invariant 7).

    Deliberately no ``requires(...)``: this needs ``conj`` on the backend and the
    identity of the trees, nothing a provider could fail to supply.

    """
    # Simplification: a provider with complex Clebsch-Gordan coefficients would need the
    # same capability gate ``ops.basic.conj`` already flags — one line, once such a
    # provider exists. Not a second speculative protocol.
    from tenet.tensor import SymmetricTensor

    plan = adjoint_plan(t.structure)
    return SymmetricTensor(
        plan.new_structure, tuple(ar.do("conj", t.blocks[src]) for src in plan.sources)
    )
