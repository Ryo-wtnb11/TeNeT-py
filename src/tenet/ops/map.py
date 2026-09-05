"""Morphism composition ``a ∘ b``, ``identity`` and the adjoint ``T†``.

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
reordering" would need a within-side transpose, which is a braid.

[adjoint][tenet.adjoint] is the dagger, and it is one of four operations that are easy to
confuse and are deliberately kept apart (invariant 2: conjugation, duality and the
adjoint are distinct):

=================  ==========  ==========  =======================  ==============
operation          ``side``    ``dual``    blocks                   coefficients
=================  ==========  ==========  =======================  ==============
``conj``           unchanged   unchanged   conjugated               none
``leg.dualized``   unchanged   flipped     Z-isomorphism            FS signs
``repartition``    some flip   same flip   bent                     B-symbols
``adjoint``        all flip    unchanged   conjugated, key-swapped  none
=================  ==========  ==========  =======================  ==============

``adjoint`` needs no bending coefficient precisely because it flips *every* side
at once: ``⊕_c B_c ⊗ id_c`` is simply re-read as ``⊕_c B_c† ⊗ id_c``, so trees,
coupled sectors and multiplicities all survive and only their row/column roles
trade places. And it needs no block transpose because reduced axes travel with
their own legs (invariant 7) and the public axis order is untouched: the whole
transpose is absorbed into the key swap ``(ot, it) → (it, ot)``.

No ``to_dense`` here and no provider branching. NumPy appears as
[identity][tenet.identity]'s default dtype and as
[random_isometry][tenet.random_isometry]'s draw — a
constructor runs at setup time, outside any trace, and ``to_backend`` is the
documented route onto a device.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any

import autoray as ar
import numpy as np

from tenet.backend import lib_fn, promote
from tenet.cache import plan_cache
from tenet.leg import IN, OUT, Leg
from tenet.map_view import check_square, from_matrices, map_layout
from tenet.ops.embed import embed
from tenet.space import ProductSpace
from tenet.structure import FusionBlockKey, TensorStructure, _pattern

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = [
    "AdjointPlan",
    "adjoint",
    "adjoint_plan",
    "compose",
    "identity",
    "isometry",
    "map_diagonal",
    "random_isometry",
]


def _check_composable(a: TensorStructure, b: TensorStructure) -> None:
    """Raise ``ValueError`` naming the offending axis on *both* tensors.

    Follows ``ops.basic._check_same_structure``'s per-axis style: the position in
    the domain/codomain is useless on its own when the public axis order
    interleaves the sides, so both public axis indices are printed.

    Takes the two *structures* rather than the tensors: compatibility is a question
    about legs alone, so [compose_plan][tenet.ops.map.compose_plan] can ask it once
    per structure pair instead of once per composition.
    """
    domain = ProductSpace(tuple(a.legs[i] for i in a.in_axes))
    codomain = ProductSpace(tuple(b.legs[i] for i in b.out_axes))
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
        f"(public axis {a.in_axes[i]} of a, public axis {b.out_axes[i]} "
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

    Parameters
    ----------
    a : SymmetricTensor
        The outer morphism; its domain consumes ``b``'s codomain.
    b : SymmetricTensor
        The inner morphism. Its codomain must carry the same ``(space, dual)``
        sequence, in the same order, as ``a``'s domain — ``side`` is not
        compared and ``name`` is ignored; dimensions alone are never enough.

    Returns
    -------
    SymmetricTensor
        The composition; its public axis order is ``a``'s OUT legs followed by
        ``b``'s IN legs, each in its own public order. A coupled sector that only
        one operand carries — every sector, when an operand is block-less because
        its legs cannot couple — is zero, and those zeros take their backend and
        dtype from the other operand, NumPy ``float64`` when neither has a block.

    Raises
    ------
    ValueError
        If ``a``'s domain and ``b``'s codomain differ in length, or at any
        position in ``(space, dual)`` — the message names the offending axis
        on *both* tensors. Composition never reorders legs within a side; use
        [tenet.transpose][] for that, or [repartition][tenet.SymmetricTensor.repartition] if a
        leg has to change side.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> bool(tenet.allclose(tenet.identity(a.codomain) @ a, a))
    True

    Notes
    -----
    For a long chain at a *fixed* partition in eager NumPy, the matrix form can
    be kept between steps by hand -- worth about 1.1x asymptotically, and the reason
    the library does not persist the layout itself is that real ``tensordot`` chains
    never hit such a cache and ``from_matrices`` is already zero-copy::

        acc = to_matrices(ts[0])
        for t in ts[1:]:
            mb = to_matrices(t)
            acc = {c: acc[c] @ mb[c] for c in acc}
        out = from_matrices(TensorStructure((*ts[0].codomain, *ts[-1].domain)), acc)
    """
    return compose_lowered(a.structure, a.data, b.structure, b.data, block_ref(a, b))


def block_ref(*tensors: "SymmetricTensor") -> Any:
    """A block to take a backend and a dtype from, first block-carrying tensor wins.

    Parameters
    ----------
    *tensors : SymmetricTensor
        The operands, in the order they should be consulted.

    Returns
    -------
    array
        One array of the first ``i`` that carries a block -- whichever of its two
        forms it already holds -- and a NumPy ``float64`` scalar when none does.

    Notes
    -----
    A tensor whose legs cannot couple to any total charge is block-less and legal, and
    it therefore carries no backend and no dtype of its own. Where such an operand meets
    a structure that *does* admit blocks, the zeros that stand for it have to be built
    somewhere: they take the backend and the dtype of the other operand, and NumPy
    ``float64`` only when every operand is block-less and there is nothing else to ask.
    """
    for t in tensors:
        if t.structure.num_blocks:
            # whichever form the tensor holds: a block and its coupled-sector matrix
            # share both, and forcing the other one to read a dtype would gather (or cut)
            # every block for nothing -- a graph node each, under a trace
            return t._first_block()
    return np.zeros((), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ComposePlan:
    """Which of two operands' matrices multiply into which of the result's. Array-free.

    Parameters
    ----------
    structure : TensorStructure
        The composition's structure, ``(*a.codomain, *b.domain)``.
    products : tuple of (int, int, int)
        ``(result position, a position, b position)`` for every coupled sector both
        operands carry, in the result's sector order. Positions index the three
        tensors' ``data`` tuples, i.e. their ``map_layout(...).sectors`` order.
    missing : tuple of (int, (int, int))
        ``(result position, matrix shape)`` for a coupled sector only one operand
        carries, which is zero.
    num_sectors : int
        How many matrices the result holds.
    batches : tuple
        Products grouped by operand matrix shapes for batched Torch matmul.
    """

    structure: TensorStructure
    products: tuple[tuple[int, int, int], ...]
    missing: tuple[tuple[int, tuple[int, int]], ...]
    num_sectors: int
    batches: tuple[tuple[tuple[int, int, int], ...], ...]


@plan_cache(cost=lambda plan: plan.num_sectors)
def compose_plan(a: TensorStructure, b: TensorStructure) -> ComposePlan:
    """The composition's structure and its sector alignment, computed once per pair.

    Parameters
    ----------
    a, b : TensorStructure
        The operands' structures. ``a``'s domain must be ``b``'s codomain as
        ``(space, dual)`` in order.

    Returns
    -------
    ComposePlan
        The plan [compose_lowered][tenet.ops.map.compose_lowered] applies.

    Raises
    ------
    ValueError
        If the two are not composable; the message names the offending axis on both.

    Notes
    -----
    Everything a composition does that is not arithmetic: the compatibility check, the
    result structure, and the map from the result's coupled sectors to the two operands'.
    All three are pure functions of the legs, so a chain that composes the same shapes
    every step pays them once -- measured at the production SU(2) rank-3 point, 4.0x the
    cost of the ``gemm``\\ s themselves before, 1.2x after. The alignment is *positional*
    for the same reason ``_slots`` holds a sector index rather than a sector: the matrices
    are a tuple in that order, so a dict keyed on sectors would hash one per composition.
    """
    _check_composable(a, b)
    structure = TensorStructure((*(a.legs[i] for i in a.out_axes), *(b.legs[i] for i in b.in_axes)))
    at = {c: i for i, c in enumerate(map_layout(a).sectors)}
    bt = {c: i for i, c in enumerate(map_layout(b).sectors)}
    layout = map_layout(structure)
    products: list[tuple[int, int, int]] = []
    missing: list[tuple[int, tuple[int, int]]] = []
    for k, c in enumerate(layout.sectors):
        i, j = at.get(c), bt.get(c)
        if i is None or j is None:
            # A coupled sector carried by only one operand contributes nothing: the
            # missing side has zero trees there, so the product is zero. Deliberate
            # (the result structure still declares the sector), not a KeyError.
            missing.append((k, layout.shapes[k]))
        else:
            # The join's band orders agree by construction (see the module docstring);
            # assert it rather than trust it, the way ``map_layout`` asserts its grid.
            # Cold, once per structure pair -- and it is what ``from_matrices``' shape
            # check used to say on every single composition.
            want = (map_layout(a).shapes[i][0], map_layout(b).shapes[j][1])
            if want != layout.shapes[k]:
                raise RuntimeError(
                    f"compose_plan: sector {c!r} would multiply to {want}, but the "
                    f"composition's layout declares {layout.shapes[k]}; the two sides of "
                    "the join do not enumerate the same trees in the same order"
                )
            products.append((k, i, j))
    batches: dict[tuple, list[tuple[int, int, int]]] = {}
    for k, i, j in products:
        shapes = (map_layout(a).shapes[i], map_layout(b).shapes[j])
        batches.setdefault(shapes, []).append((k, i, j))
    return ComposePlan(
        structure,
        tuple(products),
        tuple(missing),
        len(layout.sectors),
        tuple(tuple(group) for group in batches.values()),
    )


def compose_lowered(
    sa: TensorStructure,
    da: "Sequence[Any]",
    sb: TensorStructure,
    db: "Sequence[Any]",
    ref: Any,
) -> "SymmetricTensor":
    """[compose][tenet.compose]'s body once both operands are already lowered.

    Parameters
    ----------
    sa, sb : TensorStructure
        The operands' structures.
    da, db : sequence of array
        The operands' coupled-sector matrices, in ``map_layout(...).sectors`` order --
        the order a tensor's ``data`` and [lower_plan][tenet.map_view.lower_plan] both
        already use.
    ref : array
        A block to take a backend and a dtype from when a coupled sector is
        missing from both products, as [block_ref][tenet.ops.map.block_ref] returns it.

    Returns
    -------
    SymmetricTensor
        The composition.

    Notes
    -----
    Split out of [compose][tenet.compose] so that ``ops.contraction`` can hand it
    matrices assembled straight from the *unrepartitioned* operands
    ([lower_plan][tenet.map_view.lower_plan]); the matmul and
    the missing-sector rule are the ones ``compose`` always applied.

    The ``matmul`` is resolved once per composition rather than once per sector: at the
    production SU(2) rank-3 point the blocks are 38x3 and ``ar.do``'s lookup is a
    measurable fraction of the ``gemm`` it dispatches. Same function, same values --
    ``lib_fn`` resolves what ``ar.do`` would have reached (see ``tenet.backend``).
    """
    from tenet.tensor import SymmetricTensor

    plan = compose_plan(sa, sb)
    mats: list[Any] = [None] * plan.num_sectors
    if plan.products:
        # ``promote``, not ``infer_backend`` of one side: a NumPy operand meeting a
        # traced JAX one has to dispatch to JAX, which is exactly the promotion
        # ``ar.do`` performed here before -- one CTMRG corner is a constant times a
        # traced tensor -- and a NumPy operand meeting a torch one has to be converted
        # as well, because ``torch.matmul`` refuses an ``ndarray``. Once per call: a
        # tensor's own matrices all share a backend, so one pair decides for the rest.
        backend, da, db = promote(da, db)
        matmul = lib_fn(backend, "matmul")
        if backend == "torch":
            # ponytail: stacking copies operands; tiny CPU matrices can favor the loop.
            # Add a size/device gate only with representative workload measurements.
            stack = lib_fn(backend, "stack")
            for group in plan.batches:
                if len(group) == 1:
                    k, i, j = group[0]
                    mats[k] = matmul(da[i], db[j])
                    continue
                values = matmul(
                    stack(tuple(da[i] for _, i, _ in group)),
                    stack(tuple(db[j] for _, _, j in group)),
                )
                for (k, _, _), value in zip(group, values, strict=True):
                    mats[k] = value
        else:
            for k, i, j in plan.products:
                mats[k] = matmul(da[i], db[j])
    if plan.missing:
        if plan.products:
            ref = mats[plan.products[0][0]]
        elif da and db:
            # No product decided a backend, and the two operands need not be on one:
            # ``block_ref`` answers with the *first* operand that carries a block, which
            # is argument order and not a property of the pair, so a NumPy tensor
            # composed with a traced JAX one would come back NumPy. Cold -- reached only
            # when the operands share no coupled sector at all -- so it costs the hot
            # path nothing to ask the same question ``promote`` asks.
            wanted = ar.infer_backend_multi(da[0], db[0])
            ref = da[0] if ar.infer_backend(da[0]) == wanted else db[0]
        # ``ref.dtype``, not ``ar.get_dtype_name`` and back: every backend this package
        # runs on hands its own ``zeros`` its own dtype object, and reading ``.dtype``
        # off a block is what ``from_matrices`` and ``SymmetricTensor.dtype`` already do.
        zeros, dtype = lib_fn(ar.infer_backend(ref), "zeros"), ref.dtype
        for k, shape in plan.missing:
            mats[k] = zeros(shape, dtype=dtype)
    return SymmetricTensor.from_data(plan.structure, tuple(mats))


def identity(
    legs: Sequence[Leg], *, dtype: Any = np.float64, like: Any = "numpy"
) -> "SymmetricTensor":
    """``id`` on ``ProductSpace(legs)``: the legs mirrored as ``(OUT..., IN...)``.

    Parameters
    ----------
    legs : sequence of Leg
        The legs to build the identity on. ``space``, ``dual`` and ``name``
        are kept and only ``side`` is set, so that
        ``identity(t.codomain) @ t == t``.
    dtype : dtype-like, optional
        The blocks' dtype. Default ``np.float64``.
    like : str or array, optional
        Anything ``ar.do`` accepts — a backend name or a reference array.
        Default ``"numpy"``.

    Returns
    -------
    SymmetricTensor
        The identity morphism, legs ``(*legs OUT, *legs IN)``, one ``eye``
        block per coupled sector.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> i = tenet.identity((Leg(V, OUT),))
    >>> i.shape, bool(tenet.allclose(i @ a, a))
    ((2, 2), True)

    Notes
    -----
    ``space``, ``dual`` and ``name`` are kept and only ``side`` is set, so that
    ``identity(t.codomain) @ t == t``. Dualizing the mirror would build a cup, a
    different morphism.

    ``B_c = eye`` for every coupled sector, and nothing else — which is also the
    sharpest test of ``MapLayout``: this is the identity morphism only because the
    row and column orderings are *derived* from ``block_order`` rather than
    invented, so mirrored legs give mirrored bands.

    ``like`` is anything ``ar.do`` accepts — a backend name or a reference array —
    and defaults to today's ``"numpy"``: ``identity(legs)`` has no tensor to infer
    a backend from, so a caller that *does* have one (``ops/contraction.py::trace``)
    passes it; hard-coding it would make ``trace`` on a torch tensor contract torch
    blocks against NumPy ones.
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

    Parameters
    ----------
    codomain : sequence of Leg
        The larger side. ``codomain[i]`` must contain ``domain[i]``
        sector-wise — same provider, same ``dual``, every sector of the
        domain present in the codomain with a degeneracy at least as large.
    domain : sequence of Leg
        The smaller side being included. ``side`` is *set*, not compared,
        exactly as [identity][tenet.identity] does.
    dtype : dtype-like, optional
        The blocks' dtype. Default ``np.float64``.

    Returns
    -------
    SymmetricTensor
        The inclusion isometry, legs ``(*codomain OUT, *domain IN)``.

    Raises
    ------
    ValueError
        [embed][tenet.SymmetricTensor.embed]'s refusals: a leg count, provider or ``dual``
        mismatch, or a domain sector missing from its codomain partner (or
        present with a smaller degeneracy).

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> W = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> w = tenet.isometry((Leg(W, OUT),), (Leg(V, IN),))
    >>> bool(tenet.allclose(tenet.adjoint(w) @ w, tenet.identity((Leg(V, IN),))))
    True

    Notes
    -----
    The whole body, and every refusal, is [embed][tenet.SymmetricTensor.embed] of
    [identity][tenet.identity]: the blocks come from the identity morphism and the placement
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

    Parameters
    ----------
    codomain : sequence of Leg
        The larger side. Requires ``rows_c >= cols_c`` in every coupled
        sector — the *fused* containment condition, read structurally off
        ``MapLayout``, which is weaker than [isometry][tenet.isometry]'s
        per-leg one.
    domain : sequence of Leg
        The side the isometry is an isometry *of*: ``W† W = id(domain)``.
    seed : int or None, optional
        Seed for ``np.random.default_rng``. ``None`` (the default) is
        non-reproducible.
    dtype : dtype-like, optional
        The blocks' dtype; a complex dtype gets a genuinely complex (Ginibre)
        draw. Default ``np.float64``.

    Returns
    -------
    SymmetricTensor
        A Haar-random isometry, legs ``(*codomain OUT, *domain IN)``.

    Raises
    ------
    ValueError
        If some coupled sector has ``rows_c < cols_c``, so no isometry exists
        there — named with the sector and both dimensions, raised before any
        draw.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> W = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> w = tenet.random_isometry((Leg(W, OUT),), (Leg(V, IN),), seed=0)
    >>> bool(tenet.allclose(tenet.adjoint(w) @ w, tenet.identity((Leg(V, IN),))))
    True

    Notes
    -----
    Per coupled sector ``c``, a ``(rows_c, cols_c)`` Gaussian draw, a QR, and the
    sign fix that makes the result Haar-distributed.

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
    """The dagger plan for ``structure``. Cached: repeat calls return one object.

    Notes
    -----
    The **body** is cached one level down, on ``_pattern(structure)`` -- the same legs
    with every degeneracy 1 -- as ``permutation_plan`` and ``repartition_plan`` are.
    ``sources`` is read off ``block_order`` and ``index_of``, both of which are functions
    of the legs' sectors, sides and duals alone, so this entry holds nothing but
    ``new_structure`` and shares the pattern entry's ``sources`` tuple. Without the split
    a sweep that moves a bond's degeneracies rebuilds a permutation of ``num_blocks``
    that is identical entry for entry, and keeps one copy per bond dimension.
    """
    plan = _pattern_adjoint_plan(_pattern(structure))
    flipped = tuple(replace(leg, side=IN if leg.side is OUT else OUT) for leg in structure.legs)
    if flipped == plan.new_structure.legs:
        return plan
    return replace(plan, new_structure=TensorStructure(flipped))


@cache
def _pattern_adjoint_plan(structure: TensorStructure) -> AdjointPlan:
    """[adjoint_plan][tenet.adjoint_plan]'s body, on a degeneracy-free structure."""
    flipped = tuple(replace(leg, side=IN if leg.side is OUT else OUT) for leg in structure.legs)
    new_structure = TensorStructure(flipped)
    sources = tuple(
        structure.index_of(FusionBlockKey(key.input_tree, key.output_tree))
        for key in new_structure.block_order
    )
    return AdjointPlan(new_structure, sources)


def adjoint(t: "SymmetricTensor") -> "SymmetricTensor":
    """``T†``: the Euclidean-adjoint morphism in ``Hom(codomain, domain)``.

    Parameters
    ----------
    t : SymmetricTensor
        The morphism to dagger.

    Returns
    -------
    SymmetricTensor
        ``T†``: every leg keeps its ``space``, ``dual`` and ``name`` and flips
        its ``side``; the public axis order is unchanged.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> d = tenet.adjoint(a)
    >>> d.legs[0].side, bool(tenet.allclose(tenet.adjoint(d), a))
    (<Side.IN: 'in'>, True)

    Notes
    -----
    Every leg keeps its ``space``, ``dual`` and ``name`` and flips its ``side``;
    the public axis order is unchanged; the block for key ``(ot, it)`` becomes the
    conjugate of the block for key ``(it, ot)`` — no axis permutation is needed,
    because reduced axes travel with their own legs (invariant 7).

    Deliberately no ``requires(...)``: this needs ``conj`` on the backend and the
    identity of the trees, nothing a provider could fail to supply. The dagger
    structure this leans on is named by
    [DaggerData][tenet.symmetry.DaggerData], today a marker every provider
    satisfies.

    """
    # Simplification: a provider with complex Clebsch-Gordan coefficients would need the
    # same capability gate ``ops.basic.conj`` already flags — one ``requires(provider,
    # DaggerData)`` line, once DaggerData grows content. Not a second speculative protocol.
    from tenet.tensor import SymmetricTensor

    plan = adjoint_plan(t.structure)
    # One conjugate-transpose per coupled sector, not one conjugate per block.
    # ``adjoint_plan`` says what this is: swap the two trees of every key. On the
    # coupled-sector matrix that swap *is* the transpose, because flipping ``side``
    # moves no axis and touches no space, so the adjoint's row bands are this tensor's
    # column bands and vice versa -- band order being a pure function of one side's
    # ordered legs, and ``space_sector`` reading ``dual`` rather than ``side``. The
    # degeneracy multi-index of a side is flattened over that side's public axis order,
    # which is likewise unchanged, so cell for cell ``B'_c = conj(B_c)^T``.
    return SymmetricTensor.from_data(
        plan.new_structure,
        tuple(ar.do("conj", ar.do("transpose", m)) for m in t.data),
    )


def map_diagonal(m: "SymmetricTensor") -> "SymmetricTensor":
    """The diagonal of a square map, in the reduced basis, on its codomain legs.

    Parameters
    ----------
    m : SymmetricTensor
        A square map: its domain must be its codomain as ``(space, dual)`` in the
        same order. ``side`` is not compared and ``name`` is ignored.

    Returns
    -------
    SymmetricTensor
        The diagonal entries, on ``m``'s codomain legs — the same structure the
        vectors ``m`` acts on carry, so
        [zip_blocks][tenet.zip_blocks] pairs the two block for block.

    Raises
    ------
    ValueError
        If the map is not square — ``check_square``'s refusal, which names the
        first offending position and both legs.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> legs = (Leg(V, OUT), Leg(V, OUT, dual=True))
    >>> d = tenet.map_diagonal(tenet.identity(legs))  # the identity's diagonal is all ones
    >>> d.legs == legs
    True
    >>> [b.tolist() for b in d.blocks]
    [[[1.0, 1.0], [1.0, 1.0]], [[1.0]]]

    Notes
    -----
    **Coefficient space, not dense space**, in
    [apply_blocks][tenet.apply_blocks]' sense: the result holds the diagonal of the
    matrix a solver iterates on, not the diagonal of the dense expansion of ``m``. The two
    coincide entry for entry only where the Clebsch-Gordan factor is all-ones.
    For a vector ``v`` on the same legs, entry ``k`` of the result is
    ``(m @ v)[k]`` when ``v`` is the ``k``-th reduced-basis unit vector — which is
    also ``<v|m|v> / <v|v>``, and *that* reading is basis-free and survives dense
    expansion.

    **Which basis, and why no recoupling coefficient appears.** Composition is one
    ``matmul`` per coupled sector and nothing else — the module docstring above
    explains why — so the matrix of ``m`` in the reduced storage basis *is*
    ``to_matrices(m)``, and the diagonal of that matrix is the diagonal of the
    blocks whose two fusion trees coincide. The trees are drawn from one set:
    ``Leg.fused_sector`` reads ``dual``, never ``side``, so a square map's
    codomain and domain contribute identical uncoupled labels. No F-symbol, no
    R-symbol, no twist and no bend is read here, and the operation therefore
    requires no capability beyond the fusion rules every structure already needs.

    That is not in tension with the fusion-tree basis being relational
    (invariant 4). The reduced basis of a rank-``N`` map is labelled by a *pair of
    trees*, not by a tuple of external sectors, and this reads the labels: for
    SU(2) at external tuple ``(1, ½, ½, 1)`` two inner lines share one sector
    tuple and are two distinct blocks with unrelated diagonal entries. What cannot
    be done is to *manufacture* the diagonal by
    contracting per-leg diagonals of the operator's factors, which is a per-leg
    reading of that relational basis and loses both the inner line and the
    graded braiding sign. Given the map itself, both are already in its blocks.

    **The whole operation is one ``diagonal`` per coupled sector.** The result carries
    ``m``'s codomain legs and no domain, so its own domain admits only the empty fusion
    tree and it has exactly one coupled sector: the unit. Its matrix is one column, its
    row bands are the codomain trees, and a row band's index flattens the codomain
    degeneracies in ``out_axes`` order -- which is the order ``m``'s own rows flatten in,
    and, ``m`` being square, the order its columns do too. The block at ``(tau, tau)``
    is therefore the diagonal square of ``m``'s unit matrix at that band, and the whole
    column is that matrix's main diagonal. No block is cut out and no einsum runs
    (invariant 8); the 26-subscript ceiling the einsum spelling carried is gone with it.

    **The unit-coupled reading.** The result carries ``m``'s codomain legs, whose
    blocks are the unit-coupled ones, and that is the whole diagonal of the
    operator on the vectors tenet can represent: a ``SymmetricTensor`` on those
    legs is invariant by construction (invariant 1), and a targeted charge is
    carried by an explicit charge leg — which is then a leg of ``m`` too. A vector
    stored on a different partition of the same legs (an MPS two-site tensor keeps
    its right bond IN) reaches this basis by [bend][tenet.bend], which is one
    scalar per block: a diagonal similarity, so it leaves both this diagonal and
    the quotient ``q / (lambda - diag)`` entry for entry unchanged.
    """
    from tenet.tensor import SymmetricTensor

    check_square(m, "map_diagonal")
    structure = TensorStructure(tuple(m.codomain))
    at = {c: i for i, c in enumerate(map_layout(m.structure).sectors)}
    return SymmetricTensor.from_data(
        structure,
        tuple(
            ar.do("reshape", ar.do("diagonal", m.data[at[c]]), (-1, 1))
            for c in map_layout(structure).sectors
        ),
    )
