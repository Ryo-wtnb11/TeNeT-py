"""C4v CTMRG, written out: the classical Ising model against Onsager, then an iPEPS gradient.

Run it standalone::

    uv run --extra jax python examples/toy_codes/ctmrg.py

The algorithm is the file: the corner and edge tensors, the two absorbers, the projector,
the move and the sweep. Nothing is imported from ``tenet.network``, which ships all of
it; ``examples/ising2d.py`` is the Ising half through the library.

The tensor operations it is built on: ``SymmetricTensor.from_blocks`` and
``SymmetricTensor.random`` for the inputs, ``tenet.einsum`` for every contraction,
``tenet.adjoint``, ``tenet.transpose`` and ``tenet.repartition`` for the leg moves,
``tenet.linalg.svd_truncated`` and ``tenet.linalg.svd(bond=)`` for the projector,
``tenet.identity`` for the corner seed, ``tenet.norm``, ``tenet.full_trace`` for the
closed networks, and ``tenet.to_matrices`` to read a corner spectrum.

Two physical problems, **one** CTMRG core:

* the classical 2D Ising partition function, whose free energy per site has a closed form
  (Onsager) and whose internal energy ``d(beta f)/d beta`` is therefore an oracle for
  ``jax.grad`` through the unrolled sweeps;
* a single-site U(1) (or SU(2)) iPEPS with a random symmetric two-site ``h``, which
  exercises graded truncation, ``svd(bond=)`` across sectors and multiplet degeneracies.

**This file lives on both sides of a trace**, so every function below opens by saying
which. :func:`converge` runs ``tenet.linalg.svd_truncated`` **outside** ``jax.grad`` -- it
decides a bond :class:`~tenet.GradedSpace` from singular *values*, so it raises
``tenet.StructureChangingError`` under any trace -- and :func:`unrolled` runs
``tenet.linalg.svd(bond=)`` **inside** it at exactly that frozen bond: shape-static, one
trace, differentiable. The ``GradedSpace`` is the only thing that crosses the boundary,
and it is metadata -- frozen, hashable, array-free, a legitimate jit *cache key* and never
a jit *argument*.

**Leg conventions**, the part worth reading before the code:

* bulk ``(l OUT, u OUT, r IN, d IN)`` -- ``l``/``u`` share a side and ``r``/``d`` share
  one, so the C4v diagonal mirror is the plain transpose ``(1, 0, 3, 2)`` and *one* edge
  tensor serves both the top and the left of a corner;
* corner ``c`` ``(X OUT, X IN)`` -- for **both** models -- and edge ``e``
  ``(X IN, X OUT, V IN)`` for a single layer, ``(X IN, X OUT, V_ket IN, V_bra IN dual)``
  for a double one. The double-layer edge carries the ket bond and its conjugate as two
  separate legs and **never fuses them** (froSTspin ``ctm_environment.py``:16-33): the
  ``dual=True`` is the leg bend the fused convention used to hide, only the ``X`` leg is
  ever truncated, and the site enters as a ket and then a bra rather than as a product.
  Both edges are oriented maps on the environment space, so the boundary ring closes as
  ``c -> e -> ... -> adjoint(c) -> adjoint(e) -> ...`` -- see :func:`ring`;
* the enlarged corner is a *bilinear form*, not a map -- its two index groups are related
  by the diagonal mirror, so they sit on the same side -- and the single leg bend that
  ``svd(axes=...)`` performs to make it a map is exactly that mirror. It is why the
  projector ``u`` contracts the *incoming* group of an enlarged edge while ``adjoint(u)``
  contracts the *outgoing* one. The groups are the tensor's two halves, which is why
  :func:`move` partitions at ``ndim // 2`` rather than branching on the model: rank 4 for
  a single layer, rank 6 for a double one.

**The Ising half is Z2-graded**, for the reason YASTN's CTMRG Ising example passes
``sym='Z2'``: it stops a finite-chi environment from breaking the symmetry spuriously in
the ordered phase, which is what lets this file run at ``beta > beta_c`` against Onsager
at all. Two further things the grading buys: zero magnetization is *structural* -- a spin
insertion is a Z2-odd tensor, which no invariant ``SymmetricTensor`` can hold -- and the
ordered-phase corner spectrum acquires **exact** two-fold degeneracy across the parity
sectors. Because that doubling is *cross*-sector and ``tenet.ad`` broadens *per coupled
sector*, the graded run never hands one SVD a degenerate pair.

**The iPEPS half is a plumbing result, not a physics result, and cannot be otherwise with
a one-site unit cell**, so it makes **no benchmark-energy claim**. Liao et al. get a
single-site AFM Heisenberg cell by rotating one sublattice by pi about y, which turns
``S^x S^x - S^y S^y`` into ``(S^+S^+ + S^-S^-)/2`` -- an operator that changes
``S^z_tot`` by +-2 and so *destroys the U(1) the ansatz is graded by*. The alternatives
are a two-site unit cell (out of scope) or dropping the symmetry (which deletes the
reason this half exists). So it follows ``examples/toy_codes/vmc_mps.py``: random
symmetric ``h``, no comparison against ``-0.669437(5)``, said out loud right here.

Simplification: **one C4v move, not four directional ones**, and no multi-site unit cell.
One corner and one edge describe the whole environment only for a mirror-symmetric bulk
on a 1x1 cell, which is why :func:`c4v` symmetrizes the *ansatz*. The upgrade path is
named and not started: YASTN's ``EnvCTM`` (eight tensors per site and four ``update_``
moves), froSTspin's four ``contract_*`` wrappers over one ``contract_enlarged_corner``.

Simplification: **truncated backprop through K unrolled moves, never the implicit fixed
point** (PRX 9, 031041 Sec. III C) -- the implicit route is a second numerical framework
inside a VJP, with its own tolerance and data-dependent exit, which cannot warn under a
trace.

Simplification: **no gradient checkpointing** -- at ``k=4`` and ``chi=16`` the tape fits,
and ``jax.checkpoint`` on :func:`move` is the one-line addition when it does not.

Simplification: **no pre-QR before the projector SVD** -- YASTN takes an intermediate QR
(``use_qr=True``) for stability; ``tenet.linalg.qr`` exists and the composition is three
lines, and at ``chi <= 16`` in float64 nothing has lost digits yet.

Simplification: **``svd``, not ``eigh``, for the projector**, even though C4v CTMRG
classically diagonalizes a Hermitian corner: the fixed-bond differentiable route exists
only for ``svd``. tensorgrad does the same.

**There is an XLA compile floor here**, unlike in ``dmrg.py``: the unrolled sweeps are one
traced region, and tracing plus compiling them dominates a short run.
"""

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.symmetry import SU2, U1, Z2, SU2Sector, U1Sector, Z2Sector

BETA_C = 0.4406867935097714  # ln(1 + sqrt(2)) / 2

# ``opt_einsum``'s greedy path, for the ring contractions only. Simplification: greedy, not
# "auto" -- at ten-plus operands "auto"'s dynamic-programming *search* costs an order of
# magnitude more than the contraction it plans (4.5 s against 0.4 s for the two-site
# energy). Upgrade path: an explicit path, or cotengra.
PATH = "greedy"

# Environment dimension for the iPEPS half, per provider. Simplification: not one number.
# ``max_bond`` bounds the *dense* bond, which for SU(2) is ``sum_c (2j+1) m_c``: a budget
# of 4 stops in the middle of the second multiplet -- the split Francuz-Schuch-Vanhecke's
# Appendix C warns about, and slower to converge and to differentiate than the 6 that
# closes it. U(1) has no multiplets and 4 is plenty.
CHI_IPEPS = {"u1": 4, "su2": 6}

# Physical and virtual spaces, per provider, as ``vmc_mps.SPACES`` does. The virtual space
# must contain the unit sector or a spin-1/2 site tensor has no allowed block at all.
SPACES = {
    "u1": (
        GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1}),
        GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1}),
    ),
    "su2": (
        GradedSpace.new(SU2, {SU2Sector(1): 1}),
        GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 1}),
    ),
}


# --- the two bulk tensors ----------------------------------------------------------


def ising_bulk(beta):
    """Classical 2D Ising partition-function tensor, legs ``(l OUT, u OUT, r IN, d IN)``.

    ``a[l,u,r,d] = sum_s W[s,l] W[s,u] W[s,r] W[s,d]`` with ``W W^T`` the bond Boltzmann
    matrix ``[[e^b, e^-b], [e^-b, e^b]]``, i.e. the symmetric splitting
    ``W = [[sqrt cosh b, sqrt sinh b], [sqrt cosh b, -sqrt sinh b]]``. That ``W`` *is
    already the parity basis*: ``W[s, 0]`` does not depend on ``s`` and ``W[s, 1]`` is odd
    under ``s -> -s``, so summing over ``s`` doubles every term with an even number of odd
    legs and annihilates the rest.

    The ``Z2`` legs are therefore not a claim checked afterwards, they are the statement:
    the blocks the grading allows are exactly the surviving entries, and each one is
    ``2 W[0,l] W[0,u] W[0,r] W[0,d]``. The eight structurally zero entries have no block to
    live in and are never built.

    ``beta`` may be a *traced scalar*, so the block values are built with ``jax.numpy``.
    """
    c, s = jnp.sqrt(jnp.cosh(beta)), jnp.sqrt(jnp.sinh(beta))
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    legs = (Leg(space, OUT), Leg(space, OUT), Leg(space, IN), Leg(space, IN))
    structure = TensorStructure(legs)
    blocks = {}
    for key in structure.block_order:  # the key names (l, u) and (r, d)
        w = [c if sector.parity == 0 else s for sector in key.output_tree.uncoupled]
        w += [c if sector.parity == 0 else s for sector in key.input_tree.uncoupled]
        blocks[key] = jnp.full((1, 1, 1, 1), 2.0 * (w[0] * w[1] * w[2] * w[3]))
    return SymmetricTensor.from_blocks(legs, blocks)


def c4v(a: SymmetricTensor) -> SymmetricTensor:
    """Symmetrize an iPEPS tensor under the C4v diagonal mirror ``l <-> u``, ``r <-> d``.

    An **ansatz constraint**, and :func:`double_layer_ctm` documents it as a precondition
    rather than enforcing it: one corner and one edge describe the environment only if the
    bulk is mirror-symmetric, a random ansatz is not, and symmetrizing the caller's state
    is not the environment's business. Because ``l``/``u`` share a side and ``r``/``d``
    share one, the mirror is the plain transpose ``(0, 2, 1, 4, 3)``: no bend, no bending
    coefficient, and linear, so it differentiates for free.
    """
    return (a + tenet.transpose(a, (0, 2, 1, 4, 3))) / 2


def build_ipeps(provider: str = "u1", seed: int = 1) -> SymmetricTensor:
    """A random single-site iPEPS, legs ``(P OUT, l OUT, u OUT, r IN, d IN)``."""
    phys, virt = SPACES[provider]
    legs = (Leg(phys, OUT), Leg(virt, OUT), Leg(virt, OUT), Leg(virt, IN), Leg(virt, IN))
    return SymmetricTensor.random(legs, seed=seed).to_backend("jax")


def build_h(provider: str = "u1", seed: int = 100) -> SymmetricTensor:
    """A random two-site operator on ``(P OUT, P OUT, P IN, P IN)``: symmetric by
    construction, hence ``Sz``-conserving for U(1) and a scalar under SU(2). A plumbing
    operator, exactly as ``vmc_mps.build_h`` is."""
    phys = SPACES[provider][0]
    legs = (Leg(phys, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(phys, IN))
    return SymmetricTensor.random(legs, seed=seed).to_backend("jax")


# --- the environment: seeds, absorbers, the move, the sweep -------------------------


class Absorb(NamedTuple):
    """How one model grows an environment: ``corner(c, e)`` and ``edge(e, p)``.

    The type is the *definition* of a model's absorption, not the absorption step -- that
    is :func:`move`. The two callable contracts:

    * ``corner(c, e) -> big_c`` of rank ``2n``, whose two index groups are the diagonal
      mirror of each other -- a bilinear form, not a map -- which is what licenses
      :func:`move` partitioning at ``ndim // 2``;
    * ``edge(e, p) -> new_e`` at the same rank as ``e``, on the projector's new bond.

    Write those two and a third model needs nothing from :func:`move`'s body. A
    ``NamedTuple`` of two closures rather than a ``Protocol``: the closures must be able to
    capture *traced* values -- a gradient with respect to the bulk flows through
    :func:`single_layer` built inside the traced region -- and a ``NamedTuple`` is
    hashable, so it can be passed at ``static_argnums`` with no ``__hash__`` to write.
    """

    corner: Callable[[SymmetricTensor, SymmetricTensor], SymmetricTensor]
    edge: Callable[[SymmetricTensor, SymmetricTensor], SymmetricTensor]


def ones(legs: Sequence[Leg]) -> SymmetricTensor:
    """A tensor with every structurally allowed entry equal to 1.

    ``TensorStructure`` already knows which blocks the grading allows and how big each one
    is, so the seed is "fill the blocks that exist"; there is no dense array here to build
    and project.
    """
    structure = TensorStructure(tuple(legs))
    blocks = {key: np.ones(structure.block_shape(key)) for key in structure.block_order}
    return SymmetricTensor.from_blocks(legs, blocks)


def spectrum(s: SymmetricTensor) -> list[float]:
    """The corner spectrum, descending.

    ``s`` comes from :func:`tenet.linalg.svd_truncated` and is diagonal by construction, so
    this reads its diagonal; the ``sqrt(qdim)`` weight is the same one :func:`tenet.norm`
    carries, and it is 1 for Z2 and U(1) and ``sqrt(2j+1)`` for SU(2). Weighting is what
    makes the values comparable across sectors, which is the whole point of using them as
    the convergence criterion of a graded sweep.
    """
    qdim = s.provider.qdim
    out = [
        float(m[i, i]) * qdim(sector) ** 0.5
        for sector, m in tenet.to_matrices(s).items()
        for i in range(m.shape[0])
    ]
    return sorted(out, reverse=True)


def normalized(t: SymmetricTensor) -> SymmetricTensor:
    """**Inside** the traced region. ``t / ||t||``, after every move.

    This is a division by ``tenet.norm``, not the renormalization -- the projector
    truncation :func:`move` performs -- that the R in CTMRG names.

    Not cosmetic: ``tenet.ad``'s Lorentzian ``epsilon`` is in units of sigma squared and the
    PRX default ``1e-12`` assumes an ``O(1)``-normalized spectrum. A CTMRG that does not
    renormalize sees the corner norm grow like the partition function itself, and the
    broadening would then be either a no-op or a sledgehammer depending on the coupling.
    """
    return t / tenet.norm(t)


def ring(c: SymmetricTensor, e: SymmetricTensor) -> tuple[SymmetricTensor, ...]:
    """**Inside** the traced region. ``(c, adjoint(c), e, adjoint(e))``.

    One line, and it is here for the convention rather than for the line: the far side of a
    C4v boundary ring is the ``tenet.adjoint`` of the near side, which for a real
    environment is what "the same tensor seen from the other side" means.
    """
    return c, tenet.adjoint(c), e, tenet.adjoint(e)


def single_layer(bulk: SymmetricTensor) -> Absorb:
    """**Inside** the traced region. An :class:`Absorb` for any rank-4 bulk tensor.

    ``bulk`` is ``(l OUT, u OUT, r IN, d IN)`` and nothing here knows which model produced
    it -- an Ising Boltzmann tensor, a six-vertex weight, any single-layer transfer tensor.
    There is no bra and no ket, so there is no pair to fuse and nothing hiding a leg bend.
    """

    def corner(c: SymmetricTensor, e: SymmetricTensor) -> SymmetricTensor:
        """The 2x2 object the projector diagonalizes: corner, two edges, one bulk tensor.

        Legs ``(X OUT, V IN, X IN, V IN)``. The two index pairs are the diagonal mirror of
        each other, so this is a bilinear form rather than a map.
        """
        return tenet.einsum("ab,ace,fbg,gehi->chfi", c, e, e, bulk)

    def edge(e: SymmetricTensor, p: SymmetricTensor) -> SymmetricTensor:
        """One edge with one bulk tensor absorbed -- legs ``(X IN, X OUT, V OUT, V IN,
        V IN)`` -- then projected with ``p`` on its incoming pair and ``adjoint(p)`` on its
        outgoing one."""
        big_e = tenet.einsum("abe,gehi->abghi", e, bulk)
        return tenet.einsum("abghi,agx,bhy->xyi", big_e, p, tenet.adjoint(p))

    return Absorb(corner, edge)


def layers(ket: SymmetricTensor) -> tuple[SymmetricTensor, SymmetricTensor]:
    """**Inside** the traced region. ``(ket, bra)`` for a rank-5 iPEPS ket.

    * ket ``(P OUT, l OUT, u OUT, r IN, d IN)``, returned untouched;
    * bra ``repartition(adjoint(ket), (1, 2), (0, 3, 4))``, legs
      ``(L OUT dual, U OUT dual, s IN, R IN dual, D IN dual)``.

    **Their product is never formed.** The bend is done here, at rank 5, and it is *named*
    rather than hidden: bending is what flips ``dual``, and it is that flip which makes the
    bra's bonds meet the ``dual=True`` bra bonds of the rank-4 environment edge. The fused
    convention this replaced hid the same flip inside a ``fuse`` of a ``(V, V*)`` pair.
    """
    return ket, tenet.repartition(tenet.adjoint(ket), (1, 2), (0, 3, 4))


def double_layer(ket: SymmetricTensor, bra: SymmetricTensor) -> Absorb:
    """**Inside** the traced region. An :class:`Absorb` for a rank-5 iPEPS ket:
    **environment first, then the ket, then the bra**.

    That is froSTspin's ``contract_enlarged_corner`` order (``ctmrg/ctm_contract.py``:42,
    :53, which closes the physical legs at the moment the bra enters) and YASTN's,
    mirrored (``fpeps/envs/_env_contractions.py``:221-224). YASTN *can* materialize the
    product -- ``DoublePepsTensor.fuse_layers()`` -- and makes you ask for it by name;
    nothing here asks. The peak is froSTspin's ``2*a*d*chi**2*D**4`` (the comment at :52):
    rank 6, against a fused double layer's ``d**2 D**8``.

    The edge is **projected before it absorbs**, per ``ctm_renormalize.py``:145-166
    (``nT = T @ P``, then ``A``, then ``A.dagger()``, then ``Pt``), which is why no rank-8
    enlarged edge is materialized either: the environment leg is cut to ``chi'`` first and
    every later step carries it instead of a full ``chi``.
    """

    def corner(c: SymmetricTensor, e: SymmetricTensor) -> SymmetricTensor:
        """Rank 6, legs ``(X OUT, r_ket, r_bra, X IN, d_ket, d_bra)`` -- froSTspin's
        ``contract_enlarged_corner`` return, ``permute((2,0,4),(3,1,5))``, in tenet
        spelling. Its two index triples are the diagonal mirror of each other, exactly as
        the single-layer corner's two pairs are."""
        env = tenet.einsum("ab,acjJ,fbgG->cfgGjJ", c, e, e)  # (X, X, l_k, l_b, u_k, u_b)
        env = tenet.einsum("cfgGjJ,sgjri->csfGJri", env, ket)  # rank 7, physical open
        return tenet.einsum("csfGJri,GJsRI->crRfiI", env, bra)

    def edge(e: SymmetricTensor, p: SymmetricTensor) -> SymmetricTensor:
        """``T @ P``, ket, bra, ``Pt`` -- four steps, peak rank 7, result rank 4."""
        t = tenet.einsum("abuU,alLx->buUlLx", e, p)
        t = tenet.einsum("buUlLx,slurd->bULxsrd", t, ket)
        t = tenet.einsum("bULxsrd,LUsRD->bxrRdD", t, bra)
        return tenet.einsum("bxrRdD,brRy->xydD", t, tenet.adjoint(p))

    return Absorb(corner, edge)


def init_env(site: SymmetricTensor, *bonds: Leg) -> tuple[SymmetricTensor, SymmetricTensor]:
    """**Outside** the traced region. Corner and edge on a *one-dimensional* environment
    space.

    ``bonds`` are the edge's virtual legs: one for a single-layer bulk, two -- the ket bond
    and its dual bra partner -- for a double layer. ``site`` supplies only the provider, the
    dtype and the backend.

    The environment then grows one bulk leg per move -- ``X -> X (x) V`` truncated to
    ``chi`` -- which is the original "grow the lattice out of a corner" reading of CTMRG and
    needs no partial trace of the bulk to seed it. The corner is the identity on the unit
    sector and the edge is all ones, i.e. YASTN's free boundary.

    **All ones rather than a random draw.** A seed whose component along the dominant
    eigenvector is small is a boundary the sweep has to climb out of -- measurably, a
    per-sweep contraction of 0.97 instead of 0.75 for the graded Ising bulk at
    ``beta = 0.4``. On a one-dimensional *unit-sector* environment space the only allowed
    block is the unit one, so under a grading the seed is right by construction rather than
    by luck.

    Simplification: a 1-dimensional seed, not YASTN's ``init='dl'`` partial trace and not
    ``tenet.random_isometry``. The isometry seed is what a ``chi > D**2`` start needs, where
    growing from one dimension takes an extra sweep or two to fill the space;
    ``tenet.isometry``/``random_isometry`` slot straight in at that point.
    """
    unit = GradedSpace.new(site.provider, {site.provider.unit: 1})
    c = tenet.identity((Leg(unit, OUT),), dtype=site.dtype, like=site.backend)
    return c, ones((Leg(unit, IN), Leg(unit, OUT), *bonds)).to_backend(site.backend)


def single_layer_ctm(bulk: SymmetricTensor) -> tuple[Absorb, SymmetricTensor, SymmetricTensor]:
    """**Outside** the traced region. ``(absorber, c, e)`` for a rank-4 bulk tensor.

    One virtual bond per edge, so ``converge(*single_layer_ctm(bulk), chi=16)`` is the whole
    call.
    """
    return single_layer(bulk), *init_env(bulk, Leg(bulk.legs[0].space, IN))


def double_layer_ctm(ket: SymmetricTensor) -> tuple[Absorb, SymmetricTensor, SymmetricTensor]:
    """**Outside** the traced region. ``(absorber, c, e)`` for a single-site iPEPS ket.

    **Precondition, not policy:** ``ket`` must already be invariant under the diagonal
    mirror ``tenet.transpose(ket, (0, 2, 1, 4, 3))`` -- see :func:`c4v` -- or one corner and
    one edge do not describe the environment. Nothing here symmetrizes it: that would
    silently edit the caller's state, and a caller whose ansatz is genuinely C4v-symmetric
    would pay for a no-op.

    The edge is **rank 4**, legs ``(X IN, X OUT, V_ket IN, V_bra IN dual)``: the ket bond
    and its conjugate as two separate legs, never fused. It is why ``e`` can stay rank 4
    forever -- only the ``X`` leg is ever truncated -- and why seeding an environment does
    not need a double layer to exist first.
    """
    bra = layers(ket)[1]
    virt = ket.legs[1].space
    seed = init_env(ket, Leg(virt, IN), Leg(virt, IN, dual=True))
    return double_layer(ket, bra), *seed


def move(
    c: SymmetricTensor,
    e: SymmetricTensor,
    absorb: Absorb,
    *,
    bond: GradedSpace | None = None,
    chi: int | None = None,
) -> tuple[SymmetricTensor, SymmetricTensor, GradedSpace]:
    """One C4v move, and the only function here that is on **both** sides of the trace.

    **Outside** ``jit``/``grad`` with ``chi=``: the projector comes from
    ``tenet.linalg.svd_truncated``, which reads the singular *values* to decide which
    sectors survive and therefore raises ``tenet.StructureChangingError`` under
    ``jax.jit``/``jax.grad``. That half decides a structure.

    **Inside** ``jax.jit(jax.grad(...))`` with ``bond=B``: the projector comes from
    ``tenet.linalg.svd(..., bond=B)`` -- the same factorization projected onto a space the
    caller decided out there, fully shape-static and differentiable. That half reuses one.

    **The new corner is ``s`` itself**, because ``s = adjoint(u) . big_c . v`` by
    definition: projecting the enlarged corner with ``u`` on one side and ``v`` on the other
    *is* the singular-value matrix, and forming it explicitly would be the same numbers
    through two more contractions. The new edge takes ``u`` on its incoming pair and
    ``adjoint(u)`` on its outgoing one -- the two pairs are related by the leg bend inside
    ``svd(axes=...)``, which is the C4v diagonal mirror written in leg metadata.

    ``absorb`` is the only thing that knows which model this is, and the partition is
    ``ndim // 2`` rather than a branch, because a bilinear form's two index groups are
    always its two halves: rank 4 for a single layer, rank 6 for a double one.

    **Precondition:** a single isometry ``u`` projects both index groups, which is exact
    only for a *positive* enlarged corner. A single-layer Ising corner is positive, which is
    why that model reproduces Onsager to float64; a double-layer corner with an indefinite
    spectrum still gets a self-consistent contraction, but its corner and edge then differ
    by a diagonal of signs.

    Simplification: one isometry for a bilinear corner whose ``u`` and ``v`` coincide only
    when it is positive. Fixing the indefinite case wants a fixed-bond ``eigh`` -- which the
    library does not offer -- or four directional moves.
    """
    big_c = absorb.corner(c, e)
    n = big_c.ndim // 2  # (0..n-1 | n..2n-1): 2 for a single layer, 3 for a double one
    axes = (tuple(range(n)), tuple(range(n, 2 * n)))
    if bond is None:
        p, s, _ = tenet.linalg.svd_truncated(big_c, axes, max_bond=chi)
    else:
        p, s, _ = tenet.linalg.svd(big_c, axes, bond=bond)
    return normalized(s), normalized(absorb.edge(e, p)), p.legs[-1].space


def _spectrum_change(old: list[float], new: list[float]) -> float:
    """Max entrywise change, zero-padded to the longer spectrum. While the environment is
    still growing the two have different lengths, and the padding makes that a large change
    rather than an error -- which is what it is."""
    n = max(len(old), len(new))
    old, new = old + [0.0] * (n - len(old)), new + [0.0] * (n - len(new))
    return max(abs(a - b) for a, b in zip(old, new, strict=True))


def converge(
    absorb: Absorb,
    c: SymmetricTensor,
    e: SymmetricTensor,
    chi: int = 16,
    tol: float = 1e-10,
    max_sweeps: int = 100,
) -> tuple[SymmetricTensor, SymmetricTensor, GradedSpace]:
    """**Outside** ``jit``/``grad``, and it cannot be otherwise. Sweep to a fixed spectrum.

    Returns ``(c, e, bond)``: the converged environment and the frozen bond the last
    :func:`move` decided -- the one and only thing that crosses into the differentiated
    region. The loop reads singular values to decide a bond and a corner spectrum to decide
    when to stop; a data-dependent loop exit is not a tracing edge case, it is the thing the
    outside/inside split exists to keep outside.

    Simplification: the sweep record is dropped on the floor. ``tenet.network.ctmrg``
    returns a record carrying the per-sweep spectrum change, so a caller can *assert*
    convergence; here the check is the loop's own exit and the caller is :func:`main`,
    which prints a number an oracle judges.
    """
    bond, previous = None, spectrum(c)
    for _ in range(max_sweeps):
        c, e, bond = move(c, e, absorb, chi=chi)
        current = spectrum(c)
        change, previous = _spectrum_change(previous, current), current
        if change < tol:
            break
    assert bond is not None  # max_sweeps >= 1, so move() ran at least once
    return c, e, bond


def unrolled(
    c: SymmetricTensor,
    e: SymmetricTensor,
    absorb: Absorb,
    bond: GradedSpace,
    k: int = 4,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """**Inside** ``jax.jit(jax.grad(...))``. Exactly ``k`` fixed-structure moves.

    Takes ``c``, ``e`` and ``bond`` as three arguments rather than as one registered
    container, because ``bond`` is a ``static_argnums`` cache key and would become a pytree
    *leaf* if it arrived inside one: a :class:`~tenet.GradedSpace` is metadata, and jit must
    key on it, never flatten it.

    A static Python loop: at ``k = 4`` there is nothing for ``jax.lax.scan`` to buy, and the
    loop being static is what makes the whole region one trace.
    """
    for _ in range(k):
        c, e, _ = move(c, e, absorb, bond=bond)
    return c, e


# --- observables -------------------------------------------------------------------


def log_kappa(beta, env, k: int = 4):
    """``ln`` of the partition function per site, from ``k`` unrolled moves at ``beta``.

    ``kappa = Z(L+1,L+1) Z(L,L) / Z(L+1,L) Z(L,L+1)``, Baxter's corner-transfer
    telescoping: four corners cover an ``L x L`` patch, adding four edges and one bulk
    tensor covers ``(L+1) x (L+1)``, and adding only the left and right edges covers
    ``L x (L+1)``. Every leg closes except one bond, which ``tenet.full_trace`` closes.
    ``env`` is the ``(c, e, bond)`` triple :func:`converge` returned: the truncated
    backprop's *initial condition*, which carries no gradient, while the ``k`` moves inside
    do.
    """
    c0, e0, bond = env
    bulk = ising_bulk(beta)
    c, e = unrolled(c0, e0, single_layer(bulk), bond, k=k)
    cc, ca, ec, ea = ring(c, e)
    z_c = tenet.full_trace(tenet.einsum("ab,ac,dc,eb->de", cc, ca, cc, ca))
    z_h = tenet.full_trace(
        tenet.einsum("ab,ac,dcf,ed,eg,ghf->hb", cc, ca, ea, cc, ca, ec, optimize=PATH)
    )
    z_a = tenet.full_trace(
        tenet.einsum(
            "ab,acp,cd,edq,fe,gfr,gh,hks,spqr->kb",
            cc,
            ec,
            ca,
            ea,
            cc,
            ea,
            ca,
            ec,
            bulk,
            optimize=PATH,
        )
    )
    return jnp.log(z_a * z_c / z_h**2)


def free_energy(beta, env, k: int = 4):
    """``-ln(kappa)/beta``, the free energy per site. Compare :func:`onsager`."""
    return -log_kappa(beta, env, k=k) / beta


def beta_free_energy(beta, env, k: int = 4):
    """``beta f = -ln kappa``. This is the function differentiated: ``d(beta f)/d beta`` is
    the internal energy per site, and the Onsager oracle has it in closed form."""
    return -log_kappa(beta, env, k=k)


def onsager(beta: float, points: int = 200_001) -> float:
    """``beta f`` from Onsager's closed form, by direct quadrature. NumPy, no ``scipy``.

    ``-beta f = ln 2 + (1/2pi) int_0^pi dtheta ln[cosh^2(2b) + (1/k) sqrt(1 + k^2 - 2k cos
    2theta)]``, ``k = 1/sinh^2(2b)``. The equivalent elliptic form is cross-checked in
    ``tests/integration/test_ctmrg.py`` before this is used to judge anything.
    """
    kk = 1.0 / np.sinh(2.0 * beta) ** 2
    theta = np.linspace(0.0, np.pi, points)
    integrand = np.log(
        np.cosh(2.0 * beta) ** 2 + np.sqrt(1.0 + kk**2 - 2.0 * kk * np.cos(2.0 * theta)) / kk
    )
    return -(np.log(2.0) / 2.0 + np.trapezoid(integrand, theta) / (2.0 * np.pi))


def _halves(r, ket, bra, phys1: str = "", phys2: str = ""):
    """The 2x1 environment, split down the middle into two halves.

    ``left`` is the bottom-left corner, the left edge, the top-left corner, the first top
    and bottom edges and the first site: legs ``(*phys1, b, c, k, h, r, R)`` where
    ``b``/``k`` are the ring's one open bond, ``c``/``h`` the cut through the top and
    bottom rows and ``r``/``R`` the bonds between the two sites; ``right`` is the mirror
    image. ``phys`` is ``""`` (physical legs closed, the denominator) or ``"Ww"`` -- bra
    first, then ket -- for the numerator. Each half is built the way :func:`double_layer`
    builds a corner (environment, ket, bra), so the peak is rank 7 -- rank 8 with the
    physical legs open, froSTspin ``rdm.py``:30-69, ``a*d*chi**2*D**4`` -- and no double
    layer is formed.

    Simplification: two hand-written halves instead of one twelve-operand equation. The
    contraction is identical; what changes is that the intermediates are rank 5 and rank 3
    by construction rather than by whatever path ``opt_einsum`` picks from *physical* leg
    sizes -- which for an unevenly filled graded tensor it picks badly and unpredictably:
    the same network measured 0.7 s and 3.6 s for two SU(2) environments differing only in
    how ``chi`` split across sectors. Upgrade path: a path planner that costs a graded
    network by its *blocks*.
    """
    cc, ca, ec, ea = r
    k1, b1 = (phys1[1], phys1[0]) if phys1 else ("s", "s")
    k2, b2 = (phys2[1], phys2[0]) if phys2 else ("s", "s")
    left = tenet.einsum("ij,jklL,ihdD->khlLdD", ca, ec, ea)
    left = tenet.einsum(f"khlLdD,{k1}lurd->khLD{k1}ur", left, ket)
    left = tenet.einsum(f"khLD{k1}ur,LU{b1}RD->{phys1}khuUrR", left, bra)
    left = tenet.einsum(f"{phys1}khuUrR,ab,acuU->{phys1}bckhrR", left, cc, ec, optimize=PATH)
    right = tenet.einsum("cduU,de,ferR->cfuUrR", ec, ca, ea)
    right = tenet.einsum(f"cfuUrR,{k2}lurd->cfUR{k2}ld", right, ket)
    right = tenet.einsum(f"cfUR{k2}ld,LU{b2}RD->{phys2}cflLdD", right, bra)
    right = tenet.einsum(f"{phys2}cflLdD,gf,hgdD->{phys2}chlL", right, cc, ea, optimize=PATH)
    return left, right


def energy(a: SymmetricTensor, h: SymmetricTensor, env, k: int = 4):
    """``<h> / <1>`` on a 2x1 patch, from ``k`` unrolled moves at the current ``a``.

    The ring is four corners, two top edges, two bottom edges and one edge on each side;
    each site enters as a ket and a bra absorbed one after the other, physical legs left
    **open** in the numerator so ``h`` closes them and closed against each other in the
    denominator. One bond stays open for ``tenet.full_trace``. With :func:`_halves` this is
    a reduced-density-matrix API at one geometry, which is why the library's environment
    module stops short of it.

    Simplification: ``h`` closes two open physical legs (froSTspin ``contract_open_corner``)
    rather than being inserted into the ket (YASTN's ``DoublePepsTensor(op=...)``), whose
    route is cheaper only for a *one-site* operator: ``h`` is two-site here, so inserting it
    means an SVD of ``h``, a new bond space and a truncation decision -- a third
    factorization in a file that already teaches two.
    """
    c0, e0, bond = env
    ket, bra = layers(c4v(a))
    r = ring(*unrolled(c0, e0, double_layer(ket, bra), bond, k=k))
    left, right = _halves(r, ket, bra, "Ww", "Xx")
    numerator = tenet.full_trace(
        tenet.einsum("WwbckhrR,XxchrR,WXwx->kb", left, right, h, optimize=PATH)
    )
    left, right = _halves(r, ket, bra)
    denominator = tenet.full_trace(tenet.einsum("bckhrR,chrR->kb", left, right))
    return numerator / denominator


def step(a: SymmetricTensor, h: SymmetricTensor, env, lr: float, k: int = 4):
    """One plain SGD step on ``a``, ``vmc_mps.step``-style. ``optax`` would slot in here."""
    import jax

    value, grad = jax.value_and_grad(energy)(a, h, env, k)
    return jax.tree.map(lambda p, g: p - lr * g, a, grad), value


def main(chi_ising: int = 16, chi_ipeps: dict | None = None, k: int = 4, steps: int = 3):
    """Print both halves: Ising against Onsager with its gradient, then the iPEPS trace."""
    import jax

    # the pytree registration plus tenet.ad's broadened SVD/eigh VJPs, which the
    # degenerate CTM spectra below need; `ad=True` is opted into by name because
    # that half is process-global (tenet.ad's module docstring)
    tenet.enable_jax(ad=True)

    for beta in (0.3, 0.4, 0.5):
        env = converge(*single_layer_ctm(ising_bulk(beta)), chi=chi_ising)
        bf = float(beta_free_energy(beta, env, k=k))
        grad = float(jax.grad(beta_free_energy)(beta, env, k))
        print(
            f"ising beta={beta:.2f}  beta*f={bf:+.10f}  onsager={onsager(beta):+.10f}  "
            f"rel={abs(bf / onsager(beta) - 1):.2e}  d(beta f)/dbeta={grad:+.8f}"
        )

    for provider in ("u1", "su2"):
        a, h = build_ipeps(provider), build_h(provider)
        env = converge(*double_layer_ctm(c4v(a)), chi=(chi_ipeps or CHI_IPEPS)[provider])
        trace = []
        for _ in range(steps):
            a, value = step(a, h, env, lr=0.01, k=k)
            trace.append(float(value))
        print(f"ipeps {provider}: " + " ".join(f"{v:+.8f}" for v in trace))


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)  # tests/conftest.py does this for the suite
    main()
