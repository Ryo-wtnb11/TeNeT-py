"""Differentiable CTMRG: classical Ising against Onsager, then a U(1)/SU(2) iPEPS gradient.

Run it standalone::

    uv run --extra jax python examples/ctmrg.py

Two physical problems, **one** CTMRG core, because the core does not care which rank-4
bulk tensor it is fed -- which is itself part of what this file demonstrates:

* the classical 2D Ising partition function, whose free energy per site has a closed
  form (Onsager) and whose internal energy ``d(beta f)/d beta`` is therefore an oracle
  for ``jax.grad`` through the unrolled sweeps;
* a single-site U(1) (or SU(2)) iPEPS with a random symmetric two-site ``h``, which
  exercises graded truncation, ``svd(bond=)`` across sectors and multiplet
  degeneracies -- and makes **no benchmark-energy claim**, see below.

The pairing #77 designed and ``examples/vmc_mps.py::compress`` only documents:
:func:`converge` runs ``svd_truncated`` **outside** ``jax.grad`` (it decides a bond
:class:`~tenet.GradedSpace` from singular *values*, so it raises
``tenet.StructureChangingError`` under any trace), and :func:`unrolled` runs
``svd(bond=)`` **inside** it at exactly that frozen bond -- shape-static, one trace,
differentiable. The ``GradedSpace`` is the only thing that crosses the boundary, and
it is metadata: frozen, hashable, array-free, a legitimate jit cache key.

**The Ising half is Z2-graded**, on the bosonic ``Z2`` provider (#104), for the reason
YASTN's CTMRG Ising example passes ``sym='Z2'``: it stops a finite-chi environment from
breaking the symmetry spuriously in the ordered phase, which is what lets this file run
at ``beta > beta_c = ln(1+sqrt 2)/2`` against Onsager at all. Two further things the
grading buys, both asserted in ``tests/integration/test_ctmrg.py``: zero magnetization
becomes *structural* -- a spin insertion is a Z2-odd tensor, which no invariant
``SymmetricTensor`` can hold, so ``from_dense`` refuses it and the refusal is the
statement -- and the ordered-phase corner spectrum acquires **exact** two-fold
degeneracy, every partner pair carrying opposite parity. Because that doubling is
*cross*-sector and ``tenet.ad`` broadens *per coupled sector*, the graded run never hands
one SVD a degenerate pair: grading is what removes the ``NaN``, not what creates it.

The grading changed no arithmetic. The symmetric splitting
``W = [[sqrt cosh b, sqrt sinh b], [sqrt cosh b, -sqrt sinh b]]`` *is* the parity change
of basis -- ``W[s,0]`` is independent of ``s`` and ``W[s,1]`` is odd under ``s -> -s`` --
so the sum over ``s`` already killed every entry with an odd number of odd legs. The
example was not missing a symmetry; it was declining to declare one.

**The iPEPS half is a plumbing result, not a physics result, and cannot be otherwise
with a one-site unit cell.** Liao et al. get a single-site AFM Heisenberg cell by
rotating one sublattice by pi about y, which turns ``S^x S^x - S^y S^y`` into
``(S^+S^+ + S^-S^-)/2`` -- an operator that changes ``S^z_tot`` by +-2 and so *destroys
the U(1) the ansatz is graded by*. The alternatives are a two-site unit cell (out of
scope) or dropping the symmetry (which deletes the reason this half exists). So it
follows the ``examples/vmc_mps.py`` precedent exactly: random symmetric ``h``, no
comparison against ``-0.669437(5)``, said out loud right here.

Other deliberate simplifications, each with its ceiling:

ponytail: **truncated backprop through K unrolled moves, never the implicit fixed
point** (PRX 9, 031041 Sec. III C). The implicit route is a second numerical framework
inside a VJP -- its own tolerance, iteration cap and data-dependent exit -- which
cannot warn under a trace. Unrolling from an already-converged environment costs K
bounded steps and lets the tests *measure* the K-dependence instead of assuming it.
Upgrade path: only if a criterion here shows K=8 still moving.

ponytail: **no gradient checkpointing.** At K=4 and chi=16 the tape fits;
``jax.checkpoint`` on :func:`move` is the one-line addition when it does not.

ponytail: **one C4v move, not four directional ones.** Both bulk tensors are symmetric
under the C4v of the square lattice with a 1x1 unit cell, so one corner ``c`` and one
edge ``e`` describe the whole environment and the left/up/right/down moves are the same
function. Four directional moves is what a multi-site unit cell needs; that is the
upgrade path, and it is also what buys the sublattice-rotated Heisenberg energy.

ponytail: **no pre-QR before the projector SVD.** YASTN takes an intermediate QR
(``use_qr=True``) for stability. ``tenet.linalg.qr`` exists and the composition is three
lines; at chi <= 16 in float64 nothing here has lost digits. Add it when a criterion
does.

ponytail: **``svd``, not ``eigh``, for the projector**, even though C4v CTMRG classically
diagonalizes a Hermitian corner: #77 left ``eigh(t, bond=)`` out of scope, so the
fixed-bond differentiable route exists only for ``svd``. tensorgrad takes the same route.

ponytail: **``tenet.cast`` (#92) is mentioned and not used.** Building an SU(2) ansatz
and casting it to U(1) is attractive and is a third concept in a file that already has
two models; the SU(2) provider is instead run through the *same* iPEPS path via a
``provider`` parameter, ``vmc_mps.py``-style.

**Leg conventions**, the part worth reading before the code:

* bulk ``(l OUT, u OUT, r IN, d IN)`` -- ``l``/``u`` share a side and ``r``/``d`` share
  one, so the C4v diagonal mirror is the plain transpose ``(1, 0, 3, 2)`` and *one*
  edge tensor serves both the top and the left of a corner;
* corner ``c`` ``(X OUT, X IN)`` -- for **both** halves -- and edge ``e``
  ``(X IN, X OUT, V IN)`` for the Ising bulk, ``(X IN, X OUT, V_ket IN, V_bra IN dual)``
  for the iPEPS. The iPEPS edge carries the ket bond and its conjugate as two separate
  legs and **never fuses them** (#107, froSTspin ``ctm_environment.py``:16-33): the
  ``dual=True`` is the leg bend the fused convention used to hide, only the ``X`` leg is
  ever truncated, and the site enters as a ket and then a bra rather than as a product.
  Both edges are oriented maps on the environment space, so the boundary ring closes as
  ``c -> e -> ... -> adjoint(c) -> adjoint(e) -> ...``: the far corners and edges of the
  ring are the ``tenet.adjoint`` of the near ones, which for a real environment is what
  "the same tensor seen from the other side" means;
* the enlarged corner is a *bilinear form*, not a map -- its two index groups are
  related by the diagonal mirror, so they sit on the same side -- and the single leg
  bend that ``svd(axes=...)`` performs to make it a map is exactly that mirror. It is
  why the projector ``u`` contracts the *incoming* group of an enlarged edge while
  ``adjoint(u)`` contracts the *outgoing* one. The groups are the tensor's two halves,
  which is why :func:`move` partitions at ``ndim // 2`` rather than branching on the
  model: rank 4 for Ising, rank 6 for the iPEPS.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import autoray as ar

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, U1, Z2, SU2Sector, U1Sector, Z2Sector

BETA_C = 0.4406867935097714  # ln(1 + sqrt(2)) / 2

# ``opt_einsum``'s greedy path, for the ring contractions only. ponytail: greedy, not
# "auto". At ten-plus operands "auto" runs the dynamic-programming search, and for these
# rings that *search* costs an order of magnitude more than the contraction it plans --
# measured 4.5 s against 0.4 s for the two-site energy. The contraction is identical
# either way; only the plan differs. Upgrade path: an explicit path, or cotengra, when a
# network here is large enough for greedy's plan to be the expensive part.
PATH = "greedy"

# Environment dimension for the iPEPS half, per provider. ponytail: not one number.
# ``svd_truncated``'s ``max_bond`` bounds the *dense* bond, and for SU(2) that is
# ``sum_c (2j+1) m_c``: a budget of 4 buys a singlet and a triplet and then stops in the
# middle of the next multiplet -- exactly the split Francuz-Schuch-Vanhecke's Appendix C
# warns about, and measurably slower both to converge and to differentiate than the budget
# of 6 that closes it. U(1) has no multiplets and 4 is plenty.
CHI_IPEPS = {"u1": 4, "su2": 6}

# Physical and virtual spaces for the iPEPS half, one entry per provider, as
# ``vmc_mps.SPACES`` does. The virtual space must contain the unit sector or a
# single-site tensor with a spin-1/2 physical leg has no allowed block at all.
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


# --- leaving the tensor world ------------------------------------------------------


def scalar(t: SymmetricTensor):
    """The categorical trace of a rank-2 map ``(X OUT, X IN)``: ``sum_c d_c tr(M_c)``.

    Every closed network here contracts down to a ring with one bond left open, because
    ``SymmetricTensor`` has no rank 0 (``tensordot`` refuses a contraction that leaves
    no free leg). Closing that last bond is a trace, and a trace of a symmetric morphism
    carries the same ``qdim`` weight :func:`tenet.norm` carries -- the reduced index in
    sector ``c`` stands for ``d_c`` dense basis states. This is where the tensor world is
    left explicitly, exactly as ``vmc_mps.scalar`` does it.
    """
    qdim = t.provider.qdim
    mats = tenet.to_matrices(t)
    return sum(qdim(c) * ar.do("trace", m) for c, m in mats.items())


def renormalized(t: SymmetricTensor) -> SymmetricTensor:
    """``t / ||t||``. Every environment tensor is renormalized after every move.

    Not cosmetic: ``tenet.ad``'s Lorentzian ``epsilon`` is in units of sigma squared and
    the PRX default ``1e-12`` assumes an ``O(1)``-normalized spectrum. A CTMRG that does
    not renormalize sees the corner norm grow like the partition function itself, and
    the broadening would then be either a no-op or a sledgehammer depending on ``beta``.
    """
    return t / tenet.norm(t)


# --- the two bulk tensors ----------------------------------------------------------


def ising_bulk(beta):
    """Classical 2D Ising partition-function tensor, legs ``(l OUT, u OUT, r IN, d IN)``.

    ``a[l,u,r,d] = sum_s W[s,l] W[s,u] W[s,r] W[s,d]`` with ``W W^T`` the bond Boltzmann
    matrix ``[[e^b, e^-b], [e^-b, e^b]]``, i.e. the standard symmetric splitting
    ``W = [[sqrt cosh b, sqrt sinh b], [sqrt cosh b, -sqrt sinh b]]``.

    That ``W`` is *already the parity basis*: ``W[s, 0]`` does not depend on ``s`` and
    ``W[s, 1]`` is odd under ``s -> -s``, so summing over ``s`` gives
    ``a[l,u,r,d] = 2 (cosh b)^{n0/2} (sinh b)^{n1/2} [n1 even]`` with ``n1`` the number of
    odd legs. Eight of the sixteen entries are *structurally* zero, and the ``Z2`` legs
    over ``{Z2Sector(0): 1, Z2Sector(1): 1}`` are what stops us storing them.

    ``beta`` may be a *traced scalar*: the block is built through ``autoray``, so
    ``jax.grad`` has something to differentiate and no backend is hard-coded. That is why
    ``from_dense`` is called with ``atol=math.inf`` -- #82's documented "project, don't
    check" spelling, because the symmetry check is a concrete-value question and would
    raise under a trace. The check is not lost, it is *moved*: an untraced test in
    ``tests/integration/test_ctmrg.py`` runs the same array through ``from_dense`` at the
    **default** relative ``atol`` and it passes.

    ponytail: dense-then-gather at setup on a 16-element array. The ceiling is
    ``prod dim_i`` and there is no upgrade path worth naming at this size.
    """
    c, s = ar.do("sqrt", ar.do("cosh", beta)), ar.do("sqrt", ar.do("sinh", beta))
    w = ar.do("stack", (ar.do("stack", (c, s)), ar.do("stack", (c, -s))))
    block = ar.do("einsum", "sl,su,sr,sd->lurd", w, w, w, w)
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    legs = (Leg(space, OUT), Leg(space, OUT), Leg(space, IN), Leg(space, IN))
    return SymmetricTensor.from_dense(block, legs, atol=math.inf)


def c4v(a: SymmetricTensor) -> SymmetricTensor:
    """Symmetrize an iPEPS tensor under the C4v diagonal mirror ``l <-> u``, ``r <-> d``.

    Required, not cosmetic: one corner and one edge tensor describe the whole environment
    only if the bulk is mirror-symmetric, and a random ansatz is not. Because ``l``/``u``
    share a side and ``r``/``d`` share one, the mirror is the plain transpose
    ``(0, 2, 1, 4, 3)`` -- no bend, no bending coefficient, and the average of a tensor
    with its mirror is again a valid symmetric tensor. It is also linear, so it
    differentiates for free and a finite difference on one block sees exactly what
    ``jax.grad`` sees.

    ponytail: symmetrizing the *ansatz* rather than carrying four directional moves. A
    general (non-mirror-symmetric) single-site iPEPS needs the four-move environment, and
    that is the same upgrade the multi-site unit cell needs.
    """
    return (a + tenet.transpose(a, (0, 2, 1, 4, 3))) / 2


def ipeps_layers(a: SymmetricTensor) -> tuple[SymmetricTensor, SymmetricTensor]:
    """``(ket, bra)``: the two rank-5 layers the environment absorbs one after the other.

    * ket ``c4v(a)``, legs ``(P OUT, l OUT, u OUT, r IN, d IN)``;
    * bra ``repartition(adjoint(ket), (1, 2), (0, 3, 4))``, legs
      ``(L OUT dual, U OUT dual, s IN, R IN dual, D IN dual)``.

    **Their product is never formed** (#107). The old ``ipeps_bulk``/``ipeps_bulk_open``
    contracted these two into a rank-10 intermediate and fused its four bra/ket pairs into
    one bond per side; at SU(2) chi=6 that intermediate carries 841 blocks and cost 18 546
    ``ar.do`` calls before any environment tensor had been touched. Both reference
    implementations decline to build it -- froSTspin absorbs ``A`` then ``A.dagger()``
    (``ctmrg/ctm_contract.py``:42,53), YASTN keeps ``bra`` and ``ket`` as separate fields
    of a ``DoublePepsTensor`` and contracts them one at a time
    (``fpeps/envs/_env_contractions.py``:221-224) -- and so does :func:`ipeps_absorber`.

    The bra is bent here, at rank 5 and 12 blocks (#105's fix B, kept): bending is what
    flips ``dual``, and it is the flip that makes the bra's bonds meet the ``dual=True``
    bra bonds of the rank-4 environment edge. The old convention hid that flip inside a
    ``fuse`` of a ``(V, V*)`` pair; naming it is half the point of the redesign.
    """
    ket = c4v(a)
    return ket, tenet.repartition(tenet.adjoint(ket), (1, 2), (0, 3, 4))


def build_ipeps(provider: str = "u1", seed: int = 1) -> SymmetricTensor:
    """A random single-site iPEPS, legs ``(P OUT, l OUT, u OUT, r IN, d IN)``."""
    phys, virt = SPACES[provider]
    legs = (Leg(phys, OUT), Leg(virt, OUT), Leg(virt, OUT), Leg(virt, IN), Leg(virt, IN))
    return SymmetricTensor.random(legs, seed=seed).to_backend("jax")


def build_h(provider: str = "u1", seed: int = 100) -> SymmetricTensor:
    """A random two-site operator on legs ``(P OUT, P OUT, P IN, P IN)``.

    Symmetric by construction, hence ``Sz``-conserving for U(1) and a scalar under SU(2);
    a plumbing operator, exactly as ``vmc_mps.build_h`` is.
    """
    phys = SPACES[provider][0]
    legs = (Leg(phys, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(phys, IN))
    return SymmetricTensor.random(legs, seed=seed).to_backend("jax")


# --- CTMRG: one C4v move -----------------------------------------------------------


class Absorb(NamedTuple):
    """How one model grows an environment: ``corner(c, e)`` and ``edge(e, p)``.

    Everything else -- :func:`move`, :func:`converge`, :func:`unrolled`,
    :func:`spectrum` -- is model-independent and shared. That is the whole reason this
    type exists, and it is why it is allowed to: it abstracts over *two* implementations
    whose environments have different ranks, not over one. The alternative, two full
    ``move``/``converge`` stacks, would duplicate the #77 outside-decide/inside-
    differentiate pairing, which is the single most load-bearing thing in this file.
    """

    corner: Callable[[SymmetricTensor, SymmetricTensor], SymmetricTensor]
    edge: Callable[[SymmetricTensor, SymmetricTensor], SymmetricTensor]


def ising_absorber(bulk: SymmetricTensor) -> Absorb:
    """The Ising half's absorber: one rank-4 Boltzmann tensor, one bond per side.

    Unchanged arithmetic -- both bodies are #102's ``enlarged_corner`` and
    ``enlarged_edge`` moved here verbatim. #107's redesign does not apply to this half
    and does not touch it: there is no bra and no ket, so there is no pair to fuse and
    nothing that was hiding a leg bend.
    """

    def corner(c: SymmetricTensor, e: SymmetricTensor) -> SymmetricTensor:
        """The 2x2 object the projector diagonalizes: corner, two edges, one bulk tensor.

        Legs ``(X OUT, V IN, X IN, V IN)``. The two index pairs are the diagonal mirror of
        each other, so this is a bilinear form rather than a map -- see the module
        docstring.
        """
        return tenet.einsum("ab,ace,fbg,gehi->chfi", c, e, e, bulk)

    def edge(e: SymmetricTensor, p: SymmetricTensor) -> SymmetricTensor:
        """One edge with one bulk tensor absorbed -- legs ``(X IN, X OUT, V OUT, V IN,
        V IN)`` -- then projected with ``p`` on its incoming pair and ``adjoint(p)`` on
        its outgoing one."""
        big_e = tenet.einsum("abe,gehi->abghi", e, bulk)
        return tenet.einsum("abghi,agx,bhy->xyi", big_e, p, tenet.adjoint(p))

    return Absorb(corner, edge)


def ipeps_absorber(ket: SymmetricTensor, bra: SymmetricTensor) -> Absorb:
    """The iPEPS half's absorber: **environment first, then the ket, then the bra**.

    This is froSTspin's ``contract_enlarged_corner`` order (``ctmrg/ctm_contract.py``:
    ``ul = T1 @ C1``, ``ul = ul @ T4``, then ``ul = A @ ul`` at :42 and
    ``ul = A.permute(...).dagger() @ ul`` at :53, which closes the physical legs at the
    moment the bra enters) and YASTN's, mirrored (``fpeps/envs/_env_contractions.py``:
    ``unfuse_legs`` at :221 splits the environment vector's ``[t t']``/``[l l']`` back
    into separate bra and ket bonds, then :223 contracts the bra and :224 the ket).
    YASTN *can* materialize the product -- ``DoublePepsTensor.fuse_layers()``,
    ``_doublePepsTensor.py``:291-307 -- and makes you ask for it by name; nothing here
    asks. The peak is froSTspin's ``2*a*d*chi**2*D**4`` (the comment at :52): rank 6 and
    51 blocks at SU(2) chi=6, against the deleted rank-10's 841, and never ``d**2 D**8``.

    The edge is **projected before it absorbs**, per ``ctm_renormalize.py``:145-166
    (``nT = T @ P``, then ``A``, then ``A.dagger()``, then ``Pt``), which is why no
    rank-8 enlarged edge is materialized either: the environment leg is cut to ``chi'``
    first and every later step carries it instead of a full ``chi``.
    """

    def corner(c: SymmetricTensor, e: SymmetricTensor) -> SymmetricTensor:
        """Rank 6, legs ``(X OUT, r_ket, r_bra, X IN, d_ket, d_bra)`` -- froSTspin's
        ``contract_enlarged_corner`` return, ``permute((2,0,4),(3,1,5))``, in tenet
        spelling. Its two index triples are the diagonal mirror of each other, exactly as
        the Ising corner's two pairs are."""
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


def ising_ctm(bulk: SymmetricTensor) -> tuple[Absorb, SymmetricTensor, SymmetricTensor]:
    """``(absorber, c, e)`` for a rank-4 bulk tensor: one virtual bond per edge."""
    return ising_absorber(bulk), *init_env(bulk, Leg(bulk.legs[0].space, IN))


def ipeps_ctm(a: SymmetricTensor) -> tuple[Absorb, SymmetricTensor, SymmetricTensor]:
    """``(absorber, c, e)`` for a single-site iPEPS. The edge is **rank 4**.

    Legs ``(X IN, X OUT, V_ket IN, V_bra IN dual)``: the ket bond and its conjugate as two
    separate legs, never fused. ``dual=True`` on the bra bond is the bend flip that the
    deleted ``_fuse_pair`` used to hide inside a fused ``(V, V*)`` pair, and writing it
    down is what lets :func:`ipeps_absorber` meet the bent bra without a fuse. froSTspin
    makes the same choice -- ``ctmrg/ctm_environment.py``:16-33 builds every ``T`` with
    signature ``[..., site_tensor.signature[k], ~site_tensor.signature[k], ...]`` -- and
    it is why ``T`` can stay rank 4 forever: only the ``X`` leg is ever truncated, the two
    ``D`` bonds never are.

    Strictly cheaper than the deleted route, too: seeding an environment no longer needs a
    double layer to exist first.
    """
    ket, bra = ipeps_layers(a)
    virt = ket.legs[1].space
    seed = init_env(ket, Leg(virt, IN), Leg(virt, IN, dual=True))
    return ipeps_absorber(ket, bra), *seed


def init_env(site: SymmetricTensor, *bonds: Leg) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Corner ``c`` and edge ``e`` on a *one-dimensional* environment space.

    ``bonds`` are the edge's virtual legs: one for the Ising bulk, two -- the ket bond and
    its dual bra partner -- for an iPEPS. ``site`` supplies only the provider, a block to
    match dtype against and the backend.

    The environment then grows one bulk leg per move -- ``X -> X (x) V`` truncated to
    ``chi`` -- which is the original "grow the lattice out of a corner" reading of CTMRG
    and needs no partial trace of the bulk to seed it. The corner is the identity on the
    unit sector and the edge is all ones, i.e. YASTN's free boundary.

    **All ones rather than a random draw, and for the Ising bulk the grading now makes
    that structural.** In the ``W`` basis this file splits the Boltzmann weight into,
    index 0 of a bulk leg is the even (``cosh``) component and index 1 the odd (``sinh``)
    one. A seed whose even component is small is a boundary made almost entirely of
    domain walls: it has almost no overlap with the dominant even eigenvector, and CTMRG
    then spends dozens of sweeps climbing out of the odd sector -- measurably, a per-sweep
    contraction of 0.97 instead of 0.75 at ``beta = 0.4``. On a one-dimensional
    *unit-sector* environment space the only allowed edge block is the even one, so under
    the ``Z2`` grading the seed is even by construction rather than by luck; the ungraded
    iPEPS route still relies on ``ones_like`` for the same effect.

    ponytail: a 1-dimensional seed, not YASTN's ``init='dl'`` partial trace and not
    ``tenet.random_isometry``. The isometry seed is what a ``chi > D**2`` start needs,
    where growing from one dimension takes an extra sweep or two to fill the space;
    ``tenet.isometry``/``random_isometry`` slot straight in at that point.
    """
    unit = GradedSpace.new(site.provider, {site.provider.unit: 1})
    c = tenet.identity((Leg(unit, OUT),), like=site.blocks[0])
    e = SymmetricTensor.zeros((Leg(unit, IN), Leg(unit, OUT), *bonds))
    return c, e.apply_blocks(lambda b: ar.do("ones_like", b)).to_backend(site.backend)


def move(
    c: SymmetricTensor,
    e: SymmetricTensor,
    absorb: Absorb,
    *,
    bond: GradedSpace | None = None,
    chi: int | None = None,
) -> tuple[SymmetricTensor, SymmetricTensor, GradedSpace]:
    """One C4v move. ``bond=`` is the traceable half, ``chi=`` the structure-deciding one.

    With ``bond=None`` the projector comes from ``tenet.linalg.svd_truncated``, which reads
    the singular *values* to decide which sectors survive and therefore raises
    ``tenet.StructureChangingError`` under ``jax.jit``/``jax.grad``. With ``bond=B`` it comes
    from ``tenet.linalg.svd(..., bond=B)``: the same factorization projected onto a space the
    caller decided out here, fully shape-static and differentiable.

    **The new corner is ``s`` itself**, because ``s = adjoint(u) . big_c . v`` by definition:
    projecting the enlarged corner with ``u`` on one side and ``v`` on the other *is* the
    singular-value matrix, and forming it explicitly would be the same numbers through two
    more contractions. The new edge takes ``u`` on its incoming pair and ``adjoint(u)`` on
    its outgoing one -- the two pairs are related by the leg bend inside ``svd(axes=...)``,
    which is the C4v diagonal mirror written in leg metadata.

    ponytail: one isometry, ``u``, for a *bilinear* enlarged corner whose ``u`` and ``v``
    coincide only when it is positive. The Ising bulk's corner is, which is why that half
    reproduces Onsager to float64; a double-layer corner with indefinite spectrum gets a
    consistent contraction whose corner and edge differ by a diagonal of signs, which is
    one more reason the iPEPS half claims plumbing and not physics. Fixing it wants
    ``eigh(t, bond=)`` -- #77's explicit non-goal -- or four directional moves.

    ``absorb`` is the only thing that knows which model this is: the Ising corner is rank
    4 and the iPEPS one rank 6, and the partition is ``ndim // 2`` rather than a branch,
    because a bilinear form's two index groups are always its two halves.
    """
    big_c = absorb.corner(c, e)
    n = big_c.ndim // 2  # (0..n-1 | n..2n-1): 2 for Ising, 3 for iPEPS
    axes = (tuple(range(n)), tuple(range(n, 2 * n)))
    if bond is None:
        p, s, _ = tenet.linalg.svd_truncated(big_c, axes, max_bond=chi)
    else:
        p, s, _ = tenet.linalg.svd(big_c, axes, bond=bond)
    return renormalized(s), renormalized(absorb.edge(e, p)), p.legs[-1].space


def spectrum(c: SymmetricTensor) -> list[float]:
    """The corner spectrum, descending. The corner is diagonal by construction (it *is*
    the singular-value matrix of the enlarged corner), so this reads its diagonal."""
    qdim = c.provider.qdim
    out = [
        float(v)
        for sector, m in tenet.to_matrices(c).items()
        for v in ar.do("diag", m) * qdim(sector) ** 0.5
    ]
    return sorted(out, reverse=True)


def _spectrum_change(old: list[float], new: list[float]) -> float:
    """Max entrywise change, zero-padded to the longer spectrum. While the environment is
    still growing the two have different lengths, and the padding makes that a large
    change rather than an error -- which is what it is."""
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
) -> tuple[SymmetricTensor, SymmetricTensor, GradedSpace, list[float]]:
    """**Outside** ``jit``/``grad``. Sweep to a fixed corner spectrum.

    Takes the ``(absorber, c, e)`` triple :func:`ising_ctm` / :func:`ipeps_ctm` return, so
    ``converge(*ising_ctm(bulk), chi=16)`` is the whole call.

    Returns ``(c, e, bond, history)`` where ``bond`` is the frozen environment
    :class:`~tenet.GradedSpace` -- the one and only thing that crosses into the
    differentiated region -- and ``history`` is the per-sweep corner-spectrum change, so a
    caller can *assert* convergence rather than assume it.
    """
    bond, previous, history = None, spectrum(c), []
    for _ in range(max_sweeps):
        c, e, bond = move(c, e, absorb, chi=chi)
        current = spectrum(c)
        history.append(_spectrum_change(previous, current))
        previous = current
        if history[-1] < tol:
            break
    return c, e, bond, history


def unrolled(
    c: SymmetricTensor,
    e: SymmetricTensor,
    absorb: Absorb,
    bond: GradedSpace,
    k: int = 4,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """**Inside** ``jax.jit(jax.grad(...))``. Exactly ``k`` fixed-structure moves.

    A static Python loop: at ``k = 4`` there is nothing for ``jax.lax.scan`` to buy, and
    the loop being static is what makes the whole region one trace.
    """
    for _ in range(k):
        c, e, _ = move(c, e, absorb, bond=bond)
    return c, e


# --- observables -------------------------------------------------------------------


def _ring(c: SymmetricTensor, e: SymmetricTensor):
    """``(c, adjoint(c), e, adjoint(e))``. The far side of the boundary ring is the
    adjoint of the near side -- see the module docstring's leg conventions."""
    return c, tenet.adjoint(c), e, tenet.adjoint(e)


def log_kappa(beta, env, k: int = 4):
    """``ln`` of the partition function per site, from ``k`` unrolled moves at ``beta``.

    ``kappa = Z(L+1,L+1) Z(L,L) / Z(L+1,L) Z(L,L+1)``, Baxter's corner-transfer
    telescoping: four corners cover an ``L x L`` patch, adding four edges and one bulk
    tensor covers ``(L+1) x (L+1)``, and adding only the left and right edges covers
    ``L x (L+1)``. Every leg of every ring closes except one bond, which :func:`scalar`
    traces.

    ``env`` is ``(c, e, bond)`` from :func:`converge` at this ``beta``; the environment is
    the *initial condition* of the truncated backprop and carries no gradient, while the
    ``k`` moves inside do.
    """
    c0, e0, bond = env
    bulk = ising_bulk(beta)
    c, e = unrolled(c0, e0, ising_absorber(bulk), bond, k=k)
    cc, ca, ec, ea = _ring(c, e)
    z_c = scalar(tenet.einsum("ab,ac,dc,eb->de", cc, ca, cc, ca))
    z_h = scalar(tenet.einsum("ab,ac,dcf,ed,eg,ghf->hb", cc, ca, ea, cc, ca, ec, optimize=PATH))
    z_a = scalar(
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
    return ar.do("log", z_a * z_c / z_h**2)


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
    import numpy as np

    kk = 1.0 / np.sinh(2.0 * beta) ** 2
    theta = np.linspace(0.0, np.pi, points)
    integrand = np.log(
        np.cosh(2.0 * beta) ** 2 + np.sqrt(1.0 + kk**2 - 2.0 * kk * np.cos(2.0 * theta)) / kk
    )
    return -(np.log(2.0) / 2.0 + np.trapezoid(integrand, theta) / (2.0 * np.pi))


def _halves(ring, ket, bra, phys1: str = "", phys2: str = ""):
    """The 2x1 environment, split down the middle into two halves.

    ``left`` is the bottom-left corner, the left edge, the top-left corner, the first top
    and bottom edges and the first site: legs ``(*phys1, b, c, k, h, r, R)`` where
    ``b``/``k`` are the ring's one open bond, ``c``/``h`` the cut through the top and
    bottom rows and ``r``/``R`` the ket and bra bonds between the two sites. ``right`` is
    the mirror image. ``phys`` is ``""`` (the physical legs closed, the denominator) or
    ``"Ww"`` -- bra first, then ket -- for the numerator.

    Each half is built the way :func:`ipeps_absorber` builds a corner: environment first,
    then the ket, then the bra, so the peak is rank 7 (rank 8 with the physical legs open
    -- froSTspin's ``rdm.py``:30-69 ``contract_open_corner``, ``a*d*chi**2*D**4``) and the
    two double layers this used to need are never formed.

    ponytail: two hand-written halves instead of one twelve-operand equation. The
    contraction is identical; what changes is that the intermediates are rank 5 and rank 3
    by construction rather than by whatever path ``opt_einsum`` picks from *physical* leg
    sizes -- which, for a graded tensor whose sectors are unevenly filled, it picks badly
    and unpredictably: the same network was measured at 0.7 s and at 3.6 s for two SU(2)
    environments differing only in how ``chi`` split across sectors. Upgrade path: a
    contraction-path planner that costs a graded network by its *blocks*, which is M9.
    """
    cc, ca, ec, ea = ring
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

    The environment ring is four corners, two top edges, two bottom edges and one edge on
    each side; each site enters as a ket and a bra absorbed one after the other, with the
    physical legs left **open** in the numerator so that ``h`` closes them, and closed
    against each other in the denominator. One bond of the ring is left open for
    :func:`scalar`, as everywhere else.

    ponytail: ``h`` closes two open physical legs (froSTspin ``contract_open_corner``)
    rather than being inserted into the ket (YASTN's ``DoublePepsTensor(op=...)``,
    ``_env_contractions.py``:216-217). YASTN's route is genuinely cheaper for a *one-site*
    operator; ``h`` here is two-site, so inserting it means splitting it across the bond --
    an SVD of ``h``, a new bond space and a truncation decision, i.e. a third factorization
    in a file that already teaches two. Measured, the open route costs 2 315 ``ar.do``
    calls per corner against 18 546 for the double layer it replaces; there is nothing
    left to buy.
    """
    c0, e0, bond = env
    ket, bra = ipeps_layers(a)
    ring = _ring(*unrolled(c0, e0, ipeps_absorber(ket, bra), bond, k=k))
    left, right = _halves(ring, ket, bra, "Ww", "Xx")
    numerator = scalar(tenet.einsum("WwbckhrR,XxchrR,WXwx->kb", left, right, h, optimize=PATH))
    left, right = _halves(ring, ket, bra)
    denominator = scalar(tenet.einsum("bckhrR,chrR->kb", left, right))
    return numerator / denominator


def step(a: SymmetricTensor, h: SymmetricTensor, env, lr: float, k: int = 4):
    """One plain SGD step on ``a``, ``vmc_mps.step``-style. ``optax`` would slot in here."""
    import jax

    value, grad = jax.value_and_grad(energy)(a, h, env, k)
    return jax.tree.map(lambda p, g: p - lr * g, a, grad), value


def main(chi_ising: int = 16, chi_ipeps: dict | None = None, k: int = 4, steps: int = 3):
    """Print both halves: Ising against Onsager with its gradient, then the iPEPS trace."""
    import jax

    import tenet.ad
    import tenet.pytree  # noqa: F401  # registration is the import's side effect

    tenet.ad.install()

    for beta in (0.3, 0.4, 0.5):
        env = converge(*ising_ctm(ising_bulk(beta)), chi=chi_ising)[:3]
        bf = float(beta_free_energy(beta, env, k=k))
        grad = float(jax.grad(beta_free_energy)(beta, env, k))
        print(
            f"ising beta={beta:.2f}  beta*f={bf:+.10f}  onsager={onsager(beta):+.10f}  "
            f"rel={abs(bf / onsager(beta) - 1):.2e}  d(beta f)/dbeta={grad:+.8f}"
        )

    for provider in ("u1", "su2"):
        a, h = build_ipeps(provider), build_h(provider)
        env = converge(*ipeps_ctm(a), chi=(chi_ipeps or CHI_IPEPS)[provider])[:3]
        trace = []
        for _ in range(steps):
            a, value = step(a, h, env, lr=0.01, k=k)
            trace.append(float(value))
        print(f"ipeps {provider}: " + " ".join(f"{v:+.8f}" for v in trace))


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)  # tests/conftest.py does this for the suite
    main()
