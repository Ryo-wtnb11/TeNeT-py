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
* corner ``c`` ``(X OUT, X IN)`` and edge ``e`` ``(X IN, X OUT, V IN)``, i.e. both are
  oriented maps on the environment space, so the boundary ring closes as
  ``c -> e -> ... -> adjoint(c) -> adjoint(e) -> ...``: the far corners and edges of the
  ring are the ``tenet.adjoint`` of the near ones, which for a real environment is what
  "the same tensor seen from the other side" means;
* the enlarged corner is a *bilinear form*, not a map -- its two index pairs are
  related by the diagonal mirror, so they sit on the same side -- and the single leg
  bend that ``svd(axes=...)`` performs to make it a map is exactly that mirror. It is
  why the projector ``u`` contracts the *incoming* pair of an enlarged edge while
  ``adjoint(u)`` contracts the *outgoing* one.
"""

import math

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


def _fuse_pair(t: SymmetricTensor, i: int, j: int) -> SymmetricTensor:
    """Fuse axes ``i``, ``j`` (same side) into a leading leg. ``tenet.fuse`` wants the
    pair to be the first legs of its side, so transpose them there first."""
    order = (i, j, *(k for k in range(t.ndim) if k not in (i, j)))
    return tenet.fuse(tenet.transpose(t, order), (0, 1))


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


def ipeps_bulk(a: SymmetricTensor) -> SymmetricTensor:
    """Double layer of a single-site iPEPS ``a``, legs ``(l OUT, u OUT, r IN, d IN)``.

    ``a`` has legs ``(P OUT, l OUT, u OUT, r IN, d IN)`` and is :func:`c4v`-symmetrized
    first. One ``tenet.einsum`` call now that #67 landed contracts the physical leg of
    ``adjoint(a)`` against ``a``; one ``repartition`` puts every bra/ket pair on one side
    (bending flips ``dual``, which is exactly what makes ``V (x) V*`` fuse); four ``fuse``
    calls collapse the pairs.

    The bra layer is bent **before** the ``einsum``, not after (#105). Bending afterwards
    means bending a rank-10, 841-block intermediate: 4 808 ``repartition_plan`` terms, one
    ``ar.do("transpose", ...)`` each. Bending ``adjoint(a)`` while it is still rank 5 and
    12 blocks costs a handful, and then the ``repartition`` below is handed a tensor whose
    partition it already wants, so it takes the "no leg crosses" early return in
    ``src/tenet/ops/repartition.py`` (line 305) down to a plain transpose -- whose
    permutation is the identity, which ``permutation_plan`` (``src/tenet/ops/
    permutation.py``, line 116, case A) hands back for free. Both mechanisms were already
    there; the old order simply never reached them. Measured on the SU(2) chi=6 case:
    ``ipeps_bulk_open`` 23 163 -> 18 546 ``ar.do`` calls, 0.413 s -> 0.309 s.
    """
    a = c4v(a)
    bra = tenet.repartition(tenet.adjoint(a), (1, 2), (0, 3, 4))  # (L, U, s, R, D)
    dl = tenet.einsum("LUsRD,slurd->lLuUrRdD", bra, a)
    dl = tenet.repartition(dl, (0, 1, 2, 3), (4, 5, 6, 7))  # already this partition: free
    for i in range(4):
        dl = _fuse_pair(dl, i, i + 1)
    return tenet.transpose(dl, (3, 2, 1, 0))


def ipeps_bulk_open(a: SymmetricTensor) -> SymmetricTensor:
    """As :func:`ipeps_bulk` but keeping the two physical legs open.

    Legs ``(P IN, P OUT, l OUT, u OUT, r IN, d IN)``: the bra's ``P IN`` meets an
    operator's ``P OUT``, and the ket's ``P OUT`` meets its ``P IN``. Its virtual legs are
    identical to :func:`ipeps_bulk`'s, which is what lets one environment serve both.

    Same bend-early ordering as :func:`ipeps_bulk`, and this is where it pays: the rank-10
    intermediate is this function's, and the ``einsum`` output is written in exactly the
    order the ``repartition`` below wants so that it early-returns (``repartition.py``:305)
    to an identity transpose (``permutation.py``:116, case A).
    """
    a = c4v(a)
    bra = tenet.repartition(tenet.adjoint(a), (1, 2), (0, 3, 4))  # (L, U, S, R, D)
    dl = tenet.einsum("LUSRD,slurd->slLuUSrRdD", bra, a)
    dl = tenet.repartition(dl, (0, 1, 2, 3, 4), (5, 6, 7, 8, 9))  # already this partition
    # (s, l, L, u, U | S, r, R, d, D); each fuse lands at 0 and shifts the rest along
    for i, j in ((1, 2), (2, 3), (4, 5), (5, 6)):
        dl = _fuse_pair(dl, i, j)
    # now (dD, rR, uU, lL, s, S): put it back into (S, s, l, u, r, d)
    return tenet.transpose(dl, (5, 4, 3, 2, 1, 0))


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


def init_env(bulk: SymmetricTensor) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Corner ``c`` and edge ``e`` on a *one-dimensional* environment space.

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
    unit = GradedSpace.new(bulk.provider, {bulk.provider.unit: 1})
    virt = bulk.legs[0].space
    c = tenet.identity((Leg(unit, OUT),), like=bulk.blocks[0])
    e = SymmetricTensor.zeros((Leg(unit, IN), Leg(unit, OUT), Leg(virt, IN)))
    return c, e.apply_blocks(lambda b: ar.do("ones_like", b)).to_backend(bulk.backend)


def enlarged_corner(
    c: SymmetricTensor, e: SymmetricTensor, bulk: SymmetricTensor
) -> SymmetricTensor:
    """The 2x2 object the projector diagonalizes: corner, two edges and one bulk tensor.

    Legs ``(X OUT, V IN, X IN, V IN)``. The two index pairs are the diagonal mirror of
    each other, so this is a bilinear form rather than a map -- see the module docstring.
    """
    return tenet.einsum("ab,ace,fbg,gehi->chfi", c, e, e, bulk)


def enlarged_edge(e: SymmetricTensor, bulk: SymmetricTensor) -> SymmetricTensor:
    """One edge with one bulk tensor absorbed: legs ``(X IN, X OUT, V OUT, V IN, V IN)``."""
    return tenet.einsum("abe,gehi->abghi", e, bulk)


def move(
    c: SymmetricTensor,
    e: SymmetricTensor,
    bulk: SymmetricTensor,
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
    """
    axes = ((0, 1), (2, 3))
    big_c = enlarged_corner(c, e, bulk)
    if bond is None:
        p, s, _ = tenet.linalg.svd_truncated(big_c, axes, max_bond=chi)
    else:
        p, s, _ = tenet.linalg.svd(big_c, axes, bond=bond)
    new_e = tenet.einsum("abghi,agx,bhy->xyi", enlarged_edge(e, bulk), p, tenet.adjoint(p))
    return renormalized(s), renormalized(new_e), p.legs[-1].space


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
    bulk: SymmetricTensor, chi: int = 16, tol: float = 1e-10, max_sweeps: int = 100
) -> tuple[SymmetricTensor, SymmetricTensor, GradedSpace, list[float]]:
    """**Outside** ``jit``/``grad``. Sweep to a fixed corner spectrum.

    Returns ``(c, e, bond, history)`` where ``bond`` is the frozen environment
    :class:`~tenet.GradedSpace` -- the one and only thing that crosses into the
    differentiated region -- and ``history`` is the per-sweep corner-spectrum change, so a
    caller can *assert* convergence rather than assume it.
    """
    c, e = init_env(bulk)
    bond, previous, history = None, spectrum(c), []
    for _ in range(max_sweeps):
        c, e, bond = move(c, e, bulk, chi=chi)
        current = spectrum(c)
        history.append(_spectrum_change(previous, current))
        previous = current
        if history[-1] < tol:
            break
    return c, e, bond, history


def unrolled(
    c: SymmetricTensor,
    e: SymmetricTensor,
    bulk: SymmetricTensor,
    bond: GradedSpace,
    k: int = 4,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """**Inside** ``jax.jit(jax.grad(...))``. Exactly ``k`` fixed-structure moves.

    A static Python loop: at ``k = 4`` there is nothing for ``jax.lax.scan`` to buy, and
    the loop being static is what makes the whole region one trace.
    """
    for _ in range(k):
        c, e, _ = move(c, e, bulk, bond=bond)
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
    c, e = unrolled(c0, e0, bulk, bond, k=k)
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


def _halves(ring, site1, site2, phys1: str = "", phys2: str = ""):
    """The 2x1 environment, split down the middle into two halves.

    ``left`` is the bottom-left corner, the left edge, the top-left corner, the first top
    and bottom edges and the first site: legs ``(*phys1, b, c, k, h, m)`` where ``b``/``k``
    are the ring's one open bond, ``c``/``h`` the cut through the top and bottom rows and
    ``m`` the bulk leg between the two sites. ``right`` is the mirror image.

    ponytail: two hand-written halves instead of one twelve-operand equation. The
    contraction is identical; what changes is that the intermediates are rank 5 and rank 3
    by construction rather than by whatever path ``opt_einsum`` picks from *physical* leg
    sizes -- which, for a graded tensor whose sectors are unevenly filled, it picks badly
    and unpredictably: the same network was measured at 0.7 s and at 3.6 s for two SU(2)
    environments differing only in how ``chi`` split across sectors. Upgrade path: a
    contraction-path planner that costs a graded network by its *blocks*, which is M9.
    """
    cc, ca, ec, ea = ring
    left = tenet.einsum(
        f"ab,acp,ij,jku,iht,{phys1}upmt->{phys1}bckhm", cc, ec, ca, ec, ea, site1, optimize=PATH
    )
    right = tenet.einsum(
        f"cdq,de,fer,gf,hgs,{phys2}mqrs->{phys2}chm", ec, ca, ea, cc, ea, site2, optimize=PATH
    )
    return left, right


def energy(a: SymmetricTensor, h: SymmetricTensor, env, k: int = 4):
    """``<h> / <1>`` on a 2x1 patch, from ``k`` unrolled moves at the current ``a``.

    The environment ring is four corners, two top edges, two bottom edges and one edge on
    each side; the two sites are :func:`ipeps_bulk_open` double layers, so ``h`` closes
    their physical legs in the numerator and :func:`ipeps_bulk` closes them in the
    denominator. One bond of the ring is left open for :func:`scalar`, as everywhere else.
    """
    c0, e0, bond = env
    bulk = ipeps_bulk(a)
    ring = _ring(*unrolled(c0, e0, bulk, bond, k=k))
    # One tensor, not two: both sites of the 2x1 patch are the same double layer, and
    # calling ipeps_bulk_open twice built it twice and differentiated it twice (#105).
    site = ipeps_bulk_open(a)
    left, right = _halves(ring, site, site, "Ww", "Xx")
    numerator = scalar(tenet.einsum("Wwbckhm,Xxchm,WXwx->kb", left, right, h, optimize=PATH))
    left, right = _halves(ring, bulk, bulk)
    denominator = scalar(tenet.einsum("bckhm,chm->kb", left, right))
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
        env = converge(ising_bulk(beta), chi=chi_ising)[:3]
        bf = float(beta_free_energy(beta, env, k=k))
        grad = float(jax.grad(beta_free_energy)(beta, env, k))
        print(
            f"ising beta={beta:.2f}  beta*f={bf:+.10f}  onsager={onsager(beta):+.10f}  "
            f"rel={abs(bf / onsager(beta) - 1):.2e}  d(beta f)/dbeta={grad:+.8f}"
        )

    for provider in ("u1", "su2"):
        a, h = build_ipeps(provider), build_h(provider)
        env = converge(ipeps_bulk(a), chi=(chi_ipeps or CHI_IPEPS)[provider])[:3]
        trace = []
        for _ in range(steps):
            a, value = step(a, h, env, lr=0.01, k=k)
            trace.append(float(value))
        print(f"ipeps {provider}: " + " ".join(f"{v:+.8f}" for v in trace))


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)  # tests/conftest.py does this for the suite
    main()
