"""C4v corner-transfer-matrix renormalization: the half of this package that is traced.

Promoted from ``examples/ctmrg.py`` (#102/#104/#105/#107) by #114 with no arithmetic
change; the example keeps the physics -- the Boltzmann tensor, the C4v ansatz constraint,
the observables -- and this module keeps the environment machinery.

**This is the first module in ``src/tenet/`` that lives on both sides of a trace**, so
every public function below opens by saying which. The pairing is #77's:
:func:`ctmrg` runs ``svd_truncated`` **outside** ``jax.grad`` (it decides a bond
:class:`~tenet.GradedSpace` from singular *values*, so it raises
``tenet.StructureChangingError`` under any trace), and :func:`ctmrg_unrolled` runs
``svd(bond=)`` **inside** it at exactly that frozen bond -- shape-static, one trace,
differentiable. The ``GradedSpace`` is the only thing that crosses the boundary, and it
is metadata: frozen, hashable, array-free, a legitimate jit *cache key* and never a jit
*argument*.

**Leg conventions**, the part worth reading before the code:

* bulk ``(l OUT, u OUT, r IN, d IN)`` -- ``l``/``u`` share a side and ``r``/``d`` share
  one, so the C4v diagonal mirror is the plain transpose ``(1, 0, 3, 2)`` and *one*
  edge tensor serves both the top and the left of a corner;
* corner ``c`` ``(X OUT, X IN)`` -- for **both** models -- and edge ``e``
  ``(X IN, X OUT, V IN)`` for a single layer, ``(X IN, X OUT, V_ket IN, V_bra IN dual)``
  for a double one. The double-layer edge carries the ket bond and its conjugate as two
  separate legs and **never fuses them** (#107, froSTspin ``ctm_environment.py``:16-33):
  the ``dual=True`` is the leg bend the fused convention used to hide, only the ``X`` leg
  is ever truncated, and the site enters as a ket and then a bra rather than as a product.
  Both edges are oriented maps on the environment space, so the boundary ring closes as
  ``c -> e -> ... -> adjoint(c) -> adjoint(e) -> ...``: the far corners and edges of the
  ring are the ``tenet.adjoint`` of the near ones, which for a real environment is what
  "the same tensor seen from the other side" means -- see :func:`ring`;
* the enlarged corner is a *bilinear form*, not a map -- its two index groups are
  related by the diagonal mirror, so they sit on the same side -- and the single leg
  bend that ``svd(axes=...)`` performs to make it a map is exactly that mirror. It is
  why the projector ``u`` contracts the *incoming* group of an enlarged edge while
  ``adjoint(u)`` contracts the *outgoing* one. The groups are the tensor's two halves,
  which is why :func:`move` partitions at ``ndim // 2`` rather than branching on the
  model: rank 4 for a single layer, rank 6 for a double one.

**One C4v move, not four directional ones**, and no multi-site unit cell: one corner and
one edge describe the whole environment only for a mirror-symmetric bulk on a 1x1 cell.
That restriction is a documented *precondition* on the caller's tensor (see
:func:`double_layer_ctm`), never a symmetrization this module performs. The upgrade path
is named and not started: YASTN's ``EnvCTM`` (a ``Peps`` subclass with eight tensors per
site and four ``update_`` moves), and froSTspin's four ``contract_*`` wrappers over one
``contract_enlarged_corner``.

Four ceilings come with the code, unchanged from the example that had them.
ponytail: **truncated backprop through K unrolled moves, never the implicit fixed point**
(PRX 9, 031041 Sec. III C) -- the implicit route is a second numerical framework inside a
VJP, with its own tolerance and data-dependent exit, which cannot warn under a trace.
ponytail: **no gradient checkpointing** -- at K=4 and chi=16 the tape fits, and
``jax.checkpoint`` on :func:`move` is the one-line addition when it does not.
ponytail: **no pre-QR before the projector SVD** -- YASTN takes an intermediate QR
(``use_qr=True``) for stability; ``tenet.linalg.qr`` exists and the composition is three
lines, and at chi <= 16 in float64 nothing has lost digits yet.
ponytail: **``svd``, not ``eigh``, for the projector**, even though C4v CTMRG classically
diagonalizes a Hermitian corner: #77 left ``eigh(t, bond=)`` out of scope, so the
fixed-bond differentiable route exists only for ``svd``. tensorgrad takes the same route.
"""

from collections.abc import Callable
from typing import NamedTuple

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network.common import ones, spectrum

__all__ = [
    "Absorb",
    "CTMEnv",
    "ctmrg",
    "ctmrg_unrolled",
    "double_layer",
    "double_layer_ctm",
    "init_env",
    "layers",
    "move",
    "normalized",
    "ring",
    "single_layer",
    "single_layer_ctm",
]


class CTMEnv(NamedTuple):
    """A C4v environment and the frozen bond that crosses into the trace.

    This is the *outside* container, and it unpacks positionally, so
    ``c, e, bond = move(...)`` reads exactly as it did before the type existed.

    :func:`ctmrg_unrolled` deliberately does **not** take one. A ``NamedTuple`` is a registered
    pytree, so handing this object to ``jax.jit`` would try to flatten ``bond`` -- a
    :class:`~tenet.GradedSpace` -- into a leaf. ``bond`` is metadata: frozen, hashable,
    array-free, a jit *cache key* (``static_argnums``) and never a jit *argument*. The
    inside therefore takes ``c``, ``e`` and ``bond`` as three separate arguments, and that
    asymmetry is the #77 boundary written into the types instead of described in prose.
    """

    c: SymmetricTensor
    e: SymmetricTensor
    bond: GradedSpace


class Absorb(NamedTuple):
    """How one model grows an environment: ``corner(c, e)`` and ``edge(e, p)``.

    The two callable contracts, pinned here because they are now public:

    * ``corner(c, e) -> big_c`` of rank ``2n``, whose two index groups are the diagonal
      mirror of each other -- a bilinear form, not a map -- which is what licenses
      :func:`move` partitioning at ``ndim // 2``;
    * ``edge(e, p) -> new_e`` at the same rank as ``e``, on the projector's new bond.

    Write those two and a third model needs nothing from :func:`move`'s body.

    A ``Protocol`` or an ABC is **refused, not deferred**, for three reasons. (i) There
    are two real implementations of three ``tenet.einsum`` calls each; a ``Protocol``
    would ask a user to write a class with two methods where two closures suffice, and
    the only product would be a type name. (ii) The closures must be able to capture
    *traced* values -- a gradient with respect to the bulk tensor flows through
    ``single_layer(bulk)`` built inside a ``jax.jit`` region, which is load-bearing
    behaviour a stateful class would tempt someone to break. (iii) ``Absorb`` is passed at
    ``static_argnums``, and a ``NamedTuple`` of functions is hashable with no ``__hash__``
    to write.
    """

    corner: Callable[[SymmetricTensor, SymmetricTensor], SymmetricTensor]
    edge: Callable[[SymmetricTensor, SymmetricTensor], SymmetricTensor]


def normalized(t: SymmetricTensor) -> SymmetricTensor:
    """**Inside** ``jax.jit(jax.grad(...))``. ``t / ||t||``, after every move.

    This is a division by ``tenet.norm``, not the renormalization -- the projector
    truncation :func:`move` performs -- that the R in CTMRG names.

    Not cosmetic: ``tenet.ad``'s Lorentzian ``epsilon`` is in units of sigma squared and
    the PRX default ``1e-12`` assumes an ``O(1)``-normalized spectrum. A CTMRG that does
    not renormalize sees the corner norm grow like the partition function itself, and the
    broadening would then be either a no-op or a sledgehammer depending on the coupling.
    """
    return t / tenet.norm(t)


def ring(c: SymmetricTensor, e: SymmetricTensor) -> tuple[SymmetricTensor, ...]:
    """**Inside** ``jax.jit(jax.grad(...))``. ``(c, adjoint(c), e, adjoint(e))``.

    One line, and it is here for the convention rather than for the line: the far side of
    a C4v boundary ring is the ``tenet.adjoint`` of the near side, which is the module
    docstring's leg conventions in code, next to the code that assumes them.
    """
    return c, tenet.adjoint(c), e, tenet.adjoint(e)


def single_layer(bulk: SymmetricTensor) -> Absorb:
    """**Inside** ``jax.jit(jax.grad(...))``. An :class:`Absorb` for any rank-4 bulk.

    ``bulk`` is ``(l OUT, u OUT, r IN, d IN)`` and nothing here knows which model produced
    it -- an Ising Boltzmann tensor, a six-vertex weight, any single-layer transfer
    tensor. There is no bra and no ket, so there is no pair to fuse and nothing hiding a
    leg bend.
    """

    def corner(c: SymmetricTensor, e: SymmetricTensor) -> SymmetricTensor:
        """The 2x2 object the projector diagonalizes: corner, two edges, one bulk tensor.

        Legs ``(X OUT, V IN, X IN, V IN)``. The two index pairs are the diagonal mirror of
        each other, so this is a bilinear form rather than a map.
        """
        return tenet.einsum("ab,ace,fbg,gehi->chfi", c, e, e, bulk)

    def edge(e: SymmetricTensor, p: SymmetricTensor) -> SymmetricTensor:
        """One edge with one bulk tensor absorbed -- legs ``(X IN, X OUT, V OUT, V IN,
        V IN)`` -- then projected with ``p`` on its incoming pair and ``adjoint(p)`` on
        its outgoing one."""
        big_e = tenet.einsum("abe,gehi->abghi", e, bulk)
        return tenet.einsum("abghi,agx,bhy->xyi", big_e, p, tenet.adjoint(p))

    return Absorb(corner, edge)


def layers(ket: SymmetricTensor) -> tuple[SymmetricTensor, SymmetricTensor]:
    """**Inside** ``jax.jit(jax.grad(...))``. ``(ket, bra)`` for a rank-5 iPEPS ket.

    * ket ``(P OUT, l OUT, u OUT, r IN, d IN)``, returned untouched;
    * bra ``repartition(adjoint(ket), (1, 2), (0, 3, 4))``, legs
      ``(L OUT dual, U OUT dual, s IN, R IN dual, D IN dual)``.

    **Their product is never formed** (#107). The bend is done here, at rank 5, and it is
    *named* rather than hidden: bending is what flips ``dual``, and it is that flip which
    makes the bra's bonds meet the ``dual=True`` bra bonds of the rank-4 environment edge.
    The fused convention this replaced hid the same flip inside a ``fuse`` of a ``(V, V*)``
    pair. Public because a measurement built on this environment needs the same pair.
    """
    return ket, tenet.repartition(tenet.adjoint(ket), (1, 2), (0, 3, 4))


def double_layer(ket: SymmetricTensor, bra: SymmetricTensor) -> Absorb:
    """**Inside** ``jax.jit(jax.grad(...))``. An :class:`Absorb` for any rank-5 iPEPS ket:
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
    """**Outside** ``jit``/``grad``. Corner ``c`` and edge ``e`` on a *one-dimensional*
    environment space.

    ``bonds`` are the edge's virtual legs: one for a single-layer bulk, two -- the ket
    bond and its dual bra partner -- for a double layer. ``site`` supplies only the
    provider, the dtype and the backend. It is a free function taking ``*bonds`` rather
    than a :class:`CTMEnv` constructor because there *is* no frozen bond until the first
    :func:`move` decides one, and a third model with neither bond pattern can still use it.

    The environment then grows one bulk leg per move -- ``X -> X (x) V`` truncated to
    ``chi`` -- which is the original "grow the lattice out of a corner" reading of CTMRG
    and needs no partial trace of the bulk to seed it. The corner is the identity on the
    unit sector and the edge is all ones, i.e. YASTN's free boundary.

    **All ones rather than a random draw.** A seed whose component along the dominant
    eigenvector is small is a boundary the sweep has to climb out of -- measurably, a
    per-sweep contraction of 0.97 instead of 0.75 for the graded Ising bulk at
    ``beta = 0.4``. On a one-dimensional *unit-sector* environment space the only allowed
    block is the unit one, so under a grading the seed is right by construction rather
    than by luck.

    ponytail: a 1-dimensional seed, not YASTN's ``init='dl'`` partial trace and not
    ``tenet.random_isometry``. The isometry seed is what a ``chi > D**2`` start needs,
    where growing from one dimension takes an extra sweep or two to fill the space;
    ``tenet.isometry``/``random_isometry`` slot straight in at that point.
    """
    unit = GradedSpace.new(site.provider, {site.provider.unit: 1})
    c = tenet.identity((Leg(unit, OUT),), dtype=site.dtype, like=site.backend)
    return c, ones((Leg(unit, IN), Leg(unit, OUT), *bonds)).to_backend(site.backend)


def single_layer_ctm(bulk: SymmetricTensor) -> tuple[Absorb, SymmetricTensor, SymmetricTensor]:
    """**Outside** ``jit``/``grad``. ``(absorber, c, e)`` for a rank-4 bulk tensor.

    One virtual bond per edge, so ``ctmrg(*single_layer_ctm(bulk), chi=16)`` is the
    whole call.
    """
    return single_layer(bulk), *init_env(bulk, Leg(bulk.legs[0].space, IN))


def double_layer_ctm(ket: SymmetricTensor) -> tuple[Absorb, SymmetricTensor, SymmetricTensor]:
    """**Outside** ``jit``/``grad``. ``(absorber, c, e)`` for a single-site iPEPS ket.

    **Precondition, not policy:** ``ket`` must already be invariant under the diagonal
    mirror ``tenet.transpose(ket, (0, 2, 1, 4, 3))``, or one corner and one edge do not
    describe the environment. This module does not symmetrize it -- that would silently
    edit the caller's state, and a caller whose ansatz is genuinely C4v-symmetric would
    pay for a no-op. Symmetrizing an ansatz is an ansatz constraint and belongs to
    whoever chose the ansatz.

    The edge is **rank 4**, legs ``(X IN, X OUT, V_ket IN, V_bra IN dual)``: the ket bond
    and its conjugate as two separate legs, never fused (froSTspin
    ``ctmrg/ctm_environment.py``:16-33). It is why ``e`` can stay rank 4 forever -- only
    the ``X`` leg is ever truncated -- and why seeding an environment no longer needs a
    double layer to exist first.
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
) -> CTMEnv:
    """One C4v move, and the only function here that is on **both** sides of the trace.

    **Outside** ``jit``/``grad`` with ``chi=``: the projector comes from
    ``tenet.linalg.svd_truncated``, which reads the singular *values* to decide which
    sectors survive and therefore raises ``tenet.StructureChangingError`` under
    ``jax.jit``/``jax.grad``. That half decides a structure.

    **Inside** ``jax.jit(jax.grad(...))`` with ``bond=B``: the projector comes from
    ``tenet.linalg.svd(..., bond=B)`` -- the same factorization projected onto a space the
    caller decided out there, fully shape-static and differentiable. That half reuses one.

    **The new corner is ``s`` itself**, because ``s = adjoint(u) . big_c . v`` by
    definition: projecting the enlarged corner with ``u`` on one side and ``v`` on the
    other *is* the singular-value matrix, and forming it explicitly would be the same
    numbers through two more contractions. The new edge takes ``u`` on its incoming pair
    and ``adjoint(u)`` on its outgoing one -- the two pairs are related by the leg bend
    inside ``svd(axes=...)``, which is the C4v diagonal mirror written in leg metadata.

    ``absorb`` is the only thing that knows which model this is, and the partition is
    ``ndim // 2`` rather than a branch, because a bilinear form's two index groups are
    always its two halves: rank 4 for a single layer, rank 6 for a double one.

    ponytail: one isometry, ``u``, for a *bilinear* enlarged corner whose ``u`` and ``v``
    coincide only when it is positive. A single-layer Ising corner is, which is why that
    model reproduces Onsager to float64; a double-layer corner with indefinite spectrum
    gets a consistent contraction whose corner and edge differ by a diagonal of signs.
    Fixing it wants ``eigh(t, bond=)`` -- #77's explicit non-goal -- or four directional
    moves.
    """
    big_c = absorb.corner(c, e)
    n = big_c.ndim // 2  # (0..n-1 | n..2n-1): 2 for a single layer, 3 for a double one
    axes = (tuple(range(n)), tuple(range(n, 2 * n)))
    if bond is None:
        p, s, _ = tenet.linalg.svd_truncated(big_c, axes, max_bond=chi)
    else:
        p, s, _ = tenet.linalg.svd(big_c, axes, bond=bond)
    return CTMEnv(normalized(s), normalized(absorb.edge(e, p)), p.legs[-1].space)


def _spectrum_change(old: list[float], new: list[float]) -> float:
    """Max entrywise change, zero-padded to the longer spectrum. While the environment is
    still growing the two have different lengths, and the padding makes that a large
    change rather than an error -- which is what it is."""
    n = max(len(old), len(new))
    old, new = old + [0.0] * (n - len(old)), new + [0.0] * (n - len(new))
    return max(abs(a - b) for a, b in zip(old, new, strict=True))


class CTMRG_out(NamedTuple):
    """:class:`~tenet.network.DMRG_out`'s twin, with YASTN's names (``_env_ctm.py``:32-36).

    ``sweeps`` is ``len(history)``, ``max_dsv`` its last entry and ``converged`` the
    ``history[-1] < tol`` the loop already evaluates -- stored instead of discarded,
    because two test files were re-deriving it from the tail. ``history`` stays a field,
    so no caller loses information.

    YASTN's ``max_D`` is deliberately absent: tenet's ``chi`` is an *input*, and the
    realized environment bond is :attr:`CTMEnv.bond`.
    """

    sweeps: int
    max_dsv: float
    converged: bool
    env: CTMEnv
    history: list[float]


def ctmrg(
    absorb: Absorb,
    c: SymmetricTensor,
    e: SymmetricTensor,
    chi: int = 16,
    tol: float = 1e-10,
    max_sweeps: int = 100,
) -> CTMRG_out:
    """**Outside** ``jit``/``grad``, and it cannot be otherwise. Sweep to a fixed spectrum.

    It reads singular values to decide a bond and a corner spectrum to decide when to
    stop; a data-dependent loop exit is not a tracing edge case, it is the thing the
    outside/inside split exists to keep outside. Under ``jax.jit`` this raises before it
    ever reaches an SVD.

    Takes the ``(absorber, c, e)`` triple :func:`single_layer_ctm` /
    :func:`double_layer_ctm` return, so ``ctmrg(*single_layer_ctm(bulk), chi=16)`` is
    the whole call.

    Returns a :class:`CTMRG_out`. Its ``env`` is a :class:`CTMEnv` whose ``bond`` is the
    frozen environment :class:`~tenet.GradedSpace` -- the one and only thing that crosses
    into the differentiated region -- and its ``history`` the per-sweep corner-spectrum
    change, so a caller can *assert* convergence rather than assume it; ``converged`` is
    that assertion already made.
    """
    bond, previous, history = None, spectrum(c), []
    for _ in range(max_sweeps):
        c, e, bond = move(c, e, absorb, chi=chi)
        current = spectrum(c)
        history.append(_spectrum_change(previous, current))
        previous = current
        if history[-1] < tol:
            break
    return CTMRG_out(len(history), history[-1], history[-1] < tol, CTMEnv(c, e, bond), history)


def ctmrg_unrolled(
    c: SymmetricTensor,
    e: SymmetricTensor,
    absorb: Absorb,
    bond: GradedSpace,
    k: int = 4,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """**Inside** ``jax.jit(jax.grad(...))``. Exactly ``k`` fixed-structure moves.

    Takes ``c``, ``e`` and ``bond`` separately rather than a :class:`CTMEnv`, because
    ``bond`` is a ``static_argnums`` cache key here and a pytree leaf if handed in as part
    of a ``NamedTuple`` -- see :class:`CTMEnv`.

    A static Python loop: at ``k = 4`` there is nothing for ``jax.lax.scan`` to buy, and
    the loop being static is what makes the whole region one trace.
    """
    for _ in range(k):
        c, e, _ = move(c, e, absorb, bond=bond)
    return c, e
