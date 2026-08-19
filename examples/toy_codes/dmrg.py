"""Finite-chain two-site DMRG written out by hand: U(1) Heisenberg against exact diagonalization.

Run it standalone::

    uv run python examples/toy_codes/dmrg.py

**The lane rule, and what this file is.** ``examples/toy_codes/`` teaches what the library
does not own: the algorithm is written here, on ``tenet``'s *tensor* layer --
``SymmetricTensor``, ``tenet.einsum``, ``tenet.linalg`` -- and **nothing is imported from
``tenet.network``**, which ``tests/test_examples.py`` asserts for every file in this
directory. The MPS list, the canonical form, the directed-bond environment cache and its
invalidation, the Krylov step and the two-site sweep are what the reader is here to see.
The library ships all five (``MPS``, ``Env``, ``lanczos``, ``sweep_``, ``dmrg_``); this is
the same code before it was promoted (#112), restored (#183), and the split is tenpy's --
``tenpy_toycodes`` writes MPS/TEBD/DMRG out with numpy only, tenpy's own ``examples/``
calls the library. Adapted, because the symmetric-tensor layer *is* tenet's subject matter:
a toy code here may use it, and must not use the algorithms built on it.
``examples/heisenberg_walkthrough.py`` is the same physics through ``tenet.network``, and
is where a reader who wants the library's version should go.

**The per-helper rule, stated once and applied throughout.** Some of what follows exists
in ``tenet`` already. The rule is: **a helper stays local when writing it out is the
lesson, and is called from ``tenet`` when it is the tensor layer this file is built on.**
So :func:`scalar`, :func:`inner` and :func:`spectrum` are written here even though
``tenet.full_trace``, ``tenet.inner`` and ``tenet.network.spectrum`` exist -- *how a
tensor network produces a number* is exactly what a first reader is missing, and each is
five lines. So is :func:`_composed`, whose bend is the file's one categorical subtlety.
Everything below the tensor layer -- ``einsum``, ``svd_truncated``, ``lq``,
``repartition``, ``norm``, ``adjoint``, ``add``/``subtract`` -- is called, never
reimplemented: rewriting a graded SVD would teach nothing about DMRG.

What it demonstrates (no ``scipy``, no ``quimb``, no ``jax``):

* a **U(1) MPS whose target sector is fixed by the boundary legs alone**. The physical
  charge is ``t = 2 S^z in {-1, +1}`` (``vmc_mps.SPACES["u1"]``'s spin doublet) and both
  boundary legs are :data:`BOUNDARY`, the unit sector with degeneracy 1. Invariance of
  every site tensor then forces ``Sum_i 2 S^z_i = 0``, i.e. **``S^z_tot = 0``, the sector
  the ground state of an even chain lives in, enforced structurally and for free** -- no
  penalty term, no projector, no ``project=`` argument;
* the Heisenberg MPO built by ``tenet.SymmetricTensor.from_dense`` (#82) at the
  **default** relative ``atol`` from the 5x5 ``W`` matrix written out in the carrier
  basis, on the graded MPO bond :data:`MPO_BOND`. A wrong grading makes ``from_dense``
  *raise*, and that refusal -- asserted in ``tests/integration/test_dmrg.py`` -- is the
  proof the grading is right. A passing ``allclose`` would not be;
* ``tenet.linalg.svd_truncated`` (#77) deciding a bond :class:`~tenet.GradedSpace` at
  **every bond of every sweep**, with the discarded weight reported by Pythagoras --
  the mirror image of CTMRG's frozen bond;
* an **iterative Krylov eigensolver written over ``SymmetricTensor`` as a vector**:
  :func:`lanczos` needs only ``tenet.add``/``subtract``, scalar multiply/divide,
  ``tenet.norm`` and :func:`inner`. No ``scipy.sparse.linalg``, no dense reshaping of the
  local problem.

**Why there is no ``jit`` and no ``grad`` here, and why that is a decision.** DMRG is a
fixed-point solver whose control flow is data-dependent at every level: the truncation
re-decides the bond space each sweep (``tenet.StructureChangingError`` under a trace, by
design), :func:`lanczos`'s happy breakdown tests a norm against ``tol``, and :func:`dmrg`'s
loop exits on a measured energy change. Every one of those is precisely what tenet refuses
to trace, and correctly. So this module runs on the eager NumPy backend and makes no
differentiability claim of any kind. The #77 pairing -- ``svd_truncated`` *outside* the
trace, ``svd(bond=)`` *inside* it -- has **two** legitimate halves, and this file uses one:
``ctmrg.py`` needs the inside half because it differentiates through its sweeps; DMRG needs
only the outside half because it does not. There is also, for the same reason, no XLA
compile floor here.

**Leg conventions**, the part worth reading before the code:

* MPS site ``A_n``: ``(left bond OUT, physical OUT, right bond IN)``, the
  ``examples/toy_codes/vmc_mps.py`` convention. Charge flows left to right,
  ``bond_n (x) phys_n -> bond_{n+1}``, and both end bonds are :data:`BOUNDARY`;
* MPO site ``W_n``: ``(wl IN, p OUT, p IN, wr OUT)``. Invariance reads
  ``q(p_out) + q(wr) = q(wl) + q(p_in)``, so an ``S^-`` emitted from the start channel
  sends the MPO bond to ``+2`` and an ``S^+`` sends it to ``-2``. The first and last
  sites carry a ``D=1`` :data:`BOUNDARY` MPO bond, which is what makes *every* ``W_n``
  rank 4 and removes the boundary-vector special case;
* environment ``F[(n, n+1)]``: ``(ket IN, mpo OUT, bra OUT)``, built from sites ``<= n``;
  environment ``F[(n, n-1)]``: ``(ket OUT, mpo IN, bra IN)``, built from sites ``>= n``.

**Operand order is part of the arithmetic, not a style choice.** Every ``tenet.einsum``
below is a *composition*: **operand 1 supplies the ``IN`` end of every shared wire**
(M23's rule, ``docs/design.md`` "Milestone 11"). Meeting ``IN`` against ``OUT`` is not enough
-- that condition is symmetric, while the cap direction, and hence the Koszul sign a
fermionic provider pays, depends on which operand supplies which end. The wires that
genuinely turn around -- the MPS bond arrow and the MPO bond arrow cross the two-site cell
in opposite directions, so closing either cap reverses one rail -- are bent *explicitly*
by :func:`_composed` rather than left to ``einsum``. This chain is U(1), where every such
sign is ``+1`` and nothing here can measure the difference; the orders are still written
correctly, because a reader copying this file for a fermionic model would otherwise copy a
silent sign error. The pre-promotion revision of this file argued the symmetric reading
out loud and was wrong; ``tests/integration/test_dmrg.py`` now asserts the rule on every
contraction below, and ``tests/network/test_hygiene.py`` measures the library's copy of
these same contractions against a dense Jordan-Wigner oracle.

Deliberate simplifications, each with its ceiling:

Simplification: **two-site DMRG only; single-site plus subspace expansion is deferred, and
``tenet.linalg.left_null`` (#88) therefore gets no demo here.** Two-site is what makes
``svd_truncated`` the bond-deciding step, which is the tenet feature this example exists
to exercise, and it grows a bond by a factor of ``d`` per site with no extra concept.
Strictly-single-site DMRG at fixed bond *cannot grow a bond at all*, so it is only honest
with subspace expansion (Hubig-McCulloch-Schollwoeck-Wall, PRB 91, 155115 (2015)) -- which
needs ``left_null``, a mixing factor ``alpha``, its own schedule and a second ``heff1``
contraction chain. Named upgrade path, and a good one.

Simplification: **hand-written pairwise contraction orders, not ``optimize=`` on a
five-operand einsum.** Same reason as ``ctmrg._halves``: ``opt_einsum`` costs a graded
network from *physical* leg sizes, and a U(1) MPS bond whose sectors are unevenly filled
is exactly where that estimate is wrong. The orders here are YASTN's own
(``yastn/tn/mps/_env.py``:496-518), which its ``_dmrg.py``:102-108 documents as
``O(D^3 M d + D^2 M^2 d^2)`` per matvec -- optimal for *one* matvec, which is all a Krylov
step ever wants.

Simplification: **the MPO is written out, not generated, and there is no sweep schedule.**
Deriving MPO bonds from a term list and ramping ``chi`` with noise are library features
(``MPO.from_terms``, ``Sweep``); both are demonstrated in
``examples/heisenberg_walkthrough.py``. Here the Hamiltonian is one 5x5 matrix of physics
and the loop runs at one ``chi``, which is what a first reading wants.

See :func:`lanczos` for the no-reorthogonalization and numpy-``eigh`` notes.
"""

import string
from typing import NamedTuple

import autoray as ar
import numpy as np

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import U1, U1Sector

# Physical space: charge t = 2 S^z, so the spin doublet is {-1, +1} -- exactly
# ``vmc_mps.SPACES["u1"]``'s physical leg. BOUNDARY is the unit sector with degeneracy 1,
# used for *both* ends of the MPS (fixing S^z_tot = 0) and for both ends of the MPO.
PHYS = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
BOUNDARY = GradedSpace.new(U1, {U1Sector(0): 1})

# The MPO bond: three charge-0 channels (start, S^z, end) and the two S^± channels at
# +-2, because S^± shifts 2 S^z by +-2. The sign convention is **not derived by hand**:
# the dense W below is handed to ``from_dense`` at the default relative atol, and a wrong
# grading makes it raise. See ``test_dmrg.py::test_mpo_refuses_a_perturbed_grading``.
MPO_BOND = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(2): 1, U1Sector(-2): 1})

# Dense positions of the five MPO channels. ``GradedSpace`` sorts its sectors ascending,
# so the -2 channel is index 0, the three charge-0 channels are 1..3 and the +2 channel is
# index 4. The names, not the numbers, are what the W matrix is written in below:
# ``_SM_CHANNEL`` is the channel *entered* by emitting an S^-, which raises the MPO bond
# charge by +2, and ``_SP_CHANNEL`` the one entered by emitting an S^+.
_SP_CHANNEL, _END, _SZ_CHANNEL, _START, _SM_CHANNEL = 0, 1, 2, 3, 4

# The thermodynamic limit, 1/4 - ln 2 (Bethe 1931; Hulthen 1938), for main()'s report.
E_INF = -0.4431471805599453


# --- leaving the tensor world ------------------------------------------------------


def scalar(t: SymmetricTensor):
    """The categorical trace of a rank-2 map ``(X OUT, X IN)``: ``sum_c d_c tr(M_c)``.

    ``SymmetricTensor`` has no rank 0, so every fully closed network here contracts down
    to a rank-2 tensor with one bond left open, and closing that bond is a trace carrying
    the same ``qdim`` weight :func:`tenet.norm` carries. **This is where the tensor world
    is left**, and writing it out is the point: the library spells it
    ``tenet.full_trace`` (#126), which is these five lines plus a square-map refusal.
    """
    qdim = t.provider.qdim
    return sum(qdim(c) * ar.do("trace", m) for c, m in tenet.to_matrices(t).items())


def inner(a: SymmetricTensor, b: SymmetricTensor):
    """``<a|b>``: contract every axis but the first, then :func:`scalar` the rest.

    Works at any rank, which is what lets :func:`lanczos` be a plain vector-space
    algorithm: the adjoint flips every leg, so axis 0 of ``adjoint(a)`` is IN and axis 0
    of ``b`` is OUT, and the leftover rank-2 map is exactly what :func:`scalar` traces.
    ``tenet.inner`` (#126) is this function verbatim.

    A sesquilinear pairing is **not** a composition, and this is the module's one
    contraction that is not: ``adjoint`` turns every leg around, so the *domain* wires of
    a rank-3 vector meet OUT against IN here by construction. That is the pairing's own
    arithmetic, and ``tenet.inner`` writes the identical einsum.
    """
    rest = string.ascii_lowercase[1 : a.ndim]
    return scalar(tenet.einsum(f"L{rest},l{rest}->lL", tenet.adjoint(a), b))


def spectrum(s: SymmetricTensor) -> list[float]:
    """The Schmidt values on a bond, descending -- ``tenet.network.spectrum``'s body.

    ``s`` comes from :func:`tenet.linalg.svd_truncated` and is diagonal by construction,
    so this reads its diagonal; the ``sqrt(qdim)`` weight is the same one
    :func:`tenet.norm` carries, and it is 1 throughout for U(1).
    """
    qdim = s.provider.qdim
    out = [
        float(v)
        for sector, m in tenet.to_matrices(s).items()
        for v in ar.do("diag", m) * qdim(sector) ** 0.5
    ]
    return sorted(out, reverse=True)


def _composed(equation: str, a: SymmetricTensor, b: SymmetricTensor, bend: str = ""):
    """A two-operand ``tenet.einsum`` with the wires named in ``bend`` bent first.

    Operand 1 must supply the ``IN`` end of every shared wire (the module docstring's
    composition rule). A wire that turns around in the intended planar diagram -- one
    running through an environment's cap -- cannot meet that rule as drawn, and letting
    ``einsum`` bend it implicitly would leave the cap direction to operand order. So the
    bend is spelled: both ends of each named wire move to the other side with
    ``tenet.repartition``, which pays the categorical bend coefficient by construction,
    and the einsum that follows is a plain composition again. ``bend=""`` is a straight
    composition and could as well be ``tenet.einsum``.
    """
    if bend:
        lhs, out = equation.split("->")
        ta, tb = lhs.split(",")

        def bent(t: SymmetricTensor, term: str) -> tuple[SymmetricTensor, str]:
            flip = set(bend)
            outs = tuple(i for i, c in enumerate(term) if (t.legs[i].side is OUT) != (c in flip))
            ins = tuple(i for i in range(len(term)) if i not in outs)
            return tenet.repartition(t, outs, ins), "".join(term[i] for i in (*outs, *ins))

        a, ta = bent(a, ta)
        b, tb = bent(b, tb)
        equation = f"{ta},{tb}->{out}"
    return tenet.einsum(equation, a, b)


# --- the Hamiltonian, as an MPO through from_dense ---------------------------------


def _spin_half() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(I, Sz, S+, S-)`` as 2x2 dense arrays in this module's physical basis.

    Index 0 is charge ``-1`` (spin down) and index 1 is charge ``+1`` (spin up), because
    :data:`PHYS` sorts its sectors ascending. The matrices are ``O[out, in]``.
    """
    eye = np.eye(2)
    sz = np.diag([-0.5, 0.5])
    sp = np.array([[0.0, 0.0], [1.0, 0.0]])  # |down> -> |up>
    return eye, sz, sp, sp.T


def mpo_array() -> np.ndarray:
    """The 5x5 Heisenberg ``W``, dense, indexed ``[wl, p_out, p_in, wr]``.

    ``H = Sum_i (S^z_i S^z_{i+1} + (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) / 2)``, J = 1,
    **open** boundaries. In the channel names of :data:`_START` and friends the standard
    lower-triangular matrix reads

    * ``W[START, START] = I``   -- nothing emitted yet;
    * ``W[START, SM] = S^-/2``, ``W[SM, END] = S^+``   -- the ``S^- S^+/2`` term;
    * ``W[START, SP] = S^+/2``, ``W[SP, END] = S^-``   -- the ``S^+ S^-/2`` term;
    * ``W[START, SZ] = S^z``,  ``W[SZ, END] = S^z``    -- the ``S^z S^z`` term;
    * ``W[END, END] = I``      -- the term is finished.

    The left boundary vector is ``e_START`` and the right one ``e_END``; both are spelled
    as ``D=1`` MPO bond legs in :func:`mpo`, which is what makes every ``W_n`` rank 4.
    """
    eye, sz, sp, sm = _spin_half()
    w = np.zeros((5, 2, 2, 5))
    w[_START, :, :, _START] = eye
    w[_END, :, :, _END] = eye
    w[_START, :, :, _SM_CHANNEL] = 0.5 * sm
    w[_SM_CHANNEL, :, :, _END] = sp
    w[_START, :, :, _SP_CHANNEL] = 0.5 * sp
    w[_SP_CHANNEL, :, :, _END] = sm
    w[_START, :, :, _SZ_CHANNEL] = sz
    w[_SZ_CHANNEL, :, :, _END] = sz
    return w


def mpo(n_sites: int, bond: GradedSpace = MPO_BOND) -> list[SymmetricTensor]:
    """The Heisenberg MPO, one rank-4 ``SymmetricTensor`` per site.

    Legs ``(wl IN, p OUT, p IN, wr OUT)``. The bulk tensor is :func:`mpo_array` on
    ``bond`` at both ends; the first site keeps only the ``START`` row and the last only
    the ``END`` column, each on a ``D=1`` :data:`BOUNDARY` MPO leg.

    ``from_dense`` is called at its **default** (relative) ``atol``, so it *refuses* an
    array that is not symmetric for the declared legs. ``bond`` is a parameter for
    exactly one reason: so a test can hand it a perturbed grading and assert the refusal.
    """
    w = mpo_array()

    def make(array, left, right):
        legs = (Leg(left, IN), Leg(PHYS, OUT), Leg(PHYS, IN), Leg(right, OUT))
        return SymmetricTensor.from_dense(array, legs)

    first = make(w[_START : _START + 1], BOUNDARY, bond)
    bulk = make(w, bond, bond)
    last = make(w[:, :, :, _END : _END + 1], bond, BOUNDARY)
    return [first, *[bulk] * (n_sites - 2), last]


# --- the MPS -----------------------------------------------------------------------


def bond_spaces(n_sites: int) -> list[GradedSpace]:
    """The ``n_sites + 1`` virtual spaces, degeneracy 1 in every reachable sector.

    The charge on bond ``i`` is the sum of ``i`` physical charges (each ``+-1``) and must
    still be able to reach 0 by site ``n_sites``, so it runs over
    ``-w, -w+2, ..., w`` with ``w = min(i, n_sites - i)``; bonds 0 and ``n_sites`` are
    :data:`BOUNDARY`, which is the whole ``S^z_tot = 0`` statement. Which charges are
    reachable is physics, and it is the only thing about this chain a generic MPS
    container cannot know -- which is why the library takes bond *spaces* and not a chi.

    Simplification: degeneracy 1 per sector, not a full-rank seed. Two-site DMRG multiplies a
    bond by ``d = 2`` per sweep, so a chi=64 bond is three sweeps away and the first
    sweeps are correspondingly cheap. A larger seed is one ``min(..., cap)`` here if a
    workload ever wants the bond full on sweep one.
    """
    spaces = []
    for i in range(n_sites + 1):
        w = min(i, n_sites - i)
        spaces.append(GradedSpace.new(U1, {U1Sector(q): 1 for q in range(-w, w + 1, 2)}))
    return spaces


def random_mps(n_sites: int, seed: int = 0) -> list[SymmetricTensor]:
    """A random U(1) MPS, ``A_n`` on ``(left bond OUT, phys OUT, right bond IN)``."""
    spaces = bond_spaces(n_sites)
    return [
        SymmetricTensor.random(
            (Leg(spaces[i], OUT), Leg(PHYS, OUT), Leg(spaces[i + 1], IN)), seed=seed + i
        )
        for i in range(n_sites)
    ]


def _as_site(t: SymmetricTensor) -> SymmetricTensor:
    """Put a rank-3 factor back on the MPS partition ``(l, p | r)``.

    Every factorization in ``tenet.linalg`` lowers its input to a *map* first, so the
    legs it hands to the domain come back bent -- a physical leg that was ``OUT`` in the
    codomain is ``IN dual`` in the domain, which is the leg bend spelled out rather than
    hidden. One :func:`tenet.repartition` puts it back, and only then do a site tensor
    from :func:`canonicalize` and one from :func:`sweep` share a structure. (``MPS``
    does this in a write barrier on ``__setitem__``; here it is a named call, because
    seeing it is the lesson.)
    """
    return tenet.repartition(t, (0, 1), (2,))


def canonicalize(psi: list[SymmetricTensor]) -> list[SymmetricTensor]:
    """Right-canonicalize in place of YASTN's ``canonize_(to='first')`` (``_mps_obc.py``:390).

    One ``tenet.linalg.lq`` per site from the right, mirroring ``orthogonalize_site_``
    (:245-300): ``A_n = L . Q`` with ``Q`` on ``(bond OUT, phys OUT, right IN)`` -- the
    MPS leg convention unchanged -- and ``L`` absorbed into ``A_{n-1}``. ``lq`` rather
    than ``qr`` because ``qr`` would put the new bond on the *right* of the factor and
    leave the site tensor's left leg IN, which is not this file's convention.

    Setup only. A two-site sweep leaves the state canonical by construction on the side
    it came from, so this runs once, before the first environment is built.
    """
    psi = list(psi)
    for n in range(len(psi) - 1, 0, -1):
        left, q = tenet.linalg.lq(psi[n], ((0,), (1, 2)))
        psi[n] = _as_site(q)
        psi[n - 1] = tenet.einsum("apx,xy->apy", psi[n - 1], left)
    return [psi[0] / tenet.norm(psi[0]), *psi[1:]]


# --- the environment: YASTN's directed-bond dict ------------------------------------


def _ones(legs) -> SymmetricTensor:
    """A tensor of ones on ``legs`` -- ``ctmrg.init_env``'s seed spelling."""
    t = SymmetricTensor.zeros(tuple(legs))
    return t.apply_blocks(lambda b: ar.do("ones_like", b))


def boundary_envs(n_sites: int) -> dict[tuple[int, int], SymmetricTensor]:
    """``{(-1, 0): left, (n, n-1): right}``, both the trivial 1x1x1 tensor.

    The environment is a plain ``dict`` keyed by *directed* bond, exactly YASTN's ``Env``
    (``yastn/tn/mps/_env.py``:104-125 ``setup_``): ``F[(n, n+1)]`` is built from sites
    ``<= n`` and ``F[(n, n-1)]`` from sites ``>= n``. A list-of-left / list-of-right would
    hide the invalidation discipline, which is the entire correctness content of an
    environment cache -- a stale ``F[(n, n+1)]`` after site ``n`` changed gives an energy
    that is *plausible and wrong*, the worst failure mode a DMRG has.
    """
    return {
        (-1, 0): _ones((Leg(BOUNDARY, IN), Leg(BOUNDARY, OUT), Leg(BOUNDARY, OUT))),
        (n_sites, n_sites - 1): _ones((Leg(BOUNDARY, OUT), Leg(BOUNDARY, IN), Leg(BOUNDARY, IN))),
    }


def update_env(envs, psi, w, n: int, to: str) -> None:
    """Write one directed-bond entry from its neighbour -- YASTN ``_env.py``:152-168.

    ``to='last'`` writes ``F[(n, n+1)]`` from ``F[(n-1, n)]``; ``to='first'`` writes
    ``F[(n, n-1)]`` from ``F[(n+1, n)]``. Three pairwise contractions each: environment
    first, then the ket, then the MPO, then the bra. The ``'first'`` direction runs
    against the arrows -- the physical wire ``p`` and the bra wire ``P`` each turn around
    in the cap -- so those two are :func:`_composed` with the bend named.
    """
    a, bra = psi[n], tenet.adjoint(psi[n])
    if to == "last":
        t = tenet.einsum("axB,apr->xBpr", envs[n - 1, n], a)
        t = tenet.einsum("xPpm,xBpr->BrPm", w[n], t)
        envs[n, n + 1] = tenet.einsum("BPs,BrPm->rms", bra, t)
    else:
        t = tenet.einsum("apr,rys->apys", a, envs[n + 1, n])
        t = _composed("apys,xPpy->axPs", t, w[n], bend="p")
        envs[n, n - 1] = _composed("axPs,BPs->axB", t, bra, bend="P")


def invalidate(envs, *sites: int) -> None:
    """Pop every entry a changed site invalidates -- YASTN ``clear_site_``, :127-134.

    Both directed bonds touching each site go, and they go *before* the replacement is
    written, so a missed update is a ``KeyError`` rather than a wrong number.
    """
    for n in sites:
        envs.pop((n, n - 1), None)
        envs.pop((n, n + 1), None)


def setup_envs(psi, w) -> dict[tuple[int, int], SymmetricTensor]:
    """Every right-directed environment, for a right-canonical ``psi`` -- ``setup_(to='first')``."""
    envs = boundary_envs(len(psi))
    for n in range(len(psi) - 1, 0, -1):
        update_env(envs, psi, w, n, "first")
    return envs


# --- the local problem -------------------------------------------------------------


def heff2(envs, w1, w2, n: int, aa: SymmetricTensor) -> SymmetricTensor:
    """``H_eff`` on the two-site tensor at bond ``(n, n+1)``. Four pairwise contractions.

    Right environment, then ``W2``, then ``W1``, then the left environment: YASTN's
    ``Env_mps_mpo_mps.Heff2`` order (``_env.py``:496-518) with ``precompute=False``,
    which ``_dmrg.py``:102-108 documents as ``O(D^3 M d + D^2 M^2 d^2)`` -- optimal for
    a single matvec, which is all a Krylov step ever wants.

    In and out on ``(left bond OUT, p OUT, q OUT, right bond IN)``: the *bra* legs of the
    two environments become the output's bonds while the *ket* legs close against the
    input's, which is why the result has ``aa``'s structure exactly and
    :func:`lanczos` can add the two. Three of the four contractions run through a cap and
    name their bent wire; the first, which only rides the right environment, does not.
    """
    t = tenet.einsum("apqr,rys->apqys", aa, envs[n + 2, n + 1])
    t = _composed("apqys,mQqy->apQms", t, w2, bend="q")
    t = _composed("apQms,xPpm->aPQxs", t, w1, bend="p")
    return _composed("aPQxs,axB->BPQs", t, envs[n - 1, n], bend="a")


def lanczos(matvec, v: SymmetricTensor, ncv: int = 3, tol: float = 1e-13):
    """Ground eigenpair ``(value, vector)`` of a Hermitian ``matvec`` over SymmetricTensors.

    YASTN's three-term recurrence (``yastn/tensor/_krylov.py``:34-42) and its happy
    breakdown (``H[(j+1,j)] < tol`` -> stop and drop the row, :39-43), then ``eigh`` of
    the ``(m, m)`` tridiagonal and one recombination
    (``yastn/krylov/_krylov.py``:226-239, a single iteration with no restart at :217-219).
    ``hermitian=True, ncv=3, which='SR'`` are YASTN's own DMRG defaults
    (``_dmrg.py``:151-152) and are not knobs this example tunes.

    The only tensor operations are ``tenet.add``/``subtract``, scalar multiply/divide,
    ``tenet.norm`` and :func:`inner` -- a Krylov solver needs a vector space and nothing
    else, and a ``SymmetricTensor`` is one.

    Simplification: **no reorthogonalization**, and neither has YASTN. At ``ncv=3`` the
    recurrence has not had time to lose orthogonality, and the vector is reseeded from the
    current MPS at every bond -- this is an inner solver inside an outer sweep, not a
    standalone eigensolver. Ceiling: raise ``ncv`` past ~10 and full reorthogonalization
    against the stored ``vecs`` becomes the two-line addition.

    Simplification: numpy ``eigh`` on the ``(3, 3)`` tridiagonal, not ``tenet.linalg.eigh``. The
    projected matrix has no symmetry structure to respect -- it is 9 floats.
    """
    vecs = [v / tenet.norm(v)]
    alphas: list[float] = []
    betas: list[float] = []
    for j in range(ncv):
        w = matvec(vecs[j])
        alphas.append(float(inner(vecs[j], w)))
        w = tenet.subtract(w, vecs[j] * alphas[j])
        if j:
            w = tenet.subtract(w, vecs[j - 1] * betas[j - 1])
        beta = float(tenet.norm(w))
        if j + 1 == ncv or beta < tol:  # happy breakdown: drop the row, keep the space
            break
        betas.append(beta)
        vecs.append(w / beta)
    tri = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    values, states = np.linalg.eigh(tri)
    ground = states[:, 0]
    out = vecs[0] * float(ground[0])
    for k in range(1, len(vecs)):
        out = tenet.add(out, vecs[k] * float(ground[k]))
    return float(values[0]), out / tenet.norm(out)


# --- the sweep ---------------------------------------------------------------------


def sweep(psi, w, envs, schmidt, *, chi: int, cutoff: float, ncv: int = 3):
    """One left-to-right then right-to-left two-site sweep. ``psi`` is updated in place.

    YASTN's ``_dmrg_sweep_2site_`` (``_dmrg.py``:222-249) and its
    ``(('last', 0), ('first', 1))`` two-direction loop, five steps per bond:
    merge, solve, split, invalidate, update the environment.

    ``svd_truncated`` decides the bond :class:`~tenet.GradedSpace` here, every bond and
    every sweep, and the discarded weight is Pythagoras exactly as its docstring
    prescribes: ``U S Vh`` is isometric on both sides, so ``norm(U S Vh) = norm(S)`` and
    the dropped fraction of the (unit-norm) two-site tensor is ``1 - norm(S)**2``.

    Returns ``(energy, max_discarded_weight)``; ``schmidt`` is updated in place with the
    per-bond Schmidt spectra, which is the second convergence criterion's input.
    """
    n_sites = len(psi)
    energy, max_dw = 0.0, 0.0
    for direction in ("right", "left"):
        bonds = range(n_sites - 1) if direction == "right" else range(n_sites - 2, -1, -1)
        for n in bonds:
            aa = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
            w1, w2 = w[n], w[n + 1]
            energy, aa = lanczos(
                lambda v, w1=w1, w2=w2, n=n: heff2(envs, w1, w2, n, v), aa, ncv=ncv
            )
            u, s, vh = tenet.linalg.svd_truncated(aa, ((0, 1), (2, 3)), max_bond=chi, cutoff=cutoff)
            vh = _as_site(vh)
            norm_s = tenet.norm(s)
            max_dw = max(max_dw, 1.0 - float(norm_s / tenet.norm(aa)) ** 2)
            s = s / norm_s  # the two-site tensor is normalized; keep the MPS so
            if direction == "right":
                psi[n], psi[n + 1] = u, tenet.einsum("xy,yqr->xqr", s, vh)
            else:
                psi[n], psi[n + 1] = tenet.einsum("apx,xy->apy", u, s), vh
            schmidt[n] = spectrum(s)
            invalidate(envs, n, n + 1)
            if direction == "right":
                update_env(envs, psi, w, n, "last")
            else:
                update_env(envs, psi, w, n + 1, "first")
    return energy, max_dw


def _schmidt_change(old: dict, new: dict) -> float:
    """``max_k ||S_k - S_k^old||`` over bonds, zero-padded -- YASTN ``_dmrg.py``:154-195.

    A bond present in only one of the two, or a spectrum whose length changed because
    ``svd_truncated`` moved the bond space, counts as a large change rather than an
    error, which is what it is. Infinite before the first sweep has any history.
    """
    if not old:
        return float("inf")
    worst = 0.0
    for n, current in new.items():
        previous = old.get(n, [])
        m = max(len(previous), len(current))
        a = previous + [0.0] * (m - len(previous))
        b = current + [0.0] * (m - len(current))
        worst = max(worst, sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5)
    return worst


class DMRG_out(NamedTuple):
    """YASTN's ``DMRG_out`` (``_dmrg.py``:33-39), plus the two things a test needs.

    ``history`` is one ``(energy, denergy, dSchmidt, discarded)`` tuple per sweep -- the
    ``ctmrg.converge`` precedent, and it says everything YASTN's ``iterator=True``
    generator protocol (:124-128, :196-198) says to a test without the protocol. ``psi``
    is the converged MPS, as a plain list of site tensors.
    """

    sweeps: int
    energy: float
    denergy: float
    max_dSchmidt: float
    max_discarded_weight: float
    history: list
    psi: list


def dmrg(
    n_sites: int,
    chi: int = 64,
    *,
    cutoff: float = 1e-14,
    energy_tol: float = 1e-12,
    schmidt_tol: float = 1e-8,
    max_sweeps: int = 40,
    seed: int = 0,
    ncv: int = 3,
) -> DMRG_out:
    """Sweep to the ground state and return a :class:`DMRG_out`.

    Convergence uses **both** of YASTN's criteria (``_dmrg.py``:180-195): the energy
    change ``|E_old - E| < energy_tol`` *and* the worst-cut Schmidt change
    ``max_k ||S_k - S_k^old|| < schmidt_tol``, and the loop stops only when both are met
    in one sweep. The Schmidt criterion is the sensitive one, and it is what catches a run
    whose energy has plateaued on a wrong bond structure.
    """
    psi = canonicalize(random_mps(n_sites, seed=seed))
    w = mpo(n_sites)
    envs = setup_envs(psi, w)
    schmidt: dict[int, list[float]] = {}
    energy, history, out = float("inf"), [], None
    for it in range(1, max_sweeps + 1):
        old_energy, old_schmidt = energy, dict(schmidt)
        energy, max_dw = sweep(psi, w, envs, schmidt, chi=chi, cutoff=cutoff, ncv=ncv)
        denergy = abs(old_energy - energy)
        d_schmidt = _schmidt_change(old_schmidt, schmidt)
        history.append((energy, denergy, d_schmidt, max_dw))
        out = DMRG_out(it, energy, denergy, d_schmidt, max_dw, history, psi)
        if denergy < energy_tol and d_schmidt < schmidt_tol:
            break
    return out


def main(n_sites: int = 12, chi: int = 64, big_sites: int = 32, big_chi: int = 64):
    """N=12 at chi=64 against the exact ground state, then N=32 at chi=64 against ``e_inf``.

    The N=12 reference printed here is the **open**-boundary energy
    ``-5.142090632840532``; the periodic chain's ``-5.387390917445203`` is a different
    number for a different model and an OBC MPS cannot reproduce it.
    ``tests/integration/test_dmrg.py`` computes the OBC value rather than trusting it.
    """
    small = dmrg(n_sites, chi)
    print(f"N={n_sites} chi={chi}  E={small.energy:+.12f}  exact=-5.142090632840532")
    for i, (e, de, ds, dw) in enumerate(small.history, start=1):
        print(f"  sweep {i:2d}  E={e:+.12f}  dE={de:.3e}  dS={ds:.3e}  dw={dw:.3e}")

    big = dmrg(big_sites, big_chi)
    print(
        f"N={big_sites} chi={big_chi}  E={big.energy:+.12f}  "
        f"E/N={big.energy / big_sites:+.12f}  e_inf={E_INF:+.12f}  "
        f"sweeps={big.sweeps}  max_dw={big.max_discarded_weight:.3e}"
    )
    return small, big


if __name__ == "__main__":
    main()
