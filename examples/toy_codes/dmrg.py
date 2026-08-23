"""Finite-chain two-site DMRG, written out: the U(1) Heisenberg chain against exact diagonalization.

Run it standalone::

    uv run python examples/toy_codes/dmrg.py

The algorithm is the file: the MPS list and its canonical form, the Heisenberg MPO, the
directed-bond environment cache and its invalidation, a Lanczos step over the two-site
tensor, and the sweep whose truncation re-decides each bond space. Nothing is imported
from ``tenet.network``, which ships all of it; ``examples/heisenberg_walkthrough.py`` is
the same physics through the library.

The tensor operations it is built on: ``SymmetricTensor.from_blocks`` and
``SymmetricTensor.random`` for the inputs, ``tenet.einsum`` for every contraction,
``tenet.repartition`` for the leg bends, ``tenet.linalg.lq`` and
``tenet.linalg.svd_truncated`` for the factorizations, ``tenet.add``, ``tenet.subtract``,
``tenet.norm`` and ``tenet.inner`` for the Krylov vector space, and ``tenet.to_matrices``
to read the Schmidt values off a bond.

**Leg conventions**, the part worth reading before the code:

* MPS site ``A_n``: ``(left bond OUT, physical OUT, right bond IN)``, the
  ``examples/toy_codes/vmc_mps.py`` convention. Charge flows left to right,
  ``bond_n (x) phys_n -> bond_{n+1}``, and both end bonds are :data:`BOUNDARY`, the unit
  sector with degeneracy 1 -- which forces ``Sum_i 2 S^z_i = 0``, i.e. ``S^z_tot = 0``,
  structurally and for free;
* MPO site ``W_n``: ``(wl IN, p OUT, p IN, wr OUT)``. Invariance reads
  ``q(p_out) + q(wr) = q(wl) + q(p_in)``, so an ``S^-`` emitted from the start channel
  sends the MPO bond to ``+2`` and an ``S^+`` sends it to ``-2``. The first and last
  sites carry a ``D=1`` :data:`BOUNDARY` MPO bond, which is what makes *every* ``W_n``
  rank 4 and removes the boundary-vector special case;
* environment ``F[(n, n+1)]``: ``(ket IN, mpo OUT, bra OUT)``, built from sites ``<= n``;
  environment ``F[(n, n-1)]``: ``(ket OUT, mpo IN, bra IN)``, built from sites ``>= n``.

**Operand order is part of the arithmetic, not a style choice.** Every ``tenet.einsum``
below is a *composition*: operand 1 supplies the ``IN`` end of every shared wire. Meeting
``IN`` against ``OUT`` is not enough -- that condition is symmetric, while the cap
direction, and hence the Koszul sign a fermionic provider pays, depends on which operand
supplies which end. The wires that genuinely turn around are bent *explicitly* by
:func:`_composed`. This chain is U(1), where every such sign is ``+1``; the orders are
still written correctly, because a reader copying this file for a fermionic model would
otherwise copy a silent sign error.

There is no ``jit`` and no ``grad`` here, and that is a decision: DMRG's control flow is
data-dependent at every level -- the truncation re-decides a bond space each sweep,
:func:`lanczos` tests a norm against a tolerance, :func:`dmrg` exits on a measured energy
change -- so this module runs on the eager NumPy backend and makes no differentiability
claim. ``ctmrg.py`` is the half of the library that lives under a trace.

Simplification: **two-site DMRG only.** It is what makes ``svd_truncated`` the
bond-deciding step, and it grows a bond by a factor of ``d`` per site with no extra
concept. Single-site DMRG cannot grow a bond at all, so it is only honest with subspace
expansion (Hubig-McCulloch-Schollwoeck-Wall, PRB 91, 155115 (2015)), which wants
``tenet.linalg.left_null``, a mixing factor and a second contraction chain.

Simplification: **hand-written pairwise contraction orders, not ``optimize=`` on a
five-operand einsum.** ``opt_einsum`` costs a graded network from *physical* leg sizes,
and a U(1) MPS bond whose sectors are unevenly filled is exactly where that estimate is
wrong. The orders here are YASTN's own (``yastn/tn/mps/_env.py``:496-518), documented as
``O(D^3 M d + D^2 M^2 d^2)`` per matvec -- optimal for *one* matvec, which is all a
Krylov step ever wants.

Simplification: **the MPO is written out, not generated, and there is no sweep
schedule.** Deriving MPO bonds from a term list and ramping ``chi`` with noise are
library features (``MPO.from_terms``, ``Sweep``), both demonstrated in
``examples/heisenberg_walkthrough.py``. Here the Hamiltonian is one page of blocks and
the loop runs at one ``chi``.
"""

from typing import NamedTuple

import numpy as np

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.symmetry import U1, U1Sector

# Physical space: charge t = 2 S^z, so the spin doublet is {-1, +1} -- exactly
# ``vmc_mps.SPACES["u1"]``'s physical leg. BOUNDARY is the unit sector with degeneracy 1,
# used for *both* ends of the MPS (fixing S^z_tot = 0) and for both ends of the MPO.
PHYS = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
BOUNDARY = GradedSpace.new(U1, {U1Sector(0): 1})

# The MPO bond: three charge-0 channels (start, S^z, end) and the two S^± channels at
# +-2, because S^± shifts 2 S^z by +-2.
MPO_BOND = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(2): 1, U1Sector(-2): 1})

# The sectors the blocks below are named in: physical down and up, and the three MPO bond
# charges. ``S^-`` raises the bond charge by 2 and ``S^+`` lowers it by 2.
DOWN, UP = U1Sector(-1), U1Sector(1)
ZERO, PLUS, MINUS = U1Sector(0), U1Sector(2), U1Sector(-2)

# Degeneracy indices inside the charge-0 MPO channel: the "nothing emitted yet" channel,
# the S^z channel and the "term finished" channel. ``GradedSpace`` keeps a sector's
# degeneracies in the order they were given, so these three names are the whole layout.
_END, _SZ, _START = 0, 1, 2

# The thermodynamic limit, 1/4 - ln 2 (Bethe 1931; Hulthen 1938), for main()'s report.
E_INF = -0.4431471805599453


# --- reading numbers off a tensor --------------------------------------------------


def spectrum(s: SymmetricTensor) -> list[float]:
    """The Schmidt values on a bond, descending.

    ``s`` comes from :func:`tenet.linalg.svd_truncated` and is diagonal by construction,
    so this reads the diagonal of each coupled-sector matrix ``tenet.to_matrices`` hands
    back -- the public way to read block values. The ``sqrt(qdim)`` weight is the same one
    :func:`tenet.norm` carries, and it is 1 throughout for U(1).
    """
    qdim = s.provider.qdim
    out = [
        float(m[i, i]) * qdim(sector) ** 0.5
        for sector, m in tenet.to_matrices(s).items()
        for i in range(m.shape[0])
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


# --- the Hamiltonian, as an MPO built block by block --------------------------------


def _blocks(legs, values: dict) -> SymmetricTensor:
    """Build from ``{(sector on each leg, in axis order): block}``.

    ``SymmetricTensor.from_blocks`` is keyed by a ``FusionBlockKey``, which for these legs
    carries the ``OUT`` sectors and the ``IN`` sectors in axis order; naming the sector on
    each leg is the same statement, read left to right off the leg list. Keys not named
    are zero. A sector combination the legs do not allow is not in ``block_order`` at all
    and raises here -- which is why a wrong MPO bond grading is a *refusal* rather than a
    silent projection onto some other operator.
    """
    structure = TensorStructure(tuple(legs))
    outs = [i for i, leg in enumerate(legs) if leg.side is OUT]
    ins = [i for i, leg in enumerate(legs) if leg.side is IN]
    keys = {}
    for key in structure.block_order:
        sectors = dict(zip(outs, key.output_tree.uncoupled, strict=True))
        sectors.update(zip(ins, key.input_tree.uncoupled, strict=True))
        keys[tuple(sectors[i] for i in range(len(legs)))] = key
    return SymmetricTensor.from_blocks(legs, {keys[s]: values[s] for s in values})


def mpo_blocks() -> dict:
    """The Heisenberg ``W``, one block per allowed sector tuple ``(wl, p_out, p_in, wr)``.

    ``H = Sum_i (S^z_i S^z_{i+1} + (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) / 2)``, J = 1, open
    boundaries. As the standard lower-triangular MPO, with ``SM``/``SP`` naming the channel
    entered by emitting an ``S^-``/``S^+``,

    * ``W[START, START] = I``   -- nothing emitted yet;
    * ``W[START, SM] = S^-/2``, ``W[SM, END] = S^+``   -- the ``S^- S^+/2`` term;
    * ``W[START, SP] = S^+/2``, ``W[SP, END] = S^-``   -- the ``S^+ S^-/2`` term;
    * ``W[START, SZ] = S^z``,  ``W[SZ, END] = S^z``    -- the ``S^z S^z`` term;
    * ``W[END, END] = I``      -- the term is finished.

    The symmetry is what splits that matrix into blocks rather than something checked
    afterwards. ``I`` and ``S^z`` keep the physical charge and so live in the two blocks
    on ``wl = wr = 0``, indexed ``[wl channel, 1, 1, wr channel]``; each ``S^±`` moves the
    bond charge by ``-+2`` and so is a block of its own, of extent 1 on that end.
    """
    return {
        # I and S^z: the charge-0 corner of the bond, as a 3x3 channel matrix
        (ZERO, DOWN, DOWN, ZERO): np.array(
            [[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, -0.5, 1.0]]
        ).reshape(3, 1, 1, 3),
        (ZERO, UP, UP, ZERO): np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 1.0]]).reshape(
            3, 1, 1, 3
        ),
        # S^-/2 and S^+/2 leaving the START channel
        (ZERO, DOWN, UP, PLUS): np.array([0.0, 0.0, 0.5]).reshape(3, 1, 1, 1),
        (ZERO, UP, DOWN, MINUS): np.array([0.0, 0.0, 0.5]).reshape(3, 1, 1, 1),
        # S^+ and S^- arriving at the END channel
        (PLUS, UP, DOWN, ZERO): np.array([1.0, 0.0, 0.0]).reshape(1, 1, 1, 3),
        (MINUS, DOWN, UP, ZERO): np.array([1.0, 0.0, 0.0]).reshape(1, 1, 1, 3),
    }


def mpo(n_sites: int, bond: GradedSpace = MPO_BOND) -> list[SymmetricTensor]:
    """The Heisenberg MPO, one rank-4 ``SymmetricTensor`` per site.

    Legs ``(wl IN, p OUT, p IN, wr OUT)``. The bulk tensor is :func:`mpo_blocks` on
    ``bond`` at both ends; the first site is its ``START`` row and the last its ``END``
    column, each on a ``D=1`` :data:`BOUNDARY` MPO leg -- which is what makes every ``W_n``
    rank 4 and removes the boundary-vector special case.

    ``bond`` is a parameter for one reason: so a test can hand it a grading the blocks do
    not fit and assert the refusal.
    """
    blocks = mpo_blocks()

    def legs(left: GradedSpace, right: GradedSpace):
        return (Leg(left, IN), Leg(PHYS, OUT), Leg(PHYS, IN), Leg(right, OUT))

    bulk = _blocks(legs(bond, bond), blocks)
    first = _blocks(
        legs(BOUNDARY, bond),
        {key: blocks[key][_START : _START + 1] for key in blocks if key[0] == ZERO},
    )
    last = _blocks(
        legs(bond, BOUNDARY),
        {key: blocks[key][..., _END : _END + 1] for key in blocks if key[3] == ZERO},
    )
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
    """A tensor with every structurally allowed entry equal to 1.

    ``TensorStructure`` already knows which blocks the grading allows and how big each one
    is, so the seed is "fill the blocks that exist": there is no dense array here to build
    and project.
    """
    structure = TensorStructure(tuple(legs))
    blocks = {key: np.ones(structure.block_shape(key)) for key in structure.block_order}
    return SymmetricTensor.from_blocks(legs, blocks)


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
    ``tenet.norm`` and ``tenet.inner`` -- a Krylov solver needs a vector space and nothing
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
        alphas.append(float(tenet.inner(vecs[j], w)))
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
    for n in new:
        previous, current = old.get(n, []), new[n]
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
