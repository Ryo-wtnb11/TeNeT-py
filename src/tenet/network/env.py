"""The environment cache: ``<psi|H|psi>`` partial contractions, keyed by *directed* bond.

Promoted from ``examples/dmrg.py`` (#110) with no arithmetic change: ``boundary_envs``
:318-331, ``update_env`` :334-349, ``invalidate`` :352-360, ``setup_envs`` :363-368 and
``heff2`` :374-390. :meth:`Env.measure` is the one genuinely new capability in M11a.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

import tenet
from tenet import IN, OUT, Leg, SymmetricTensor
from tenet.network.common import ones
from tenet.network.mps import MPO, MPS

__all__ = ["Env"]


class _Prepared(NamedTuple):
    """One bond's prepared two-site operator: MPSKit's ten fields, merged for the matvec.

    ``JordanMPO_AC2_Hamiltonian`` (``hamiltonian_derivatives.jl``:168-181) transcribed
    and then merged the way its ``prepare_operator!!`` (:371-460) merges degenerate
    cases, so the apply is few, large contractions rather than ten small ones:

    * ``grt``/``gl`` -- every identity-through path (``II``, ``EE`` and the sign-free
      spectator part of ``AA``, per :class:`~tenet.network.mps.JordanBlocks`'s
      classification) as *one* rank-2 channel map folded into the right environment,
      closed by the untouched left environment; exact for any state;
    * ``caf`` -- every ``IdL``-anchored path (``IC``, ``ID``, ``CB``, ``CA``) summed
      into one field with the *right* environment slices folded in exactly; the left
      closure is MPSKit's one-sided move: the ``IdL`` channel of a left environment
      over left-orthonormal sites is the identity, so no contraction spells it;
    * ``abf`` -- every ``IdR``-anchored path (``AB``, ``BE``, ``DE``), mirrored: left
      environment folded exactly, right ``IdR`` channel the gauge identity;
    * ``ra1``/``ra2``/``sr2`` with ``gl2``/``gr2`` -- the operator-carrying
      open-to-open remainder of ``AA`` (which, for a provider that braids with signs,
      includes the spectators whose ride would drop the string, #160), four factors
      exactly as MPSKit's own ``AA`` (:93): folding an environment into an open-channel
      block would cost ``chi^2 d^2 D_w`` memory.

    ``None`` fields cost nothing in :func:`_apply2` -- MPSKit's ``ismissing`` guards
    (:485-500) -- and they are where the structural zeros go.
    """

    grt: SymmetricTensor | None
    gl: SymmetricTensor | None
    caf: SymmetricTensor | None
    abf: SymmetricTensor | None
    gl2: SymmetricTensor | None
    gr2: SymmetricTensor | None
    ra1: SymmetricTensor | None
    ra2: SymmetricTensor | None
    sr2: SymmetricTensor | None


def _composed(
    equation: str, a: SymmetricTensor, b: SymmetricTensor, bend: str = ""
) -> SymmetricTensor:
    """A two-operand ``tenet.einsum`` with the wires named in ``bend`` bent first.

    The composition rule (``network/__init__.py``) requires operand 1 to supply the
    ``IN`` end of every shared wire. A wire that turns around in the intended planar
    diagram -- one that runs through an environment's cap -- cannot meet that rule as
    drawn, and letting ``einsum`` bend it implicitly would leave the cap direction to
    operand order, which is #147's gate-1 sign. So the bend is spelled: both ends of
    each named wire are moved to the other side with :func:`tenet.repartition`, which
    pays the categorical bend coefficient by construction, and the einsum that follows
    is a plain composition again. Every call site below states its bent wires
    explicitly; a call with ``bend=""`` is a straight composition and could as well be
    ``tenet.einsum``.
    """
    if bend:
        lhs, out = equation.split("->")
        ta, tb = lhs.split(",")

        def bent(t: SymmetricTensor, term: str) -> tuple[SymmetricTensor, str]:
            flip = set(bend)
            outs = tuple(i for i, c in enumerate(term) if (t.legs[i].side is OUT) != (c in flip))
            ins = tuple(i for i in range(len(term)) if i not in outs)
            new = "".join(term[i] for i in (*outs, *ins))
            return tenet.repartition(t, outs, ins), new

        a, ta = bent(a, ta)
        b, tb = bent(b, tb)
        equation = f"{ta},{tb}->{out}"
    return tenet.einsum(equation, a, b)


def _sl(gl: SymmetricTensor, emb: SymmetricTensor | None) -> SymmetricTensor | None:
    """Slice a left environment down to one Jordan group of its MPO leg."""
    return None if emb is None else tenet.einsum("xv,axB->avB", emb, gl)


def _sr(emb: SymmetricTensor | None, gr: SymmetricTensor) -> SymmetricTensor | None:
    """Slice a right environment down to one Jordan group of its MPO leg."""
    return None if emb is None else tenet.einsum("rys,wy->rws", gr, emb)


def _drop(t: SymmetricTensor, i: int) -> SymmetricTensor:
    """Contract away a ``D=1`` unit leg with a ones cap -- MPSKit's ``removeunit``."""
    leg = t.legs[i]
    cap = SymmetricTensor.from_dense(
        np.ones(1), (Leg(leg.space, OUT if leg.side == IN else IN, leg.dual),)
    )
    labels = "abcdefgh"[: t.ndim]
    kept = labels[:i] + labels[i + 1 :]
    if leg.side is IN:  # the cap supplies the OUT end
        return tenet.einsum(f"{labels},{labels[i]}->{kept}", t, cap)
    return tenet.einsum(f"{labels[i]},{labels}->{kept}", cap, t)


#: The ten MPSKit field names, in ``hamiltonian_derivatives.jl``:168-181 order. Presence
#: is recorded per bond next to the (merged) prepared operator, so the sparsity that
#: survived instantiation stays countable after the merge.
_FIELDS = ("II", "IC", "ID", "CB", "CA", "AB", "AA", "BE", "DE", "EE")


def _present(t1, t2) -> tuple[str, ...]:
    """Which of the ten fields this bond populates, read off the two block tables."""
    paths = {
        "II": t1.idl_l is not None and t1.idl_r is not None and t2.idl_r is not None,
        "IC": t1.idl_l is not None and t2.c_op is not None,
        "ID": t1.idl_l is not None and t2.d_op is not None,
        "CB": t1.c_op is not None and t2.b_op is not None,
        "CA": t1.c_op is not None and t2.a_op is not None,
        "AB": t1.a_op is not None and t2.b_op is not None,
        "AA": t1.a_op is not None and t2.a_op is not None,
        "BE": t1.b_op is not None and t2.idr_r is not None,
        "DE": t1.d_op is not None and t2.idr_r is not None,
        "EE": t1.idr_l is not None and t1.idr_r is not None and t2.idr_r is not None,
    }
    return tuple(name for name in _FIELDS if paths[name])


class _Cores(NamedTuple):
    """The environment-independent half of one bond's preparation, built once ever.

    Everything here is a function of the two sites' block tables alone -- MPSKit's
    merged cores with the group embeddings already folded back onto the full bonds --
    so ``Env`` computes it on the bond's first visit and every later
    :func:`_build2` is three environment folds.
    """

    thru: SymmetricTensor | None  # composed identity channels, bond_n -> bond_{n+2}
    caf: SymmetricTensor | None  # [P, p, Q, q, y_full], awaiting the right environment
    abf: SymmetricTensor | None  # [x_full, P, p, Q, q], awaiting the left environment
    ra1: SymmetricTensor | None
    ra2: SymmetricTensor | None
    sr2: SymmetricTensor | None
    open_l: SymmetricTensor | None  # slicers for the AA-remainder chains
    open_r: SymmetricTensor | None


def _cores2(t1, t2, eye_p) -> _Cores:
    """Merge the bond's Jordan blocks into MPSKit's prepared cores, environment-free.

    ``hamiltonian_derivatives.jl``:272-345 in ``tenet.einsum`` -- ``CB = C1 . B2``,
    ``AB = A1 . B2`` -- followed by its ``prepare_operator!!`` merging: the
    one-site-identity fields are padded with ``eye_p`` into the two-site fields that
    share their closure, the two corner channels are one composed rank-2 map, and each
    merged core's group leg is re-embedded onto the full bond so that :func:`_build2`
    folds one whole environment per family. ``None`` stands for every absent piece.

    ``idmap``/``spec_op`` already exclude any spectator whose state braids with signs
    (:class:`~tenet.network.mps.JordanBlocks`): its string crossing lives in the
    rank-4 blocks, so it reaches the matvec through the ``AA`` chains below instead of
    the phys-free ``thru`` ride (#160).
    """
    thru = None
    if t1.idmap is not None and t2.idmap is not None:
        thru = tenet.einsum("yz,xy->xz", t2.idmap, t1.idmap)

    caf = None  # the dropped IdL leg is the left gauge identity
    if t1.idl_l is not None:
        core_e = core_2 = None  # by right target: the IdR channel against the open group
        if t2.c_op is not None:  # IC, starting right
            core_2 = tenet.einsum("Pp,vQqw->vPpQqw", eye_p, t2.c_op)
        if t2.d_op is not None:  # ID, onsite right
            core_e = tenet.einsum("Pp,vQqw->vPpQqw", eye_p, t2.d_op)
        if t1.c_op is not None and t2.b_op is not None:  # CB
            part = tenet.einsum("mQqw,vPpm->vPpQqw", t2.b_op, t1.c_op)
            core_e = part if core_e is None else tenet.add(core_e, part)
        if t1.c_op is not None and t2.a_op is not None:  # CA
            part = tenet.einsum("mQqw,vPpm->vPpQqw", t2.a_op, t1.c_op)
            core_2 = part if core_2 is None else tenet.add(core_2, part)
        if core_e is not None and t2.idr_r is not None:
            caf = tenet.einsum("wy,PpQqw->PpQqy", t2.idr_r, _drop(core_e, 0))
        if core_2 is not None and t2.open_r is not None:
            part = tenet.einsum("wy,PpQqw->PpQqy", t2.open_r, _drop(core_2, 0))
            caf = part if caf is None else tenet.add(caf, part)

    abf = None  # the dropped IdR leg is the right gauge identity
    if t2.idr_r is not None:
        core_w = None  # [w, P, p, Q, q] on the left open group
        if t1.a_op is not None and t2.b_op is not None:  # AB
            core_w = _drop(tenet.einsum("mQqv,wPpm->wPpQqv", t2.b_op, t1.a_op), 5)
        if t1.b_op is not None:  # BE, ending left
            part = tenet.einsum("Qq,wPp->wPpQq", eye_p, _drop(t1.b_op, 3))
            core_w = part if core_w is None else tenet.add(core_w, part)
        if core_w is not None and t1.open_l is not None:
            abf = tenet.einsum("wPpQq,xw->xPpQq", core_w, t1.open_l)
        if t1.d_op is not None and t1.idl_l is not None:  # DE, onsite left
            core_v = tenet.einsum("Qq,vPp->vPpQq", eye_p, _drop(t1.d_op, 3))
            part = tenet.einsum("vPpQq,xv->xPpQq", core_v, t1.idl_l)
            abf = part if abf is None else tenet.add(abf, part)

    # The operator-carrying open-to-open remainder of AA: A1.A2 minus the free-riding
    # spectator.spectator, split as R1.A2 plus S1.R2 so that part never pays a per-edge
    # anything. For a provider that braids with signs, R includes the spectators whose
    # rank-2 ride would drop the string (docstring above).
    ra1 = ra2 = sr2 = None
    if t1.open_l is not None and t2.open_r is not None:
        if t1.a_real_op is not None and t2.a_op is not None:
            ra1, ra2 = t1.a_real_op, t2.a_op
        if t1.spec_op is not None and t2.a_real_op is not None:
            sr2 = tenet.einsum("uQqy,wu->wQqy", t2.a_real_op, t1.spec_op)

    return _Cores(thru, caf, abf, ra1, ra2, sr2, t1.open_l, t2.open_r)


def _build2(cores: _Cores, gl: SymmetricTensor, gr: SymmetricTensor) -> _Prepared:
    """Fold the two environments into the bond's static cores, once per Krylov solve.

    MPSKit's ``AC2_hamiltonian`` moment: everything block-sized was merged ahead of time
    by :func:`_cores2`, so this is one whole-environment contraction per populated
    family -- paid once and used for all of ``lanczos``'s matvecs at the bond.
    """
    grt = None
    if cores.thru is not None:
        grt = tenet.einsum("rzs,xz->rxs", gr, cores.thru)
    caf = None
    if cores.caf is not None:
        caf = tenet.einsum("rys,PpQqy->PpQqrs", gr, cores.caf)
    abf = None
    if cores.abf is not None:
        abf = tenet.einsum("xPpQq,axB->aPpQqB", cores.abf, gl)
    gl2 = gr2 = None
    if cores.ra1 is not None or cores.sr2 is not None:
        gl2 = _sl(gl, cores.open_l)
        gr2 = _sr(cores.open_r, gr)
    return _Prepared(
        grt,
        gl if grt is not None else None,
        caf,
        abf,
        gl2,
        gr2,
        cores.ra1,
        cores.ra2,
        cores.sr2,
    )


def _apply2(p: _Prepared, aa: SymmetricTensor) -> SymmetricTensor:
    """The prepared two-site matvec: MPSKit :485-500 after merging, in tenet legs.

    A pure function of ``(prepared operator, aa)`` with fixed contraction structure --
    every branch is decided by which fields are ``None``, which is part of the structure
    key -- so it is traceable and is what :meth:`Env.heff2` hands to ``compile=``. Four
    contractions for a string-built Hamiltonian -- the identity-through pair plus one
    one-sided field per anchor -- against the dense path's four over the full
    ``D_w``-wide ``W`` pair, and only the first pair still carries an MPO-bond leg.

    The one-sided ``caf``/``abf`` terms use the two-site sweep's mixed-canonical gauge,
    exactly as MPSKit's matvec does: sites left of the bond left-orthonormal, sites
    right of it right-orthonormal, so the ``IdL``/``IdR`` environment channels are
    identities nobody contracts. :func:`~tenet.network.sweep_` maintains that gauge at
    every bond it visits; :meth:`Env.heff2`'s docstring states the precondition.
    """
    y = None

    def acc(t: SymmetricTensor) -> None:
        nonlocal y
        y = t if y is None else tenet.add(y, t)

    if p.grt is not None:  # II + EE + sign-free spectator AA: identity through both sites
        t = _composed("rxs,apqr->apqxs", p.grt, aa, bend="r")
        acc(_composed("axB,apqxs->Bpqs", p.gl, t, bend="x"))
    if p.caf is not None:  # IC + ID + CB + CA: the IdL channel left of the bond is gauge-1
        acc(_composed("PpQqrs,apqr->aPQs", p.caf, aa, bend="r"))
    if p.abf is not None:  # AB + BE + DE: the IdR channel right of the bond is gauge-1
        acc(tenet.einsum("aPpQqB,apqr->BPQr", p.abf, aa))
    if p.ra1 is not None:  # AA remainder, left factor operator-carrying
        t = tenet.einsum("apqr,rws->apqws", aa, p.gr2)
        t = _composed("apqws,mQqw->apQms", t, p.ra2, bend="q")
        t = _composed("apQms,wPpm->aPQws", t, p.ra1, bend="p")
        acc(_composed("aPQws,awB->BPQs", t, p.gl2, bend="a"))
    if p.sr2 is not None:  # AA remainder, free spectator left then operator right
        t = tenet.einsum("apqr,rws->apqws", aa, p.gr2)
        t = _composed("apqws,mQqw->apQms", t, p.sr2, bend="q")
        acc(_composed("apQms,amB->BpQs", t, p.gl2, bend="a"))
    return y


def _fold_last(t, f, a, bra) -> SymmetricTensor:
    """One prepared left-to-right environment step, exact for any state.

    The half-transferred ``F . A`` is shared by every path; the identity channels --
    corners and free-riding spectators at once -- then ride ``idmap`` with no ``W`` contraction at
    all, and only the operator-carrying blocks (``c``/``d`` from ``IdL``, ``b`` and the
    real part of ``a`` from the open group) pay a block-sized ``W`` step each.
    """
    t1 = tenet.einsum("axB,apr->xBpr", f, a)
    out = None
    if t.idmap is not None:
        comp = tenet.einsum("Bps,xBpr->rxs", bra, t1)
        out = tenet.einsum("xm,rxs->rms", t.idmap, comp)

    def flow(src, w, emb):
        s = tenet.einsum("vPpw,vBpr->BrPw", w, src)
        s = tenet.einsum("BPs,BrPw->rws", bra, s)
        return tenet.einsum("wm,rws->rms", emb, s)

    if t.idl_l is not None and (t.c_op is not None or t.d_op is not None):
        ti = tenet.einsum("xv,xBpr->vBpr", t.idl_l, t1)
        if t.c_op is not None:
            part = flow(ti, t.c_op, t.open_r)
            out = part if out is None else tenet.add(out, part)
        if t.d_op is not None:
            part = flow(ti, t.d_op, t.idr_r)
            out = part if out is None else tenet.add(out, part)
    if t.open_l is not None and (t.b_op is not None or t.a_real_op is not None):
        to = tenet.einsum("xv,xBpr->vBpr", t.open_l, t1)
        if t.b_op is not None:
            part = flow(to, t.b_op, t.idr_r)
            out = part if out is None else tenet.add(out, part)
        if t.a_real_op is not None:
            part = flow(to, t.a_real_op, t.open_r)
            out = part if out is None else tenet.add(out, part)
    return out


def _fold_first(t, f, a, bra) -> SymmetricTensor:
    """One prepared right-to-left environment step -- :func:`_fold_last` mirrored.

    Mirrored in the cap sense, not only in the loop direction: a right-directed
    environment is built from the right boundary inward, so in the intended planar
    diagram every bond rail -- the ket bond ``r``, the MPO bond ``y``/``w``/``v``, the
    bra bond ``s`` -- runs through the right cap and turns around, while the physical
    wires compose straight. Each contraction therefore bends its bond-rail wire
    explicitly (:func:`_composed`) and composes the rest; the dense Jordan-Wigner
    oracle fixes every one of these choices (#160), and :func:`_fold_last`, whose
    rails run *out* of the left cap, needs no bend at all.
    """
    t1 = _composed("rys,apr->apys", f, a, bend="r")
    out = None
    if t.idmap is not None:
        comp = _composed("Bps,apys->ayB", bra, t1, bend="s")
        out = _composed("xy,ayB->axB", t.idmap, comp, bend="y")

    def flow(src, w, emb):
        s = _composed("vPpw,apws->avPs", w, src, bend="w")
        s = _composed("BPs,avPs->avB", bra, s, bend="s")
        return _composed("xv,avB->axB", emb, s, bend="v")

    if t.idr_r is not None and (t.b_op is not None or t.d_op is not None):
        te = _composed("wy,apys->apws", t.idr_r, t1, bend="y")
        if t.d_op is not None:
            part = flow(te, t.d_op, t.idl_l)
            out = part if out is None else tenet.add(out, part)
        if t.b_op is not None:
            part = flow(te, t.b_op, t.open_l)
            out = part if out is None else tenet.add(out, part)
    if t.open_r is not None and (t.c_op is not None or t.a_real_op is not None):
        to = _composed("wy,apys->apws", t.open_r, t1, bend="y")
        if t.c_op is not None:
            part = flow(to, t.c_op, t.idl_l)
            out = part if out is None else tenet.add(out, part)
        if t.a_real_op is not None:
            part = flow(to, t.a_real_op, t.open_l)
            out = part if out is None else tenet.add(out, part)
    return out


class Env:
    """``<psi|H|psi>`` partial contractions for one ``(psi, h)`` pair.

    ``F[(n, n + 1)]``: ``(ket IN, mpo OUT, bra OUT)``, built from sites ``<= n``;
    ``F[(n, n - 1)]``: ``(ket OUT, mpo IN, bra IN)``, built from sites ``>= n``.

    The two orientations make every contraction in :meth:`update_` and :meth:`heff2`
    meet IN against OUT -- and that condition is **not enough**, because it is
    symmetric: it fixes contractibility only, while the cap sign depends on *which
    operand supplies which end* (the composition rule, ``network/__init__.py``).
    An earlier revision argued exactly the symmetric reading -- "IN against OUT with
    no leg bend anywhere in this module" -- and #147's gate 1 measured it to be
    insufficient: the operand orders were individually well-formed and still paid one
    Koszul sign per odd wire capped in the wrong direction. Every contraction here is
    therefore a composition with operand 1 supplying IN, and the wires that genuinely
    bend -- the MPS bond arrow and the MPO bond arrow cross the two-site cell in
    opposite directions, so closing either cap turns one rail around -- are bent
    explicitly through :func:`_composed`, each choice pinned by the dense
    Jordan-Wigner oracle (#160).

    A plain ``dict`` keyed by *directed* bond, exactly YASTN's ``Env``
    (``yastn/tn/mps/_env.py``:94-125). A list-of-left / list-of-right would hide the
    invalidation discipline, which is the entire correctness content of an environment
    cache -- a stale ``F[(n, n+1)]`` after site ``n`` changed gives an energy that is
    *plausible and wrong*, the worst failure mode a DMRG has. :meth:`clear_` therefore
    pops **both** directed bonds per site, and it runs *before* the replacement is
    written, so a missed update is a ``KeyError`` rather than a wrong number.

    **One class, not YASTN's factory over eight.** ``yastn.tn.mps.Env`` is a function
    dispatching into ``Env2``, ``Env_mps_mpo_mps``, ``…_precompute``, ``Env_mpo_mpo_mpo``,
    ``Env_mps_mpopbc_mps``, ``Env_sum``, ``Env_project`` (``_env.py``:26-89), and every
    one of those serves a feature M11a does not ship -- MPO products, PBC, sums of
    Hamiltonians, excited-state penalties. The dispatch arrives if and when a second
    target does.
    """

    F: dict[tuple[int, int], SymmetricTensor]

    def __init__(self, psi: MPS, h: MPO, *, compile: Callable | None = None) -> None:
        self.psi = psi
        self.h = h
        # ``compile=`` wraps the prepared matvec once per structure key -- ``jax.jit`` at
        # the application level (``examples/bench_dmrg.py``); this layer names no
        # accelerator and ``None`` runs the plain Python function.
        self.compile = compile
        self._prepared: dict[int, tuple] = {}  # bond -> (GL, GR, _Prepared, fields), one each
        self._cores: dict[int, _Cores] = {}  # bond -> environment-free merged blocks
        self._compiled: dict[int, tuple] = {}  # bond -> (leg key, _Prepared, callable)
        self._eye_p: SymmetricTensor | None = None  # the physical identity, for field padding
        n = len(psi)
        kl, kr = psi[0].legs[0], psi[n - 1].legs[2]
        wl, wr = h[0].legs[0], h[n - 1].legs[3]
        # The boundary legs carry the state's and the operator's own ``dual`` flags, so
        # a dual boundary bond contracts instead of refusing about the wrong wire.
        self.F = {
            (-1, 0): ones(
                (
                    Leg(kl.space, IN, kl.dual),
                    Leg(wl.space, OUT, wl.dual),
                    Leg(kl.space, OUT, kl.dual),
                )
            ),
            (n, n - 1): ones(
                (
                    Leg(kr.space, OUT, kr.dual),
                    Leg(wr.space, IN, wr.dual),
                    Leg(kr.space, IN, kr.dual),
                )
            ),
        }

    def setup_(self, to: int = 0) -> "Env":
        """Build every environment directed towards site ``to``, and return ``self``.

        ``to=0`` is YASTN's ``setup_(to='first')`` (``_env.py``:104-125): for a
        right-canonical ``psi`` this is every right-directed environment, and it is the
        state a left-to-right sweep starts from.
        """
        if to != 0:
            raise NotImplementedError("only to=0 is implemented; canonize_ has the same note")
        for n in range(len(self.psi) - 1, 0, -1):
            self.update_(n, to="first")
        return self

    def update_(self, n: int, *, to: str) -> None:
        """Write one directed-bond entry from its neighbour -- YASTN ``_env.py``:152-168.

        ``to='last'`` writes ``F[(n, n+1)]`` from ``F[(n-1, n)]``; ``to='first'`` writes
        ``F[(n, n-1)]`` from ``F[(n+1, n)]``. Dense path: three pairwise ``tenet.einsum``
        calls each -- environment first, then the ket, then the MPO, then the bra. With a
        Jordan block table present the step goes edge-aware instead
        (:func:`_fold_last` / :func:`_fold_first`): the identity channels ride ``idmap``
        with no ``W`` contraction, only the operator-carrying blocks pay one, and unlike
        :meth:`heff2` this path is **exact for any state** -- no gauge assumption.
        """
        a, bra = self.psi[n], tenet.adjoint(self.psi[n])
        blocks = self.h.jordan(n)
        if to == "last":
            if blocks is not None:
                self.F[n, n + 1] = _fold_last(blocks, self.F[n - 1, n], a, bra)
                return
            t = tenet.einsum("axB,apr->xBpr", self.F[n - 1, n], a)
            t = tenet.einsum("xPpm,xBpr->BrPm", self.h[n], t)
            self.F[n, n + 1] = tenet.einsum("BPs,BrPm->rms", bra, t)
        else:
            if blocks is not None:
                self.F[n, n - 1] = _fold_first(blocks, self.F[n + 1, n], a, bra)
                return
            t = tenet.einsum("apr,rys->apys", a, self.F[n + 1, n])
            t = _composed("apys,xPpy->axPs", t, self.h[n], bend="p")
            self.F[n, n - 1] = _composed("axPs,BPs->axB", t, bra, bend="P")

    def clear_(self, *sites: int) -> None:
        """Pop **both** directed bonds touching each changed site -- YASTN ``clear_site_``."""
        for n in sites:
            self.F.pop((n, n - 1), None)
            self.F.pop((n, n + 1), None)

    def heff2(self, n: int, aa: SymmetricTensor) -> SymmetricTensor:
        """``H_eff`` on the two-site tensor at bond ``(n, n+1)``. Two paths, one output.

        **Prepared path**, taken when ``self.h`` carries a Jordan block table -- which
        requires ``MPO.from_terms(..., cutoff=None)``, the precondition, because a
        compressed or ``from_w`` MPO has no edge structure left to prepare. The two
        environments are folded into the site blocks **once per bond** (:func:`_build2`,
        MPSKit's ``AC2_hamiltonian``) and cached against the environment tensors'
        identity, so one ``lanczos`` solve at ``ncv=3`` pays the fold once and applies
        :func:`_apply2` three times; absent fields are ``None`` and are skipped, which is
        where the structural zeros go. Like MPSKit's matvec, the ``IdL``/``IdR``-anchored
        terms are one-sided: they use the sweep's **mixed-canonical gauge** -- sites left
        of the bond left-orthonormal, sites right of it right-orthonormal, which
        :func:`~tenet.network.sweep_` maintains at every bond -- as the second
        precondition; environments built from a differently-gauged state belong to the
        dense path. The apply itself is compiled through ``compile=`` once per structure
        key -- the tuple of ``aa``'s legs plus the prepared operator's identity -- and
        the cache holds one entry per bond, replaced when the key moves.

        **Dense path**, for every MPO without a table: right environment, then ``W2``,
        then ``W1``, then the left environment -- YASTN's ``Env_mps_mpo_mps.Heff2`` order
        (``_env.py``:496-518) with ``precompute=False``, which ``_dmrg.py``:102-108
        documents as ``O(D^3 M d + D^2 M^2 d^2)``.

        The two paths agree as operators but sum their terms in a different order, so
        they agree to solver precision, never bitwise. In and out on ``(left bond OUT, p
        OUT, q OUT, right bond IN)``: the *bra* legs of the two environments become the
        output's bonds while the *ket* legs close against the input's, which is why the
        result has ``aa``'s structure exactly and :func:`~tenet.network.lanczos` can add
        the two.
        """
        if self.h.jordan(n) is not None:
            p = self._prepare2(n)
            key = tuple(aa.legs)
            hit = self._compiled.get(n)
            if hit is None or hit[0] != key or hit[1] is not p:
                fn = _apply2 if self.compile is None else self.compile(_apply2)
                hit = (key, p, fn)
                self._compiled[n] = hit
            return hit[2](p, aa)
        t = tenet.einsum("apqr,rys->apqys", aa, self.F[n + 2, n + 1])
        t = _composed("apqys,mQqy->apQms", t, self.h[n + 1], bend="q")
        t = _composed("apQms,xPpm->aPQxs", t, self.h[n], bend="p")
        return _composed("aPQxs,axB->BPQs", t, self.F[n - 1, n], bend="a")

    def _prepare2(self, n: int) -> _Prepared:
        """The bond's prepared operator, rebuilt only when either environment moved.

        Cached one per bond against the ``F`` entries *by identity*: environments are
        frozen tensors replaced on every :meth:`update_`, so holding the two used to
        build is both the invalidation test and the guarantee a stale operator can
        never be served -- the discipline :attr:`F` itself uses, one level up.
        """
        fl, fr = self.F[n - 1, n], self.F[n + 2, n + 1]
        hit = self._prepared.get(n)
        if hit is not None and hit[0] is fl and hit[1] is fr:
            return hit[2]
        t1, t2 = self.h.jordan(n), self.h.jordan(n + 1)
        if n not in self._cores:  # environment-free, so never invalidated
            if self._eye_p is None:
                self._eye_p = tenet.identity((self.psi[0].legs[1],))
            self._cores[n] = _cores2(t1, t2, self._eye_p)
        p = _build2(self._cores[n], fl, fr)
        self._prepared[n] = (fl, fr, p, _present(t1, t2))
        return p

    def measure(self) -> float:
        """``<psi|H|psi>`` without the eigensolver, on a private left-to-right pass.

        The first thing in this repository that measures a converged energy independently
        of the ``lanczos`` Rayleigh quotient that produced it. YASTN's ``measure`` is the
        same closing contraction one level down (``_env.py``:462-468, ``vdot(vecL,
        vecR)``); the pass is built in a fresh :class:`Env` so a measurement never writes
        into a sweep's cache.
        """
        n = len(self.psi)
        env = Env(self.psi, self.h)
        for site in range(n):
            env.update_(site, to="last")
        closed = tenet.einsum("Rms,rms->Rr", env.F[n, n - 1], env.F[n - 1, n])
        return float(tenet.full_trace(closed))

    def __repr__(self) -> str:
        return f"Env(sites={len(self.psi)}, bonds={sorted(self.F)})"

    def __contains__(self, key: Any) -> bool:
        return key in self.F
