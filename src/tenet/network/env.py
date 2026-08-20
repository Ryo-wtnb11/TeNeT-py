"""The environment cache: ``<psi|H|psi>`` partial contractions, keyed by *directed* bond.

Promoted from ``examples/toy_codes/dmrg.py`` (#110) with no arithmetic change: ``boundary_envs``
:318-331, ``update_env`` :334-349, ``invalidate`` :352-360, ``setup_envs`` :363-368 and
``heff2`` :374-390. [Env.measure][tenet.network.Env.measure] is the one genuinely new
capability in M11a.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

import tenet
from tenet import IN, OUT, Leg, SymmetricTensor
from tenet.network.common import Recent, ones
from tenet.network.mps import MPO, MPS, EdgeBlocks, EdgeTable

__all__ = ["Env"]


class _Prepared(NamedTuple):
    """One bond's prepared two-site operator: MPSKit's ten fields, merged for the matvec.

    ``JordanMPO_AC2_Hamiltonian`` (``hamiltonian_derivatives.jl``:168-181) transcribed
    and then merged the way its ``prepare_operator!!`` (:371-460) merges degenerate
    cases, so the apply is few, large contractions rather than ten small ones:

    * ``grt``/``gl`` -- every identity-through path (``II``, ``EE`` and the sign-free
      spectator part of ``AA``, per ``EdgeBlocks``'s
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

    ``None`` fields cost nothing in ``_apply2`` -- MPSKit's ``ismissing`` guards
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

    The composition rule (``docs/design.md`` "Milestone 11") requires operand 1 to supply
    the ``IN`` end of every shared wire. A wire that turns around in the intended planar
    diagram -- one that runs through an environment's cap -- cannot meet that rule as
    drawn, and letting ``einsum`` bend it implicitly would leave the cap direction to
    operand order, which is #147's gate-1 sign. So the bend is spelled: both ends of
    each named wire are moved to the other side with [tenet.repartition][], which
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
    """Slice a left environment down to one edge group of its MPO leg."""
    return None if emb is None else tenet.einsum("xv,axB->avB", emb, gl)


def _sr(emb: SymmetricTensor | None, gr: SymmetricTensor) -> SymmetricTensor | None:
    """Slice a right environment down to one edge group of its MPO leg."""
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
    ``_build2`` is three environment folds.
    """

    thru: SymmetricTensor | None  # composed identity channels, bond_n -> bond_{n+2}
    caf: SymmetricTensor | None  # [P, p, Q, q, y_full], awaiting the right environment
    abf: SymmetricTensor | None  # [x_full, P, p, Q, q], awaiting the left environment
    ra1: SymmetricTensor | None
    ra2: SymmetricTensor | None
    sr2: SymmetricTensor | None
    open_l: SymmetricTensor | None  # slicers for the AA-remainder chains
    open_r: SymmetricTensor | None


def _cores2(edges: EdgeTable, n: int, eye_p: SymmetricTensor) -> _Cores:
    """Merge bond ``n``'s edge blocks into MPSKit's prepared cores, environment-free.

    Built from the MPO's **edge description**, not from its site tensors (#200): the two
    sites' [EdgeBlocks][tenet.network.EdgeBlocks] are asked for here, which places their
    operators against the group slot maps and builds the group embeddings the merge
    needs, so nothing on this path is ever a full-width rank-4 ``W``. That is the
    instantiation boundary #184 staged as candidate (a), and it is where the next stage's
    per-cut assembler plugs in: this function's contract is "give me bond ``n``'s cores",
    and how the description answers is the assembler's business.

    The merge itself is ``hamiltonian_derivatives.jl``:272-345 in ``tenet.einsum`` --
    ``CB = C1 . B2``, ``AB = A1 . B2`` -- followed by its ``prepare_operator!!`` merging:
    the one-site-identity fields are padded with ``eye_p`` into the two-site fields that
    share their closure, the two corner channels are one composed rank-2 map, and each
    merged core's group leg is re-embedded onto the full bond so that ``_build2``
    folds one whole environment per family. ``None`` stands for every absent piece.

    ``idmap``/``spec_op`` already exclude any spectator whose state braids with signs
    (``EdgeBlocks``): its string crossing lives in the
    rank-4 blocks, so it reaches the matvec through the ``AA`` chains below instead of
    the phys-free ``thru`` ride (#160).
    """
    t1, t2 = edges.edge_blocks(n), edges.edge_blocks(n + 1)
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
    by ``_cores2``, so this is one whole-environment contraction per populated
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


def _families2(p: _Prepared, aa: SymmetricTensor) -> list[SymmetricTensor]:
    """The matvec's term families, each applied to ``aa`` separately and not yet summed.

    One entry per populated family of the prepared operator, in ``_apply2``'s accumulation
    order; absent families contribute nothing, exactly as they contribute nothing to the
    sum. [Env.heff2_families][tenet.network.Env.heff2_families] is the public read, and
    [sweep_][tenet.network.sweep_]'s perturbative noise is what wants the pieces apart
    rather than added up.
    """
    parts: list[SymmetricTensor] = []
    # The per-line ignores below: _prepare sets _Prepared's fields in groups —
    # ``gl`` with ``grt``, ``gl2``/``gr2`` with ``ra1`` or ``sr2``, ``ra2`` with
    # ``ra1`` — so inside each branch the sibling field is non-None; a cross-field
    # invariant no checker narrows.
    if p.grt is not None:  # II + EE + sign-free spectator AA: identity through both sites
        t = _composed("rxs,apqr->apqxs", p.grt, aa, bend="r")
        parts.append(_composed("axB,apqxs->Bpqs", p.gl, t, bend="x"))  # ty: ignore[invalid-argument-type]
    if p.caf is not None:  # IC + ID + CB + CA: the IdL channel left of the bond is gauge-1
        parts.append(_composed("PpQqrs,apqr->aPQs", p.caf, aa, bend="r"))
    if p.abf is not None:  # AB + BE + DE: the IdR channel right of the bond is gauge-1
        parts.append(tenet.einsum("aPpQqB,apqr->BPQr", p.abf, aa))
    if p.ra1 is not None:  # AA remainder, left factor operator-carrying
        t = tenet.einsum("apqr,rws->apqws", aa, p.gr2)  # ty: ignore[invalid-argument-type]
        t = _composed("apqws,mQqw->apQms", t, p.ra2, bend="q")  # ty: ignore[invalid-argument-type]
        t = _composed("apQms,wPpm->aPQws", t, p.ra1, bend="p")
        parts.append(_composed("aPQws,awB->BPQs", t, p.gl2, bend="a"))  # ty: ignore[invalid-argument-type]
    if p.sr2 is not None:  # AA remainder, free spectator left then operator right
        t = tenet.einsum("apqr,rws->apqws", aa, p.gr2)  # ty: ignore[invalid-argument-type]
        t = _composed("apqws,mQqw->apQms", t, p.sr2, bend="q")
        parts.append(_composed("apQms,amB->BpQs", t, p.gl2, bend="a"))  # ty: ignore[invalid-argument-type]
    return parts


def _apply2(p: _Prepared, aa: SymmetricTensor) -> SymmetricTensor:
    """The prepared two-site matvec: MPSKit :485-500 after merging, in tenet legs.

    A pure function of ``(prepared operator, aa)`` with fixed contraction structure --
    every branch is decided by which fields are ``None``, which is part of the structure
    key -- so it is traceable and is what [Env.heff2][tenet.network.Env.heff2] hands to
    ``compile=``. Four
    contractions for a string-built Hamiltonian -- the identity-through pair plus one
    one-sided field per anchor -- against the compatibility entry's four over the full
    ``D_w``-wide ``W`` pair, and only the first pair still carries an MPO-bond leg.

    The one-sided ``caf``/``abf`` terms use the two-site sweep's mixed-canonical gauge,
    exactly as MPSKit's matvec does: sites left of the bond left-orthonormal, sites
    right of it right-orthonormal, so the ``IdL``/``IdR`` environment channels are
    identities nobody contracts. [sweep_][tenet.network.sweep_] maintains that gauge at
    every bond it visits; [Env.heff2][tenet.network.Env.heff2]'s docstring states the precondition.

    The families are applied by ``_families2`` and summed here in its order, which is the
    order the accumulator used before the two were separated -- the sum is unchanged term
    for term.
    """
    parts = _families2(p, aa)
    y = parts[0]  # some branch always fired: h has at least one term
    for t in parts[1:]:
        y = tenet.add(y, t)
    return y


def _fold_last(
    t: EdgeBlocks, f: SymmetricTensor, a: SymmetricTensor, bra: SymmetricTensor
) -> SymmetricTensor:
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

    def flow(src, w, emb) -> SymmetricTensor:
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
    # at least one of the operator groups is present in every table row, so the
    # accumulator is assigned before the return; ty sees only the None seed
    return out  # ty: ignore[invalid-return-type]


def _fold_first(
    t: EdgeBlocks, f: SymmetricTensor, a: SymmetricTensor, bra: SymmetricTensor
) -> SymmetricTensor:
    """One prepared right-to-left environment step -- ``_fold_last`` mirrored.

    Mirrored in the cap sense, not only in the loop direction: a right-directed
    environment is built from the right boundary inward, so in the intended planar
    diagram every bond rail -- the ket bond ``r``, the MPO bond ``y``/``w``/``v``, the
    bra bond ``s`` -- runs through the right cap and turns around, while the physical
    wires compose straight. Each contraction therefore bends its bond-rail wire
    explicitly (``_composed``) and composes the rest; the dense Jordan-Wigner
    oracle fixes every one of these choices (#160), and ``_fold_last``, whose
    rails run *out* of the left cap, needs no bend at all.
    """
    t1 = _composed("rys,apr->apys", f, a, bend="r")
    out = None
    if t.idmap is not None:
        comp = _composed("Bps,apys->ayB", bra, t1, bend="s")
        out = _composed("xy,ayB->axB", t.idmap, comp, bend="y")

    def flow(src, w, emb) -> SymmetricTensor:
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
    # at least one of the operator groups is present in every table row, so the
    # accumulator is assigned before the return; ty sees only the None seed
    return out  # ty: ignore[invalid-return-type]


def _heff2_full(h: MPO, fl: SymmetricTensor, fr: SymmetricTensor, n: int, aa: SymmetricTensor):
    """``<bra env| W_n W_{n+1} |aa, ket env>`` by the four full contractions, no gauge asked.

    YASTN's ``Env_mps_mpo_mps.Heff2`` order (``_env.py``:496-518) with
    ``precompute=False``: right environment, then ``W2``, then ``W1``, then the left
    environment. Exact for any state and for a bra that is not the ket, because nothing
    here reads an ``IdL``/``IdR`` channel as a gauge identity -- which is what makes it
    both [Env.heff2][tenet.network.Env.heff2]'s compatibility entry *and*
    [Env.project2][tenet.network.Env.project2]'s whole body.

    ``aa`` lives on the *ket* bonds and the result on the *bra* bonds; the two coincide
    for a one-state [Env][tenet.network.Env] and that is why ``heff2`` can iterate on it.
    """
    t = tenet.einsum("apqr,rys->apqys", aa, fr)
    t = _composed("apqys,mQqy->apQms", t, h[n + 1], bend="q")
    t = _composed("apQms,xPpm->aPQxs", t, h[n], bend="p")
    return _composed("aPQxs,axB->BPQs", t, fl, bend="a")


class Env:
    """``<bra|H|psi>`` partial contractions for one ``(psi, h)`` pair -- ``bra = psi`` by default.

    Parameters
    ----------
    psi : MPS
        The **ket**; the cache holds views into its current tensors.
    h : MPO
        The Hamiltonian.
    bra : MPS or None, optional
        The **bra**, when it is not ``psi``. Default ``None``, meaning
        ``bra is psi`` -- today's one-state environment, ``<psi|H|psi>``.
        With a second state given, every environment is the mixed transfer
        contraction and ``Env(psi, h, bra=phi).measure()`` **is**
        ``<phi|H|psi>``. Keyword-only.
    compile : Callable or None, optional
        Wraps the prepared matvec once per structure key -- ``jax.jit`` at the
        application level; this layer names no accelerator and ``None`` (the
        default) runs the plain Python function. Keyword-only.

    Notes
    -----
    ``F[(n, n + 1)]``: ``(ket IN, mpo OUT, bra OUT)``, built from sites ``<= n``;
    ``F[(n, n - 1)]``: ``(ket OUT, mpo IN, bra IN)``, built from sites ``>= n``.

    The two orientations make every contraction in [update_][tenet.network.Env.update_] and
    [heff2][tenet.network.Env.heff2]
    meet IN against OUT -- and that condition is **not enough**, because it is
    symmetric: it fixes contractibility only, while the cap sign depends on *which
    operand supplies which end*. Every contraction here is therefore a composition
    with operand 1 supplying IN, and the wires that genuinely bend -- the MPS bond
    arrow and the MPO bond arrow cross the two-site cell in opposite directions, so
    closing either cap turns one rail around -- are bent explicitly through
    ``_composed``, each choice pinned by the dense Jordan-Wigner oracle (#160). The
    rule and why the symmetric reading is insufficient: ``docs/design.md``
    "Milestone 11".

    A plain ``dict`` keyed by *directed* bond, exactly YASTN's ``Env``
    (``yastn/tn/mps/_env.py``:94-125). A list-of-left / list-of-right would hide the
    invalidation discipline, which is the entire correctness content of an environment
    cache -- a stale ``F[(n, n+1)]`` after site ``n`` changed gives an energy that is
    *plausible and wrong*, the worst failure mode a DMRG has. [clear_][tenet.network.Env.clear_]
    therefore
    pops **both** directed bonds per site, and it runs *before* the replacement is
    written, so a missed update is a ``KeyError`` rather than a wrong number.

    Why one class rather than YASTN's factory over eight: ``docs/design.md``
    "Milestone 11".

    **The two-state form** (``bra=phi``) builds ``<phi| ... |psi>`` instead, which is the
    engine half of the excited-state and measurement machinery: block2's ``ext_mes`` are
    moving environments between two *different* states
    (``sweep_algorithm.hpp``:1195-1206). What it asks of its two states is **nothing**:
    ``update_``'s folds (``_fold_last``/``_fold_first`` and the site-tensor branch alike)
    are exact for any pair of chains, and ``MPS._braket``'s docstring already records the
    same fact one level down -- the two chains may carry different bond spaces, because
    the transfer tensor holds one index from each. Two-state environments are therefore
    gauge-free, and [measure][tenet.network.Env.measure] on one is ``<phi|H|psi>``
    undivided, for a ``phi`` and a ``psi`` in any gauge and at any norm.

    What the two-state form does **not** support is
    [heff2][tenet.network.Env.heff2]: the prepared matvec's one-sided terms read the
    ``IdL``/``IdR`` environment channels as gauge identities, which is true of a
    left-orthonormal chain against *itself* and false of a mixed transfer, so it refuses
    rather than returning a plausible wrong operator. block2 does not iterate on its
    ``ext_mes`` either -- it calls ``multiply`` on them, once per bond, to produce a
    *projection vector*. [project2][tenet.network.Env.project2] is that call.
    """

    F: dict[tuple[int, int], SymmetricTensor]

    def __init__(
        self, psi: MPS, h: MPO, *, bra: MPS | None = None, compile: Callable | None = None
    ) -> None:
        self.psi = psi
        self.bra = psi if bra is None else bra
        self.h = h
        # ``compile=`` wraps the prepared matvec once per structure key -- ``jax.jit`` at
        # the application level (``examples/bench_dmrg.py``); this layer names no
        # accelerator and ``None`` runs the plain Python function.
        self.compile = compile
        # Three per-bond caches, all held to one byte budget by ``common.CACHE_BUDGET``
        # (#202). The merged cores are environment-free and so are never *invalidated* --
        # but a sweep visits every bond, so an unbounded cache of them holds the whole
        # prepared operator at once, which is the memory the deferred instantiation
        # boundary exists not to spend. Recency, not correctness, is what decides an
        # eviction here; the environments in ``F`` keep their own invalidation discipline
        # and are untouched by this.
        self._prepared: dict[int, tuple] = Recent()  # bond -> (GL, GR, _Prepared, fields)
        self._cores: dict[int, _Cores] = Recent()  # bond -> environment-free merged blocks
        self._compiled: dict[int, tuple] = Recent()  # bond -> (leg key, _Prepared, callable)
        self._eye_p: SymmetricTensor | None = None  # the physical identity, for field padding
        n = len(psi)
        kl, kr = psi[0].legs[0], psi[n - 1].legs[2]
        # The bra's own boundary legs, which are the ket's for a one-state Env. Keeping
        # them apart is what lets the two chains carry different boundary charges: the
        # seed is then invariant only where the two agree, so an overlap between two
        # sectors is structurally zero rather than numerically small.
        bl, br = self.bra[0].legs[0], self.bra[len(self.bra) - 1].legs[2]
        # Read them off the description where there is one: asking ``h[0]`` for its leg
        # would materialise a site, which is what the deferred path exists to avoid.
        wl, wr = (
            h.edges.boundary_legs() if h.edges is not None else (h[0].legs[0], h[n - 1].legs[3])
        )
        # The boundary legs carry the state's and the operator's own ``dual`` flags, so
        # a dual boundary bond contracts instead of refusing about the wrong wire.
        self.F = {
            (-1, 0): ones(
                (
                    Leg(kl.space, IN, kl.dual),
                    Leg(wl.space, OUT, wl.dual),
                    Leg(bl.space, OUT, bl.dual),
                )
            ),
            (n, n - 1): ones(
                (
                    Leg(kr.space, OUT, kr.dual),
                    Leg(wr.space, IN, wr.dual),
                    Leg(br.space, IN, br.dual),
                )
            ),
        }

    def setup_(self, to: int = 0) -> "Env":
        """Build every environment directed towards site ``to``, and return ``self``.

        Parameters
        ----------
        to : int, optional
            The target site. Only ``0`` is implemented. Default ``0``.

        Returns
        -------
        Env
            ``self``, its right-directed environments built.

        Raises
        ------
        NotImplementedError
            If ``to`` is not ``0``; ``MPS.canonize_`` has the same note.

        Notes
        -----
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

        Parameters
        ----------
        n : int
            The site whose directed-bond entry is written.
        to : str
            The direction to write *toward*: ``'last'`` writes ``F[(n, n+1)]``
            from ``F[(n-1, n)]``, ``'first'`` writes ``F[(n, n-1)]`` from
            ``F[(n+1, n)]``. A direction, not a site -- unlike
            [MPS.canonize_][tenet.network.MPS.canonize_]'s ``to``, which is an
            ``int`` site index. Keyword-only.

        Notes
        -----
        ``to='last'`` writes ``F[(n, n+1)]`` from ``F[(n-1, n)]``; ``to='first'`` writes
        ``F[(n, n-1)]`` from ``F[(n+1, n)]``. Site-tensor path: three pairwise ``tenet.einsum``
        calls each -- environment first, then the ket, then the MPO, then the bra. With an
        edge-block table present the step goes edge-aware instead
        (``_fold_last`` / ``_fold_first``): the identity channels ride ``idmap``
        with no ``W`` contraction, only the operator-carrying blocks pay one, and unlike
        [heff2][tenet.network.Env.heff2] this path is **exact for any state** -- no gauge
        assumption.
        """
        a, bra = self.psi[n], tenet.adjoint(self.bra[n])
        blocks = self.h.edge_blocks(n)
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
        """Pop **both** directed bonds touching each changed site -- YASTN ``clear_site_``.

        Parameters
        ----------
        *sites : int
            The sites whose tensors changed.
        """
        for n in sites:
            self.F.pop((n, n - 1), None)
            self.F.pop((n, n + 1), None)

    def heff2(self, n: int, aa: SymmetricTensor) -> SymmetricTensor:
        """``H_eff`` on the two-site tensor at bond ``(n, n+1)``. Two paths, one output.

        Parameters
        ----------
        n : int
            The bond's left site.
        aa : SymmetricTensor
            The two-site tensor, ``(left bond OUT, p OUT, q OUT, right bond IN)``.

        Returns
        -------
        SymmetricTensor
            ``H_eff @ aa``, with ``aa``'s structure exactly.

        Notes
        -----
        **The engine is one path: the prepared, symbolic, term-family matvec.** Every MPO
        that carries an edge description -- [from_terms][tenet.network.MPO.from_terms] and
        [from_arrays][tenet.network.MPO.from_arrays], at *either* cutoff since #204 --
        goes through it, and **later parallelism and accelerator work attaches here and
        nowhere else**. This is block2's engine design in tenet's form: its
        ``EffectiveHamiltonian`` never forms the effective Hamiltonian and instead
        dispatches the symbolic operator sum term by term against the wavefunction
        (``effective_hamiltonian.hpp``:230-243). There is no runtime dispatch here either
        -- no bond-width threshold, no ``chi`` threshold, no ``path=`` keyword.

        **The site-tensor branch below is not a second engine; it is a compatibility entry.**
        It exists for an MPO that carries no symbols at all --
        [from_w][tenet.network.MPO.from_w] and an ``MPO`` built from bare site tensors --
        and nothing else routes to it. Recovering symbols from a numeric ``W`` is not
        possible in general: #141 measured that a compressed ``W`` retains no edge
        structure to recover. So the entry cannot be closed without refusing
        externally-built MPOs, which is a decision about the public surface and not one
        this milestone makes. block2 has no equivalent because block2 is a
        quantum-chemistry *program* and never receives an operator from outside; tenet is
        a library, and essentially every MPO in the literature is written as a ``W``
        matrix. Adopt block2's engine, not block2's role.

        **The knob that does exist is ``cutoff``, at build time.** ``cutoff=None`` keeps
        the exact finite-state machine, whose bond is already minimal for a finite-range
        lattice model and whose identity channels ride ``idmap``/``spec_op`` with no ``W``
        contraction at all; a float ``cutoff`` compresses, which is what an ab initio
        Hamiltonian needs and which -- because the rotation mixes the open states -- turns
        every open state into an operator-carrying one. Both then run through this one
        path. Measured at N=20 U(1) Heisenberg, ``chi=64``: 1.96 s at ``cutoff=None``
        against 3.53 s at ``1e-13``. So a lattice model wants ``cutoff=None`` and quantum
        chemistry wants the float, and the caller states which at build time rather than
        the engine guessing at run time -- which is also how block2 takes its algorithm
        choice. ``docs/design.md`` "Milestone 39" carries the ``chi`` scaling grid.

        **The path in detail.** The two
        environments are folded into the site blocks **once per bond** (``_build2``,
        MPSKit's ``AC2_hamiltonian``) and cached against the environment tensors'
        identity, so one ``lanczos`` solve at ``ncv=3`` pays the fold once and applies
        ``_apply2`` three times; absent fields are ``None`` and are skipped, which is
        where the structural zeros go. Like MPSKit's matvec, the ``IdL``/``IdR``-anchored
        terms are one-sided: they use the sweep's **mixed-canonical gauge** -- sites left
        of the bond left-orthonormal, sites right of it right-orthonormal, which
        [sweep_][tenet.network.sweep_] maintains at every bond -- as the standing
        precondition. It is the one thing this path asks of its caller, and it is not
        chosen at run time either: a caller whose environments come from a differently
        gauged state has to hand over ``MPO(h.sites)``, which throws the description away
        and takes the branch below. The apply itself is compiled through ``compile=``
        once per structure
        key -- the bond, and the tuple of ``aa``'s legs, which between them fix every leg
        the traced graph sees -- and the cache holds one entry per bond, its callable
        kept across a revisit and retraced only when the key moves.

        **The compatibility entry**, for an MPO with no description at all
        ([from_w][tenet.network.MPO.from_w] or bare site tensors), and no accelerator work
        targets it: right environment, then
        ``W2``, then ``W1``, then the left environment -- YASTN's ``Env_mps_mpo_mps.Heff2`` order
        (``_env.py``:496-518) with ``precompute=False``, which ``_dmrg.py``:102-108
        documents as ``O(D^3 M d + D^2 M^2 d^2)``.

        The engine and the compatibility entry agree as operators but sum their terms in
        a different order, so they agree to solver precision, never bitwise. In and out
        on ``(left bond OUT, p
        OUT, q OUT, right bond IN)``: the *bra* legs of the two environments become the
        output's bonds while the *ket* legs close against the input's, which is why the
        result has ``aa``'s structure exactly and [lanczos][tenet.network.lanczos] can add
        the two.
        """
        self._one_state("heff2")
        if self.h.edges is not None:
            p = self._prepare2(n)
            key = tuple(aa.legs)
            hit = self._compiled.get(n)
            # The traced graph is a function of ``(n, key)`` alone: at a fixed bond the
            # live fields of ``p`` are decided by the two sites' edge blocks, which never
            # move, and every one of their legs is fixed by ``aa``'s two bond legs plus
            # the operator's own. So the callable outlives a bond revisit and only a moved
            # bond space retraces it. ``p`` itself stays in the entry because its *values*
            # change every visit and because it is what weighs the entry for the byte
            # budget (#202); it is deliberately not part of the key (#225).
            # One slot per bond, and measured to be enough: over two sweeps of the three
            # models #224 counted, no bond is ever visited at two bond widths in
            # alternation, so a second slot compiles nothing extra. What does still
            # recompile is a bond whose entry the byte budget evicted -- 143 compiles
            # against 115 distinct keys on C2 at K=26, where the entry is weighed by the
            # ``_Prepared`` it carries.
            fn = (
                hit[2]
                if hit is not None and hit[0] == key
                else (_apply2 if self.compile is None else self.compile(_apply2))
            )
            self._compiled[n] = (key, p, fn)
            return fn(p, aa)
        return _heff2_full(self.h, self.F[n - 1, n], self.F[n + 2, n + 1], n, aa)

    def heff2_families(self, n: int, aa: SymmetricTensor) -> tuple[SymmetricTensor, ...]:
        """[heff2][tenet.network.Env.heff2]'s term families, applied separately, unsummed.

        Parameters
        ----------
        n : int
            The bond's left site.
        aa : SymmetricTensor
            The two-site tensor, exactly as [heff2][tenet.network.Env.heff2] takes it.

        Returns
        -------
        tuple of SymmetricTensor
            One partial application per populated family, each with ``aa``'s structure;
            their sum is ``heff2(n, aa)`` term for term.

        Examples
        --------
        >>> import tenet
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPO, MPS, Env, local_op
        >>> from tenet.symmetry import U1, U1Sector
        >>> import numpy as np
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> sz = local_op(np.diag([-0.5, 0.5]), phys=phys, charge=U1Sector(0))
        >>> h = MPO.from_terms(3, [(1.0, [(sz, i), (sz, i + 1)]) for i in range(2)])
        >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1), U1Sector(1)]).canonize_()
        >>> env = Env(psi, h).setup_()
        >>> aa = tenet.einsum("apx,xqr->apqr", psi[0], psi[1])
        >>> parts = env.heff2_families(0, aa)
        >>> total = parts[0]
        >>> for part in parts[1:]:
        ...     total = tenet.add(total, part)
        >>> bool(tenet.allclose(total, env.heff2(0, aa)))
        True

        Notes
        -----
        block2's ``perturbative_noise`` (``effective_hamiltonian.hpp``:263-360) builds one
        perturbation vector per *sub-label* of the symbolic operator sum; the families
        ``_cores2`` already holds -- the identity-through ride, the two one-sided anchored
        sums, and the two open-to-open ``AA`` remainders -- are this engine's version of
        that resolution, and this is the read [sweep_][tenet.network.sweep_]'s
        perturbative noise uses. It is a **read**, not a second engine: the same
        ``_prepare2`` cache, the same contractions, only not added up.

        The compatibility entry (an MPO with no edge description) has no families to
        resolve, so it returns the single vector ``(heff2(n, aa),)`` -- the operator's own
        action on the state, unresolved. A one-vector mixer is weaker than a
        family-resolved one, and it is what an operator that carries no symbols can offer.

        Not compiled: ``compile=`` wraps the *summed* matvec, which is what a Krylov solve
        calls thousands of times; this is called once per bond visit.
        """
        self._one_state("heff2_families")
        if self.h.edges is None:
            return (self.heff2(n, aa),)
        return tuple(_families2(self._prepare2(n), aa))

    def project2(self, n: int, aa: SymmetricTensor) -> SymmetricTensor:
        """The bond's projection vector: ``aa`` carried from the ket's bonds to the bra's.

        Parameters
        ----------
        n : int
            The bond's left site.
        aa : SymmetricTensor
            The **ket** chain's two-site tensor at that bond,
            ``(left bond OUT, p OUT, q OUT, right bond IN)``.

        Returns
        -------
        SymmetricTensor
            The same rank-4 structure on the **bra** chain's bonds:
            ``<bra env| H_n H_{n+1} |aa, ket env>``. With ``h`` the identity
            ([MPO.identity][tenet.network.MPO.identity]) this is the two-site reduced
            form of the ket state in the bra state's environment gauge, and
            ``tenet.inner(p, bb)`` is then ``<ket|bra'>`` for any ``bb`` in the bra's
            two-site variational space.

        Examples
        --------
        >>> import tenet
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPO, MPS, Env
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> states = [U1Sector(1), U1Sector(-1), U1Sector(1), U1Sector(-1)]
        >>> phi = MPS.product(phys, states).canonize_()
        >>> psi = MPS.product(phys, states).canonize_()
        >>> env = Env(phi, MPO.identity(4, phys), bra=psi).setup_()
        >>> pair = tenet.einsum("apx,xqr->apqr", phi[0], phi[1])
        >>> p = env.project2(0, pair)
        >>> bb = tenet.einsum("apx,xqr->apqr", psi[0], psi[1])
        >>> round(float(tenet.inner(p, bb)), 12)  # <psi|phi> read at one bond
        1.0

        Notes
        -----
        block2's ``i_eff->multiply`` on a two-state moving environment
        (``sweep_algorithm.hpp``:1195-1206), which is how it builds one ``ortho_bra``
        entry per converged state before handing the collection to ``eigs``
        (:1244-1249). Its ``ext_mes`` are built with the **identity** MPO -- the driver
        reads ``get_identity_mpo()`` for exactly this (``pyblock2/driver/core.py``:4817-4830)
        -- so the vector is an overlap, not an energy; [MPO.identity][tenet.network.MPO.identity]
        is the same spelling here.

        **What the contraction requires of its two states is only what the caller reads
        it as.** The contraction itself is exact in any gauge. But the *use* --
        "project the bra's two-site variational space against the ket state" -- is a
        statement about an orthonormal basis, so it holds exactly when the bra chain is
        mixed-canonical at bond ``n``, which [sweep_][tenet.network.sweep_] maintains at
        every bond it visits. The ket needs nothing: a gauge transformation on any of its
        bonds cancels between the two environments and the two-site tensor, which is why
        the converged states of an ``orthogonal_to=`` run are held fixed rather than
        canonicalized alongside the sweep the way block2 canonicalizes its ``ext_mpss``
        (:893-917).

        The four contractions are ``heff2``'s compatibility entry, shared verbatim: the
        one path in this class that reads no channel as a gauge identity, hence the one
        that survives ``bra is not psi``.
        """
        return _heff2_full(self.h, self.F[n - 1, n], self.F[n + 2, n + 1], n, aa)

    def _one_state(self, what: str) -> None:
        """Refuse ``what`` on a two-state ``Env``, naming what the object *is* for."""
        if self.bra is not self.psi:
            raise ValueError(
                f"{what} is a one-state operation and this Env carries bra is not psi. "
                "The prepared matvec reads the IdL/IdR environment channels as gauge "
                "identities, which holds for a canonical chain against itself and not "
                "for a mixed <phi|...|psi> transfer, so iterating on it would return a "
                "plausible wrong operator. A two-state Env supports setup_/update_/"
                "measure (measure() is <phi|H|psi>) and project2, which is the per-bond "
                "projection vector block2 gets from its ext_mes by multiply"
            )

    def _prepare2(self, n: int) -> _Prepared:
        """The bond's prepared operator, rebuilt only when either environment moved.

        Cached one per bond against the ``F`` entries *by identity*: environments are
        frozen tensors replaced on every [update_][tenet.network.Env.update_], so holding the two
        used to
        build is both the invalidation test and the guarantee a stale operator can
        never be served -- the discipline ``F`` itself uses, one level up.
        """
        fl, fr = self.F[n - 1, n], self.F[n + 2, n + 1]
        hit = self._prepared.get(n)
        if hit is not None and hit[0] is fl and hit[1] is fr:
            return hit[2]
        # The per-line ignores below: ``_prepare2`` is reached only through ``heff2``'s
        # edge-description branch, so ``edges`` is not ``None`` here -- a cross-method
        # invariant no checker narrows.
        edges = self.h.edges
        if n not in self._cores:  # environment-free, so evicted only by age (#202)
            if self._eye_p is None:
                self._eye_p = tenet.identity((self.psi[0].legs[1],))
            self._cores[n] = _cores2(edges, n, self._eye_p)  # ty: ignore[invalid-argument-type]
        p = _build2(self._cores[n], fl, fr)
        t1 = edges.edge_blocks(n)  # ty: ignore[unresolved-attribute]
        t2 = edges.edge_blocks(n + 1)  # ty: ignore[unresolved-attribute]
        self._prepared[n] = (fl, fr, p, _present(t1, t2))
        return p

    def measure(self) -> float:
        """``<psi|H|psi>`` without the eigensolver, on a private left-to-right pass.

        Returns
        -------
        float
            ``<bra|H|psi>``, **not** divided by any norm -- ``<psi|H|psi>`` on a
            one-state [Env][tenet.network.Env] and ``<phi|H|psi>`` on an
            ``Env(psi, h, bra=phi)``.

        Notes
        -----
        The first thing in this repository that measures a converged energy independently
        of the ``lanczos`` Rayleigh quotient that produced it. On a two-state
        [Env][tenet.network.Env] it is instead the engine fact the measurement API stands
        on (#213): ``Env(psi, h, bra=phi).measure()`` **is** ``<phi|H|psi>``, and with
        ``h`` the identity ([MPO.identity][tenet.network.MPO.identity]) it is the plain
        overlap ``<phi|psi>``. No gauge is assumed of either chain. YASTN's ``measure`` is the
        same closing contraction one level down (``_env.py``:462-468, ``vdot(vecL,
        vecR)``); the pass is built in a fresh [Env][tenet.network.Env] so a measurement never
        writes
        into a sweep's cache.
        """
        n = len(self.psi)
        env = Env(self.psi, self.h, bra=self.bra)
        for site in range(n):
            env.update_(site, to="last")
        closed = tenet.einsum("Rms,rms->Rr", env.F[n, n - 1], env.F[n - 1, n])
        return float(tenet.full_trace(closed))

    def __repr__(self) -> str:
        return f"Env(sites={len(self.psi)}, bonds={sorted(self.F)})"

    def __contains__(self, key: Any) -> bool:
        return key in self.F
