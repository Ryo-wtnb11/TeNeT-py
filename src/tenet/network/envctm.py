"""The directional corner-transfer-matrix environment: four corners and four edges per site.

YASTN's ``fpeps/envs/_env_ctm.py`` and ``_env_dataclasses.py`` (b0187c4), adopted by
M79/#277 and re-spelled on M79a's [Lattice][tenet.network.Lattice],
[Peps][tenet.network.Peps] and the twelve contraction primitives.

**Leg conventions.** Corners are rank 2 and edges rank 3 -- rank 4 for a double layer,
where the ket bond and the bra bond stay adjacent and separate, exactly as
[peps][tenet.network.peps] leaves them. Reading the boundary ring clockwise from the
top-left corner::

    C_tl --- T_t --- C_tr
     |        |        |
    T_l --- (site) --- T_r
     |        |        |
    C_bl --- T_b --- C_br

each tensor's **first** leg meets its predecessor's **last** one, all the way round:
``C_tl.1 -> T_t.0``, ``T_t.2 -> C_tr.0``, ``C_tr.1 -> T_r.0``, ``T_r.2 -> C_br.0``,
``C_br.1 -> T_b.0``, ``T_b.2 -> C_bl.0``, ``C_bl.1 -> T_l.0``, ``T_l.2 -> C_tl.0``. So
an edge is ``(env, <the site's leg, once per layer>, env)`` -- ``T_l`` is
``(down, l, up)``, ``T_r`` is ``(up, r, down)``, ``T_t`` is ``(left, t, right)``,
``T_b`` is ``(right, b, left)`` -- and a corner is ``(the edge before, the edge after)``.

**The sides are derived, not chosen.** Requiring the ring to close, requiring
horizontally adjacent ``T_t``'s (and vertically adjacent ``T_l``'s) to meet -- which the
2x2 enlarged corners do -- and requiring the two projectors of one cut to carry the two
ends of the *same* new bond leaves exactly one assignment::

    T_l (IN, ., OUT)   T_r (OUT, ., IN)   T_t (OUT, ., IN)   T_b (IN, ., OUT)
    C_tl (IN, IN)      C_tr (OUT, IN)     C_br (OUT, OUT)    C_bl (IN, OUT)

Two corners carry two legs of the same side; that is not an anomaly but the ring's
book-keeping -- a cycle of eight alternating wires needs its two turning points.

**No Hermiticity assumption, anywhere.** A projector pair is built from the QR of each
half of the 4x4 patch and an SVD of ``r0 @ r1^T`` --
[proj_corners][tenet.network.proj_corners] -- never from an eigendecomposition of an
enlarged corner and never from a single isometry used on both index groups. That is the
design point M63/#243 measured its way to: the corner's Hermiticity is a property of the
*ansatz* (the full C4v point group), so an algorithm that needs it is an algorithm with a
precondition it cannot check. This one has none.

**What is deliberately not transcribed** (M79b scope): the ``'1x2'`` projector method,
the 5x4 extended corners YASTN grows when a PEPS bond is one-dimensional (a hexagonal
lattice embedded on a square one), ``bond_metric`` and the evolution it serves (M79d),
``boundary_mps``, checkpointing, serialization and the ``patch`` mechanism.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from typing import Any, NamedTuple

import autoray as ar

import tenet
from tenet import IN, OUT, Leg, SymmetricTensor
from tenet.network.common import ones, spectrum
from tenet.network.lattice import Lattice, Site
from tenet.network.peps import (
    Peps2Layers,
    append_vec_bl,
    append_vec_br,
    append_vec_tl,
    append_vec_tr,
)

__all__ = [
    "CTM_out",
    "EnvCTM",
    "EnvLocal",
    "EnvProjectors",
    "corner2x2",
    "proj_corners",
]

#: The four corner names and the four edge names, in the order the ring visits them.
CORNERS = ("tl", "t", "tr", "r", "br", "b", "bl", "l")

#: Per direction, the psi axis it names. ``Peps`` leg order is ``(t, l, b, r, phys)``.
_AXIS = {"t": 0, "l": 1, "b": 2, "r": 3}


@dataclass
class EnvLocal:
    """One site's environment: four corners and four edges, any of them still unset.

    Attributes
    ----------
    tl, tr, bl, br : SymmetricTensor or None
        The corners, rank 2.
    t, l, b, r : SymmetricTensor or None
        The edges, rank 3 (single layer) or rank 4 (double layer).

    Notes
    -----
    A plain mutable record, YASTN's ``EnvCTM_local``. It is a dataclass rather than a
    ``NamedTuple`` because a move writes one field at a time into a *fresh* record and
    then swaps the records over -- the "all sites simultaneously" the ``'h'`` and ``'v'``
    moves promise is exactly that no site reads a field another site has already written.
    """

    tl: SymmetricTensor | None = None
    tr: SymmetricTensor | None = None
    bl: SymmetricTensor | None = None
    br: SymmetricTensor | None = None
    t: SymmetricTensor | None = None
    l: SymmetricTensor | None = None  # noqa: E741 -- the lattice direction, not an index
    b: SymmetricTensor | None = None
    r: SymmetricTensor | None = None


@dataclass
class EnvProjectors:
    """One site's eight projectors, YASTN's ``EnvCTM_projectors``.

    Attributes
    ----------
    hlt, hlb, hrt, hrb : SymmetricTensor or None
        The horizontal move's projectors: left/right, top/bottom half.
    vtl, vtr, vbl, vbr : SymmetricTensor or None
        The vertical move's.

    Notes
    -----
    Each is rank 3 (single layer) or rank 4 (double layer): the environment leg it
    absorbs, the site's leg once per layer, and the truncated bond it produces.
    """

    hlt: SymmetricTensor | None = None
    hlb: SymmetricTensor | None = None
    hrt: SymmetricTensor | None = None
    hrb: SymmetricTensor | None = None
    vtl: SymmetricTensor | None = None
    vtr: SymmetricTensor | None = None
    vbl: SymmetricTensor | None = None
    vbr: SymmetricTensor | None = None


class CTM_out(NamedTuple):
    """What a [iterate_][tenet.network.EnvCTM.iterate_] run reports, YASTN's ``CTMRG_out``.

    Attributes
    ----------
    sweeps : int
        Sweeps performed.
    max_dsv : float
        The worst corner's spectrum change over the last sweep; ``inf`` after the
        first sweep, which has nothing to compare against.
    converged : bool
        Whether ``max_dsv < corner_tol`` when the loop stopped.
    """

    sweeps: int
    max_dsv: float
    converged: bool


# -- the composition rule, once ------------------------------------------------------


def _supplies_in(leg: Leg) -> bool:
    """Whether ``leg`` is the ``IN`` end of its wire.

    ``side`` alone answers it for a leg nobody has moved, and every leg in
    ``network/env.py`` is one. Here a leg can arrive from a
    [qr][tenet.ops.linalg.qr] that repartitioned it across the map's two sides, which
    flips ``side`` and ``dual`` together and leaves the same wire in the opposite
    spelling -- so the predicate is the pair, not ``side``.
    """
    return (leg.side is IN) != leg.dual


def _composed(equation: str, a: SymmetricTensor, b: SymmetricTensor) -> SymmetricTensor:
    """A two-operand ``tenet.einsum`` whose operand order and bends are **derived**.

    Parameters
    ----------
    equation : str
        A two-operand ``einsum`` equation. The output order is the caller's and is not
        touched; only which operand ``einsum`` sees first can change.
    a, b : SymmetricTensor
        The operands.

    Returns
    -------
    SymmetricTensor
        The contraction, taken in whichever operand order bends the fewer wires, as a
        one-step ``tenet.einsum_chain`` whose ``bend`` field names those wires -- so
        what the chain composes is a composition.

    Notes
    -----
    ``network/env.py``'s ``_composed`` is handed its bend set; here both the order and
    the set are computed, because M79a settled what the answer would be anyway.
    **Bend-minimality is the criterion** (``docs/design.md``, M79a): a bend is a real
    categorical operation, so a spelling that turns two wires where one suffices lands
    on a different tensor, and the minimal one is the planar diagram's. Every wire has
    exactly one ``IN`` end here, so the two orders' bend counts sum to the number of
    shared wires and the minimum is well defined; a tie keeps the stated order, which is
    YASTN's throughout this module.
    """
    lhs, out = equation.split("->")
    ta, tb = lhs.split(",")
    shared = [c for c in ta if c in tb]
    bend = "".join(c for c in shared if not _supplies_in(a.legs[ta.index(c)]))
    if len(bend) > len(shared) - len(bend):  # the other order turns fewer wires
        a, b, ta, tb = b, a, tb, ta
        bend = "".join(c for c in shared if not _supplies_in(a.legs[ta.index(c)]))
    return tenet.einsum_chain([(f"{ta},{tb}->{out}", a, b, bend)])


def _normalized(t: SymmetricTensor) -> SymmetricTensor:
    """``t / ||t||`` after every move, so a growing environment stays ``O(1)``.

    Simplification: the Frobenius norm where YASTN takes the infinity norm. tenet has
    one public norm and the two differ by at most ``sqrt(dim)``; what the division is
    for -- keeping the corner from tracking the partition function itself -- does not
    distinguish them.
    """
    return t / tenet.norm(t)


def _dual(leg: Leg) -> Leg:
    """The leg that contracts with ``leg``: the same space and ``dual``, the other side."""
    return Leg(leg.space, IN if leg.side is OUT else OUT, dual=leg.dual)


# -- the enlarged corners and the projectors -----------------------------------------

#: Per enlarged corner: the two edges and the corner between them, the two site
#: directions the vector carries, and the primitive that absorbs the site.
_C2X2 = {
    "tl": ("l", "tl", "t", "l", "t", append_vec_tl),
    "bl": ("b", "bl", "l", "b", "l", append_vec_bl),
    "tr": ("t", "tr", "r", "t", "r", append_vec_tr),
    "br": ("r", "br", "b", "r", "b", append_vec_br),
}

#: The single-layer stand-in for each ``append_vec_*``: one plain composition, because a
#: rank-4 network has no bra to reach through.
_FLAT = {
    "tl": "xlty,tlbr->xbyr",
    "bl": "xbly,tlbr->xryt",
    "tr": "xtry,tlbr->xlyb",
    "br": "xrby,tlbr->xtyl",
}


def corner2x2(env: "EnvCTM", which: str, site: Any) -> SymmetricTensor:
    """One 2x2 enlarged corner: two edges, the corner between them, and the site.

    Parameters
    ----------
    env : EnvCTM
        The environment to read.
    which : str
        ``'tl'``, ``'tr'``, ``'bl'`` or ``'br'``.
    site : Site or tuple[int, int]
        The site whose ring supplies the three environment tensors.

    Returns
    -------
    SymmetricTensor
        Rank 4 for a single layer, rank 6 for a double one, and in **two groups of
        equal size**: the legs pointing one way out of the corner, then the legs
        pointing the other way. Which two ways depends on ``which`` -- ``'tl'`` returns
        (down, right), ``'tr'`` (left, down), ``'br'`` (up, left), ``'bl'`` (right, up)
        -- so that ``tl @ tr``, ``br @ bl``, ``bl @ tl`` and ``tr @ br`` each close the
        four halves of the 4x4 patch.

    Notes
    -----
    YASTN's ``corner2x2`` (``_env_contractions.py``:429) is ``t1 @ c @ t2`` followed by a
    ``tensordot`` onto the site, and for a double layer that ``tensordot`` *is* the
    matching ``append_vec_*``. The grouping falls out for free: M79a's
    ``append_vec_tl`` already returns ``(x, b, y, r)``, which is those two groups in
    that order, so nothing is fused and nothing is transposed here.
    """
    e1, corner, e2, d1, d2, absorb = _C2X2[which]
    p1, p2 = env.wire(d1), env.wire(d2)
    local = env[site]
    # The edge leads: its last leg is the one the corner's first leg meets, and an edge
    # supplies IN there for ``T_l``/``T_b`` (whose last leg is OUT) only after a bend --
    # which is one wire either way, so the edge-first reading of ``t1 @ c @ t2`` stands.
    vec = _composed(f"x{p1}c,cd->x{p1}d", getattr(local, e1), getattr(local, corner))
    vec = _composed(f"x{p1}d,d{p2}y->x{p1}{p2}y", vec, getattr(local, e2))
    a = env.psi[site]
    if env.double:
        return absorb(a, vec)
    return _composed(_FLAT[which], vec, a)


def proj_corners(
    r0: SymmetricTensor,
    r1: SymmetricTensor,
    *,
    max_bond: int | None = None,
    cutoff: float | None = 1e-14,
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """The projector pair across a cut, from ``r0 @ r1^T`` -- YASTN ``proj_corners``:1209.

    Parameters
    ----------
    r0, r1 : SymmetricTensor
        The two ``R`` factors, each ``(bond, *the cut's legs)``; their cut legs are
        duals of each other.
    max_bond : int or None, optional
        The environment bond-dimension cap. Default ``None``.
    cutoff : float or None, optional
        Relative singular-value cutoff for the truncation. Default ``1e-14``.

    Returns
    -------
    p0 : SymmetricTensor
        ``(*r1's cut legs, new bond IN)``.
    p1 : SymmetricTensor
        ``(*r0's cut legs, new bond OUT)``.

    Raises
    ------
    StructureChangingError
        Under ``jax.jit``/``jax.grad``:
        [svd_truncated][tenet.ops.linalg.svd_truncated] reads singular values to decide
        which sectors survive.

    Notes
    -----
    ``rr = r0 @ r1^T = u s v``, ``rs = s^(-1/2)``, ``p0 = r1 (rs v)^dagger`` and
    ``p1 = r0 (u rs)^dagger``, so ``p1^T p0`` inserts ``rs s rs = 1`` on the cut. **No
    step assumes the cut is Hermitian**: the two sides enter as two different tensors and
    leave as two different projectors, and there is no eigendecomposition and no single
    isometry reused on both index groups. That is the whole answer to #243 -- the
    property the C4v single-move projector needed and the ansatz did not supply is a
    property this construction never asks for.

    The two new bond legs come out on opposite sides, ``IN`` for ``p0`` and ``OUT`` for
    ``p1``, which is what makes the moved edge's two ends meet the moved corners'.
    """
    group = tuple(range(1, r0.ndim))
    letters = "ijk"[: len(group)]
    rr = _composed(f"a{letters},b{letters}->ab", r0, r1)
    u, s, vh = tenet.linalg.svd_truncated(rr, ((0,), (1,)), max_bond=max_bond, cutoff=cutoff)
    threshold = 0.0 if cutoff is None else cutoff * tenet.norm(s)

    def rsqrt(m: Any) -> Any:
        keep = ar.do("greater", m, threshold)
        safe = ar.do("where", keep, m, ar.do("ones_like", m))
        return ar.do("where", keep, safe ** (-0.5), ar.do("zeros_like", m))

    rs = tenet.apply_blocks(s, rsqrt)
    p1 = _composed(f"a{letters},ab->{letters}b", r0, tenet.adjoint(_composed("ab,bc->ac", u, rs)))
    p0 = _composed(f"a{letters},ba->{letters}b", r1, tenet.adjoint(_composed("ab,bc->ac", rs, vh)))
    return p0, p1


#: Every trivial projector: its name, the neighbour whose corner supplies the new bond,
#: the edge and which of its two environment legs, the site direction, the corner and
#: which of its legs, and the move letter that wants it. YASTN's ``_for_trivial``:1017.
_TRIVIAL = (
    ("hlt", "r", "l", -1, "t", "tl", 0, "l"),
    ("hlb", "r", "l", 0, "b", "bl", 1, "l"),
    ("hrt", "l", "r", 0, "t", "tr", 1, "r"),
    ("hrb", "l", "r", -1, "b", "br", 0, "r"),
    ("vtl", "b", "t", 0, "l", "tl", 1, "t"),
    ("vtr", "b", "t", -1, "r", "tr", 0, "t"),
    ("vbl", "t", "b", -1, "l", "bl", 0, "b"),
    ("vbr", "t", "b", 0, "r", "br", 1, "b"),
)

#: Per move, the 2x2 block's projector assignments: the pair of ``qr`` groupings and
#: where the two projectors land. ``0`` and ``1`` name the half's two leg groups.
_PROJ_MOVE = {
    "r": ("h", (0, 1), (1, 0), ("tr", "hrb"), ("br", "hrt")),
    "l": ("h", (1, 0), (0, 1), ("tl", "hlb"), ("bl", "hlt")),
    "t": ("v", (0, 1), (1, 0), ("tl", "vtr"), ("tr", "vtl")),
    "b": ("v", (1, 0), (0, 1), ("bl", "vbr"), ("br", "vbl")),
}


class EnvCTM:
    """A directional CTM environment: four corners and four edges per unique site.

    Parameters
    ----------
    psi : Peps
        The network. A rank-5 state becomes a [Peps2Layers][tenet.network.Peps2Layers]
        view; a rank-4 one -- a classical partition function -- is used as it is.
    init : str or None, optional
        ``'eye'`` (the default) for the one-dimensional identity boundary, ``'dl'`` for
        one un-truncated absorption on top of it, or ``None`` to leave the environment
        empty.
    bra : Peps or None, optional
        An independent bra for the double layer. Default ``None``, i.e. ``<psi|psi>``.

    Raises
    ------
    ValueError
        If ``init`` is not one of the three.

    Examples
    --------
    >>> import numpy as np
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import EnvCTM, Peps, SquareLattice
    >>> from tenet.symmetry import Z2, Z2Sector
    >>> beta = 0.3
    >>> c, s = np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))
    >>> w = np.array([[c, s], [c, -s]])
    >>> block = np.einsum("st,sl,sb,sr->tlbr", w, w, w, w)
    >>> V = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN))
    >>> psi = Peps(SquareLattice(dims=(1, 1)), SymmetricTensor.from_dense(block, legs))
    >>> env = EnvCTM(psi)
    >>> out = env.iterate_(max_bond=8, max_sweeps=50, corner_tol=1e-10)
    >>> out.converged
    True

    Notes
    -----
    The environment is a [Lattice][tenet.network.Lattice] of
    [EnvLocal][tenet.network.EnvLocal] records and the projectors a second one of
    [EnvProjectors][tenet.network.EnvProjectors], so both fold through the geometry the
    way the state does: one record per *unique* site, read at any site of the plane.
    """

    def __init__(self, psi: Any, init: str | None = "eye", bra: Any = None) -> None:
        self.geometry = psi.geometry
        self.double = psi.has_physical()
        self.psi = Peps2Layers(psi, bra) if self.double else psi
        self.env = Lattice(self.geometry, {site: EnvLocal() for site in self.sites()})
        self.proj = Lattice(self.geometry, {site: EnvProjectors() for site in self.sites()})
        if init is not None:
            if init not in ("eye", "dl"):
                raise ValueError(f"EnvCTM: init={init!r} should be 'eye', 'dl' or None")
            self.reset_(init)

    # -- the geometry, forwarded ----------------------------------------------------

    @property
    def Nx(self) -> int:
        """Rows in the unit cell."""
        return self.geometry.Nx

    @property
    def Ny(self) -> int:
        """Columns in the unit cell."""
        return self.geometry.Ny

    def sites(self, reverse: bool = False) -> tuple[Site, ...]:
        """The unique sites."""
        return self.geometry.sites(reverse)

    def nn_site(self, site: Any, d: Any) -> Site | None:
        """The neighbour of ``site`` in direction ``d``."""
        return self.geometry.nn_site(site, d)

    def __getitem__(self, site: Any) -> EnvLocal:
        return self.env[site]

    def __repr__(self) -> str:
        return f"EnvCTM(geometry={self.geometry!r}, double={self.double})"

    # -- the layer's own vocabulary -------------------------------------------------

    def wire(self, d: str) -> str:
        """The ``einsum`` letters one site leg occupies: two for a double layer, one for
        a single one. ``'t'`` is ``'tT'`` or ``'t'``, and so on for ``l``, ``b``, ``r``."""
        return {"t": "tT", "l": "lL", "b": "bB", "r": "rR"}[d] if self.double else d

    def site_legs(self, site: Any, d: str) -> tuple[Leg, ...]:
        """The legs an environment tensor needs to meet the site's ``d`` leg."""
        a = self.psi[site]
        axis = _AXIS[d]
        if self.double:
            return (_dual(a.ket.legs[axis]), _dual(a.bra.legs[axis]))
        return (_dual(a.legs[axis]),)

    # -- initialization -------------------------------------------------------------

    def reset_(self, init: str = "eye") -> None:
        """Seed every corner and edge, YASTN ``reset_``:239.

        Parameters
        ----------
        init : str, optional
            ``'eye'`` (the default) puts a one-dimensional environment bond everywhere:
            each corner is the scalar one and each edge is the identity that closes the
            ket against the bra (a single layer's free boundary, all ones). ``'dl'``
            absorbs one layer of the network into that seed **without truncating**,
            which is the environment YASTN's ``expand_outward_`` builds -- one
            un-truncated ``'hv'`` sweep reaches it, in the projectors' singular basis
            rather than YASTN's product basis, and an environment is defined only up to
            a gauge on its bonds.

        Raises
        ------
        ValueError
            If ``init`` is not ``'eye'`` or ``'dl'``.
        """
        if init not in ("eye", "dl"):
            raise ValueError(f"EnvCTM: init={init!r} should be 'eye' or 'dl'")
        first = self.psi[self.sites()[0]]
        provider = (first.ket if self.double else first).provider
        unit = tenet.GradedSpace.new(provider, {provider.unit: 1})
        for site in self.sites():
            local = self[site]
            for name in ("tl", "tr", "bl", "br"):
                s0, s1 = _CORNER_SIDES[name]
                local.__dict__[name] = ones((Leg(unit, s0), Leg(unit, s1)))
            for name in ("t", "l", "b", "r"):
                s0, s1 = _EDGE_SIDES[name]
                pair = self.site_legs(site, name)
                env = ones((Leg(unit, s0), Leg(unit, s1)))
                if not self.double:
                    local.__dict__[name] = ones((Leg(unit, s0), *pair, Leg(unit, s1)))
                    continue
                # The double layer's free boundary closes the ket against the bra --
                # YASTN's ``identity_boundary`` on a fused ``[x x']`` leg -- while a
                # single layer's is the all-ones vector, which is what ``ones`` is.
                delta = tenet.identity((Leg(pair[0].space, OUT, dual=pair[0].dual),))
                if pair[0].side is IN:
                    delta = tenet.adjoint(delta)
                local.__dict__[name] = tenet.einsum("ac,de->adec", env, delta)
        if init == "dl":
            self.update_(max_bond=None, moves="hv", cutoff=0.0)

    # -- the sweep ------------------------------------------------------------------

    def update_(
        self,
        max_bond: int | None = None,
        moves: str = "hv",
        cutoff: float | None = 1e-14,
    ) -> None:
        """One sweep: each letter of ``moves``, in order, in place.

        Parameters
        ----------
        max_bond : int or None, optional
            The environment bond-dimension cap. Default ``None``, i.e. no cap.
        moves : str, optional
            A sequence of moves. ``'h'`` and ``'v'`` update every site simultaneously
            into a fresh environment; ``'l'``, ``'r'``, ``'t'`` and ``'b'`` run causally,
            column after column or row after row. Default ``'hv'``.
        cutoff : float or None, optional
            Relative singular-value cutoff for the projector truncation.
            Default ``1e-14``.

        Raises
        ------
        ValueError
            If ``moves`` contains anything but ``'hvlrtb'``.
        """
        for move in moves:
            if move not in "hvlrtb":
                raise ValueError(f"EnvCTM: move={move!r} should be one of 'hvlrtb'")
            self._move_(move, max_bond, cutoff)

    def _move_(self, move: str, max_bond: int | None, cutoff: float | None) -> None:
        """One move, YASTN ``_update_core_``:573."""
        shift, groups = None, [self.sites()]
        if move not in "hv" and len(self.sites()) == self.Nx * self.Ny:
            if move in "lr":
                columns = range(self.Ny) if move == "l" else range(self.Ny - 1, -1, -1)
                groups = [[Site(nx, ny) for nx in range(self.Nx)] for ny in columns]
            else:
                rows = range(self.Nx) if move == "t" else range(self.Nx - 1, -1, -1)
                groups = [[Site(nx, ny) for ny in range(self.Ny)] for nx in rows]
            shift = move if move in "lt" else None
        for group in groups:
            targets = [self.nn_site(site, shift) for site in group] if shift else list(group)
            targets = [site for site in targets if site is not None]
            for letter in "lr" if move == "h" else "tb" if move == "v" else move:
                for site in targets:
                    self._update_projectors_(site, letter, max_bond, cutoff)
            self._trivial_projectors_(move, targets)
            self._absorb_(move, group)

    def _absorb_(self, move: str, group: Sequence[Any]) -> None:
        """Build every site's new tensors into a fresh record, then swap the records over."""
        fresh = {self.geometry.site2index(site): EnvLocal() for site in group}
        for site in group:
            self._update_env_(fresh[self.geometry.site2index(site)], site, move)
        for site in group:
            new = fresh[self.geometry.site2index(site)]
            local = self[site]
            for field in fields(new):
                value = getattr(new, field.name)
                if value is not None:
                    local.__dict__[field.name] = value

    # -- projectors -----------------------------------------------------------------

    def _update_projectors_(
        self, site: Any, move: str, max_bond: int | None, cutoff: float | None
    ) -> None:
        """The 2x2 block anchored at ``site``: four enlarged corners, two halves, two QRs
        and one [proj_corners][tenet.network.proj_corners]. YASTN
        ``update_extended_2x2_projectors_``:1029, without its 5x4 special case."""
        block = [self.nn_site(site, d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))]
        if None in block:
            return
        tl, tr, bl, br = block
        axis, cut_a, cut_b, (site_0, name_0), (site_1, name_1) = _PROJ_MOVE[move]
        if axis == "h":
            h1 = self._half(corner2x2(self, "tl", tl), corner2x2(self, "tr", tr))
            h2 = self._half(corner2x2(self, "br", br), corner2x2(self, "bl", bl))
        else:
            h1 = self._half(corner2x2(self, "bl", bl), corner2x2(self, "tl", tl))
            h2 = self._half(corner2x2(self, "tr", tr), corner2x2(self, "br", br))
        r0 = tenet.linalg.qr(h1, self._groups(h1, cut_a))[1]
        r1 = tenet.linalg.qr(h2, self._groups(h2, cut_b))[1]
        p0, p1 = proj_corners(r0, r1, max_bond=max_bond, cutoff=cutoff)
        anchor = {"tl": tl, "tr": tr, "bl": bl, "br": br}
        self.proj[anchor[site_0]].__dict__[name_0] = p0
        self.proj[anchor[site_1]].__dict__[name_1] = p1

    def _half(self, first: SymmetricTensor, second: SymmetricTensor) -> SymmetricTensor:
        """Two enlarged corners joined on their facing groups: half the 4x4 patch.

        ``first``'s second group and ``second``'s first group are the shared cut, which
        is the grouping [corner2x2][tenet.network.corner2x2] returns them in; the halves
        keep the two outer groups in that order.
        """
        n = first.ndim // 2
        outer, cut, tail = "ade"[:n], "ijk"[:n], "mnp"[:n]
        return _composed(f"{outer}{cut},{cut}{tail}->{outer}{tail}", first, second)

    def _groups(self, half: SymmetricTensor, order: tuple[int, int]) -> tuple[Sequence[int], ...]:
        """A half's two leg groups, in the order ``qr`` should see them."""
        n = half.ndim // 2
        both = (tuple(range(n)), tuple(range(n, 2 * n)))
        return (both[order[0]], both[order[1]])

    def _trivial_projectors_(self, move: str, sites: Sequence[Any]) -> None:
        """Fill any projector the 2x2 block could not build, YASTN
        ``_trivial_projectors_``:655. It happens only at an open boundary, where the
        environment bond the projector would truncate is one-dimensional anyway."""
        wanted = "lr" if move == "h" else "tb" if move == "v" else move
        for site in sites:
            for name, nn, edge, edge_axis, direction, corner, corner_axis, letter in _TRIVIAL:
                if letter not in wanted or getattr(self.proj[site], name) is not None:
                    continue
                neighbour = self.nn_site(site, nn)
                if neighbour is None:
                    continue
                legs = (
                    _dual(getattr(self[site], edge).legs[edge_axis]),
                    *self.site_legs(site, direction),
                    _dual(getattr(self[neighbour], corner).legs[corner_axis]),
                )
                self.proj[site].__dict__[name] = ones(legs)

    # -- the absorption -------------------------------------------------------------

    def _update_env_(self, fresh: EnvLocal, site: Any, move: str) -> None:
        """The moved corners and edges for one site, YASTN ``_update_env_``:671.

        Every contraction below states its operand order and lets ``_composed`` derive
        the bends. The order is YASTN's: the tensor that already carries the new bond
        leads, so the truncated leg is never the one waiting on a bend.
        """
        if move in "lh":
            self._move_l(fresh, site)
        if move in "rh":
            self._move_r(fresh, site)
        if move in "tv":
            self._move_t(fresh, site)
        if move in "bv":
            self._move_b(fresh, site)

    def _move_l(self, fresh: EnvLocal, site: Any) -> None:
        """The left column moves one step right."""
        left = self.nn_site(site, "l")
        if left is None:
            return
        t_, l_, b_, r_ = (self.wire(d) for d in "tlbr")
        vec = _composed(
            f"x{l_}c,c{t_}n->x{l_}{t_}n", self[left].l, self.proj[left].hlt
        )
        mid = self._absorb_site(left, "tl", vec)
        fresh.l = _normalized(
            _composed(f"x{b_}m,x{b_}n{r_}->m{r_}n", self.proj[left].hlb, mid)
        )
        above = self.nn_site(site, "tl")
        if above is not None:
            corner = _composed(f"ad,d{t_}y->a{t_}y", self[left].tl, self[left].t)
            fresh.tl = _normalized(
                _composed(f"a{t_}m,a{t_}y->my", self.proj[above].hlb, corner)
            )
        below = self.nn_site(site, "bl")
        if below is not None:
            corner = _composed(f"ae,e{b_}n->a{b_}n", self[left].bl, self.proj[below].hlt)
            fresh.bl = _normalized(_composed(f"x{b_}a,a{b_}n->xn", self[left].b, corner))

    def _move_r(self, fresh: EnvLocal, site: Any) -> None:
        """The right column moves one step left."""
        right = self.nn_site(site, "r")
        if right is None:
            return
        t_, l_, b_, r_ = (self.wire(d) for d in "tlbr")
        vec = _composed(f"x{r_}c,c{b_}n->x{r_}{b_}n", self[right].r, self.proj[right].hrb)
        mid = self._absorb_site(right, "br", vec)
        fresh.r = _normalized(
            _composed(f"x{t_}m,x{t_}n{l_}->m{l_}n", self.proj[right].hrt, mid)
        )
        above = self.nn_site(site, "tr")
        if above is not None:
            corner = _composed(f"ae,e{t_}n->a{t_}n", self[right].tr, self.proj[above].hrb)
            fresh.tr = _normalized(_composed(f"y{t_}a,a{t_}n->yn", self[right].t, corner))
        below = self.nn_site(site, "br")
        if below is not None:
            corner = _composed(f"ae,e{b_}y->a{b_}y", self[right].br, self[right].b)
            fresh.br = _normalized(
                _composed(f"a{b_}m,a{b_}y->my", self.proj[below].hrt, corner)
            )

    def _move_t(self, fresh: EnvLocal, site: Any) -> None:
        """The top row moves one step down."""
        top = self.nn_site(site, "t")
        if top is None:
            return
        t_, l_, b_, r_ = (self.wire(d) for d in "tlbr")
        vec = _composed(f"a{l_}m,a{t_}y->m{l_}{t_}y", self.proj[top].vtl, self[top].t)
        mid = self._absorb_site(top, "tl", vec)
        fresh.t = _normalized(_composed(f"m{b_}y{r_},y{r_}n->m{b_}n", mid, self.proj[top].vtr))
        left = self.nn_site(site, "tl")
        if left is not None:
            corner = _composed(f"ae,e{l_}n->a{l_}n", self[top].tl, self.proj[left].vtr)
            fresh.tl = _normalized(_composed(f"x{l_}a,a{l_}n->xn", self[top].l, corner))
        right = self.nn_site(site, "tr")
        if right is not None:
            corner = _composed(f"ae,e{r_}y->a{r_}y", self[top].tr, self[top].r)
            fresh.tr = _normalized(_composed(f"a{r_}m,a{r_}y->my", self.proj[right].vtl, corner))

    def _move_b(self, fresh: EnvLocal, site: Any) -> None:
        """The bottom row moves one step up."""
        bottom = self.nn_site(site, "b")
        if bottom is None:
            return
        t_, l_, b_, r_ = (self.wire(d) for d in "tlbr")
        vec = _composed(f"a{r_}m,a{b_}y->m{r_}{b_}y", self.proj[bottom].vbr, self[bottom].b)
        mid = self._absorb_site(bottom, "br", vec)
        fresh.b = _normalized(
            _composed(f"m{t_}y{l_},y{l_}n->m{t_}n", mid, self.proj[bottom].vbl)
        )
        left = self.nn_site(site, "bl")
        if left is not None:
            corner = _composed(f"ae,e{l_}y->a{l_}y", self[bottom].bl, self[bottom].l)
            fresh.bl = _normalized(_composed(f"a{l_}m,a{l_}y->my", self.proj[left].vbr, corner))
        right = self.nn_site(site, "br")
        if right is not None:
            corner = _composed(f"ae,e{r_}n->a{r_}n", self[bottom].br, self.proj[right].vbl)
            fresh.br = _normalized(_composed(f"x{r_}a,a{r_}n->xn", self[bottom].r, corner))

    def _absorb_site(self, site: Any, which: str, vec: SymmetricTensor) -> SymmetricTensor:
        """One site absorbed into an environment vector, through both layers or through
        the one there is."""
        a = self.psi[site]
        if self.double:
            return _C2X2[which][5](a, vec)
        return _composed(_FLAT[which], vec, a)

    # -- convergence ----------------------------------------------------------------

    def corner_spectra(self) -> dict[tuple[Any, str], list[float]]:
        """Every corner's singular values, largest scaled to one -- YASTN
        ``calculate_corner_svd``:468.

        Returns
        -------
        dict
            ``(site, corner name) -> descending singular values``.
        """
        out = {}
        for site in self.sites():
            for name in ("tl", "tr", "bl", "br"):
                s = tenet.linalg.svd(getattr(self[site], name), ((0,), (1,)))[1]
                values = spectrum(s)
                out[site, name] = [v / values[0] for v in values] if values else []
        return out

    def iterate_(
        self,
        max_bond: int | None = None,
        moves: str = "hv",
        max_sweeps: int = 100,
        corner_tol: float | None = 1e-10,
        cutoff: float | None = 1e-14,
    ) -> CTM_out:
        """Sweep until the corner spectra stop moving, YASTN ``iterate_``:841.

        Parameters
        ----------
        max_bond : int or None, optional
            The environment bond-dimension cap. Default ``None``.
        moves : str, optional
            The sweep's moves; ``'hv'`` (the default) and ``'lrtb'`` are the two
            sensible ones.
        max_sweeps : int, optional
            The sweep budget. Default ``100``.
        corner_tol : float or None, optional
            Stop when the worst corner's spectrum moves less than this. ``None`` runs
            the budget out. Default ``1e-10``.
        cutoff : float or None, optional
            Relative singular-value cutoff for the projector truncation.
            Default ``1e-14``.

        Returns
        -------
        CTM_out
            ``sweeps``, ``max_dsv`` and ``converged``.

        Raises
        ------
        StructureChangingError
            Under ``jax.jit``/``jax.grad``: the loop reads a spectrum to decide when to
            stop and reads singular values to decide a bond.
        """
        previous, max_dsv, sweep = None, float("inf"), 0
        for sweep in range(1, max_sweeps + 1):
            self.update_(max_bond, moves, cutoff)
            if corner_tol is None:
                continue
            current = self.corner_spectra()
            if previous is not None:
                max_dsv = max(_spec_diff(previous[k], current[k]) for k in current)
            previous = current
            if max_dsv < corner_tol:
                break
        return CTM_out(sweep, max_dsv, max_dsv < (corner_tol or 0.0))

    def items(self) -> Iterator[tuple[Site, EnvLocal]]:
        """``(site, environment)`` for every unique site."""
        return self.env.items()


#: The two sides of each corner, and of each edge's two environment legs. Derived in the
#: module docstring: the ring must close, adjacent enlarged corners must meet, and the
#: two projectors of one cut carry the two ends of one new bond.
_CORNER_SIDES = {"tl": (IN, IN), "tr": (OUT, IN), "br": (OUT, OUT), "bl": (IN, OUT)}
_EDGE_SIDES = {"l": (IN, OUT), "r": (OUT, IN), "t": (OUT, IN), "b": (IN, OUT)}


def _spec_diff(old: list[float], new: list[float]) -> float:
    """Euclidean distance between two corner spectra, zero-padded to the longer one.

    While the environment is still growing the two have different lengths, and the
    padding makes that a large change rather than an error -- which is what it is.
    """
    n = max(len(old), len(new))
    old = old + [0.0] * (n - len(old))
    new = new + [0.0] * (n - len(new))
    return sum((a - b) ** 2 for a, b in zip(old, new, strict=True)) ** 0.5
