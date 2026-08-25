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

**The C4v specialization.** [EnvCTMc4v][tenet.network.EnvCTMc4v] is the same machinery
with the eight tensors of a site collapsed onto the two a point-group-symmetric ansatz
leaves distinct -- its class docstring carries the leg conventions, which are the point
group's and not the ones above.

**The bond metric.** [bond_metric][tenet.network.EnvCTM.bond_metric] is the one thing
M79d added here: the six environment tensors around a bond, closed around the two
``qr``-reduced site tensors [truncate_][tenet.network.truncate_] hands it. It is the
full-update metric and nothing about it is symmetrized or checked -- that is the
caller's, and ``truncate_`` says what it found.

**What is deliberately not transcribed** (M79b scope): the ``'1x2'`` projector method,
the 5x4 extended corners YASTN grows when a PEPS bond is one-dimensional (a hexagonal
lattice embedded on a square one), ``boundary_mps``, checkpointing, serialization and the
``patch`` mechanism -- and with the last of those, any re-convergence after a truncation:
an environment is stale the moment a bond changes, and the caller decides when to sweep.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from typing import Any, NamedTuple

import autoray as ar

import tenet
from tenet import IN, OUT, Leg, SymmetricTensor
from tenet.network.common import composed, ones, spectrum, supplies_in
from tenet.network.lattice import Lattice, Site
from tenet.network.peps import (
    DoublePepsTensor,
    Peps2Layers,
    append_vec_bl,
    append_vec_br,
    append_vec_tl,
    append_vec_tr,
)

__all__ = [
    "CTMRG_out",
    "EnvCTM",
    "EnvCTMc4v",
    "EnvLocal",
    "EnvLocalC4v",
    "EnvProjectors",
    "PepsFlip",
    "corner2x2",
    "flip",
    "proj_corners",
]

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


class CTMRG_out(NamedTuple):
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
#
# ``common.composed`` is the one spelling; M79d moved it there when ``evolution.py``
# became its second caller, and the two private aliases keep this module's call sites
# and its ``ast``-driven coverage test reading exactly as they did.

_supplies_in = supplies_in
_composed = composed


def _normalized(t: SymmetricTensor) -> SymmetricTensor:
    """``t / ||t||`` after every move, so a growing environment stays ``O(1)``.

    Simplification: the Frobenius norm where YASTN takes the infinity norm. tenet has
    one public norm and the two differ by at most ``sqrt(dim)``; what the division is
    for -- keeping the corner from tracking the partition function itself -- does not
    distinguish them.
    """
    return t / tenet.norm(t)


def _dual(leg: Any) -> Leg:
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


def corner2x2(env: "EnvCTM", which: str, site: Any, a: Any = None) -> SymmetricTensor:
    """One 2x2 enlarged corner: two edges, the corner between them, and the site.

    Parameters
    ----------
    env : EnvCTM
        The environment to read.
    which : str
        ``'tl'``, ``'tr'``, ``'bl'`` or ``'br'``.
    site : Site or tuple[int, int]
        The site whose ring supplies the three environment tensors.
    a : DoublePepsTensor or SymmetricTensor or None, optional
        The tensor to absorb, in place of the one the state has at ``site``. Default
        ``None``, i.e. the state's. [bond_metric][tenet.network.EnvCTM.bond_metric]
        passes the ``qr``-reduced site here, which is the only reason the parameter
        exists: the environment ring is the site's, the tensor inside it is not.

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
    if a is None:
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
        first: Any = self.psi[self.sites()[0]]
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

    def _groups(
        self, half: SymmetricTensor, order: tuple[int, int]
    ) -> tuple[Sequence[int], Sequence[int]]:
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
        vec = _composed(f"x{l_}c,c{t_}n->x{l_}{t_}n", self[left].l, self.proj[left].hlt)
        mid = self._absorb_site(left, "tl", vec)
        fresh.l = _normalized(_composed(f"x{b_}m,x{b_}n{r_}->m{r_}n", self.proj[left].hlb, mid))
        above = self.nn_site(site, "tl")
        if above is not None:
            corner = _composed(f"ad,d{t_}y->a{t_}y", self[left].tl, self[left].t)
            fresh.tl = _normalized(_composed(f"a{t_}m,a{t_}y->my", self.proj[above].hlb, corner))
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
        fresh.r = _normalized(_composed(f"x{t_}m,x{t_}n{l_}->m{l_}n", self.proj[right].hrt, mid))
        above = self.nn_site(site, "tr")
        if above is not None:
            corner = _composed(f"ae,e{t_}n->a{t_}n", self[right].tr, self.proj[above].hrb)
            fresh.tr = _normalized(_composed(f"y{t_}a,a{t_}n->yn", self[right].t, corner))
        below = self.nn_site(site, "br")
        if below is not None:
            corner = _composed(f"ae,e{b_}y->a{b_}y", self[right].br, self[right].b)
            fresh.br = _normalized(_composed(f"a{b_}m,a{b_}y->my", self.proj[below].hrt, corner))

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
        fresh.b = _normalized(_composed(f"m{t_}y{l_},y{l_}n->m{t_}n", mid, self.proj[bottom].vbl))
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
    ) -> CTMRG_out:
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
        CTMRG_out
            ``sweeps``, ``max_dsv`` and ``converged``.

        Raises
        ------
        StructureChangingError
            Under ``jax.jit``/``jax.grad``: the loop reads a spectrum to decide when to
            stop and reads singular values to decide a bond.
        """
        previous, max_dsv, sweeps = None, float("inf"), 0
        while sweeps < max_sweeps:
            self.update_(max_bond, moves, cutoff)
            sweeps += 1
            if corner_tol is None:
                continue
            current = self.corner_spectra()
            if previous is not None:
                max_dsv = max(_spec_diff(previous[k], current[k]) for k in current)
            previous = current
            if max_dsv < corner_tol:
                break
        return CTMRG_out(sweeps, max_dsv, max_dsv < (corner_tol or 0.0))

    def bond_metric(
        self, q0: SymmetricTensor, q1: SymmetricTensor, s0: Any, s1: Any, dirn: str
    ) -> SymmetricTensor:
        """The full-update bond metric: the six environment tensors closed round a bond.

        Parameters
        ----------
        q0, q1 : SymmetricTensor
            The two reduced site tensors, rank 5, whose bond legs the metric is on --
            [truncate_][tenet.network.truncate_]'s ``qr`` isometries, not the state's
            own tensors.
        s0, s1 : Site or tuple[int, int]
            Their sites, in the fermionic order.
        dirn : str
            ``'lr'`` for a horizontal bond, ``'tb'`` for a vertical one.

        Returns
        -------
        SymmetricTensor
            Rank 4, ``(bra0, bra1, ket0, ket1)``: a map from the pair of ket bond legs
            to the pair of bra ones, **not** symmetrized and **not** checked for
            positivity. ``truncate_`` measures both and says what it found.

        Raises
        ------
        ValueError
            If the state is a single layer, which has no bond to truncate, or if the two
            sites are not adjacent in the direction ``dirn`` names.

        Examples
        --------
        >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
        >>> from tenet.network import EnvCTM, Peps, SquareLattice
        >>> from tenet.symmetry import U1, U1Sector
        >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
        >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(V, OUT))
        >>> psi = Peps(SquareLattice(dims=(2, 2)), SymmetricTensor.random(legs, seed=0))
        >>> env = EnvCTM(psi, init="dl")
        >>> env.bond_metric(psi[0, 0], psi[0, 1], (0, 0), (0, 1), "lr").ndim
        4

        Notes
        -----
        YASTN's ``bond_metric``:770. The picture, for ``dirn == 'lr'``::

            tl -- t ------- t -- tr
             |    |         |    |
             l -- Q0 --   -- Q1 -- r
             |    |         |    |
            bl -- b ------- b -- br

        The two 2x2 enlarged corners are [corner2x2][tenet.network.corner2x2] with the
        reduced tensor passed in place of the site's -- the environment ring is the
        site's, the tensor inside it is not -- so this adds no contraction primitive to
        the ones M79a wrote. The remaining four tensors close the top and bottom (or
        left and right) of the picture, two compositions each.

        **The last composition closes the environment ring, and it pays the ribbon twist
        [closed][tenet.network.closed] cannot read.** An environment leg is the boundary
        bond of the *double* layer -- one line of the bra network and one of the ket at
        once -- so joining the two halves over the two remaining environment wires closes
        a cycle in each layer. The bend rule reads one orientation per leg, and both these
        legs supply ``IN`` from the same half, so it finds nothing to bend and pays
        nothing; the closure is there all the same, and [tenet.twist][] on both of them is
        its ``theta``, one per layer. Measured against a cluster contracted site by site
        with no environment at all: exact on every loop-free cluster either way (the ring
        is one-dimensional there, so ``theta`` is 1), and 1.6 and 0.79 out on the ``lr``
        and ``tb`` bonds of a 2x2 patch under fermion parity without it -- 4e-15 and
        1.7e-15 with it,
        and the same on an interior bond of a 3x3, where both environment wires carry
        sectors and twisting only one is wrong (``docs/design.md``, M84).
        """
        if not self.double:
            raise ValueError("EnvCTM.bond_metric: a single-layer network has no bond to truncate")
        step = (0, 1) if dirn == "lr" else (1, 0)
        if self.nn_site(s0, step) != tuple(s1):
            raise ValueError(
                f"EnvCTM.bond_metric: {Site(*s0)} and {Site(*s1)} are not a {dirn} bond"
            )
        e0, e1 = self[s0], self[s1]
        vec0 = corner2x2(self, "tl", s0, DoublePepsTensor(q0, tenet.adjoint(q0)))
        vec1 = corner2x2(self, "br", s1, DoublePepsTensor(q1, tenet.adjoint(q1)))
        if dirn == "lr":
            bottom = _composed("xbBc,cd->xbBd", e0.b, e0.bl)
            vec0 = _composed("xbBd,dbByrR->xyrR", bottom, vec0)
            top = _composed("xtTc,cd->xtTd", e1.t, e1.tr)
            vec1 = _composed("xtTd,dtTylL->xylL", top, vec1)
            return _composed("pqrR,qplL->RLrl", tenet.twist(vec0, (0, 1)), vec1)
        right = _composed("cd,dxXe->cxXe", e0.tr, e0.r)
        vec0 = _composed("abByrR,yrRd->abBd", vec0, right)
        left = _composed("cd,dxXe->cxXe", e1.bl, e1.l)
        vec1 = _composed("atTylL,ylLd->atTd", vec1, left)
        return _composed("abBd,dtTa->BTbt", tenet.twist(vec0, (0, 3)), vec1)

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


# -- the C4v specialization ----------------------------------------------------------


def flip(t: SymmetricTensor) -> SymmetricTensor:
    """The sublattice partner of ``t``: every leg's ``side`` reversed, the blocks kept.

    Parameters
    ----------
    t : SymmetricTensor
        Any tensor.

    Returns
    -------
    SymmetricTensor
        ``tenet.conj(tenet.adjoint(t))`` -- each leg keeps its space, ``dual``, ``name``
        and position and reverses its ``side``, so every leg of the result contracts
        with the leg of ``t`` that sits in the same position.

    Examples
    --------
    >>> import tenet
    >>> from tenet import OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import flip
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT), Leg(V, OUT)), seed=0)
    >>> b = flip(a)
    >>> b.legs[0].side, b.legs[0].dual
    (<Side.IN: 'in'>, False)
    >>> bool(tenet.allclose(flip(b), a))
    True

    Notes
    -----
    YASTN's ``flip_signature``. ``tenet.adjoint`` alone conjugates the blocks, which
    would make the ``B`` sublattice the complex conjugate of the ``A`` one and give a
    different network; the ``tenet.conj`` puts them back.
    """
    return tenet.conj(tenet.adjoint(t))


class PepsFlip:
    """A read-only view of a network whose odd sites come back flipped.

    Parameters
    ----------
    base : Peps or Peps2Layers
        The network holding ``A``. Every other attribute is forwarded to it.

    Examples
    --------
    >>> from tenet import OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import Peps, PepsFlip, SquareLattice
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> a = SymmetricTensor.random((Leg(V, OUT),) * 4, seed=0)
    >>> psi = PepsFlip(Peps(SquareLattice(dims=(1, 1)), a))
    >>> psi[0, 0].legs[0].side, psi[0, 1].legs[0].side
    (<Side.OUT: 'out'>, <Side.IN: 'in'>)

    Notes
    -----
    YASTN's ``PsiFlip``, wrapped *outside* the double layer rather than inside it, so a
    [DoublePepsTensor][tenet.network.DoublePepsTensor]'s bra is flipped with its ket. The two
    agree: ``flip(adjoint(a))`` is ``adjoint(flip(a))``.
    """

    __slots__ = ("_base",)

    def __init__(self, base: Any) -> None:
        self._base = base

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __getitem__(self, site: Any) -> Any:
        obj = self._base[site]
        if (site[0] + site[1]) % 2 == 0:
            return obj
        ket = getattr(obj, "ket", None)
        if ket is None:
            return flip(obj)
        return DoublePepsTensor(ket=flip(ket), bra=flip(obj.bra))

    def __repr__(self) -> str:
        return f"PepsFlip({self._base!r})"


@dataclass
class EnvLocalC4v:
    """The C4v environment of one site: one corner and one edge, read under eight names.

    Attributes
    ----------
    tl : SymmetricTensor or None
        The corner, rank 2. ``tr``, ``bl`` and ``br`` read the same tensor.
    t : SymmetricTensor or None
        The edge, rank 3 (single layer) or rank 4 (double layer). ``l``, ``b`` and ``r``
        read the same tensor.

    Notes
    -----
    YASTN's ``EnvCTM_c4v_local``. The aliases are what let
    [corner2x2][tenet.network.corner2x2] and every measurement written against
    [EnvLocal][tenet.network.EnvLocal] run here unchanged.
    """

    tl: SymmetricTensor | None = None
    t: SymmetricTensor | None = None

    def __getattr__(self, name: str) -> Any:
        if name in ("tr", "bl", "br"):
            return self.tl
        if name in ("l", "b", "r"):
            return self.t
        raise AttributeError(f"EnvLocalC4v has no attribute {name!r}")


class _EnvFlip:
    """One site's environment with every tensor flipped on the way out -- YASTN's
    ``EnvFlip``. Odd sites of the checkerboard read their neighbours' tensors through it."""

    __slots__ = ("_base",)

    def __init__(self, base: EnvLocalC4v) -> None:
        self._base = base

    def __getattr__(self, name: str) -> Any:
        return flip(getattr(self._base, name))


class EnvCTMc4v(EnvCTM):
    """A C4v CTM environment: one corner, one edge, one move.

    Parameters
    ----------
    psi : Peps
        The network, on a one-site geometry. Its site tensor must have four *identical*
        virtual legs -- the C4v ansatz constraint, without which no rotation acts on it.
    init : str or None, optional
        ``'eye'`` (the default), ``'dl'`` or ``None``, as
        [EnvCTM][tenet.network.EnvCTM] takes them.
    bra : Peps or None, optional
        An independent bra for the double layer. Default ``None``.

    Raises
    ------
    ValueError
        If the geometry has more than one unique site, if the four virtual legs of the
        site tensor are not identical, or if ``init`` is not one of the three.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import EnvCTMc4v, Peps, SquareLattice
    >>> from tenet.symmetry import Z2, Z2Sector
    >>> beta = 0.3
    >>> c, s = np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))
    >>> w = np.array([[c, s], [c, -s]])
    >>> block = np.einsum("st,sl,sb,sr->tlbr", w, w, w, w)
    >>> V = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    >>> legs = (Leg(V, OUT),) * 4
    >>> psi = Peps(SquareLattice(dims=(1, 1)), SymmetricTensor.from_dense(block, legs))
    >>> env = EnvCTMc4v(psi)
    >>> env.iterate_(max_bond=8, max_sweeps=50, corner_tol=1e-10).converged
    True

    Notes
    -----
    The geometry is one unique site and the checkerboard lives in the views: every site
    of the plane folds onto the same record, and
    [PepsFlip][tenet.network.PepsFlip] (and its environment twin) flip what an odd site
    hands back. That is why there is one ``C`` and one ``T`` and not two of each.
    """

    def __init__(self, psi: Any, init: str | None = "eye", bra: Any = None) -> None:
        if len(psi.geometry.sites()) != 1:
            raise ValueError(
                f"EnvCTMc4v: a C4v environment has one unique site, got "
                f"{len(psi.geometry.sites())} -- use SquareLattice(dims=(1, 1))"
            )
        a = psi[psi.geometry.sites()[0]]
        virtual = {(leg.space, leg.side, leg.dual) for leg in a.legs[:4]}
        if len(virtual) != 1:
            raise ValueError(
                "EnvCTMc4v: the four virtual legs of a C4v site tensor must be identical "
                "-- a 90-degree rotation cycles them, so it acts only if they are one leg"
            )
        self.geometry = psi.geometry
        self.double = psi.has_physical()
        self.psi = PepsFlip(Peps2Layers(psi, bra) if self.double else psi)
        self.env = Lattice(self.geometry, {site: EnvLocalC4v() for site in self.sites()})
        if init is not None:
            if init not in ("eye", "dl"):
                raise ValueError(f"EnvCTMc4v: init={init!r} should be 'eye', 'dl' or None")
            self.reset_(init)

    def __getitem__(self, site: Any) -> Any:
        local = self.env[site]
        return local if (site[0] + site[1]) % 2 == 0 else _EnvFlip(local)

    def __repr__(self) -> str:
        return f"EnvCTMc4v(geometry={self.geometry!r}, double={self.double})"

    def reset_(self, init: str = "eye") -> None:
        """Seed the corner and the edge, YASTN ``reset_``.

        Parameters
        ----------
        init : str, optional
            ``'eye'`` (the default) puts a one-dimensional environment bond on both:
            the corner is the scalar one and the edge closes the ket against the bra (a
            single layer's free boundary is all ones). ``'dl'`` absorbs one layer into
            that seed without truncating.

        Raises
        ------
        ValueError
            If ``init`` is not ``'eye'`` or ``'dl'``.
        """
        if init not in ("eye", "dl"):
            raise ValueError(f"EnvCTMc4v: init={init!r} should be 'eye' or 'dl'")
        site = self.sites()[0]
        first: Any = self.psi[site]
        provider = (first.ket if self.double else first).provider
        unit = tenet.GradedSpace.new(provider, {provider.unit: 1})
        local = self.env[site]
        local.tl = ones((Leg(unit, OUT), Leg(unit, OUT)))
        pair = self.site_legs(site, "t")
        env = (Leg(unit, IN), Leg(unit, IN))
        if not self.double:
            local.t = ones((env[0], *pair, env[1]))
        else:
            delta = tenet.identity((Leg(pair[0].space, OUT, dual=pair[0].dual),))
            if pair[0].side is IN:
                delta = tenet.adjoint(delta)
            local.t = tenet.einsum("ac,de->adec", ones(env), delta)
        if init == "dl":
            self.update_(max_bond=None, cutoff=0.0)

    def update_(
        self,
        max_bond: int | None = None,
        moves: str = "d",
        cutoff: float | None = 1e-14,
        bond: Any = None,
    ) -> None:
        """One sweep of the single move, in place.

        Parameters
        ----------
        max_bond : int or None, optional
            The environment bond-dimension cap. Default ``None``, i.e. no cap.
        moves : str, optional
            ``'d'``, the only move a C4v environment has. Default ``'d'``.
        cutoff : float or None, optional
            Relative singular-value cutoff for the projector truncation.
            Default ``1e-14``.
        bond : GradedSpace or None, optional
            A frozen environment bond. ``None`` (the default) decides one from the
            singular values, which no trace allows; a bond reuses one decided outside
            and makes the move shape-static and differentiable.

        Raises
        ------
        ValueError
            If ``moves`` is anything but ``'d'``.
        """
        if set(moves) - {"d"}:
            raise ValueError(f"EnvCTMc4v: moves={moves!r}; a C4v environment has only 'd'")
        for _ in moves:
            self._move_d(max_bond, cutoff, bond)

    def iterate_(
        self,
        max_bond: int | None = None,
        moves: str = "d",
        max_sweeps: int = 100,
        corner_tol: float | None = 1e-10,
        cutoff: float | None = 1e-14,
    ) -> Any:
        """The parent's sweep loop, with ``'d'`` as the move -- YASTN ``iterate_``."""
        return super().iterate_(max_bond, moves, max_sweeps, corner_tol, cutoff)

    def _move_d(self, max_bond: int | None, cutoff: float | None, bond: Any) -> None:
        """The move, YASTN ``_update_2x2_``.

        The 2x2 enlarged corner is [corner2x2][tenet.network.corner2x2]'s, unchanged --
        two edges, the corner between them and the site, in two mirror leg groups. Its
        SVD gives the projector ``U``; the renormalized corner is ``V^dagger U S`` and
        the renormalized edge is the neighbour's edge with the neighbour's site absorbed
        and ``U`` closed on both of its mirror groups.
        """
        site = self.sites()[0]
        cor = corner2x2(self, "tl", site)
        n = cor.ndim // 2
        axes = (tuple(range(n)), tuple(range(n, 2 * n)))
        if bond is None:
            u, s, vh = tenet.linalg.svd_truncated(cor, axes, max_bond=max_bond, cutoff=cutoff)
        else:
            u, s, vh = tenet.linalg.svd(cor, axes, bond=bond)
        cut = "ijk"[:n]
        new_tl = _composed(f"m{cut},{cut}n->mn", tenet.adjoint(vh), u)
        new_tl = flip(_composed("mn,nk->mk", new_tl, s))
        b_, r_, t_ = (self.wire(d) for d in "brt")
        neighbour = self.nn_site(site, "r")
        vec = _composed(f"x{b_}n,x{t_}a->n{b_}{t_}a", u, self[neighbour].t)
        mid = self._absorb_site(neighbour, "tl", vec)
        new_t = _composed(f"m{b_}y{r_},y{r_}n->m{b_}n", mid, u)
        local = self.env[site]
        local.tl, local.t = _normalized(new_tl), _normalized(new_t)
