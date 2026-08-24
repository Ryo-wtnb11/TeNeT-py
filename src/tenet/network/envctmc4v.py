"""The C4v corner-transfer-matrix environment: one corner, one edge, one move.

YASTN's ``fpeps/envs/_env_ctm_c4v.py`` (b0187c4), adopted by M79/#277 as a
*specialization* of [EnvCTM][tenet.network.EnvCTM] rather than a second algorithm: the
same enlarged corner, the same absorption, the same sweep, with the eight environment
tensors of a site collapsed onto the two the point group leaves distinct.

**Why a flip is needed at all.** A C4v-symmetric site tensor transforms covariantly under
the 90-degree rotation, which cycles ``t -> l -> b -> r -> t``. A rotation can only be a
symmetry of the tensor if those four legs are the *same* leg -- same space, same ``side``,
same ``dual`` -- so a C4v ansatz cannot carry
[Peps][tenet.network.Peps]'s alternating ``(t IN, l OUT, b OUT, r IN)``. Four identical
virtual legs do not tile the plane with themselves: ``A``'s right leg meets its right
neighbour's left leg, and two identical legs never contract. The plane is therefore a
checkerboard of ``A`` and ``B = flip(A)``, and one unique ``C`` and ``T`` describe the
whole environment because the ``B`` sublattice's are ``flip(C)`` and ``flip(T)``.

**The flip.** [flip][tenet.network.flip] is ``tenet.conj(tenet.adjoint(t))``: every leg
keeps its space, ``dual`` and position and reverses its ``side``, and the blocks are
those of ``t`` itself. It is YASTN's ``flip_signature`` -- the operation that lane's
docstring calls "adjoint times complex conjugation" -- and tenet has no single primitive
spelling it. ``tenet.flip_dual`` is a different operation: it toggles ``dual`` and
relabels the space through ``provider.dual``, keeping the *same* morphism, so its result
contracts with the same partners the original did, which is the opposite of what a
sublattice partner must do.

**Leg conventions**, and they are the point-group's, not
[envctm][tenet.network.envctm]'s. The rotation identifies each side of the ring with the
next, so all four corners of a site's ring are one tensor and all four edges are one::

    C ---- T ---- C
    |      |      |
    T --- (A) --- T
    |      |      |
    C ---- T ---- C

with the corner ``(X OUT, X OUT)`` and the edge ``(X IN, <A's leg, once per layer>,
X IN)``. Both of a corner's legs sit on one side and both of an edge's environment legs on
the other, which is what lets one tensor serve four positions; the ring still closes
because corners and edges alternate around it. Two corners that meet *without* an edge
between them -- the four-corner object in Baxter's telescoping -- meet across a
sublattice boundary, so one of them enters flipped.

**No Hermiticity assumption.** The projector is the ``U`` of an SVD of the 2x2 enlarged
corner, and the renormalized corner is ``V^dagger U S`` rather than ``S``: the two index
groups enter as two different factors and the correction they differ by is kept instead
of assumed to be the identity. Assuming it is the identity is assuming the enlarged
corner is Hermitian and positive, which is the assumption M63/#243 measured and found
false for every ansatz short of the full point group.

**What is deliberately not transcribed**: YASTN's ``'1x2'`` projector (a QR of ``C @ T``,
which cannot grow the environment bond past its seed and so is a fixed-bond variant of
this one), ``leg_charge_conv_check``, and the partial-SVD spectrum prediction.
"""

from dataclasses import dataclass
from typing import Any

import tenet
from tenet import IN, OUT, Leg, SymmetricTensor
from tenet.network.common import ones
from tenet.network.envctm import EnvCTM, _composed, _normalized, corner2x2
from tenet.network.lattice import Lattice
from tenet.network.peps import DoubleLayer, Peps2Layers

__all__ = ["EnvCTMc4v", "EnvLocalC4v", "PepsFlip", "flip"]


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
    [DoubleLayer][tenet.network.DoubleLayer]'s bra is flipped with its ket. The two
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
        if isinstance(obj, DoubleLayer):
            return DoubleLayer(ket=flip(obj.ket), bra=flip(obj.bra))
        return flip(obj)

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
    >>> psi = Peps(SquareLattice(dims=(1, 1)), SymmetricTensor.from_dense(block, (Leg(V, OUT),) * 4))
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

    def _move_(self, move: str, max_bond: int | None, cutoff: float | None) -> None:
        """The parent's per-move entry point, narrowed to ``'d'``."""
        if move != "d":
            raise ValueError(f"EnvCTMc4v: move={move!r}; a C4v environment has only 'd'")
        self._move_d(max_bond, cutoff, None)

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
