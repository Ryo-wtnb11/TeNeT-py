"""PEPS on a lattice, the lazy double layer, and the twelve contraction primitives.

YASTN's ``fpeps/_peps.py``, ``_doublePepsTensor.py`` and ``envs/_env_contractions.py``
(b0187c4), adopted and re-spelled in this package's idiom.

**Leg order and signature.** A site tensor is rank 5 on
``(t IN, l OUT, b OUT, r IN, phys OUT)`` -- YASTN's ``(top, left, bottom, right,
physical)`` at YASTN's own signature ``(-1, +1, +1, -1, +1)``, with ``-1`` read as ``IN``.
Rank 4, the same order without ``phys``, is a classical/partition-function network and
is carried by [Peps][tenet.network.Peps] but not by the double layer.

**The double layer is never materialized.**
[Peps2Layers][tenet.network.Peps2Layers] is a *view*: indexing it returns a
[DoublePepsTensor][tenet.network.DoublePepsTensor], which holds a bra and a ket and knows it has
four legs, each a ket/bra pair. Every contraction below reaches through it into the two
rank-5 tensors, so nothing of size ``d^2 D^8`` is ever written.

**Pairs stay unfused.** YASTN fuses each ``[x x']`` pair into one leg and reads the
fused leg back out with ``unfuse_legs``. Here a corner comes back with the ket leg and
the bra leg *adjacent and separate* (froSTspin ``ctm_environment.py``:16-33): fusing buys
a matrix shape this layer never needs, and a fuse plus a later unfuse is two passes over
the whole tensor.

**The composition rule, and how the operand order is derived.** Every step below is a
two-operand composition -- operand 1 supplies the ``IN`` end of every shared wire -- and
a wire that turns around is bent explicitly, as the step's ``bend`` field. With
``ket = (t IN, l OUT, b OUT, r IN, phys OUT)`` and ``bra = adjoint(ket)``, each wire has
a fixed answer to *who supplies* ``IN``::

    t -> ket      l -> bra      b -> bra      r -> ket      phys -> bra

so operand 1 is whichever side wins that count over the wires a primitive contracts, and
the losers are the bent wires. **Minimality is the criterion, not a preference**: a bend
is a genuine categorical operation, so a spelling that bends two wires where one suffices
has turned a wire the planar diagram does not turn -- and it lands on a different tensor,
which ``test_the_bend_count_is_minimal`` measures rather than assumes. Two primitives,
``edge_t`` and ``edge_r``, contract one ket wire against one bra wire and so tie at one
bend either way; both are recorded as taking the bra first, and the tie is settled by
physics rather than by counting: a whole network contracted against its Onsager oracle.

Every chain here runs through [closed][tenet.network.closed] rather than
[tenet.einsum_chain][]: each of these primitives contracts more than one
wire between ket and bra, so each closes a cycle, and a step that closes one pays the
ribbon twist on the wires it bends. That is what makes the bra-first and ket-first
spellings of the tied primitives the same tensor under fermion parity as well.

The Jordan-Wigner strings YASTN writes as explicit ``swap_gate`` calls are paid here by
the braiding the lowering already performs. The bends are **not** in one-to-one
correspondence with YASTN's ``swap_gate`` lines -- those also compensate for YASTN's own
fused-leg ordering -- and the correspondence that does hold is at the level of the
resulting tensor, checked element by element against a dense oracle built from parity
vectors alone (``tests/network/test_peps.py``).

Simplification: no operator on the physical leg. YASTN's ``DoublePepsTensor`` carries an
``op`` (and pending swap charges) so a measurement can ride the same primitives; that is
one extra chain step and it arrives with the measurements.
"""

from typing import Any, NamedTuple

import tenet
from tenet import SymmetricTensor
from tenet.network.common import closed
from tenet.network.lattice import Lattice

__all__ = [
    "DoublePepsTensor",
    "Peps",
    "Peps2Layers",
    "append_vec_bl",
    "append_vec_br",
    "append_vec_tl",
    "append_vec_tr",
    "cor_bl",
    "cor_br",
    "cor_tl",
    "cor_tr",
    "edge_b",
    "edge_l",
    "edge_r",
    "edge_t",
]


class Peps(Lattice):
    """A PEPS: one rank-5 (or rank-4) tensor per unique site of a lattice.

    Parameters
    ----------
    geometry : SquareLattice or Lattice
        The lattice the state lives on.
    tensors : optional
        One tensor, a nested sequence, or a ``{site: tensor}`` mapping, as
        [Lattice][tenet.network.Lattice] takes them.

    Raises
    ------
    ValueError
        If a tensor is not rank 4 or rank 5, if the ranks disagree between sites, or
        from [Lattice][tenet.network.Lattice]'s own assignment checks.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import CheckerboardLattice, Peps
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(V, OUT))
    >>> a = SymmetricTensor.random(legs, seed=0)
    >>> psi = Peps(CheckerboardLattice(), a)
    >>> psi.has_physical(), psi[1, 1] is a
    (True, True)

    Notes
    -----
    Leg order is ``(t, l, b, r, phys)``; see the module docstring for the signature.
    Nothing here checks that a site's ``r`` leg meets its right neighbour's ``l`` leg --
    that check belongs to the first contraction that tries it, and it is the one place a
    wrong space produces a message naming both legs.
    """

    def __init__(self, geometry: Any, tensors: Any = None) -> None:
        super().__init__(geometry, tensors)
        ranks = {t.ndim for _, t in self.items() if t is not None}
        if ranks - {4, 5}:
            raise ValueError(f"Peps: site tensors should be rank 4 or rank 5, got ranks {ranks}")
        if len(ranks) > 1:
            raise ValueError(f"Peps: every site should have the same rank, got {ranks}")

    def has_physical(self) -> bool:
        """Whether the sites carry a physical leg (rank 5) or not (rank 4)."""
        return self[self.sites()[0]].ndim == 5


class DoublePepsTensor(NamedTuple):
    """One site of a bra-ket double layer, held as its two factors and never multiplied.

    Attributes
    ----------
    ket : SymmetricTensor
        The rank-5 site tensor, ``(t IN, l OUT, b OUT, r IN, phys OUT)``.
    bra : SymmetricTensor
        Its partner, same leg order with every side flipped -- ``tenet.adjoint(ket)``
        unless a different bra was supplied.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import DoublePepsTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(V, OUT))
    >>> a = SymmetricTensor.random(legs, seed=0)
    >>> t = DoublePepsTensor(a, tenet.adjoint(a))
    >>> t.ndim, len(t.legs)
    (4, 4)

    Notes
    -----
    ``ndim`` is 4 and [legs][tenet.network.DoublePepsTensor.legs] returns four *pairs*: the
    object behaves like the rank-4 tensor a CTM environment sees, while the twelve
    primitives in this module reach past it into ``ket`` and ``bra``. YASTN's
    ``DoublePepsTensor`` is the same idea with a ``trans`` field restricting transposes
    to the leg-order-preserving ones; that field is not here because no caller
    transposes a double layer -- the primitives take the direction in their *names*.
    """

    ket: SymmetricTensor
    bra: SymmetricTensor

    @property
    def ndim(self) -> int:
        """Four: the double layer hides the physical wire it already closed."""
        return 4

    @property
    def legs(self) -> tuple[tuple[Any, Any], ...]:
        """``((t_ket, t_bra), (l_ket, l_bra), (b_ket, b_bra), (r_ket, r_bra))``."""
        return tuple((self.ket.legs[i], self.bra.legs[i]) for i in range(4))


class Peps2Layers(Lattice):
    """A *view* of a bra and a ket as one double-layer network.

    Parameters
    ----------
    ket : Peps
        The state.
    bra : Peps or None
        Its partner. ``None`` (the default) means ``tenet.adjoint`` of every ket
        tensor, which is the ``<psi|psi>`` network.

    Raises
    ------
    ValueError
        If ``ket`` has no physical leg, or if ``bra``'s geometry differs.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import CheckerboardLattice, Peps, Peps2Layers
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(V, OUT))
    >>> psi = Peps(CheckerboardLattice(), SymmetricTensor.random(legs, seed=0))
    >>> net = Peps2Layers(psi)
    >>> net[0, 1].ndim
    4

    Notes
    -----
    Indexing builds a [DoublePepsTensor][tenet.network.DoublePepsTensor], which costs a tuple.
    The bra tensors are built once, at construction, rather than per read: an
    ``adjoint`` is a pass over every block, and a CTMRG sweep reads each site many
    times.
    """

    def __init__(self, ket: Peps, bra: Peps | None = None) -> None:
        super().__init__(ket.geometry)
        if not ket.has_physical():
            raise ValueError("Peps2Layers: the ket needs a physical leg (rank-5 tensors)")
        if bra is not None and bra.geometry != ket.geometry:
            raise ValueError("Peps2Layers: bra and ket must share a geometry")
        self.ket = ket
        self.bra = (
            bra
            if bra is not None
            else Peps(ket.geometry, {s: tenet.adjoint(t) for s, t in ket.items()})
        )

    def __getitem__(self, site: Any) -> DoublePepsTensor:
        return DoublePepsTensor(ket=self.ket[site], bra=self.bra[site])

    def __setitem__(self, site: Any, obj: Any) -> None:
        raise TypeError("Peps2Layers is a view; set the tensor on its ket or bra instead")


# -- corners: one site closed on two adjacent virtual legs and the physical one -------
#
# ``ket`` is ``(t IN, l OUT, b OUT, r IN, s OUT)`` and ``bra`` is its adjoint, so each
# wire has a fixed answer to "who supplies IN?":
#
#     t -> ket      l -> bra      b -> bra      r -> ket      s -> bra
#
# Operand 1 is whichever side wins that count over the wires the primitive contracts,
# and the losers are the bent wires. That is the whole operand-order proof; it is
# tabulated once here and quoted per function.


def cor_tl(a: DoublePepsTensor) -> SymmetricTensor:
    """YASTN ``cor_tl``. Close ``t``, ``l`` and ``phys``; keep ``(b_k, b_b, r_k, r_b)``.

    Parameters
    ----------
    a : DoublePepsTensor
        The site.

    Returns
    -------
    SymmetricTensor
        Rank 4, YASTN's ``[b b'] [r r']`` with each pair left unfused.

    Notes
    -----
    Wires ``t``, ``l``, ``s``; ``IN`` comes from ket, bra, bra. Two of three say bra, so
    **bra is operand 1** and ``t`` is bent. YASTN's counterpart is
    ``ctl.swap_gate(axes=((0, 2), 3))``, the ``b b' x r'`` string it writes before fusing.
    """
    return closed([("tlBRs,tlbrs->bBrR", a.bra, a.ket, "t")])


def cor_bl(a: DoublePepsTensor) -> SymmetricTensor:
    """YASTN ``cor_bl``. Close ``l``, ``b`` and ``phys``; keep ``(r_k, r_b, t_k, t_b)``.

    Parameters
    ----------
    a : DoublePepsTensor
        The site.

    Returns
    -------
    SymmetricTensor
        Rank 4, YASTN's ``[r r'] [t t']`` with each pair left unfused.

    Notes
    -----
    Wires ``l``, ``b``, ``s``; ``IN`` comes from bra on all three, so **bra is operand 1
    and nothing bends**. YASTN's ``cor_bl`` is likewise its one corner with no
    ``swap_gate`` at all -- the two conventions agree on which corner is the free one.
    """
    return closed([("TlbRs,tlbrs->rRtT", a.bra, a.ket, "")])


def cor_br(a: DoublePepsTensor) -> SymmetricTensor:
    """YASTN ``cor_br``. Close ``b``, ``r`` and ``phys``; keep ``(t_k, t_b, l_k, l_b)``.

    Parameters
    ----------
    a : DoublePepsTensor
        The site.

    Returns
    -------
    SymmetricTensor
        Rank 4, YASTN's ``[t t'] [l l']`` with each pair left unfused.

    Notes
    -----
    Wires ``b``, ``r``, ``s``; ``IN`` comes from bra, ket, bra. **Bra is operand 1** and
    ``r`` is bent -- against YASTN's ``cbr.swap_gate(axes=((1, 3), 2))``, the ``l l' x t'``
    string.
    """
    return closed([("TLbrs,tlbrs->tTlL", a.bra, a.ket, "r")])


def cor_tr(a: DoublePepsTensor) -> SymmetricTensor:
    """YASTN ``cor_tr``. Close ``t``, ``r`` and ``phys``; keep ``(l_k, l_b, b_k, b_b)``.

    Parameters
    ----------
    a : DoublePepsTensor
        The site.

    Returns
    -------
    SymmetricTensor
        Rank 4, YASTN's ``[l l'] [b b']`` with each pair left unfused.

    Notes
    -----
    Wires ``t``, ``r``, ``s``; ``IN`` comes from ket, ket, bra. This is the one corner
    where **the ket is operand 1**, and the physical wire is the bent one. YASTN reaches
    the same object from the other end: it swap-gates *both* layers up front
    (``A.swap_gate(axes=(0, 1, 2, 3))``, ``t x l`` and ``b x r``) and then needs no
    post-contraction gate.
    """
    return closed([("tlbrs,tLBrs->lLbB", a.ket, a.bra, "s")])


# -- edges: one site closed on one virtual leg and the physical one -------------------


def edge_t(a: DoublePepsTensor) -> SymmetricTensor:
    """YASTN ``edge_t``. Close ``t`` and ``phys``; keep ``l``, ``b``, ``r`` pairs.

    Parameters
    ----------
    a : DoublePepsTensor
        The site.

    Returns
    -------
    SymmetricTensor
        Rank 6, YASTN's ``[l l'] [b b'] [r r']`` with each pair left unfused.

    Notes
    -----
    Wires ``t`` (ket) and ``s`` (bra): one each, so the count does not decide and the
    tie is broken the way three of the four corners fall -- **bra is operand 1**, ``t``
    bends. One bend either way; taking the same operand order as ``cor_tl``/``cor_bl``/
    ``cor_br`` is what keeps the family readable.
    """
    return closed([("tLBRs,tlbrs->lLbBrR", a.bra, a.ket, "t")])


def edge_l(a: DoublePepsTensor) -> SymmetricTensor:
    """YASTN ``edge_l``. Close ``l`` and ``phys``; keep ``b``, ``r``, ``t`` pairs.

    Parameters
    ----------
    a : DoublePepsTensor
        The site.

    Returns
    -------
    SymmetricTensor
        Rank 6, YASTN's ``[b b'] [r r'] [t t']`` with each pair left unfused.

    Notes
    -----
    Wires ``l`` and ``s``, both supplied ``IN`` by the bra: **bra is operand 1, nothing
    bends**.
    """
    return closed([("TlBRs,tlbrs->bBrRtT", a.bra, a.ket, "")])


def edge_b(a: DoublePepsTensor) -> SymmetricTensor:
    """YASTN ``edge_b``. Close ``b`` and ``phys``; keep ``r``, ``t``, ``l`` pairs.

    Parameters
    ----------
    a : DoublePepsTensor
        The site.

    Returns
    -------
    SymmetricTensor
        Rank 6, YASTN's ``[r r'] [t t'] [l l']`` with each pair left unfused.

    Notes
    -----
    Wires ``b`` and ``s``, both supplied ``IN`` by the bra: **bra is operand 1, nothing
    bends**.
    """
    return closed([("TLbRs,tlbrs->rRtTlL", a.bra, a.ket, "")])


def edge_r(a: DoublePepsTensor) -> SymmetricTensor:
    """YASTN ``edge_r``. Close ``r`` and ``phys``; keep ``t``, ``l``, ``b`` pairs.

    Parameters
    ----------
    a : DoublePepsTensor
        The site.

    Returns
    -------
    SymmetricTensor
        Rank 6, YASTN's ``[t t'] [l l'] [b b']`` with each pair left unfused.

    Notes
    -----
    Wires ``r`` (ket) and ``s`` (bra); tie broken toward the bra as in ``edge_t``, so
    **bra is operand 1** and ``r`` bends.
    """
    return closed([("TLBrs,tlbrs->tTlLbB", a.bra, a.ket, "r")])


# -- append_vec: a rank-6 environment vector absorbing one double-layer site ----------
#
# The vector is ``(x, <first pair>, <second pair>, y)`` with the pairs named by the
# corner, always ket leg then bra leg, and ``x``/``y`` the two spectator legs the
# environment carries through. Each pair's sides are whatever contracts with the site:
# the ket leg of the vector meets the ket's leg, the bra leg meets the bra's.
#
# All four are two-step chains in YASTN's order -- **vector into the bra first, then the
# ket** -- which is what keeps the peak at one open physical leg instead of two, and
# nothing of the fused double layer is ever formed.


def append_vec_tl(a: DoublePepsTensor, vec: SymmetricTensor) -> SymmetricTensor:
    """YASTN ``append_vec_tl``. Absorb ``a`` into a top-left vector.

    Parameters
    ----------
    a : DoublePepsTensor
        The site to absorb.
    vec : SymmetricTensor
        Rank 6, ``(x, l_ket, l_bra, t_ket, t_bra, y)``.

    Returns
    -------
    SymmetricTensor
        Rank 6, ``(x, b_ket, b_bra, y, r_ket, r_bra)`` -- YASTN's ``x [b b'] y [r r']``.

    Notes
    -----
    Step 1 contracts the vector's bra pair into the bra: wire ``t_bra`` has the vector
    supplying ``IN`` and wire ``l_bra`` has the bra supplying it, so **the vector is
    operand 1** and ``l_bra`` bends. Step 2 contracts ``l_ket``, ``t_ket`` and the
    physical wire into the ket: the running result supplies ``IN`` on ``l_ket`` (from
    the vector) and on ``phys`` (from the bra), the ket supplies it on ``t_ket``, so
    **the running result is operand 1** and ``t_ket`` bends. YASTN's two gates here are
    ``t' x l l'`` before the bra and ``b b' x r'`` after the ket.
    """
    return closed(
        [
            ("xcCaAy,ACBRS->xcaySBR", vec, a.bra, "C"),
            ("xcaySBR,acbrS->xbByrR", None, a.ket, "a"),
        ]
    )


def append_vec_br(a: DoublePepsTensor, vec: SymmetricTensor) -> SymmetricTensor:
    """YASTN ``append_vec_br``. Absorb ``a`` into a bottom-right vector.

    Parameters
    ----------
    a : DoublePepsTensor
        The site to absorb.
    vec : SymmetricTensor
        Rank 6, ``(x, r_ket, r_bra, b_ket, b_bra, y)``.

    Returns
    -------
    SymmetricTensor
        Rank 6, ``(x, t_ket, t_bra, y, l_ket, l_bra)`` -- YASTN's ``x [t t'] y [l l']``.

    Notes
    -----
    The mirror of [append_vec_tl][tenet.network.append_vec_tl]. Step 1: the vector
    supplies ``IN`` on ``r_bra``, the bra on ``b_bra``, so **the vector is operand 1**
    and ``b_bra`` bends. Step 2: the running result supplies ``IN`` on ``b_ket`` and
    ``phys``, the ket on ``r_ket``, so **the running result is operand 1** and ``r_ket``
    bends. YASTN's gates: ``b b' x r'`` before the bra, ``l l' x t'`` after the ket.
    """
    return closed(
        [
            ("xdDeEy,TLEDS->xdeySTL", vec, a.bra, "E"),
            ("xdeySTL,tledS->xtTylL", None, a.ket, "d"),
        ]
    )


def append_vec_tr(a: DoublePepsTensor, vec: SymmetricTensor) -> SymmetricTensor:
    """YASTN ``append_vec_tr``. Absorb ``a`` into a top-right vector.

    Parameters
    ----------
    a : DoublePepsTensor
        The site to absorb.
    vec : SymmetricTensor
        Rank 6, ``(x, t_ket, t_bra, r_ket, r_bra, y)``.

    Returns
    -------
    SymmetricTensor
        Rank 6, ``(x, l_ket, l_bra, y, b_ket, b_bra)`` -- YASTN's ``x [l l'] y [b b']``.

    Notes
    -----
    Step 1 needs no bend: the vector supplies ``IN`` on both ``t_bra`` and ``r_bra``, so
    **the vector is operand 1** and both wires already run the right way. Step 2 is the
    one place a *site* leads: the ket supplies ``IN`` on ``t_ket`` and ``r_ket`` against
    the running result's one wire (``phys``), so **the ket is operand 1** and the
    physical wire bends -- the same asymmetry ``cor_tr`` shows, for the same reason.
    """
    return closed(
        [
            ("xaAdDy,ALBDS->xadySLB", vec, a.bra, ""),
            ("albdS,xadySLB->xlLybB", a.ket, None, "S"),
        ]
    )


def append_vec_bl(a: DoublePepsTensor, vec: SymmetricTensor) -> SymmetricTensor:
    """YASTN ``append_vec_bl``. Absorb ``a`` into a bottom-left vector.

    Parameters
    ----------
    a : DoublePepsTensor
        The site to absorb.
    vec : SymmetricTensor
        Rank 6, ``(x, b_ket, b_bra, l_ket, l_bra, y)``.

    Returns
    -------
    SymmetricTensor
        Rank 6, ``(x, r_ket, r_bra, y, t_ket, t_bra)`` -- YASTN's ``x [r r'] y [t t']``.

    Notes
    -----
    The bend-free corner, matching ``cor_bl``. Step 1: the bra supplies ``IN`` on both
    ``b_bra`` and ``l_bra``, so **the bra is operand 1**. Step 2: the running result
    supplies ``IN`` on ``b_ket``, ``l_ket`` and ``phys`` alike, so **it is operand 1**.
    Three wires, three agreements, nothing bends.
    """
    return closed(
        [
            ("TCERS,xeEcCy->xecySTR", a.bra, vec, ""),
            ("xecySTR,tcerS->xrRytT", None, a.ket, ""),
        ]
    )
