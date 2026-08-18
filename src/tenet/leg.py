"""Per-axis categorical metadata: a graded space, a side, a dual flag, a name.

``side`` (IN/OUT — domain vs codomain) and ``dual`` (``V`` vs ``V*``) are
**independent** (invariant 2). There is deliberately no way to change ``side``:
moving a leg between domain and codomain is a categorical bend requiring
(co)evaluation maps and possibly Frobenius-Schur signs — that is
``T.repartition(...)`` on the tensor (Milestone 3), never a leg-level setter.

The leg holds no fusion-tree data of any kind (invariant 4).
"""

import enum
from collections.abc import Hashable
from dataclasses import dataclass, replace

from tenet.space import GradedSpace
from tenet.symmetry.base import Sector, _DualFusionRules, _HashMemo

__all__ = ["IN", "OUT", "Leg", "Side"]


class Side(enum.Enum):
    """Which half of ``Hom(domain, codomain)`` a leg belongs to."""

    OUT = "out"
    IN = "in"


OUT = Side.OUT
IN = Side.IN


@dataclass(frozen=True, slots=True)
class Leg(_HashMemo):
    """One tensor axis: ``space``, ``side``, ``dual`` and an optional ``name``.

    Parameters
    ----------
    space : GradedSpace
        The graded representation space this axis carries.
    side : Side
        ``OUT`` (codomain) or ``IN`` (domain).
    dual : bool, optional
        Whether the axis carries ``V*`` rather than ``V``. Default ``False``.
    name : Hashable or None, optional
        User bookkeeping label. Default ``None``.

    Examples
    --------
    >>> from tenet import OUT, GradedSpace, Leg
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> leg = Leg(V, OUT, name="p")
    >>> leg.degeneracy(U1Sector(0))
    2
    >>> leg.dualized().fused_sector(U1Sector(1))
    U1Sector(charge=-1)

    Notes
    -----
    Frozen and hashable; ``name`` participates in equality like any other field.
    """

    space: GradedSpace
    side: Side
    dual: bool = False
    name: Hashable | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.space, self.side, self.dual, self.name)))

    def __hash__(self) -> int:
        return self._hash

    @property
    def provider(self) -> _DualFusionRules:
        """The space's symmetry provider.

        Returns
        -------
        provider
            ``self.space.provider``.
        """
        return self.space.provider

    @property
    def sectors(self) -> tuple[Sector, ...]:
        """Space sectors in the space's canonical order.

        Returns
        -------
        tuple of Sector
            ``tuple(self.space)``.
        """
        return tuple(self.space)

    def degeneracy(self, a: Sector) -> int:
        """``m_a`` for a **space** label ``a`` (not a fused label).

        Parameters
        ----------
        a : Sector
            A sector labelling the space, in the space's own convention.

        Returns
        -------
        int
            The degeneracy ``m_a``, or ``0`` if ``a`` is absent.

        Notes
        -----
        For a dual U(1) leg, ``fused_sector`` negates the charge, so feeding a
        fused label here would silently read the wrong degeneracy — use
        [space_sector][tenet.Leg.space_sector] first.
        """
        return self.space.degeneracy(a)

    def fused_sector(self, a: Sector) -> Sector:
        """The sector this leg contributes to a fusion tree: ``dual(a)`` if dual.

        Parameters
        ----------
        a : Sector
            A space sector of this leg.

        Returns
        -------
        Sector
            ``provider.dual(a)`` if the leg is dual, else ``a`` unchanged.
        """
        return self.provider.dual(a) if self.dual else a

    def space_sector(self, u: Sector) -> Sector:
        """Inverse of [fused_sector][tenet.Leg.fused_sector].

        ``dual`` is an involution, so lossless.

        Parameters
        ----------
        u : Sector
            A fused (tree-side) sector label.

        Returns
        -------
        Sector
            The space label: ``provider.dual(u)`` if the leg is dual, else ``u``.
        """
        return self.provider.dual(u) if self.dual else u

    def dualized(self) -> "Leg":
        """New leg with ``dual`` flipped — a relabelling of ``V ↔ V*`` only.

        Returns
        -------
        Leg
            A copy of this leg with ``dual`` negated; ``space``, ``side`` and
            ``name`` are unchanged.

        Notes
        -----
        ``tenet.flip_dual`` is the sanctioned numerical route for changing a leg's
        ``dual`` flag: it also relabels the space through ``provider.dual`` and
        pays the Z-isomorphism's scalar per fusion tree. ``repartition`` is the
        route that changes ``side`` together with ``dual``, paying the bending
        coefficient (#38). ``dualized()`` itself is metadata-only, and it is
        **not** the metadata half of ``flip_dual``: toggling the flag alone changes
        which sector the leg contributes to a fusion tree (``fused_sector`` goes
        from ``a`` to ``dual(a)``), so the block set genuinely changes and no
        scalar can express the difference — never use ``dualized()`` to build a
        flip.
        """
        return replace(self, dual=not self.dual)

    def renamed(self, name: Hashable | None) -> "Leg":
        """New leg with ``name`` replaced; everything else unchanged.

        Parameters
        ----------
        name : Hashable or None
            The new name (``None`` clears it).

        Returns
        -------
        Leg
            A copy of this leg carrying ``name``.
        """
        return replace(self, name=name)
