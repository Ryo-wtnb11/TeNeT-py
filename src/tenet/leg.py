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
from tenet.symmetry.base import FusionProvider, Sector

__all__ = ["IN", "OUT", "Leg", "Side"]


class Side(enum.Enum):
    """Which half of ``Hom(domain, codomain)`` a leg belongs to."""

    OUT = "out"
    IN = "in"


OUT = Side.OUT
IN = Side.IN


@dataclass(frozen=True, slots=True)
class Leg:
    """One tensor axis: ``space``, ``side``, ``dual`` and an optional ``name``.

    Frozen and hashable; ``name`` participates in equality like any other field.
    """

    space: GradedSpace
    side: Side
    dual: bool = False
    name: Hashable | None = None

    @property
    def provider(self) -> FusionProvider:
        return self.space.provider

    @property
    def sectors(self) -> tuple[Sector, ...]:
        """Space sectors in the space's canonical order."""
        return tuple(self.space)

    def degeneracy(self, a: Sector) -> int:
        """``m_a`` for a **space** label ``a`` (not a fused label).

        For a dual U(1) leg, ``fused_sector`` negates the charge, so feeding a
        fused label here would silently read the wrong degeneracy — use
        :meth:`space_sector` first.
        """
        return self.space.degeneracy(a)

    def fused_sector(self, a: Sector) -> Sector:
        """The sector this leg contributes to a fusion tree: ``dual(a)`` if dual."""
        return self.provider.dual(a) if self.dual else a

    def space_sector(self, u: Sector) -> Sector:
        """Inverse of :meth:`fused_sector`; ``dual`` is an involution, so lossless."""
        return self.provider.dual(u) if self.dual else u

    def dualized(self) -> "Leg":
        """New leg with ``dual`` flipped — a relabelling of ``V ↔ V*`` only.

        It does **not** move the leg between domain and codomain, and in general
        it does change the tensor's numerical content (Z-isomorphism / FS signs,
        Milestone 4), which is why no tensor-level shortcut for it exists yet.
        """
        return replace(self, dual=not self.dual)

    def renamed(self, name: Hashable | None) -> "Leg":
        return replace(self, name=name)
