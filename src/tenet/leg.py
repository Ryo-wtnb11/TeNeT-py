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
from tenet.symmetry.base import FusionProvider, Sector, _HashMemo

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

        ``tenet.flip`` is the sanctioned numerical route for changing a leg's
        ``dual`` flag: it also relabels the space through ``provider.dual`` and
        pays the Z-isomorphism's scalar per fusion tree. ``repartition`` is the
        route that changes ``side`` together with ``dual``, paying the bending
        coefficient (#38). ``dualized()`` itself is metadata-only, and it is
        **not** the metadata half of ``flip``: toggling the flag alone changes
        which sector the leg contributes to a fusion tree (``fused_sector`` goes
        from ``a`` to ``dual(a)``), so the block set genuinely changes and no
        scalar can express the difference — never use ``dualized()`` to build a
        flip.
        """
        return replace(self, dual=not self.dual)

    def renamed(self, name: Hashable | None) -> "Leg":
        return replace(self, name=name)
