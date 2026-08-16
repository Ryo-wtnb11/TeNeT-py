"""fZ2: fermion parity — Z2 fusion with *fermionic* braiding (super-vector spaces).

Everything about fZ2 except the braiding is degenerate: two sectors, XOR fusion,
self-dual labels, ``qdim == irrep_dim == 1``, all-ones CGC. The entire content of
the provider is therefore one function, :func:`koszul_sign`, and the reason it is
one function is that fZ2's F-symbols are identically ``1``:
``Fsymbol(a,b,c,d,e,f) = N(a,b,e)·N(e,c,d)·N(b,c,f)·N(a,f,d)``, never negative.
The associator being trivial is what lets a general within-side permutation of a
left-associated tree be decomposed into adjacent swaps with no F-moves, so the
coefficient is a product of R-symbols, ``Rsymbol(a,b,c) = -N`` iff both ``a`` and
``b`` are odd. Composing gives ``(-1)^(#odd-odd inversions)`` — the Koszul sign,
well defined because it is a 1-cocycle on the symmetric group.

``BraidingStyle(::Type{FermionParity}) = Fermionic()``: symmetric-with-signs, so
``R = R⁻¹`` and ``T.transpose(...)`` *does* uniquely determine the braid. That is
the condition invariant 12 demands before an ndarray spelling is allowed; no
``braid(i, j, over=...)`` is needed and none is added.

``twist(odd) == -1`` (``twist(a) = a.isodd ? -1 : +1``, same source as everything
else here). TensorKit keeps a *regular* trace with positive quantum dimensions
and absorbs the parity endomorphism into the right-evaluation map, which is
exactly why the Frobenius-Schur phase stays ``+1`` while the twist carries the
sign. Nothing in TeNeT-py consumes the twist yet — ``norm`` is qdim-weighted and
every qdim is ``1``, so it is unaffected.
Simplification: ``trace`` and ``adjoint`` (#31), and any closed fermion loop, are the
operations that will need it; add a ``Twist`` capability then, not before.

Source for every constant asserted above, TensorKitSectors ``src/fermions.jl`` at
commit ``bd1bf8296876103870d6158c0918987256396880``:
https://github.com/QuantumKitHub/TensorKitSectors.jl/blob/bd1bf8296876103870d6158c0918987256396880/src/fermions.jl
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tenet.symmetry.base import Sector, bend_unique, permute_unique_tree

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree
    from tenet.structure import FusionBlockKey

__all__ = ["FZ2_GAUGE", "FZ2Provider", "FZ2Sector", "fZ2", "koszul_sign"]

FZ2_GAUGE = "f=1;r=fermionic;fs=+1;twist=(-1)^p;perm=koszul"
"""Gauge fingerprint (TensorKitSectors ``FermionParity`` conventions); belongs in
plan-cache keys."""


@dataclass(frozen=True, slots=True, order=True)
class FZ2Sector(Sector):
    """A fermion-parity sector: ``parity in {0, 1}``. 0 is even (the unit), 1 is odd."""

    parity: int

    def __post_init__(self) -> None:
        # bool is an int subclass; reject it so `FZ2Sector(True) != FZ2Sector(1)` can
        # never become a live question (SU2Sector does the same for two_j).
        if not isinstance(self.parity, int) or isinstance(self.parity, bool):
            raise TypeError(f"parity must be an int, got {self.parity!r}")
        if self.parity not in (0, 1):
            raise ValueError(f"parity must be 0 or 1, got {self.parity}")


def koszul_sign(uncoupled: tuple[Sector, ...], perm: tuple[int, ...]) -> float:
    """``(-1)^(number of odd-odd inversions of perm)``.

    ``perm[j]`` is the OLD uncoupled position that becomes position ``j``
    (``permute_tree``'s convention). Restrict ``perm`` to the positions carrying
    odd sectors and count that subsequence's inversions.

    Direction-free: a permutation and its inverse have the same inversion count
    and restricting to odd positions commutes with inversion, so
    ``koszul_sign(u, perm) == koszul_sign(permuted(u, perm), inverse(perm))``.
    """
    odd = [i for i in perm if uncoupled[i].parity]  # old positions of odd lines, in NEW order
    inversions = sum(1 for j in range(len(odd)) for k in range(j + 1, len(odd)) if odd[j] > odd[k])
    # Simplification: O(n^2). Merge-sort inversion count if tree ranks ever leave single digits.
    return -1.0 if inversions % 2 else 1.0


@dataclass(frozen=True, slots=True)
class FZ2Provider:
    """Z2 fusion with *fermionic* braiding: SVect. Abelian, multiplicity-free, d = 1."""

    name: str = "fZ2"

    @property
    def unit(self) -> FZ2Sector:
        return FZ2Sector(0)

    def dual(self, a: FZ2Sector) -> FZ2Sector:
        """Both sectors are self-dual (``dual(f) = f``)."""
        return a

    def fusion(self, a: FZ2Sector, b: FZ2Sector) -> tuple[FZ2Sector, ...]:
        return (FZ2Sector(a.parity ^ b.parity),)

    def n_symbol(self, a: FZ2Sector, b: FZ2Sector, c: FZ2Sector) -> int:
        return int(c.parity == a.parity ^ b.parity)

    def qdim(self, a: FZ2Sector) -> float:
        return 1.0

    def irrep_dim(self, a: FZ2Sector) -> int:
        return 1

    def cgc(self, a: FZ2Sector, b: FZ2Sector, c: FZ2Sector) -> np.ndarray:
        """Shape ``(1, 1, 1, 1)``, read-only. Raises on a fusion-forbidden triple."""
        if not self.n_symbol(a, b, c):
            raise ValueError(f"fZ2 fusion forbids {a} x {b} -> {c}")
        return _CGC

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """Exactly one term: the tree is unique per ``(uncoupled, coupled)`` and the
        coefficient is the Koszul sign of ``perm`` restricted to the odd lines.

        The coefficient reads only the uncoupled parities — not the coupled sector
        and not the inner lines — because ``F ≡ 1`` makes the tree shape drop out.
        That is *not* true for SU(2), whose exchange sign ``(-1)^(j_a+j_b-j_c)``
        reads the fused channel; the spine may be ignored here and nowhere else.
        """
        ((permuted, _),) = permute_unique_tree(self, tree, perm)
        return ((permuted, koszul_sign(tree.uncoupled, perm)),)

    def bend_right(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        """One term, coefficient 1: ``dim = 1``, ``B = N``, ``FS = +1``.

        TensorKitSectors defines no ``frobenius_schur_phase`` for ``FermionParity``;
        it falls through to ``frobenius_schur_phase_from_Fsymbol(a) =
        sign(Fsymbol(a, dual(a), a, a, leftunit(a), rightunit(a)))``, and since
        ``dual(f) = f`` and every allowed F-symbol is ``+1``, ``FS(odd) == +1``.
        ``Bsymbol`` likewise falls through to ``Bsymbol_from_Fsymbol``, giving ``+1``.
        https://github.com/QuantumKitHub/TensorKitSectors.jl/blob/bd1bf8296876103870d6158c0918987256396880/src/sectors.jl#L535-L543
        The ``-1`` of fermion parity lives in ``twist``, not in FS — see the module
        docstring.
        """
        return bend_unique(self, key, right=True, dual=dual)

    def bend_left(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        return bend_unique(self, key, right=False, dual=dual)

    def z_matrix(self, a: FZ2Sector) -> np.ndarray:
        """``Z = [[1]]``, read-only: ``V_a`` is one-dimensional and ``FS == +1``."""
        return _Z


_CGC = np.ones((1, 1, 1, 1))
_CGC.flags.writeable = False

_Z = np.ones((1, 1))
_Z.flags.writeable = False

fZ2 = FZ2Provider()
"""Module-level singleton, used as ``GradedSpace(provider=fZ2, ...)``."""
