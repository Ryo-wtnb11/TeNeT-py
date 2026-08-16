"""Symmetry foundation: sector labels, fusion provider protocol, Trivial provider.

Sectors are pure labels; all symmetry behaviour (dual, fusion, dimensions) lives
on the provider. Providers are frozen, hashable, array-free values so they can
sit inside ``TensorStructure`` (invariant 8).
"""

from dataclasses import dataclass
from functools import cache
from math import sqrt
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BendingCoefficients",
    "CapabilityError",
    "ClebschGordan",
    "DualBasis",
    "FusionProvider",
    "PermutationCoefficients",
    "QuantumDimension",
    "RecouplingData",
    "Sector",
    "StructureChangingError",
    "SymmetryCast",
    "Trivial",
    "TrivialProvider",
    "TrivialSector",
    "bend_braided",
    "bend_unique",
    "permute_braided_tree",
    "permute_unique_tree",
    "requires",
]

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree
    from tenet.structure import FusionBlockKey


class _HashMemo:
    """Mixin supplying a ``_hash`` slot for frozen value types (M9, #59).

    Python's dataclass ``__hash__`` re-hashes every field on every call, so a
    nested frozen value (space → leg → structure) costs a full recursive walk per
    ``functools.cache`` lookup — 42% of a steady-state ``tensordot`` at small
    degeneracy. Subclasses compute ``hash(fields)`` **once** in ``__post_init__``
    (identical value to the generated ``__hash__``, so hashes stay unchanged and
    deterministic) and define ``__hash__`` returning it; equality is untouched.

    It is a base class rather than a ``__slots__`` line in each body because
    ``@dataclass(slots=True)`` builds ``__slots__`` itself and rejects one in the
    class body — inherited slots it accepts.
    """

    __slots__ = ("_hash",)


@dataclass(frozen=True, slots=True, order=True)
class Sector:
    """Marker base for sector labels.

    Subclasses are frozen, slotted, ordered dataclasses, so hashing and canonical
    sorting come for free. Comparison is only defined within one sector type;
    comparing different sector types raises ``TypeError``.
    """


class FusionProvider(Protocol):
    """Minimal fusion-category data every symmetry must supply."""

    @property
    def name(self) -> str: ...

    @property
    def unit(self) -> Sector: ...

    def dual(self, a: Sector) -> Sector: ...

    def fusion(self, a: Sector, b: Sector) -> tuple[Sector, ...]: ...

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        """Multiplicity ``N^c_ab``. Multiplicity-free providers return 0 or 1."""
        ...


@runtime_checkable
class QuantumDimension(Protocol):
    """Providers that define a quantum dimension."""

    def qdim(self, a: Sector) -> float: ...


@runtime_checkable
class ClebschGordan(Protocol):
    """Providers that can supply explicit Clebsch-Gordan tensors."""

    def irrep_dim(self, a: Sector) -> int: ...

    def cgc(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        """Shape ``(d_a, d_b, d_c, N^c_ab)``; last axis is the multiplicity label mu."""
        ...


@runtime_checkable
class PermutationCoefficients(Protocol):
    """Providers that can expand a fusion tree with permuted uncoupled lines."""

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """``((tree', coeff), ...)`` with the permuted tree equal to ``Σ coeff · tree'``.

        ``perm[j]`` is the *old* uncoupled position that becomes position ``j``
        (the same convention as ``transpose``'s ``axes``). Expansion is over
        canonical left-associated trees. Scalar coefficients are a
        multiplicity-free assumption: a provider with ``n_symbol > 1`` must raise
        rather than truncate a matrix-valued coefficient (Milestone 4).
        """
        ...


@runtime_checkable
class BendingCoefficients(Protocol):
    """Providers that can move one line between the two trees of a block key."""

    def bend_right(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        """Move the LAST uncoupled line of ``output_tree`` onto the END of
        ``input_tree``, dualized. ``dual`` is the moved leg's current flag (it
        selects the Frobenius-Schur factor). Returns ``((key', coeff), ...)``.
        """
        ...

    def bend_left(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        """The inverse direction: last line of ``input_tree`` onto ``output_tree``."""
        ...


@runtime_checkable
class DualBasis(Protocol):
    """Providers whose ``V_a -> V_a^*`` isomorphism is available in the dense basis."""

    def z_matrix(self, a: Sector) -> np.ndarray:
        """``Z_a``, shape ``(d_a, d_dual(a))``, in the provider's own dense basis."""
        ...


@runtime_checkable
class SymmetryCast(Protocol):
    """Providers that can be restricted to a smaller symmetry in the dense basis."""

    def branch(self, target: FusionProvider, a: Sector) -> tuple[Sector, ...]:
        """The ``target`` sector of each of ``a``'s ``d_a`` dense basis vectors.

        Length is exactly ``irrep_dim(a)``, in the provider's **own** dense
        (magnetic) order — the same order ``cgc`` and ``z_matrix`` use, so this
        composes with ``to_dense``'s ``alpha * d_a + m`` layout without a second
        convention. Every returned sector must satisfy
        ``target.irrep_dim(...) == 1``: one target label per basis vector is only
        well-defined when the target's irreps are one-dimensional, i.e. when the
        target is abelian. ``CapabilityError`` for an unsupported target.
        """
        ...


@runtime_checkable
class RecouplingData(Protocol):
    """Providers that supply associator, braiding and duality coefficients.

    All four are **scalar**-valued, which is a multiplicity-free assumption: a
    provider with ``n_symbol > 1`` must raise rather than truncate a matrix-valued
    symbol. Generic tree-braiding and tree-bending helpers dispatch on this
    protocol and know nothing about any particular symmetry.
    """

    def f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> complex:
        """``[F^{abc}_d]_{e,f}``; ``e`` is the inner line of ``((ab)c)``, ``f`` of ``(a(bc))``."""
        ...

    def r_symbol(self, a: Sector, b: Sector, c: Sector) -> complex:
        """``R^{ab}_c``, the coefficient of braiding ``a`` past ``b`` inside ``c``."""
        ...

    def b_symbol(self, a: Sector, b: Sector, c: Sector) -> complex:
        """``B^{ab}_c``, the duality coefficient bending ``b`` out of ``a x b -> c``."""
        ...

    def frobenius_schur(self, a: Sector) -> complex:
        """``chi_a``, the Frobenius-Schur phase of the ``V_a -> V_a^*`` isomorphism."""
        ...


class CapabilityError(TypeError):
    """Raised when a provider lacks a capability an operation requires."""


class StructureChangingError(TypeError):
    """Raised when an operation whose *output structure depends on block values*
    is asked to run inside a traced (jit/grad/vmap) region.

    docs/design.md "Structure-changing differentiation", invariants 9 and 10: the library
    never hides the distinction between a shape-static operation and one that
    decides its own output structure from the numbers. Lives here next to
    :class:`CapabilityError`, subclasses ``TypeError`` for the same reason it
    does, and is exported from ``tenet``.
    """


def requires(provider: FusionProvider, capability: type) -> None:
    """Raise :class:`CapabilityError` unless ``provider`` implements ``capability``."""
    if not isinstance(provider, capability):
        raise CapabilityError(
            f"{type(provider).__name__} does not provide capability {capability.__name__}"
        )


def permute_unique_tree(
    provider: FusionProvider, tree: "FusionTree", perm: tuple[int, ...]
) -> tuple[tuple["FusionTree", complex], ...]:
    """``permute_tree`` for providers whose fusion is unique and whose F/R are 1.

    Shared by Trivial and U(1): permuting the uncoupled labels leaves exactly one
    left-associated tree with the same coupled sector, so the whole expansion is
    that tree with coefficient 1. The spine is *recomputed* rather than permuted,
    which is what makes this correct rather than a relabelling.

    Not a capability check: a provider must opt in by defining ``permute_tree``
    (fermionic parity also has unique multiplicity-free fusion and still carries a
    sign, so uniqueness alone must never be taken as permission — see #21).
    """
    from tenet.fusion_tree import fusion_trees

    uncoupled = tuple(tree.uncoupled[i] for i in perm)
    trees = fusion_trees(provider, uncoupled, tree.coupled)
    if len(trees) != 1:
        raise CapabilityError(
            f"{provider.name}: permuting {tree.uncoupled} by {perm} gives {len(trees)} trees "
            f"coupling to {tree.coupled!r}; permute_unique_tree needs exactly one"
        )
    return ((trees[0], 1.0),)


@cache
def permute_braided_tree(
    provider: "RecouplingData", tree: "FusionTree", perm: tuple[int, ...]
) -> tuple[tuple["FusionTree", complex], ...]:
    """``permute_tree`` for any provider supplying F- and R-symbols.

    Bubble-decomposes ``perm`` into adjacent transpositions (Artin generators) and
    applies :func:`_artin_braid` to each, accumulating a ``{tree: coeff}`` expansion.
    ``perm[j]`` is the OLD uncoupled position that becomes position ``j``.

    Symmetric-category only: the caller gets no over/under choice, because
    ``R == R**-1`` for the providers this serves (invariant 12). A provider whose
    braiding is genuinely chiral must not reuse this helper — it needs ``levels``
    and an explicit ``braid`` API.

    The coefficients are conjugated nowhere, which is correct exactly while the
    provider's gauge is real: TensorKit braids the domain-side (fusion) tree with
    the conjugate coefficient, and for a complex-gauge provider the caller must
    conjugate on the domain side. SU(2) in the racah/TensorKitSectors gauge has
    real F and R, so the two agree and ``permutation_plan`` may treat the two
    trees of a block key symmetrically.

    No tolerance-based pruning: a structurally forbidden term is already exactly
    ``0.0`` through ``f_symbol``, and the surviving float residues from the ``sum
    over d`` cost plan size, never correctness.
    Simplification: the ceiling is term-list size for high-rank, high-spin trees; the
    upgrade path is exact sqrt-rational accumulation, not a tolerance.
    """
    terms: dict[FusionTree, complex] = {tree: 1.0}
    current = list(range(tree.rank))  # old position now sitting at each slot
    for j, target in enumerate(perm):
        for i in range(current.index(target) - 1, j - 1, -1):
            current[i], current[i + 1] = current[i + 1], current[i]
            nxt: dict[FusionTree, complex] = {}
            for t, c in terms.items():
                for braided, cb in _artin_braid(provider, t, i):
                    nxt[braided] = nxt.get(braided, 0) + c * cb
            terms = nxt
    return tuple(terms.items())


@cache
def _artin_braid(
    provider: "RecouplingData", tree: "FusionTree", i: int
) -> tuple[tuple["FusionTree", complex], ...]:
    """Swap uncoupled lines ``i`` and ``i+1`` of a left-associated tree.

    ``i == 0``: the two lines already meet at the first vertex, so a single
    R-move does it and the spine is untouched.

    ``i >= 1``: with ``a = e_{i-1}``, ``b = u_i``, ``c = u_{i+1}``, ``f = e_{i+1}``
    and the old inner line ``e = e_i``, F-move to ``(a (b c))``, R on the ``(b c)``
    vertex, F-move back::

        coeff(e -> e') = sum_d F^{abc}_f[e, d] · R^{bc}_d · conj(F^{acb}_f[e', d])

    which is a genuine expansion over the admissible ``e' in fusion(a, c)``.
    Every other spine entry, and every multiplicity label, is unchanged.
    """
    from tenet.fusion_tree import FusionTree

    u = tree.uncoupled
    spine = tree.lines()
    new_u = (*u[:i], u[i + 1], u[i], *u[i + 2 :])
    if i == 0:
        swapped = FusionTree(new_u, tree.inner, tree.multiplicities, tree.coupled)
        return ((swapped, provider.r_symbol(u[0], u[1], spine[1])),)

    a, b, c, e, f = spine[i - 1], u[i], u[i + 1], spine[i], spine[i + 1]
    out = []
    for new_e in provider.fusion(a, c):
        if not provider.n_symbol(new_e, b, f):
            continue
        coeff = sum(
            provider.f_symbol(a, b, c, f, e, d)
            * provider.r_symbol(b, c, d)
            * provider.f_symbol(a, c, b, f, new_e, d).conjugate()
            for d in provider.fusion(b, c)
        )
        inner = (*tree.inner[: i - 1], new_e, *tree.inner[i:])
        out.append((FusionTree(new_u, inner, tree.multiplicities, tree.coupled), coeff))
    return tuple(out)


def bend_unique(
    provider: FusionProvider, key: "FusionBlockKey", *, right: bool, dual: bool
) -> tuple[tuple["FusionBlockKey", complex], ...]:
    """``bend_right``/``bend_left`` for providers whose fusion is unique and B is 1.

    Shared by Trivial and U(1), exactly as :func:`permute_unique_tree` is. The
    source tree loses its last uncoupled line, the destination tree gains
    ``dual`` of it at the *end*, and the new coupled sector is the source tree's
    last inner line (the unit when the source had rank 1). Both spines are
    **recomputed** from the new uncoupled tuples, never relabelled.

    The coefficient is ``sqrt(dim(c)/dim(a)) · B(a,b,c)``, times the conjugate
    Frobenius-Schur phase of ``dual(b)`` when the moved line is already ``dual``.
    For every Abelian irrep all three factors are exactly ``1`` (all ``dim == 1``,
    ``B == N ∈ {0,1}``, ``frobenius_schur_phase == 1``), and Trivial reaches the
    same value through ``F ≡ 1``. So ``dual`` is accepted and provably ignored
    here; a provider whose FS phase is not 1 must not route through this helper.

    Not a capability check: a provider opts in by defining ``bend_right`` /
    ``bend_left``. Uniqueness of fusion alone is never permission (a fermionic
    parity provider has unique fusion and a non-trivial bend).
    """
    from tenet.fusion_tree import coupled_sectors
    from tenet.structure import FusionBlockKey

    src = key.output_tree if right else key.input_tree
    dst = key.input_tree if right else key.output_tree
    if src.rank == 0:
        raise ValueError(
            f"{provider.name}: cannot bend from an empty "
            f"{'output' if right else 'input'} tree of {key}"
        )

    moved = src.uncoupled[-1]
    remaining = src.uncoupled[:-1]
    coupled = coupled_sectors(provider, remaining)
    if len(coupled) != 1:
        raise CapabilityError(
            f"{provider.name}: dropping {moved!r} leaves {remaining} with "
            f"{len(coupled)} coupled sectors; bend_unique needs exactly one"
        )
    new_src = _unique_tree(provider, remaining, coupled[0])
    new_dst = _unique_tree(provider, (*dst.uncoupled, provider.dual(moved)), coupled[0])
    new_key = FusionBlockKey(new_src, new_dst) if right else FusionBlockKey(new_dst, new_src)
    return ((new_key, 1.0),)


def bend_braided(
    provider: "RecouplingData", key: "FusionBlockKey", *, right: bool, dual: bool
) -> tuple[tuple["FusionBlockKey", complex], ...]:
    """``bend_right``/``bend_left`` for any provider supplying B, FS and ``qdim``.

    The source tree's last vertex is ``a x b -> c`` (``a`` the unit when the
    source has rank 1). The source loses ``b`` and re-couples to ``a``; the
    destination gains ``dual(b)`` at its **end** and couples to ``a`` too. Both
    spines are **recomputed** from the new uncoupled tuples — the source spine is
    truncated and the destination's old coupled sector becomes its new last inner
    line — never relabelled.

    The coefficient is TensorKit's ``bendright``::

        coeff = sqrt(qdim(c) / qdim(a)) · B(a, b, c)          (× conj(chi_dual(b))
                                                               if the moved leg
                                                               was already dual)

    and ``bend_left`` is ``bendleft``: the *same* expression read off the input
    tree, then conjugated — which is what makes it the exact inverse of
    ``bend_right`` rather than a reciprocal guess (a unitary's inverse is its
    conjugate transpose, and multiplicity-free bending is one entry per key).
    Note the Frobenius-Schur factor is keyed on the moved leg's *current* flag,
    so it appears once across a round trip, never twice and never zero times.

    Single-term because the provider is multiplicity-free: one key in, one key
    out. ``n_symbol > 1`` needs matrix-valued coefficients and is refused.
    """
    from tenet.fusion_tree import FusionTree
    from tenet.structure import FusionBlockKey

    src = key.output_tree if right else key.input_tree
    dst = key.input_tree if right else key.output_tree
    if src.rank == 0:
        raise ValueError(
            f"{provider.name}: cannot bend from an empty "
            f"{'output' if right else 'input'} tree of {key}"
        )

    b, c = src.uncoupled[-1], src.coupled
    a = src.lines()[-2] if src.rank >= 2 else provider.unit
    if provider.n_symbol(a, b, c) > 1:
        raise CapabilityError(
            f"{provider.name}: N^{c!r}_{{{a!r},{b!r}}} > 1; bending a vertex with "
            "multiplicity needs matrix-valued B-symbols, which are not supported"
        )

    new_src = FusionTree(src.uncoupled[:-1], src.inner[:-1], src.multiplicities[:-1], a)
    new_dst = FusionTree(
        (*dst.uncoupled, provider.dual(b)),
        dst.lines()[1:],
        (*dst.multiplicities, 0) if dst.rank else (),
        a,
    )
    new_key = FusionBlockKey(new_src, new_dst) if right else FusionBlockKey(new_dst, new_src)

    coeff = sqrt(provider.qdim(c) / provider.qdim(a)) * provider.b_symbol(a, b, c)
    if dual:
        coeff *= provider.frobenius_schur(provider.dual(b)).conjugate()
    return ((new_key, coeff if right else coeff.conjugate()),)


def _unique_tree(
    provider: FusionProvider, uncoupled: tuple[Sector, ...], coupled: Sector
) -> "FusionTree":
    from tenet.fusion_tree import fusion_trees

    trees = fusion_trees(provider, uncoupled, coupled)
    if len(trees) != 1:
        raise CapabilityError(
            f"{provider.name}: {uncoupled} -> {coupled!r} has {len(trees)} trees; "
            "bend_unique needs exactly one"
        )
    return trees[0]


@dataclass(frozen=True, slots=True, order=True)
class TrivialSector(Sector):
    """The single sector of the trivial symmetry."""


@dataclass(frozen=True, slots=True)
class TrivialProvider:
    """Degenerate reference provider: one sector, everything trivial."""

    name: str = "Trivial"

    @property
    def unit(self) -> TrivialSector:
        return TrivialSector()

    def dual(self, a: Sector) -> Sector:
        return a

    def fusion(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        return (TrivialSector(),)

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1

    def qdim(self, a: Sector) -> float:
        return 1.0

    def irrep_dim(self, a: Sector) -> int:
        return 1

    def cgc(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        return np.ones((1, 1, 1, 1))

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """One term, coefficient 1: there is a single sector and ``F = R = 1``."""
        return permute_unique_tree(self, tree, perm)

    def bend_right(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        """One term, coefficient 1: ``B`` follows from ``F ≡ 1`` and ``dim ≡ 1``."""
        return bend_unique(self, key, right=True, dual=dual)

    def bend_left(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        return bend_unique(self, key, right=False, dual=dual)

    def z_matrix(self, a: Sector) -> np.ndarray:
        """``Z = [[1]]``, read-only: one-dimensional irrep, Frobenius-Schur phase 1."""
        return _Z


_Z = np.ones((1, 1))
_Z.flags.writeable = False

Trivial = TrivialProvider()
"""Module-level singleton, used as ``GradedSpace(provider=Trivial, ...)``."""
