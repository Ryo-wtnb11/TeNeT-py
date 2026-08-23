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
    "AssociatorData",
    "BMatrixData",
    "BendingCoefficients",
    "BraidingData",
    "BranchingRules",
    "CapabilityError",
    "ClebschGordanData",
    "DaggerData",
    "DualBasis",
    "DualityData",
    "FMatrixData",
    "FSIndicatorData",
    "FusionRules",
    "PermutationCoefficients",
    "PivotalData",
    "QuantumDimensionData",
    "RMatrixData",
    "Sector",
    "StructureChangingError",
    "Trivial",
    "TrivialProvider",
    "TrivialSector",
    "TwistData",
    "bend_braided",
    "bend_unique",
    "permute_braided_tree",
    "permute_unique_tree",
    "requires",
    "supports",
]

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree
    from tenet.structure import FusionBlockKey


class _HashMemo:
    """Mixin supplying a ``_hash`` slot for frozen value types.

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

    # Annotation only: subclasses fill the slot in ``__post_init__`` via
    # ``object.__setattr__``, which no checker can see. Not a dataclass field —
    # ``@dataclass`` collects annotations from the decorated class and dataclass
    # bases only, and ``_HashMemo`` is neither.
    _hash: int


@dataclass(frozen=True, slots=True, order=True)
class Sector:
    """Marker base for sector labels.

    Subclasses are frozen, slotted, ordered dataclasses, so hashing and canonical
    sorting come for free. Comparison is only defined within one sector type;
    comparing different sector types raises ``TypeError``.
    """


@runtime_checkable
class FusionRules(Protocol):
    """The fusion ring: named sector labels, the unit, channels and multiplicities.

    Deliberately *not* called a fusion category: fusion rules ``N^c_ab`` do not
    determine one — a fusion category is the rules plus an associator
    ([AssociatorData][tenet.symmetry.AssociatorData]) plus rigidity
    ([DualityData][tenet.symmetry.DualityData]), and the same
    rules with different associators are genuinely different categories
    (``Vec_G`` versus ``Vec_G^omega``). ``dual`` is duality data and lives on
    [DualityData][tenet.symmetry.DualityData]; the label-level ``dual`` map every
    provider carries is annotated through ``_DualFusionRules`` where it is
    actually read.

    Notes
    -----
    "Fusion category" is the name of a *combination* — ``FusionRules +
    AssociatorData + DualityData`` with ``tenet.symmetry.coherence.validate_pentagon``
    and ``validate_snake`` passing — and it is a name in the docs, never a class
    here.
    """

    @property
    def name(self) -> str:
        """The provider's label, e.g. ``"U1"`` — a display string, never dispatched on."""
        ...

    @property
    def unit(self) -> Sector:
        """The unit (vacuum) sector: the identity of fusion, ``unit x a -> a``."""
        ...

    def fusion(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        """The fusion channels of ``a x b``, in the provider's canonical order.

        Parameters
        ----------
        a, b : Sector
            The two sectors fused, in this provider's own sector type.

        Returns
        -------
        tuple of Sector
            Every ``c`` with ``N^c_ab > 0``, each listed **once** regardless of
            its multiplicity, in a deterministic canonical (ascending) order —
            block enumeration is derived from this order, so it must never
            depend on dict iteration or insertion history.
        """
        ...

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        """Multiplicity ``N^c_ab``. Multiplicity-free providers return 0 or 1.

        Parameters
        ----------
        a, b : Sector
            The two sectors fused.
        c : Sector
            The candidate fusion channel.

        Returns
        -------
        int
            The number of independent vertices ``a x b -> c``; ``0`` when the
            channel is forbidden.
        """
        ...


class _DualFusionRules(FusionRules, Protocol):
    """``FusionRules`` plus the label-level ``dual`` map.

    The honest annotation for the sites that store a provider and relabel
    sectors through ``dual`` (``GradedSpace``, ``Leg``, ``flip_dual``, the bend
    helpers). Deliberately **not** the full ``DualityData`` (which also
    carries ``b_symbol``), because every provider — Abelian ones included —
    satisfies this annotation while only braided/rigid ones supply B-symbols.
    Not ``runtime_checkable``: it is an annotation, never an ``isinstance``
    gate.
    """

    def dual(self, a: Sector) -> Sector: ...


@runtime_checkable
class QuantumDimensionData(Protocol):
    """Providers that define a quantum dimension.

    Notes
    -----
    ``qdim`` need not be an integer (Fibonacci's ``tau`` has ``qdim == phi``)
    and is independent of any dense expansion
    ([ClebschGordanData][tenet.symmetry.ClebschGordanData]).
    """

    def qdim(self, a: Sector) -> float:
        """The quantum dimension ``d_a``.

        Parameters
        ----------
        a : Sector
            The sector whose quantum dimension is asked.

        Returns
        -------
        float
            ``d_a > 0``. Equal to ``irrep_dim(a)`` whenever the provider also
            has [ClebschGordanData][tenet.symmetry.ClebschGordanData], but not
            an integer in general.
        """
        ...


@runtime_checkable
class ClebschGordanData(Protocol):
    """Providers that can supply explicit Clebsch-Gordan tensors.

    Notes
    -----
    A dense-basis capability: it exists exactly when the sectors are
    representations of something. An anyonic provider (Fibonacci, Ising) has no
    ``cgc`` and no ``irrep_dim`` at all, which is why quantum dimensions live on
    [QuantumDimensionData][tenet.symmetry.QuantumDimensionData] instead.
    """

    def irrep_dim(self, a: Sector) -> int:
        """The dense dimension ``d_a`` of the irrep labelled ``a``.

        Parameters
        ----------
        a : Sector
            The sector whose irrep dimension is asked.

        Returns
        -------
        int
            ``d_a >= 1``, the length of each of ``cgc``'s first three axes.
        """
        ...

    def cgc(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        """Shape ``(d_a, d_b, d_c, N^c_ab)``; last axis is the multiplicity label mu.

        Parameters
        ----------
        a, b : Sector
            The two fused sectors.
        c : Sector
            The fusion channel selected.

        Returns
        -------
        numpy.ndarray
            The Clebsch-Gordan tensor of ``a x b -> c`` in the provider's own
            dense basis, shape ``(d_a, d_b, d_c, N^c_ab)`` — the trailing axis
            is the multiplicity label ``mu`` and is present even when it has
            size 1.

        Raises
        ------
        ValueError
            If ``c`` is not a fusion channel of ``a x b``.
        """
        ...


@runtime_checkable
class PermutationCoefficients(Protocol):
    """Providers that can expand a fusion tree with permuted uncoupled lines."""

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """``((tree', coeff), ...)`` with the permuted tree equal to ``Σ coeff · tree'``.

        Parameters
        ----------
        tree : FusionTree
            The left-associated tree whose uncoupled lines are permuted.
        perm : tuple of int
            ``perm[j]`` is the *old* uncoupled position that becomes position
            ``j`` (the same convention as ``transpose``'s ``axes``).

        Returns
        -------
        tuple of (FusionTree, complex)
            The expansion of the permuted tree over canonical left-associated
            trees with the same coupled sector.

        Notes
        -----
        The coefficient stays **scalar** even under ``n_symbol > 1``, because
        the multiplicity label lives inside the tree: a matrix-valued F is
        still one number per ``(tree, tree')`` pair, it just makes the
        expansion longer.
        """
        ...


@runtime_checkable
class BendingCoefficients(Protocol):
    """Providers that can move one line between the two trees of a block key."""

    def bend_right(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        """Move the LAST uncoupled line of ``output_tree`` onto the END of
        ``input_tree``, dualized.

        Parameters
        ----------
        key : FusionBlockKey
            The block key whose output tree loses its last line.
        dual : bool
            The moved leg's *current* dual flag; it selects the
            Frobenius-Schur factor.

        Returns
        -------
        tuple of (FusionBlockKey, complex)
            ``((key', coeff), ...)``, the expansion of the bent key.
        """
        ...

    def bend_left(
        self, key: "FusionBlockKey", *, dual: bool
    ) -> tuple[tuple["FusionBlockKey", complex], ...]:
        """The inverse direction: last line of ``input_tree`` onto ``output_tree``.

        Parameters
        ----------
        key : FusionBlockKey
            The block key whose input tree loses its last line.
        dual : bool
            The moved leg's *current* dual flag.

        Returns
        -------
        tuple of (FusionBlockKey, complex)
            ``((key', coeff), ...)``, the expansion of the bent key.
        """
        ...


@runtime_checkable
class DualBasis(Protocol):
    """Providers whose ``V_a -> V_a^*`` isomorphism is available in the dense basis."""

    def z_matrix(self, a: Sector) -> np.ndarray:
        """``Z_a``, shape ``(d_a, d_dual(a))``, in the provider's own dense basis.

        Parameters
        ----------
        a : Sector
            The sector whose ``V_a -> V_a^*`` isomorphism is asked.

        Returns
        -------
        numpy.ndarray
            ``Z_a``, shape ``(d_a, d_dual(a))``, in the same gauge as ``cgc``.
        """
        ...


@runtime_checkable
class BranchingRules(Protocol):
    """Providers that can be restricted to a smaller symmetry in the dense basis."""

    def branch(self, target: FusionRules, a: Sector) -> tuple[Sector, ...]:
        """The ``target`` sector of each of ``a``'s ``d_a`` dense basis vectors.

        Parameters
        ----------
        target : FusionRules
            The smaller symmetry restricted to; must be abelian.
        a : Sector
            The sector of *this* provider being decomposed.

        Returns
        -------
        tuple of Sector
            One ``target`` sector per dense basis vector, length exactly
            ``irrep_dim(a)``.

        Raises
        ------
        CapabilityError
            For a ``target`` this provider cannot restrict to.

        Notes
        -----
        The order is the provider's **own** dense (magnetic) order — the same
        order ``cgc`` and ``z_matrix`` use, so this composes with
        ``to_dense``'s ``alpha * d_a + m`` layout without a second convention.
        Every returned sector must satisfy ``target.irrep_dim(...) == 1``: one
        target label per basis vector is only well-defined when the target's
        irreps are one-dimensional, i.e. when the target is abelian.
        """
        ...


@runtime_checkable
class AssociatorData(Protocol):
    """Providers that supply the associator ``F``, scalar-valued.

    Scalar-valued is a multiplicity-free assumption: a provider with
    ``n_symbol > 1`` must raise rather than truncate a matrix-valued symbol, and
    supply [FMatrixData][tenet.symmetry.FMatrixData] beside this protocol instead.
    """

    def f_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector) -> complex:
        """``[F^{abc}_d]_{e,f}``; ``e`` is the inner line of ``((ab)c)``, ``f`` of ``(a(bc))``.

        Parameters
        ----------
        a, b, c : Sector
            The three fused sectors, in order.
        d : Sector
            The total (coupled) sector.
        e : Sector
            The inner line of the ``((ab)c)`` association.
        f : Sector
            The inner line of the ``(a(bc))`` association.

        Returns
        -------
        complex
            The recoupling coefficient; exactly ``0`` for a structurally
            forbidden labelling.

        Raises
        ------
        ValueError
            If any vertex of the labelling has ``n_symbol > 1`` — the scalar
            symbol must refuse rather than truncate a matrix-valued one.
        """
        ...


@runtime_checkable
class BraidingData(Protocol):
    """Providers that supply the braiding ``R``, scalar-valued.

    Notes
    -----
    Having ``R`` does not make the braiding symmetric: ``transpose`` gates on
    this protocol *plus* the symmetric-braiding property (``R == R**-1``), and a
    chiral provider is refused rather than handed one of two inequivalent
    braids. Multiplicity-bearing providers supply
    [RMatrixData][tenet.symmetry.RMatrixData] beside this one.
    """

    def r_symbol(self, a: Sector, b: Sector, c: Sector) -> complex:
        """``R^{ab}_c``, the coefficient of braiding ``a`` past ``b`` inside ``c``.

        Parameters
        ----------
        a, b : Sector
            The two braided sectors, in order.
        c : Sector
            The fusion channel they are braided inside.

        Returns
        -------
        complex
            The braiding phase; unit modulus in a unitary gauge.

        Raises
        ------
        ValueError
            If ``N^c_ab > 1`` — a matrix-valued braiding must be served
            through [RMatrixData][tenet.symmetry.RMatrixData], never truncated.
        """
        ...


@runtime_checkable
class DualityData(Protocol):
    """Providers that supply rigidity: the dual label map and the bend ``B``.

    Notes
    -----
    ``dual`` alone (the label map every provider carries) is not rigidity; the
    B-symbol is what prices an evaluation/coevaluation bend. Multiplicity-bearing
    providers supply [BMatrixData][tenet.symmetry.BMatrixData] beside this one.
    """

    def dual(self, a: Sector) -> Sector:
        """The dual (conjugate) label of ``a``.

        Parameters
        ----------
        a : Sector
            The sector dualized.

        Returns
        -------
        Sector
            ``dual(a)``, with ``dual(dual(a)) == a``. A self-dual *label*
            (``dual(a) == a``) does not make the ``V_a -> V_a^*`` isomorphism
            the identity — that is ``z_matrix``'s content.
        """
        ...

    def b_symbol(self, a: Sector, b: Sector, c: Sector) -> complex:
        """``B^{ab}_c``, the duality coefficient bending ``b`` out of ``a x b -> c``.

        Parameters
        ----------
        a, b : Sector
            The vertex's two fused sectors; ``b`` is the line bent away.
        c : Sector
            The vertex's fusion channel.

        Returns
        -------
        complex
            The bend coefficient.

        Raises
        ------
        ValueError
            If any vertex of the labelling has ``n_symbol > 1`` — matrix-valued
            bends are served through [BMatrixData][tenet.symmetry.BMatrixData].
        """
        ...


@runtime_checkable
class FSIndicatorData(Protocol):
    """Providers that supply the Frobenius-Schur indicator ``chi``."""

    def frobenius_schur(self, a: Sector) -> complex:
        """``chi_a``, the Frobenius-Schur phase of the ``V_a -> V_a^*`` isomorphism.

        Parameters
        ----------
        a : Sector
            The sector whose indicator is asked.

        Returns
        -------
        complex
            ``chi_a``, a phase; ``+-1`` for every real-or-unitary gauge tenet
            ships.
        """
        ...


@runtime_checkable
class TwistData(Protocol):
    """Providers that supply the ribbon twist ``theta``.

    Notes
    -----
    Separate from the ``chi * theta`` flip product, because closed loops
    (``trace``, ``adjoint``, any fermion loop) need the *bare* twist. ``1`` for
    every symmetric bosonic category, ``(-1)^parity``
    for fermion parity, ``e^{4 pi i / 5}`` on Fibonacci's ``tau``.
    """

    def twist(self, a: Sector) -> complex:
        """``theta_a``, the ribbon twist of ``a``.

        Parameters
        ----------
        a : Sector
            The sector whose twist is asked.

        Returns
        -------
        complex
            ``theta_a``, a phase; ``1`` for every symmetric bosonic category,
            ``(-1)^parity`` for fermion parity.
        """
        ...


@runtime_checkable
class PivotalData(Protocol):
    """Marker for the pivotal convention the bend helpers hardcode.

    The pivotal isomorphism today is one fixed choice — TensorKit's
    ``bendright``, the ``sqrt(qdim(c)/qdim(a))`` split of the bend's
    normalization in [bend_braided][tenet.symmetry.bend_braided] — shared by every provider, so this
    protocol carries no method yet and every provider satisfies it.

    Notes
    -----
    It exists so the operations that *depend* on a pivotal structure
    (``bend``, ``full_trace``) can name it in their contracts now; a
    non-spherical pivotal provider is the trigger for giving it a real method
    (the per-object pivotal phase) instead of the hardcoded expression.
    """


@runtime_checkable
class DaggerData(Protocol):
    """Marker for the dagger structure ``adjoint``/``conj`` rely on.

    Every provider tenet ships is unitary in a real-or-unitary gauge, so the
    dagger needs no per-provider data yet and every provider satisfies this
    protocol. A provider with complex Clebsch-Gordan tensors or a non-unitary
    F/R gauge is the trigger for making it a real capability gate.
    """


@runtime_checkable
class FMatrixData(Protocol):
    """Array-valued associator for providers with ``N^c_ab > 1``.

    [AssociatorData][tenet.symmetry.AssociatorData]'s array-valued sibling, a
    capability *beside* it: the scalar symbol stays exactly as it is for every
    multiplicity-free provider, and ``_artin_braid`` takes the array path only
    when the provider also implements the array-valued protocols.
    """

    def f_matrix(
        self, a: Sector, b: Sector, c: Sector, d: Sector, e: Sector, f: Sector
    ) -> np.ndarray:
        """``[F^{abc}_d]_{e,f}``, shape ``(N^e_ab, N^d_ec, N^f_bc, N^d_af)``.

        Parameters
        ----------
        a, b, c : Sector
            The three fused sectors, in order.
        d : Sector
            The total (coupled) sector.
        e : Sector
            The inner line of the ``((ab)c)`` association.
        f : Sector
            The inner line of the ``(a(bc))`` association.

        Returns
        -------
        numpy.ndarray
            Shape ``(N^e_ab, N^d_ec, N^f_bc, N^d_af)``. The four axes are the
            four vertex labels of ``((ab)c)_d`` and ``(a(bc))_d``, in that
            order.
        """
        ...


@runtime_checkable
class RMatrixData(Protocol):
    """Array-valued braiding — [BraidingData][tenet.symmetry.BraidingData]'s
    sibling for ``N^c_ab > 1``."""

    def r_matrix(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        """``R^{ab}_c``, shape ``(N^c_ab, N^c_ba)``.

        Parameters
        ----------
        a, b : Sector
            The two braided sectors, in order.
        c : Sector
            The fusion channel they are braided inside.

        Returns
        -------
        numpy.ndarray
            Shape ``(N^c_ab, N^c_ba)``: the vertex label before against the
            vertex label after the braid.
        """
        ...


@runtime_checkable
class BMatrixData(Protocol):
    """Array-valued bend — [DualityData][tenet.symmetry.DualityData]'s sibling
    for ``N^c_ab > 1``."""

    def b_matrix(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        """``B^{ab}_c``, shape ``(N^c_ab, N^a_{c,dual(b)})``.

        Parameters
        ----------
        a, b : Sector
            The vertex's two fused sectors; ``b`` is the line bent away.
        c : Sector
            The vertex's fusion channel.

        Returns
        -------
        numpy.ndarray
            Shape ``(N^c_ab, N^a_{c,dual(b)})``: the source vertex label
            against the destination's new vertex label.
        """
        ...


class _TreeBraider(FusionRules, AssociatorData, BraidingData, Protocol):
    """What ``permute_braided_tree`` / ``_artin_braid`` actually read:
    fusion data (``fusion``/``n_symbol``/``unit``/``name``) plus the scalar F-
    and R-symbols. They never read ``b_symbol``, ``frobenius_schur`` or
    ``qdim``, so this annotation asks for none of them."""


class _TreeBender(
    FusionRules, DualityData, FSIndicatorData, QuantumDimensionData, PivotalData, Protocol
):
    """What ``bend_braided`` actually reads: fusion data plus ``dual``,
    ``b_symbol``, ``frobenius_schur`` and ``qdim`` (under the hardcoded pivotal
    convention). It never reads ``f_symbol`` or ``r_symbol``; ``qdim`` is
    declared by ``QuantumDimensionData``."""


@runtime_checkable
class _FRMatrices(FMatrixData, RMatrixData, Protocol):
    """What ``_artin_braid``'s array-valued path reads: ``f_matrix`` and
    ``r_matrix`` together (one F-R-F move mixes both). ``b_matrix`` is owned by
    ``bend_braided``, which gates on ``BMatrixData`` alone."""


class CapabilityError(TypeError):
    """Raised when a provider lacks a capability an operation requires."""


class StructureChangingError(TypeError):
    """Raised when an operation whose *output structure depends on block values*
    is asked to run inside a traced (jit/grad/vmap) region.

    Invariants 9 and 10: the library
    never hides the distinction between a shape-static operation and one that
    decides its own output structure from the numbers. Lives here next to
    [CapabilityError][tenet.symmetry.CapabilityError], subclasses ``TypeError``
    for the same reason it does, and is exported from ``tenet``.
    """


def requires(provider: object, capability: type) -> None:
    """Raise [CapabilityError][tenet.symmetry.CapabilityError] unless
    ``provider`` implements ``capability``.

    Parameters
    ----------
    provider : object
        The provider gated, usually read off a leg's space.
    capability : type
        One of the ``runtime_checkable`` capability protocols of
        ``tenet.symmetry``.

    Raises
    ------
    CapabilityError
        If ``provider`` does not implement ``capability``, naming both.

    Examples
    --------
    >>> from tenet.symmetry import U1, Z2, ClebschGordanData, BranchingRules, requires
    >>> requires(U1, ClebschGordanData)  # U(1) has CG tensors: no raise
    >>> requires(Z2, BranchingRules)
    Traceback (most recent call last):
        ...
    tenet.symmetry.base.CapabilityError: Z2Provider does not provide capability BranchingRules
    """
    if not isinstance(provider, capability):
        raise CapabilityError(
            f"{type(provider).__name__} does not provide capability {capability.__name__}"
        )


def supports(provider: object, capability: type) -> bool:
    """``True`` iff ``provider`` implements ``capability`` — [requires][tenet.symmetry.requires]'
    non-raising sibling, so the capability graph is queryable without
    ``try/except CapabilityError``.

    Parameters
    ----------
    provider : object
        The provider queried.
    capability : type
        One of the ``runtime_checkable`` capability protocols of
        ``tenet.symmetry``.

    Returns
    -------
    bool
        Whether ``provider`` implements ``capability``.

    Examples
    --------
    >>> from tenet.symmetry import U1, Z2, ClebschGordanData, BranchingRules, supports
    >>> supports(U1, ClebschGordanData)
    True
    >>> supports(Z2, BranchingRules)
    False
    """
    return isinstance(provider, capability)


def permute_unique_tree(
    provider: FusionRules, tree: "FusionTree", perm: tuple[int, ...]
) -> tuple[tuple["FusionTree", complex], ...]:
    """``permute_tree`` for providers whose fusion is unique and whose F/R are 1.

    Parameters
    ----------
    provider : FusionRules
        The provider whose trees are permuted; it must have opted in by
        defining ``permute_tree`` in terms of this helper.
    tree : FusionTree
        The left-associated tree whose uncoupled lines are permuted.
    perm : tuple of int
        ``perm[j]`` is the *old* uncoupled position that becomes position ``j``.

    Returns
    -------
    tuple of (FusionTree, complex)
        Exactly one term: the recomputed tree with coefficient ``1.0``.

    Raises
    ------
    CapabilityError
        If the permuted uncoupled labels admit anything other than exactly one
        tree coupling to the same sector — this helper is only correct for
        unique fusion.

    Notes
    -----
    Shared by Trivial and U(1): permuting the uncoupled labels leaves exactly one
    left-associated tree with the same coupled sector, so the whole expansion is
    that tree with coefficient 1. The spine is *recomputed* rather than permuted,
    which is what makes this correct rather than a relabelling.

    Not a capability check: a provider must opt in by defining ``permute_tree``
    (fermionic parity also has unique multiplicity-free fusion and still carries a
    sign, so uniqueness alone must never be taken as permission).
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
    provider: _TreeBraider, tree: "FusionTree", perm: tuple[int, ...]
) -> tuple[tuple["FusionTree", complex], ...]:
    """``permute_tree`` for any provider supplying F- and R-symbols.

    Parameters
    ----------
    provider : FusionRules
        A provider with [AssociatorData][tenet.symmetry.AssociatorData] and
        [BraidingData][tenet.symmetry.BraidingData] (checked with
        [requires][tenet.symmetry.requires] on entry).
    tree : FusionTree
        The left-associated tree whose uncoupled lines are permuted.
    perm : tuple of int
        ``perm[j]`` is the OLD uncoupled position that becomes position ``j``.

    Returns
    -------
    tuple of (FusionTree, complex)
        The accumulated ``{tree: coeff}`` expansion of the permuted tree.

    Raises
    ------
    CapabilityError
        If the provider lacks
        [FusionRules][tenet.symmetry.FusionRules],
        [AssociatorData][tenet.symmetry.AssociatorData] or
        [BraidingData][tenet.symmetry.BraidingData].

    Notes
    -----
    Bubble-decomposes ``perm`` into adjacent transpositions (Artin generators) and
    applies ``_artin_braid`` to each, accumulating a ``{tree: coeff}`` expansion.

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
    """
    # Simplification: the ceiling is term-list size for high-rank, high-spin trees; the
    # upgrade path is exact sqrt-rational accumulation, not a tolerance.
    for capability in (FusionRules, AssociatorData, BraidingData):
        requires(provider, capability)
    terms: dict[FusionTree, complex] = {tree: 1.0}
    current = list(range(tree.rank))  # old position now sitting at each slot
    for j, target in enumerate(perm):
        for i in range(current.index(target) - 1, j - 1, -1):
            current[i], current[i + 1] = current[i + 1], current[i]
            nxt: dict[FusionTree, complex] = {}
            for t, c in terms.items():
                # hashable by provider contract; the protocol deliberately
                # omits __hash__ (member set pinned by its test)
                for braided, cb in _artin_braid(provider, t, i):  # ty: ignore[invalid-argument-type]
                    nxt[braided] = nxt.get(braided, 0) + c * cb
            terms = nxt
    return tuple(terms.items())


@cache
def _artin_braid(
    provider: _TreeBraider, tree: "FusionTree", i: int
) -> tuple[tuple["FusionTree", complex], ...]:
    """Swap uncoupled lines ``i`` and ``i+1`` of a left-associated tree.

    ``i == 0``: the two lines already meet at the first vertex, so a single
    R-move does it and the spine is untouched.

    ``i >= 1``: with ``a = e_{i-1}``, ``b = u_i``, ``c = u_{i+1}``, ``f = e_{i+1}``
    and the old inner line ``e = e_i``, F-move to ``(a (b c))``, R on the ``(b c)``
    vertex, F-move back::

        coeff(e -> e') = sum_d F^{abc}_f[e, d] · R^{bc}_d · conj(F^{acb}_f[e', d])

    which is a genuine expansion over the admissible ``e' in fusion(a, c)``.
    Every other spine entry is unchanged.

    A provider that also implements ``FMatrixData`` and ``RMatrixData``
    takes the array-valued path: the two vertex labels the move touches (``mu_{i-1}`` on
    ``a x b -> e`` and ``mu_i`` on ``e x c -> f``) are expanded over too, since an
    F-move mixes ``(e, mu)`` with ``(f, mu')``. Every *other* multiplicity label
    is unchanged. Providers without it keep the scalar path exactly, and a
    multiplicity-free provider's ``n_symbol > 1`` never arises.
    """
    from tenet.fusion_tree import FusionTree

    u = tree.uncoupled
    spine = tree.lines()
    new_u = (*u[:i], u[i + 1], u[i], *u[i + 2 :])
    matrices = provider if isinstance(provider, _FRMatrices) else None
    if i == 0:
        if matrices is None:
            swapped = FusionTree(new_u, tree.inner, tree.multiplicities, tree.coupled)
            return ((swapped, provider.r_symbol(u[0], u[1], spine[1])),)
        row = matrices.r_matrix(u[0], u[1], spine[1])[tree.multiplicities[0]]
        return tuple(
            (
                FusionTree(new_u, tree.inner, (mu, *tree.multiplicities[1:]), tree.coupled),
                complex(row[mu]),
            )
            for mu in range(len(row))
        )

    a, b, c, e, f = spine[i - 1], u[i], u[i + 1], spine[i], spine[i + 1]
    out = []
    for new_e in provider.fusion(a, c):
        if not provider.n_symbol(new_e, b, f):
            continue
        inner = (*tree.inner[: i - 1], new_e, *tree.inner[i:])
        if matrices is None:
            coeff = sum(
                provider.f_symbol(a, b, c, f, e, d)
                * provider.r_symbol(b, c, d)
                * provider.f_symbol(a, c, b, f, new_e, d).conjugate()
                for d in provider.fusion(b, c)
            )
            out.append((FusionTree(new_u, inner, tree.multiplicities, tree.coupled), coeff))
            continue
        # Simplification: the double sum is spelled with plain loops, not einsum, so this
        # helper stays array-free (pinned by ``test_braid_helpers_touch_no_arrays``).
        # A vertex multiplicity is a handful of channels; if some future group makes
        # it large, the upgrade path is one einsum here, not a redesign.
        n_prev, n_cur = provider.n_symbol(a, c, new_e), provider.n_symbol(new_e, b, f)
        block = [[0j] * n_cur for _ in range(n_prev)]
        for d in provider.fusion(b, c):
            n_bcd, n_adf = provider.n_symbol(b, c, d), provider.n_symbol(a, d, f)
            if not (n_bcd and n_adf):
                continue
            lhs = matrices.f_matrix(a, b, c, f, e, d)[
                tree.multiplicities[i - 1], tree.multiplicities[i]
            ]
            rmat = matrices.r_matrix(b, c, d)
            rhs = matrices.f_matrix(a, c, b, f, new_e, d)
            for mu_prev in range(n_prev):
                for mu_cur in range(n_cur):
                    block[mu_prev][mu_cur] += sum(
                        lhs[nu1, nu2]
                        * rmat[nu1, nu1p]
                        * rhs[mu_prev, mu_cur, nu1p, nu2].conjugate()
                        for nu1 in range(n_bcd)
                        for nu1p in range(rmat.shape[1])
                        for nu2 in range(n_adf)
                    )
        for mu_prev in range(n_prev):
            for mu_cur in range(n_cur):
                labels = (
                    *tree.multiplicities[: i - 1],
                    mu_prev,
                    mu_cur,
                    *tree.multiplicities[i + 1 :],
                )
                out.append(
                    (
                        FusionTree(new_u, inner, labels, tree.coupled),
                        complex(block[mu_prev][mu_cur]),
                    )
                )
    return tuple(out)


def bend_unique(
    provider: _DualFusionRules, key: "FusionBlockKey", *, right: bool, dual: bool
) -> tuple[tuple["FusionBlockKey", complex], ...]:
    """``bend_right``/``bend_left`` for providers whose fusion is unique and B is 1.

    Parameters
    ----------
    provider : FusionRules
        The provider whose keys are bent; it must have opted in by defining
        ``bend_right``/``bend_left`` in terms of this helper.
    key : FusionBlockKey
        The block key one line is moved across.
    right : bool
        ``True`` moves the output tree's last line onto the input tree
        (``bend_right``); ``False`` is the inverse direction.
    dual : bool
        The moved leg's *current* dual flag. Accepted and provably ignored
        here — see Notes.

    Returns
    -------
    tuple of (FusionBlockKey, complex)
        Exactly one term: the recomputed key with coefficient ``1.0``.

    Raises
    ------
    ValueError
        If the source tree is empty (rank 0), so there is no line to bend.
    CapabilityError
        If dropping the moved line leaves uncoupled labels with anything other
        than exactly one coupled sector, or more than one tree — this helper is
        only correct for unique fusion.

    Notes
    -----
    Shared by Trivial and U(1), exactly as
    [permute_unique_tree][tenet.symmetry.permute_unique_tree] is. The
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
    provider: _TreeBender, key: "FusionBlockKey", *, right: bool, dual: bool
) -> tuple[tuple["FusionBlockKey", complex], ...]:
    """``bend_right``/``bend_left`` for any provider supplying B, FS and ``qdim``.

    Parameters
    ----------
    provider : FusionRules
        A provider with [DualityData][tenet.symmetry.DualityData],
        [FSIndicatorData][tenet.symmetry.FSIndicatorData],
        [QuantumDimensionData][tenet.symmetry.QuantumDimensionData] and
        [PivotalData][tenet.symmetry.PivotalData] (the ``_TreeBender``
        annotation — enforced by the type checker, not a runtime gate; see the
        comment in the body).
    key : FusionBlockKey
        The block key one line is moved across.
    right : bool
        ``True`` moves the output tree's last line onto the input tree
        (``bend_right``); ``False`` is the inverse direction.
    dual : bool
        The moved leg's *current* dual flag; it keys the Frobenius-Schur
        factor.

    Returns
    -------
    tuple of (FusionBlockKey, complex)
        One term for a multiplicity-free provider; genuinely multi-term when
        the provider supplies [BMatrixData][tenet.symmetry.BMatrixData].

    Raises
    ------
    ValueError
        If the source tree is empty (rank 0), so there is no line to bend.
    CapabilityError
        If the bent vertex has ``n_symbol > 1`` and the provider does not
        supply [BMatrixData][tenet.symmetry.BMatrixData].

    Notes
    -----
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
    out. ``n_symbol > 1`` needs matrix-valued coefficients, which a provider
    supplies by also implementing [BMatrixData][tenet.symmetry.BMatrixData]; then the
    destination's new vertex label is expanded over ``B``'s second axis instead
    of being pinned to ``0``, and the result is genuinely multi-term. Without
    that capability the existing [CapabilityError][tenet.symmetry.CapabilityError] still fires.
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

    # No runtime requires() here: the _TreeBender annotation names the honest
    # capability set (DualityData + FSIndicatorData + QuantumDimensionData +
    # PivotalData over FusionRules) and the type checker enforces it; a runtime
    # isinstance gate would use getattr_static and refuse the __getattr__-forwarding
    # test stubs that the pre-M24 behavior accepts.
    b, c = src.uncoupled[-1], src.coupled
    a = src.lines()[-2] if src.rank >= 2 else provider.unit
    matrices = provider if isinstance(provider, BMatrixData) else None
    if matrices is None and provider.n_symbol(a, b, c) > 1:
        raise CapabilityError(
            f"{provider.name}: N^{c!r}_{{{a!r},{b!r}}} > 1; bending a vertex with "
            "multiplicity needs matrix-valued B-symbols, which are not supported"
        )

    new_src = FusionTree(src.uncoupled[:-1], src.inner[:-1], src.multiplicities[:-1], a)
    scale = sqrt(provider.qdim(c) / provider.qdim(a))
    if dual:
        scale *= provider.frobenius_schur(provider.dual(b)).conjugate()

    def keyed(nu: int, coeff: complex) -> tuple["FusionBlockKey", complex]:
        new_dst = FusionTree(
            (*dst.uncoupled, provider.dual(b)),
            dst.lines()[1:],
            (*dst.multiplicities, nu) if dst.rank else (),
            a,
        )
        key_ = FusionBlockKey(new_src, new_dst) if right else FusionBlockKey(new_dst, new_src)
        return (key_, coeff if right else coeff.conjugate())

    if matrices is None:
        return (keyed(0, scale * provider.b_symbol(a, b, c)),)

    # The destination gains the vertex ``c x dual(b) -> a``, whose label is B's
    # second axis; the source's last vertex label ``mu`` selects B's row. A
    # rank-0 destination gains no vertex, so that axis is length 1 there.
    row = matrices.b_matrix(a, b, c)[src.multiplicities[-1] if src.rank >= 2 else 0]
    return tuple(keyed(nu, scale * complex(row[nu])) for nu in range(len(row)))


def _unique_tree(
    provider: FusionRules, uncoupled: tuple[Sector, ...], coupled: Sector
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
class TrivialSector(Sector):  # ty: ignore[subclass-of-dataclass-with-order]  # deliberate, see Sector
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

    def frobenius_schur(self, a: Sector) -> float:
        """``chi_a = 1``: one sector, everything trivial."""
        return 1.0

    def twist(self, a: Sector) -> float:
        """``theta_a = 1``: one sector, everything trivial."""
        return 1.0


_Z = np.ones((1, 1))
_Z.flags.writeable = False

Trivial = TrivialProvider()
"""Module-level singleton, used as ``GradedSpace(provider=Trivial, ...)``."""
