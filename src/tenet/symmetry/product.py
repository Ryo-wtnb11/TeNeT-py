"""Deligne products of providers: ``U(1)xU(1)``, ``U(1) x (fermion parity)``, ``SU(2)xU(1)``, ...

Every categorical quantity of a product factorizes, so this module is mechanical:
``unit``, ``dual``, ``fusion``, ``n_symbol``, ``qdim``, ``irrep_dim`` and ``cgc``
are componentwise, and each *optional* capability is forwarded iff **every**
factor has it. The only non-mechanical piece is
[permute_tree][tenet.symmetry.ProductProvider.permute_tree],
which is the distributed product over the factors' expansions (a factor may return
more than one term), resting on the ``project`` / ``assemble`` bijection

    {product trees}  <->  prod_i {factor-i trees}

which is the whole mathematical content here and is asserted, not assumed.

Two conventions, stated once and used twice:

* **``mu`` is mixed radix, first factor most significant** (C order):
  ``mu = sum_i mu_i * prod_{j>i} n_j`` — ``mu_encode`` / ``mu_decode``.
* **``cgc`` flattens its irrep indices the same way**, because it is built as an
  iterated outer product followed by a single C-order ``reshape``, and a C-order
  ``reshape`` *is* mixed radix with the leading factor most significant. The two
  flattenings therefore agree by construction rather than by matching arithmetic.

Two known gaps, deliberate:

* **``GradedSpace.new``'s sector-type check weakens under products.** It uses
  ``type(provider.unit)``, which is ``ProductSector`` for every product, so a
  ``U(1)xU(1)`` space would accept a ``U(1) x (fermion parity)`` sector. The mitigation lives
  here, not in ``GradedSpace`` (which correctly knows nothing about symmetries):
  ``dual`` / ``fusion`` / ``n_symbol`` validate component count and per-component
  sector type and raise ``TypeError`` naming the position, so the failure happens
  loudly at the first fusion query.
* **No ``BendingCoefficients`` / ``DualBasis`` forwarding.** The pattern is the
  same (the duality matrix becomes a Kronecker product with this same flattening)
  but it has its own dense-oracle criteria; a product has no bending capability
  today and fails loudly. Follow-up to #40.

**Nested products are not flattened.** ``ProductProvider((ProductProvider((A, B)),
C))`` is a legal provider whose sectors are nested ``ProductSector``s, and it is
*not* equal to ``ProductProvider((A, B, C))`` — the two carry different sector
values and so are not interchangeable anyway. Canonicalization, if anyone ever
wants it, belongs in a ``product(...)`` factory, not in the type.
"""

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tenet.symmetry.base import (
    CapabilityError,
    ClebschGordanData,
    FSIndicatorData,
    PermutationCoefficients,
    QuantumDimensionData,
    Sector,
    TwistData,
    _DualFusionRules,
    requires,
)

if TYPE_CHECKING:
    from tenet.fusion_tree import FusionTree

__all__ = [
    "ProductProvider",
    "ProductSector",
    "assemble",
    "mu_decode",
    "mu_encode",
    "project",
]

_PERM_HINT = (
    "Within-side reordering is a braid and needs that factor's own F- and R-moves "
    "(Milestone 4 for SU(2)); it will not be faked by permuting block axes."
)
_CGC_HINT = "A product's Clebsch-Gordan tensor is the outer product of the factors'."


@dataclass(frozen=True, slots=True, order=True)
class ProductSector(Sector):  # ty: ignore[subclass-of-dataclass-with-order]  # deliberate, see Sector
    """One sector per factor, in factor order. Ordering is lexicographic by component.

    Parameters
    ----------
    components : tuple of Sector
        One sector per factor of the [ProductProvider][tenet.symmetry.ProductProvider],
        in factor order.
    """

    components: tuple[Sector, ...]


def mu_encode(ns: tuple[int, ...], mus: tuple[int, ...]) -> int:
    """Mixed radix, first factor most significant: ``sum_i mu_i * prod_{j>i} n_j``.

    Parameters
    ----------
    ns : tuple of int
        The per-factor vertex multiplicities ``n_i``.
    mus : tuple of int
        The per-factor vertex labels, ``0 <= mu_i < n_i``.

    Returns
    -------
    int
        The product vertex label.

    Examples
    --------
    >>> from tenet.symmetry.product import mu_encode, mu_decode
    >>> mu_encode((2, 3), (1, 2))
    5
    >>> mu_decode((2, 3), 5)
    (1, 2)

    Notes
    -----
    Every product of multiplicity-free factors is multiplicity-free, but a factor
    with ``n > 1`` reaches this, and the mixed radix is what keeps the product's
    ``mu`` a single integer. Covered by the ``n_symbol == 2`` stub in
    ``tests/symmetry/test_product.py`` and by the real SU(3) x U(1) product in
    ``tests/symmetry/test_su3_multiplicity.py``.
    """
    mu = 0
    for n, m in zip(ns, mus, strict=True):
        mu = mu * n + m
    return mu


def mu_decode(ns: tuple[int, ...], mu: int) -> tuple[int, ...]:
    """Inverse of ``mu_encode`` on ``range(prod(ns))``."""
    out = []
    for n in reversed(ns):
        out.append(mu % n)
        mu //= n
    return tuple(reversed(out))


@dataclass(frozen=True, slots=True)
class ProductProvider:
    """The Deligne product of two or more providers. Frozen, hashable, array-free.

    Every core quantity (``unit``, ``dual``, ``fusion``, ``n_symbol``,
    ``qdim``, ``irrep_dim``, ``cgc``, the phases) is componentwise, so the
    capability contract stays the one documented on the protocols in
    ``tenet.symmetry``; the methods here document only what forwarding adds.

    Parameters
    ----------
    factors : tuple of provider
        Two or more providers, in order; the order is the component order of
        every [ProductSector][tenet.symmetry.ProductSector].

    Raises
    ------
    ValueError
        If fewer than two factors are given — a one-factor product is a
        confusing no-op wrapper, not a feature.

    Examples
    --------
    >>> from tenet.symmetry import U1, U1Sector, Z2, Z2Sector
    >>> from tenet.symmetry import ProductProvider, ProductSector
    >>> p = ProductProvider((U1, Z2))
    >>> p.name
    'U1 x Z2'
    >>> a = ProductSector((U1Sector(1), Z2Sector(1)))
    >>> p.fusion(a, a)
    (ProductSector(components=(U1Sector(charge=2), Z2Sector(parity=0))),)
    >>> p.dual(a)
    ProductSector(components=(U1Sector(charge=-1), Z2Sector(parity=1)))

    Notes
    -----
    Optional capabilities are forwarded iff *every* factor has them, and the
    forwarding methods are defined **unconditionally** so that the error can name
    the offending factor. The consequence is a real wart, handled by raising
    rather than by lying: ``runtime_checkable`` protocols check only for the
    *presence* of a method, so ``isinstance(ProductProvider((SU2, U1)),
    PermutationCoefficients)`` is ``True`` even though the call raises. **The call
    is authoritative, ``isinstance`` is optimistic.** Synthesizing a subclass per
    capability combination — a factory for one product — is rejected on sight.

    Equality and hashing are on ``factors`` alone; ``name`` is a property, not a
    field, so it can never be a second source of truth.
    """

    factors: tuple[_DualFusionRules, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        if len(self.factors) < 2:
            raise ValueError(
                f"a ProductProvider needs at least 2 factors, got {len(self.factors)}: "
                "a one-factor product is a confusing no-op wrapper, not a feature"
            )

    @property
    def name(self) -> str:
        return " x ".join(f.name for f in self.factors)

    # --- core fusion data -----------------------------------------------------

    @property
    def unit(self) -> ProductSector:
        return ProductSector(tuple(f.unit for f in self.factors))

    def dual(self, a: Sector) -> ProductSector:
        return ProductSector(tuple(f.dual(x) for f, x in self._zip(a)))

    def fusion(self, a: Sector, b: Sector) -> tuple[ProductSector, ...]:
        """Componentwise, and ascending: ``itertools.product`` is lexicographic and
        [ProductSector][tenet.symmetry.ProductSector] compares
        lexicographically, so the canonical order
        of each factor's ``fusion`` survives (asserted by test, not assumed)."""
        return tuple(
            ProductSector(cs)
            for cs in itertools.product(*(f.fusion(x, y) for f, x, y in self._zip(a, b)))
        )

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return math.prod(f.n_symbol(x, y, z) for f, x, y, z in self._zip(a, b, c))

    # --- forwarded capabilities: present iff EVERY factor has them -------------

    def qdim(self, a: Sector) -> float:
        self._require_all(QuantumDimensionData)
        return math.prod(f.qdim(x) for f, x in self._zip(a))

    def frobenius_schur(self, a: Sector) -> complex:
        """``chi`` of a Deligne product factorizes, like every other phase."""
        self._require_all(FSIndicatorData)
        return math.prod(f.frobenius_schur(x) for f, x in self._zip(a))

    def twist(self, a: Sector) -> complex:
        """``theta`` of a Deligne product factorizes, like every other phase."""
        self._require_all(TwistData)
        return math.prod(f.twist(x) for f, x in self._zip(a))

    def irrep_dim(self, a: Sector) -> int:
        self._require_all(ClebschGordanData, _CGC_HINT)
        return math.prod(f.irrep_dim(x) for f, x in self._zip(a))

    def cgc(self, a: Sector, b: Sector, c: Sector) -> np.ndarray:
        """Shape ``(prod d_a, prod d_b, prod d_c, prod n)``, the iterated outer
        product of the factors' ``cgc``s flattened in C order — which is exactly
        the mixed radix of ``mu_encode``. Raises ``ValueError`` (from the
        factor) on a fusion-forbidden triple."""
        self._require_all(ClebschGordanData, _CGC_HINT)
        out = None
        for f, x, y, z in self._zip(a, b, c):
            g = f.cgc(x, y, z)
            if out is None:
                out = g
                continue
            shape = tuple(p * q for p, q in zip(out.shape, g.shape, strict=True))
            out = np.einsum("abcm,ABCM->aAbBcCmM", out, g).reshape(shape)
        # a product has at least one factor, so the loop assigned out
        return out  # ty: ignore[invalid-return-type]

    def permute_tree(
        self, tree: "FusionTree", perm: tuple[int, ...]
    ) -> tuple[tuple["FusionTree", complex], ...]:
        """The distributed product of the factors' expansions.

        The braiding of a Deligne product is the tensor product of the factors'
        braidings and the associator factorizes, so the permuted product tree is
        literally ``⊗_i (Σ_α c_α^(i) t_α^(i))``, which multiplies out to this.
        Written for many terms per factor, which SU(2)'s braid expansion (#36)
        actually reaches — a single-term special case would be a correctness trap.
        """
        self._require_all(PermutationCoefficients, _PERM_HINT)
        per_factor = [
            # _require_all(PermutationCoefficients) above; no narrowing
            f.permute_tree(project(self, tree, i), perm)  # ty: ignore[unresolved-attribute]
            for i, f in enumerate(self.factors)
        ]
        out = []
        for terms in itertools.product(*per_factor):
            coeff = math.prod(c for _, c in terms)
            if coeff == 0:
                continue
            out.append((assemble(self, tuple(t for t, _ in terms)), coeff))
        return tuple(out)

    # --- validation -----------------------------------------------------------

    def _require_all(self, capability: type, hint: str = "") -> None:
        """Raise naming the first factor that lacks ``capability``.

        The only dispatch in this module: ``requires(factor, Capability)``, never
        a provider identity test.
        """
        for i, f in enumerate(self.factors):
            try:
                requires(f, capability)
            except CapabilityError as exc:
                raise CapabilityError(
                    f"{self.name}: factor {i} ({f.name}) does not implement "
                    f"{capability.__name__}, so the product does not either — a capability "
                    f"is forwarded only when every factor has it. {hint}"
                ) from exc

    def _split(self, a: Sector, position: int = 0) -> tuple[Sector, ...]:
        """``a``'s components, validated against the factors.

        ``GradedSpace`` cannot make this check (every product's sector type is
        ``ProductSector``), so a foreign product sector must fail loudly here.
        """
        if not isinstance(a, ProductSector):
            raise TypeError(
                f"{self.name}: argument {position} is {type(a).__name__}, expected a ProductSector"
            )
        if len(a.components) != len(self.factors):
            raise TypeError(
                f"{self.name}: argument {position} {a!r} has {len(a.components)} components, "
                f"expected {len(self.factors)}"
            )
        for i, (f, x) in enumerate(zip(self.factors, a.components, strict=True)):
            expected = type(f.unit)
            if type(x) is not expected:
                raise TypeError(
                    f"{self.name}: argument {position}, component {i} is "
                    f"{type(x).__name__}, expected {expected.__name__} for factor {f.name}"
                )
        return a.components

    def _zip(self, *sectors: Sector):
        """``(factor, comp_0, comp_1, ...)`` per factor, all arguments validated."""
        columns = [self._split(a, position) for position, a in enumerate(sectors)]
        return zip(self.factors, *columns, strict=True)


# --- the project / assemble bijection ---------------------------------------------


def project(provider: ProductProvider, tree: "FusionTree", i: int) -> "FusionTree":
    """The factor-``i`` component of a product tree.

    Parameters
    ----------
    provider : ProductProvider
        The product the tree belongs to.
    tree : FusionTree
        A tree whose sectors are this product's ``ProductSector`` values.
    i : int
        The factor index.

    Returns
    -------
    FusionTree
        A valid left-associated tree for ``provider.factors[i]`` of the same
        rank: componentwise uncoupled / inner / coupled, with each vertex's
        ``mu`` decoded from mixed radix.

    Raises
    ------
    IndexError
        If ``i`` is out of range for the factors.
    """
    from tenet.fusion_tree import FusionTree

    factors = provider.factors
    if not 0 <= i < len(factors):
        raise IndexError(f"{provider.name}: factor index {i} out of range for {len(factors)}")
    split = provider._split
    mus = []
    for e, u, f, mu in tree.vertices():
        ns = tuple(
            g.n_symbol(x, y, z)
            for g, x, y, z in zip(factors, split(e), split(u), split(f), strict=True)
        )
        mus.append(mu_decode(ns, mu)[i])
    return FusionTree(
        tuple(split(u)[i] for u in tree.uncoupled),
        tuple(split(e)[i] for e in tree.inner),
        tuple(mus),
        split(tree.coupled)[i],
    )


def assemble(provider: ProductProvider, trees: tuple["FusionTree", ...]) -> "FusionTree":
    """Inverse of ``project``: one tree per factor, same rank, into one product tree.

    Parameters
    ----------
    provider : ProductProvider
        The product the trees are assembled for.
    trees : tuple of FusionTree
        One tree per factor, all of the same rank, in factor order.

    Returns
    -------
    FusionTree
        The product tree whose components the inputs are; each vertex's ``mu``
        is re-encoded in mixed radix.

    Raises
    ------
    ValueError
        If the number of trees does not match the factors, or their ranks
        differ.
    """
    from tenet.fusion_tree import FusionTree

    factors = provider.factors
    if len(trees) != len(factors):
        raise ValueError(f"{provider.name}: expected {len(factors)} trees, got {len(trees)}")
    ranks = {t.rank for t in trees}
    if len(ranks) != 1:
        raise ValueError(f"{provider.name}: factor trees have differing ranks {sorted(ranks)}")
    mus = []
    for vertices in zip(*(t.vertices() for t in trees), strict=True):
        ns = tuple(f.n_symbol(a, b, c) for f, (a, b, c, _) in zip(factors, vertices, strict=True))
        mus.append(mu_encode(ns, tuple(v[3] for v in vertices)))
    return FusionTree(
        tuple(ProductSector(cs) for cs in zip(*(t.uncoupled for t in trees), strict=True)),
        tuple(ProductSector(cs) for cs in zip(*(t.inner for t in trees), strict=True)),
        tuple(mus),
        ProductSector(tuple(t.coupled for t in trees)),
    )
