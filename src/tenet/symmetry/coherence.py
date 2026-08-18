"""Coherence validators and derived categorical properties (M24a, #158).

Properties and coherence conditions — pentagon, hexagon, snake, sphericality,
unitarity, non-degeneracy of the braiding — are **validators, not protocols**:
a provider is data/capabilities (:mod:`tenet.symmetry.base`) plus validated
properties (this module). Each validator takes a provider and an explicit
sector budget, checks every instance of its identity that the budget reaches,
raises ``ValueError`` on the first violation and returns the number of
instances checked. Nothing here runs on an operation's hot path — operations
gate through ``requires``/``supports``; validators run in tests and at the two
property gates (``transpose``'s symmetric-braiding check, ``full_trace``'s
sphericality check), both cached.

Tolerances follow the arithmetic: exact identities (quantum dimensions,
``R**2`` for a symmetric provider) are compared exactly; float-folded ones
(pentagon, unitarity) use the same ``atol`` the pre-M24 test assertions used.

Multiplicity-bearing providers (SU(N)) validate their coherence racah-side —
the scalar validators here raise on an ``n_symbol > 1`` vertex by construction
of the scalar symbols. The symmetric-braiding check alone has an array path,
because ``transpose`` must classify SU(N) too.
"""

from dataclasses import dataclass
from functools import cache
from itertools import product
from math import sqrt
from typing import Protocol, runtime_checkable

import numpy as np

from tenet.symmetry.base import (
    AssociatorData,
    BraidingData,
    DaggerData,
    FusionRules,
    PivotalData,
    QuantumDimensionData,
    RMatrixData,
    Sector,
    TwistData,
    _DualFusionRules,
    supports,
)

__all__ = [
    "CategoricalProperties",
    "properties",
    "symmetric_braiding",
    "validate_hexagon",
    "validate_non_degenerate_braiding",
    "validate_pentagon",
    "validate_snake",
    "validate_spherical",
    "validate_unitary",
]

Sectors = tuple[Sector, ...]


@runtime_checkable
class SectorEnumeration(Protocol):
    """Providers with finitely many sectors that can list them all.

    Opt-in probe for :func:`properties` when no explicit sector budget is
    given; an infinite-sector provider (U(1), SU(2)) cannot implement it and
    must be handed sectors explicitly for a meaningful answer.
    """

    def all_sectors(self) -> tuple[Sector, ...]: ...


def _channels(provider, a, b):
    return provider.fusion(a, b)


def validate_pentagon(provider: FusionRules, sectors: Sectors, *, atol: float = 1e-12) -> int:
    """Check the pentagon identity of ``f_symbol`` over ``sectors``.

    ``[F^{fcd}_e]_{g,l} [F^{abl}_e]_{f,k} = sum_h [F^{abc}_g]_{f,h}
    [F^{ahd}_e]_{g,k} [F^{bcd}_k]_{h,l}`` for every admissible labelling with
    ``a, b, c, d`` drawn from ``sectors`` (inner lines follow fusion and may
    leave the budget). Needs :class:`~tenet.symmetry.AssociatorData` and
    :class:`~tenet.symmetry.FusionRules`.

    Parameters
    ----------
    provider : FusionRules
        The provider whose associator is checked.
    sectors : tuple of Sector
        The budget the four outer labels are drawn from.
    atol : float, optional
        Absolute tolerance per instance.

    Returns
    -------
    int
        The number of pentagon instances checked.

    Raises
    ------
    ValueError
        On the first violated instance, naming its labels.
    """
    F = provider.f_symbol  # ty: ignore[unresolved-attribute]  # AssociatorData, caller-gated
    checked = 0
    for a, b, c, d in product(sectors, repeat=4):
        for f in _channels(provider, a, b):
            for g in _channels(provider, f, c):
                for ell in _channels(provider, c, d):
                    for e in _channels(provider, g, d):
                        if not provider.n_symbol(f, ell, e):
                            continue
                        for k in _channels(provider, b, ell):
                            if not provider.n_symbol(a, k, e):
                                continue
                            lhs = F(f, c, d, e, g, ell) * F(a, b, ell, e, f, k)
                            rhs = sum(
                                F(a, b, c, g, f, h) * F(a, h, d, e, g, k) * F(b, c, d, k, h, ell)
                                for h in _channels(provider, b, c)
                            )
                            if abs(lhs - rhs) > atol:
                                raise ValueError(
                                    f"{provider.name}: pentagon violated at "
                                    f"{(a, b, c, d, e, f, g, k, ell)}: "
                                    f"lhs {lhs!r} != rhs {rhs!r}"
                                )
                            checked += 1
    return checked


def validate_hexagon(provider: FusionRules, sectors: Sectors, *, atol: float = 1e-12) -> int:
    """Check the (R-move) hexagon identity of ``f_symbol``/``r_symbol``.

    ``R^{ca}_e [F^{acb}_d]_{e,g} R^{cb}_g = sum_f [F^{cab}_d]_{e,f} R^{cf}_d
    [F^{abc}_d]_{f,g}`` — braiding ``c`` leftward past ``a`` then ``b`` equals
    braiding it past ``a x b`` at once. Needs
    :class:`~tenet.symmetry.AssociatorData` and
    :class:`~tenet.symmetry.BraidingData`.

    Parameters
    ----------
    provider : FusionRules
        The provider whose braiding is checked against its associator.
    sectors : tuple of Sector
        The budget ``a, b, c, d`` are drawn from.
    atol : float, optional
        Absolute tolerance per instance.

    Returns
    -------
    int
        The number of hexagon instances checked.

    Raises
    ------
    ValueError
        On the first violated instance, naming its labels.
    """
    F = provider.f_symbol  # ty: ignore[unresolved-attribute]  # AssociatorData, caller-gated
    R = provider.r_symbol  # ty: ignore[unresolved-attribute]  # BraidingData, caller-gated
    checked = 0
    for a, b, c, d in product(sectors, repeat=4):
        for e in _channels(provider, c, a):
            if not provider.n_symbol(e, b, d):
                continue
            for g in _channels(provider, c, b):
                if not provider.n_symbol(a, g, d):
                    continue
                lhs = R(c, a, e) * F(a, c, b, d, e, g) * R(c, b, g)
                rhs = sum(
                    F(c, a, b, d, e, f) * R(c, f, d) * F(a, b, c, d, f, g)
                    for f in _channels(provider, a, b)
                )
                if abs(lhs - rhs) > atol:
                    raise ValueError(
                        f"{provider.name}: hexagon violated at {(a, b, c, d, e, g)}: "
                        f"lhs {lhs!r} != rhs {rhs!r}"
                    )
                checked += 1
    return checked


def validate_snake(provider: _DualFusionRules, sectors: Sectors, *, atol: float = 1e-12) -> int:
    """Check the zig-zag (snake) consistency of ``b_symbol`` with F and qdim.

    ``B^{ab}_c == sqrt(qdim(a) qdim(b) / qdim(c)) [F^{a b dual(b)}_a]_{c, 1}``
    — the ``Bsymbol_from_Fsymbol`` relation that encodes evaluation followed by
    coevaluation being the identity. Needs
    :class:`~tenet.symmetry.DualityData`, :class:`~tenet.symmetry.AssociatorData`
    and :class:`~tenet.symmetry.QuantumDimensionData`.

    Parameters
    ----------
    provider : FusionRules
        The provider whose bend coefficients are checked.
    sectors : tuple of Sector
        The budget ``a, b`` are drawn from; ``c`` runs over their fusion.
    atol : float, optional
        Absolute tolerance per instance.

    Returns
    -------
    int
        The number of snake instances checked.

    Raises
    ------
    ValueError
        On the first violated instance, naming its labels.
    """
    checked = 0
    for a, b in product(sectors, repeat=2):
        for c in _channels(provider, a, b):
            expected = sqrt(provider.qdim(a) * provider.qdim(b) / provider.qdim(c)) * (  # ty: ignore[unresolved-attribute]
                provider.f_symbol(a, b, provider.dual(b), a, c, provider.unit)  # ty: ignore[unresolved-attribute]
            )
            got = provider.b_symbol(a, b, c)  # ty: ignore[unresolved-attribute]
            if abs(got - expected) > atol:
                raise ValueError(
                    f"{provider.name}: snake violated at {(a, b, c)}: "
                    f"b_symbol {got!r} != {expected!r}"
                )
            checked += 1
    return checked


def validate_spherical(provider: _DualFusionRules, sectors: Sectors, *, atol: float = 0.0) -> int:
    """Check sphericality: left and right traces agree, ``qdim(a) == qdim(dual(a))``.

    Exact by default — every provider's quantum dimensions come from exact
    arithmetic. Needs :class:`~tenet.symmetry.PivotalData` (today a marker) and
    :class:`~tenet.symmetry.QuantumDimensionData`.

    Parameters
    ----------
    provider : FusionRules
        The provider whose pivotal structure is checked.
    sectors : tuple of Sector
        The sectors checked.
    atol : float, optional
        Absolute tolerance; ``0.0`` (exact) by default.

    Returns
    -------
    int
        The number of sectors checked.

    Raises
    ------
    ValueError
        On the first sector whose two traces differ.
    """
    checked = 0
    for a in sectors:
        left, right = provider.qdim(a), provider.qdim(provider.dual(a))  # ty: ignore[unresolved-attribute]
        if abs(left - right) > atol:
            raise ValueError(
                f"{provider.name}: not spherical at {a!r}: qdim {left!r} != dual's {right!r}"
            )
        checked += 1
    return checked


def validate_unitary(provider: FusionRules, sectors: Sectors, *, atol: float = 1e-13) -> int:
    """Check unitarity of the recoupling data: F-matrices unitary, ``|R| == 1``.

    For every ``a, b, c, d`` in ``sectors`` the matrix ``[F^{abc}_d]_{e,f}``
    over admissible inner lines must be unitary, and every ``R^{ab}_c`` must
    have unit modulus. Needs :class:`~tenet.symmetry.DaggerData` (today a
    marker) plus :class:`~tenet.symmetry.AssociatorData`;
    :class:`~tenet.symmetry.BraidingData` is checked when present.

    Parameters
    ----------
    provider : FusionRules
        The provider whose gauge is checked.
    sectors : tuple of Sector
        The budget ``a, b, c, d`` are drawn from.
    atol : float, optional
        Absolute tolerance, matching the pre-M24 test assertion.

    Returns
    -------
    int
        The number of F-matrices checked.

    Raises
    ------
    ValueError
        On the first non-unitary F-matrix or non-unimodular R.
    """
    checked = 0
    for a, b, c, d in product(sectors, repeat=4):
        es = [e for e in _channels(provider, a, b) if provider.n_symbol(e, c, d)]
        fs = [f for f in _channels(provider, b, c) if provider.n_symbol(a, f, d)]
        if not es:
            if fs:
                raise ValueError(
                    f"{provider.name}: F^{{{a!r},{b!r},{c!r}}}_{d!r} has inner lines "
                    "on one association only"
                )
            continue
        m = np.array(
            [[provider.f_symbol(a, b, c, d, e, f) for f in fs] for e in es]  # ty: ignore[unresolved-attribute]
        )
        eye = np.eye(len(es))
        if not (
            np.allclose(m @ m.conj().T, eye, atol=atol)
            and np.allclose(m.conj().T @ m, eye, atol=atol)
        ):
            raise ValueError(f"{provider.name}: F^{{{a!r},{b!r},{c!r}}}_{d!r} is not unitary")
        checked += 1
    if supports(provider, BraidingData):
        for a, b in product(sectors, repeat=2):
            for c in _channels(provider, a, b):
                r = provider.r_symbol(a, b, c)  # ty: ignore[unresolved-attribute]
                if abs(abs(r) - 1.0) > atol:
                    raise ValueError(
                        f"{provider.name}: |R^{{{a!r},{b!r}}}_{c!r}| = {abs(r)!r} != 1"
                    )
    return checked


def validate_non_degenerate_braiding(
    provider: _DualFusionRules, sectors: Sectors, *, atol: float = 1e-10
) -> int:
    """Check that the (unnormalized) S-matrix over ``sectors`` is invertible.

    ``S_ab = sum_c N^c_{dual(a), b} (theta_c / (theta_a theta_b)) qdim(c)``;
    full rank over the given sectors is the modularity criterion. Needs
    :class:`~tenet.symmetry.FusionRules`, :class:`~tenet.symmetry.TwistData`,
    :class:`~tenet.symmetry.QuantumDimensionData` and the dual label map. Only
    meaningful when ``sectors`` is the complete sector set (Fibonacci's two;
    any symmetric provider with more than one sector fails, as it must).

    Parameters
    ----------
    provider : FusionRules
        The provider whose braiding is checked.
    sectors : tuple of Sector
        The complete sector set of the (finite) theory.
    atol : float, optional
        Rank tolerance handed to ``numpy.linalg.matrix_rank``.

    Returns
    -------
    int
        The size of the S-matrix checked.

    Raises
    ------
    ValueError
        If the S-matrix is singular over ``sectors``.
    """
    theta = provider.twist  # ty: ignore[unresolved-attribute]  # TwistData, caller-gated
    s = np.array(
        [
            [
                sum(
                    provider.n_symbol(provider.dual(a), b, c)
                    * theta(c)
                    / (theta(a) * theta(b))
                    * provider.qdim(c)  # ty: ignore[unresolved-attribute]
                    for c in _channels(provider, provider.dual(a), b)
                )
                for b in sectors
            ]
            for a in sectors
        ]
    )
    if np.linalg.matrix_rank(s, tol=atol) != len(sectors):
        raise ValueError(
            f"{provider.name}: the braiding is degenerate — the S-matrix over "
            f"{sectors} has rank {np.linalg.matrix_rank(s, tol=atol)} < {len(sectors)}"
        )
    return len(sectors)


@cache
def symmetric_braiding(provider: FusionRules, sectors: Sectors) -> bool:
    """``True`` iff ``R == R**-1`` on every fusion channel over ``sectors``.

    The property ``transpose`` gates on beside
    :class:`~tenet.symmetry.BraidingData`: a symmetric braiding is what makes
    ``axes`` alone determine the braid (invariant 12). Takes the array path for
    a provider with :class:`~tenet.symmetry.RMatrixData`, so multiplicity-bearing
    providers are classified too. Cached per ``(provider, sectors)``.
    """
    matrices = provider if isinstance(provider, RMatrixData) else None
    for a, b in product(sectors, repeat=2):
        for c in _channels(provider, a, b):
            if matrices is not None:
                m = matrices.r_matrix(a, b, c) @ matrices.r_matrix(b, a, c)
                if not np.allclose(m, np.eye(m.shape[0]), atol=1e-10):
                    return False
            else:
                r = provider.r_symbol(a, b, c)  # ty: ignore[unresolved-attribute]
                if abs(r * provider.r_symbol(b, a, c) - 1.0) > 1e-12:  # ty: ignore[unresolved-attribute]
                    return False
    return True


@dataclass(frozen=True, slots=True)
class CategoricalProperties:
    """Derived properties of a provider over a sector budget — computed, never
    declared. Returned by :func:`properties`."""

    braided: bool
    symmetric: bool
    spherical: bool
    modular: bool
    unitary: bool


@cache
def properties(provider: _DualFusionRules, sectors: Sectors | None = None) -> CategoricalProperties:
    """The derived categorical properties of ``provider``, cached.

    A free function, not a provider attribute: providers stay frozen, hashable,
    array-free values with no new fields. With ``sectors=None`` the budget is
    the provider's own ``all_sectors()`` when it enumerates finitely
    (:class:`SectorEnumeration`), else the unit alone — pass explicit sectors
    for a meaningful answer on an infinite-sector provider.

    Parameters
    ----------
    provider : FusionRules
        The provider classified.
    sectors : tuple of Sector or None, optional
        The sector budget the checks run over.

    Returns
    -------
    CategoricalProperties
        ``braided`` / ``symmetric`` / ``spherical`` / ``modular`` / ``unitary``.
    """
    if sectors is None:
        sectors = (
            tuple(provider.all_sectors())
            if isinstance(provider, SectorEnumeration)
            else (provider.unit,)
        )
    braided = supports(provider, BraidingData) or supports(provider, RMatrixData)
    # hashable by provider contract; the protocol deliberately omits __hash__
    symmetric = braided and symmetric_braiding(provider, sectors)  # ty: ignore[invalid-argument-type]
    spherical = (
        supports(provider, PivotalData)
        and supports(provider, QuantumDimensionData)
        and _passes(validate_spherical, provider, sectors)
    )
    modular = (
        braided
        and spherical
        and supports(provider, TwistData)
        and _passes(validate_non_degenerate_braiding, provider, sectors)
    )
    # Simplification: for a multiplicity-bearing provider (scalar F raises) the
    # F-unitarity half is delegated racah-side and only DaggerData/AssociatorData
    # presence is consulted here; the trigger for an array path is a caller.
    unitary = supports(provider, DaggerData) and (
        not supports(provider, AssociatorData)
        or supports(provider, RMatrixData)
        or _passes(validate_unitary, provider, sectors)
    )
    return CategoricalProperties(braided, symmetric, spherical, modular, unitary)


def _passes(validator, provider, sectors) -> bool:
    try:
        validator(provider, sectors)
    except ValueError:
        return False
    return True
