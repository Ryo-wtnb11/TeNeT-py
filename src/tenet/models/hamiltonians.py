"""The named Hamiltonians: one function per model, the symmetry a parameter.

Every function here returns an [MPO][tenet.network.MPO] over an open chain of ``n``
sites, nearest-neighbour only, built from the sites of ``tenet.models.sites`` through
[MPO.from_terms][tenet.network.MPO.from_terms]. ``symbolic=`` is passed straight through
to that builder and decides nothing else.

A model's *grading* is part of its identity here, not a decoration: a Hamiltonian is only
expressible over a site whose symmetry its terms conserve, so the symmetry argument
selects the site and the term list together, and a grading a model has no terms for is
refused by name.
"""

import numpy as np

from tenet.leg import IN, OUT, Leg
from tenet.models.sites import spin_half, spinful_fermion, spinless_fermion
from tenet.network import MPO, local_op
from tenet.space import GradedSpace
from tenet.structure import TensorStructure
from tenet.symmetry import SU2, U1, Z2, SUNProvider, SUNSector, Z2Sector

# ``_DualFusionRules`` is the provider parameter's spelling throughout the core; the
# docstrings say ``FusionRules``, which is the public name of the same contract.
from tenet.symmetry.base import _DualFusionRules
from tenet.tensor import SymmetricTensor

__all__ = [
    "heisenberg",
    "hubbard",
    "spinless_tv",
    "sun_exchange",
    "sun_heisenberg",
    "transverse_field_ising",
    "xxz",
]

# The Z2-graded spin-1/2 site of the transverse-field Ising chain, in the ``sigma^x``
# eigenbasis ``(|+>, |->)`` -- charge order, since the Z2 charge is the spin flip and
# ``|+->`` are its eigenvectors. In that basis ``sigma^x`` is the diagonal even operator
# and ``sigma^z`` the off-diagonal odd one.
_SIGMA_X = np.diag([1.0, -1.0])
_SIGMA_Z = np.array([[0.0, 1.0], [1.0, 0.0]])


def _spin_half_only(spin: float) -> None:
    """The shipped spin site is the doublet; anything else is refused by name."""
    if spin != 0.5:
        raise ValueError(f"spin={spin!r}: the shipped spin site is spin 1/2 (spin=0.5)")


def _xxz_terms(n: int, jz: float, jxy: float) -> list:
    """``sum_i jz S^z_i S^z_{i+1} + jxy (S^+_i S^-_{i+1} + S^-_i S^+_{i+1})`` over U(1) ops."""
    ops = spin_half().ops
    op_sz, op_sp, op_sm = ops["Sz"], ops["S+"], ops["S-"]
    terms = []
    for i in range(n - 1):
        terms.append((jz, [(op_sz, i), (op_sz, i + 1)]))
        terms.append((jxy, [(op_sp, i), (op_sm, i + 1)]))
        terms.append((jxy, [(op_sm, i), (op_sp, i + 1)]))
    return terms


def heisenberg(
    n: int,
    symmetry: _DualFusionRules = U1,  # ty: ignore[invalid-parameter-default]
    *,
    J: float = 1.0,
    spin: float = 0.5,
    symbolic: bool = False,
) -> MPO:
    r"""The XXX chain $H = J \sum_i \vec{S}_i \cdot \vec{S}_{i+1}$, open boundaries.

    Parameters
    ----------
    n : int
        Chain length.
    symmetry : FusionRules, optional
        ``U1`` (the default), the charge $2S^z$, where the term list is
        $S^z S^z + \frac{1}{2}(S^+ S^- + S^- S^+)$ over the rank-3 operators of
        [spin_half][tenet.models.spin_half]; or ``SU2``, where the whole bond term is
        that site's single invariant operator ``S.S``. Any other provider is refused.
    J : float, optional
        The coupling. Positive is antiferromagnetic. Default ``1.0``. Keyword-only.
    spin : float, optional
        The site spin. Only ``0.5`` is shipped. Default ``0.5``. Keyword-only.
    symbolic : bool, optional
        Passed to [MPO.from_terms][tenet.network.MPO.from_terms]. Default ``False``.
        Keyword-only.

    Returns
    -------
    MPO
        The Hamiltonian on ``n`` sites.

    Raises
    ------
    ValueError
        If ``spin`` is not ``0.5``, or if ``symmetry`` is neither ``U1`` nor ``SU2``.

    Examples
    --------
    >>> from tenet.models import heisenberg
    >>> from tenet.symmetry import SU2
    >>> heisenberg(4).to_dense().shape
    (16, 16)
    >>> len(heisenberg(4, SU2))
    4
    """
    _spin_half_only(spin)
    if symmetry is SU2:
        op = spin_half(SU2).ops["S.S"]
        terms = [(J, [(op, (i, i + 1))]) for i in range(n - 1)]
        return MPO.from_terms(n, terms, symbolic=symbolic)
    if symmetry is U1:
        return MPO.from_terms(n, _xxz_terms(n, J, J * 0.5), symbolic=symbolic)
    raise ValueError(f"heisenberg: the shipped gradings are U1 and SU2, got {symmetry!r}")


def xxz(
    n: int,
    *,
    Delta: float = 1.0,
    J: float = 1.0,
    spin: float = 0.5,
    symbolic: bool = False,
) -> MPO:
    r"""$H = J \sum_i S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + \Delta S^z_i S^z_{i+1}$, on U(1).

    The transverse half is written as $\frac{1}{2}(S^+_i S^-_{i+1} + S^-_i S^+_{i+1})$,
    which is what the U(1) grading (charge $2S^z$) has operators for: $S^\pm$ carry charge
    $\mp 2$ each and enter paired across a bond. ``Delta = 1`` is
    [heisenberg][tenet.models.heisenberg] under the same grading, term for term.

    Parameters
    ----------
    n : int
        Chain length.
    Delta : float, optional
        The Ising anisotropy of the $S^z S^z$ term. Default ``1.0``. Keyword-only.
    J : float, optional
        The overall coupling. Positive is antiferromagnetic. Default ``1.0``.
        Keyword-only.
    spin : float, optional
        The site spin. Only ``0.5`` is shipped. Default ``0.5``. Keyword-only.
    symbolic : bool, optional
        Passed to [MPO.from_terms][tenet.network.MPO.from_terms]. Default ``False``.
        Keyword-only.

    Returns
    -------
    MPO
        The Hamiltonian on ``n`` sites, graded by ``U1``.

    Raises
    ------
    ValueError
        If ``spin`` is not ``0.5``.

    Examples
    --------
    >>> from tenet.models import xxz
    >>> xxz(4, Delta=0.5).to_dense().shape
    (16, 16)
    """
    _spin_half_only(spin)
    return MPO.from_terms(n, _xxz_terms(n, J * Delta, J * 0.5), symbolic=symbolic)


def transverse_field_ising(
    n: int, *, J: float = 1.0, g: float = 1.0, symbolic: bool = False
) -> MPO:
    r"""$H = -J\left(\sum_i \sigma^z_i \sigma^z_{i+1} + g \sum_i \sigma^x_i\right)$, on Z2.

    The Pauli matrices, not the spin operators: the critical point is $g = 1$. The site is
    graded by the Z2 spin flip, so the physical basis is the $\sigma^x$ eigenbasis
    $(\lvert +\rangle, \lvert -\rangle)$ in charge order -- $\sigma^x$ is the diagonal
    even operator there and $\sigma^z$ the off-diagonal odd one, and it is $\sigma^z$
    that the grading forbids alone and admits in pairs.

    Parameters
    ----------
    n : int
        Chain length.
    J : float, optional
        The coupling, multiplying both terms. Default ``1.0``. Keyword-only.
    g : float, optional
        The transverse field, in units of ``J``. Default ``1.0``, the critical point.
        Keyword-only.
    symbolic : bool, optional
        Passed to [MPO.from_terms][tenet.network.MPO.from_terms]. Default ``False``.
        Keyword-only.

    Returns
    -------
    MPO
        The Hamiltonian on ``n`` sites, graded by ``Z2``.

    Examples
    --------
    >>> from tenet.models import transverse_field_ising
    >>> transverse_field_ising(4, g=0.5).to_dense().shape
    (16, 16)
    """
    phys = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})  # ty: ignore[invalid-argument-type]
    op_x = local_op(_SIGMA_X, phys=phys, charge=Z2Sector(0))
    op_z = local_op(_SIGMA_Z, phys=phys, charge=Z2Sector(1))
    terms = [(-J, [(op_z, i), (op_z, i + 1)]) for i in range(n - 1)]
    terms += [(-J * g, [(op_x, i)]) for i in range(n)]
    return MPO.from_terms(n, terms, symbolic=symbolic)


def hubbard(n: int, *, t: float = 1.0, U: float = 4.0, symbolic: bool = False) -> MPO:
    r"""$H = -t \sum_{i\sigma} (c^\dagger_{i\sigma} c_{i+1\sigma} + h.c.) + U \sum_i n_{i\uparrow} n_{i\downarrow}$.

    On [spinful_fermion][tenet.models.spinful_fermion], graded by ``fZ2``: there is no
    Jordan-Wigner operator in the terms, because the string is the braiding an odd MPO
    bond pays when it crosses a physical line. Both hopping directions are written out --
    $c^\dagger_i c_{i+1}$ and $c^\dagger_{i+1} c_i$, spin by spin, up before down -- and
    the on-site repulsion uses the site's pre-multiplied ``n_up n_dn``.

    Parameters
    ----------
    n : int
        Chain length.
    t : float, optional
        The hopping. It enters as $-t$, so ``t > 0`` is the usual sign. Default ``1.0``.
        Keyword-only.
    U : float, optional
        The on-site repulsion, positive for repulsive. Default ``4.0``. Keyword-only.
    symbolic : bool, optional
        Passed to [MPO.from_terms][tenet.network.MPO.from_terms]. Default ``False``.
        Keyword-only.

    Returns
    -------
    MPO
        The Hamiltonian on ``n`` sites, graded by ``fZ2``.

    Examples
    --------
    >>> from tenet.models import hubbard
    >>> hubbard(3, U=8.0).to_dense().shape
    (64, 64)
    """  # noqa: E501
    ops = spinful_fermion().ops
    terms = []
    for i in range(n - 1):
        for flavour in ("up", "dn"):
            cd, c = ops[f"c+_{flavour}"], ops[f"c_{flavour}"]
            terms.append((-t, [(cd, i), (c, i + 1)]))
            terms.append((-t, [(cd, i + 1), (c, i)]))
    terms += [(U, [(ops["n_up n_dn"], i)]) for i in range(n)]
    return MPO.from_terms(n, terms, symbolic=symbolic)


def spinless_tv(n: int, *, t: float = 1.0, V: float = 1.0, symbolic: bool = False) -> MPO:
    r"""$H = -t \sum_i (c^\dagger_i c_{i+1} + h.c.) + V \sum_i n_i n_{i+1}$.

    On [spinless_fermion][tenet.models.spinless_fermion], graded by ``fZ2``, with the
    same convention as [hubbard][tenet.models.hubbard]: both hopping directions written
    out, and no Jordan-Wigner operator in the terms.

    Parameters
    ----------
    n : int
        Chain length.
    t : float, optional
        The hopping, entering as $-t$. Default ``1.0``. Keyword-only.
    V : float, optional
        The nearest-neighbour repulsion, positive for repulsive. Default ``1.0``.
        Keyword-only.
    symbolic : bool, optional
        Passed to [MPO.from_terms][tenet.network.MPO.from_terms]. Default ``False``.
        Keyword-only.

    Returns
    -------
    MPO
        The Hamiltonian on ``n`` sites, graded by ``fZ2``.

    Examples
    --------
    >>> from tenet.models import spinless_tv
    >>> spinless_tv(4, V=2.0).to_dense().shape
    (16, 16)
    """
    ops = spinless_fermion().ops
    cd, c, op_n = ops["c+"], ops["c"], ops["n"]
    terms = []
    for i in range(n - 1):
        terms.append((-t, [(cd, i), (c, i + 1)]))
        terms.append((-t, [(cd, i + 1), (c, i)]))
        terms.append((V, [(op_n, i), (op_n, i + 1)]))
    return MPO.from_terms(n, terms, symbolic=symbolic)


def sun_exchange(N: int) -> SymmetricTensor:
    r"""The two-site exchange $P$ on a pair of SU($N$) fundamentals, as a rank-4 operator.

    $\mathbf{N} \otimes \mathbf{N}$ is the symmetric part (Dynkin label
    $(2, 0, \ldots)$) plus the antisymmetric one ($(0, 1, 0, \ldots)$, the singlet at
    $N = 2$), and $P$ is $+1$ on the first and $-1$ on the second. That is one block per
    coupled sector, so the operator is written with
    [SymmetricTensor.from_blocks][tenet.SymmetricTensor.from_blocks] and no
    Clebsch-Gordan array is spelled out. The legs are two ``OUT`` (ket) then two ``IN``
    (bra), the ordering [MPO.from_terms][tenet.network.MPO.from_terms] and
    [expectation_2site][tenet.network.expectation_2site] both read a term through.

    Parameters
    ----------
    N : int
        The number of colours, ``N >= 2``.

    Returns
    -------
    SymmetricTensor
        $P$, on ``(phys, phys, phys*, phys*)`` over one fundamental multiplet.

    Raises
    ------
    ValueError
        If ``N < 2``.

    Examples
    --------
    >>> from tenet.models import sun_exchange
    >>> sun_exchange(3).to_dense().shape
    (3, 3, 3, 3)
    """
    if N < 2:
        raise ValueError(f"sun_exchange: N >= 2, got {N}")
    provider = SUNProvider(N)
    phys = GradedSpace.new(provider, {SUNSector((1,) + (0,) * (N - 2)): 1})  # ty: ignore[invalid-argument-type]
    symmetric = SUNSector((2,) + (0,) * (N - 2))
    legs = (Leg(phys, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(phys, IN))
    structure = TensorStructure(legs)
    return SymmetricTensor.from_blocks(
        legs,
        {
            key: np.full(
                structure.block_shape(key),
                1.0 if key.output_tree.coupled == symmetric else -1.0,
            )
            for key in structure.block_order
        },
    )


def sun_heisenberg(n: int, N: int, *, J: float = 1.0, symbolic: bool = False) -> MPO:
    r"""$H = J \sum_i P_{i,i+1}$ on a chain of SU($N$) fundamentals, open boundaries.

    $P$ is [sun_exchange][tenet.models.sun_exchange], the permutation of two neighbouring
    fundamentals, and each bond is one invariant two-site term.
    At $N = 2$ it is the spin-1/2 chain up to a constant, $P = 2\,\vec{S}\cdot\vec{S} +
    \frac{1}{2}$ per bond.

    Parameters
    ----------
    n : int
        Chain length.
    N : int
        The number of colours, ``N >= 2``.
    J : float, optional
        The coupling. Positive is antiferromagnetic. Default ``1.0``. Keyword-only.
    symbolic : bool, optional
        Passed to [MPO.from_terms][tenet.network.MPO.from_terms]. Default ``False``.
        Keyword-only.

    Returns
    -------
    MPO
        The Hamiltonian on ``n`` sites, graded by SU($N$).

    Raises
    ------
    ValueError
        If ``N < 2``.

    Examples
    --------
    >>> from tenet.models import sun_heisenberg
    >>> len(sun_heisenberg(4, 3))
    4
    """
    op = sun_exchange(N)
    return MPO.from_terms(n, [(J, [(op, (i, i + 1))]) for i in range(n - 1)], symbolic=symbolic)
