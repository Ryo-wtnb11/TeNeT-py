"""The standard local sites: a physical space and its operator set, built by ``local_op``.

Every operator here goes through [local_op][tenet.network.local_op], so it arrives with
that function's refusals attached -- a matrix that is not what the grading says it is
raises inside ``from_dense`` at its default relative tolerance rather than being
projected. Nothing in this module calls ``SymmetricTensor.from_dense`` itself, and
``tests/models/test_sites.py`` asserts that mechanically.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tenet.network import local_op
from tenet.space import GradedSpace
from tenet.symmetry import (
    SU2,
    U1,
    FusionRules,
    FZ2Sector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)
from tenet.tensor import SymmetricTensor

# The spin-1/2 matrices in the ``2 S^z`` order (|down>, |up>), which is the canonical
# sector order of every grading below that resolves the doublet, and of the single
# SU(2) j=1/2 multiplet as well.
_SZ = np.diag([-0.5, 0.5])
_SP = np.array([[0.0, 0.0], [1.0, 0.0]])  # |down> -> |up>
_SS = np.kron(_SZ, _SZ) + (np.kron(_SP, _SP.T) + np.kron(_SP.T, _SP)) / 2

# The one-mode annihilation operator on (|0>, |1>).
_A = np.array([[0.0, 1.0], [0.0, 0.0]])

# The spinful d=4 site in *graded* order (|0>, |ud>, |u>, |d>) -- even sector first --
# with modes ordered up before down, so |ud> = c+_up c+_dn |0>. ``c_up`` carries no
# intra-site sign (the up mode is first) and ``c_dn`` pays the Jordan-Wigner Z on it.
# This is the basis M21/#147 pinned; ``tests/network/test_hubbard.py`` is its oracle.
_C_UP = np.zeros((4, 4))
_C_UP[0, 2] = 1.0  # c_up |u> = |0>
_C_UP[3, 1] = 1.0  # c_up |ud> = +|d>
_C_DN = np.zeros((4, 4))
_C_DN[0, 3] = 1.0  # c_dn |d> = |0>
_C_DN[2, 1] = -1.0  # c_dn |ud> = -|u>


@dataclass(frozen=True)
class Site:
    """One lattice site: its physical space, its term operators, and their matrices.

    Parameters
    ----------
    phys : GradedSpace
        The physical space. Every operator in ``ops`` lives on it, and it is what
        [MPS.product][tenet.network.MPS.product] / [MPS.random][tenet.network.MPS.random]
        take.
    ops : Mapping[str, SymmetricTensor]
        Name to **term** operator, in whichever of
        [local_op][tenet.network.local_op]'s two forms the grading admits: rank 3 with a
        charge leg where the symmetry is Abelian, rank 2*k* invariant where it is not.
        Where it is rank 3 this mapping is exactly what
        [MPO.from_arrays][tenet.network.MPO.from_arrays] calls ``ops``; where it is not,
        ``from_arrays`` cannot express the term at all and
        [MPO.from_terms][tenet.network.MPO.from_terms] is the door.
    matrices : Mapping[str, numpy.ndarray]
        The dense matrix behind each operator, under the same key, **plus** any matrix
        the site knows whose grading admits no entry in ``ops`` (the spin-1/2 ``S.S``
        under ``U1``). It is here for the forms the term API does not build -- rank-2
        measurement operators for
        [expectation_1site][tenet.network.expectation_1site], invariant *k*-site ones
        for [expectation_2site][tenet.network.expectation_2site], and dense oracles --
        so that reaching for one is not a reason to write a spin matrix out again.

    Examples
    --------
    >>> from tenet.models import spin_half
    >>> site = spin_half()
    >>> site.phys.dim
    2
    >>> sorted(site.ops)
    ['S+', 'S-', 'Sz']
    >>> site.ops["S+"].ndim  # the charge-leg form
    3
    >>> sorted(site.matrices)  # S.S has no rank-3 form, so it is a matrix here
    ['S+', 'S-', 'S.S', 'Sz']
    """

    phys: GradedSpace
    ops: Mapping[str, SymmetricTensor]
    matrices: Mapping[str, np.ndarray]


def _build(
    phys: GradedSpace,
    table: Mapping[str, tuple[np.ndarray, Any]],
    extra: Mapping[str, np.ndarray] = {},  # noqa: B006  (read-only default)
) -> Site:
    """A ``Site`` from ``name -> (matrix, charge)``; ``charge=None`` is the invariant form.

    ``extra`` names matrices this grading has no ``ops`` entry for -- they reach
    ``matrices`` and nothing else.
    """
    ops = {name: local_op(m, phys=phys, charge=q) for name, (m, q) in table.items()}
    return Site(phys, ops, {name: m for name, (m, _) in table.items()} | dict(extra))


def spin_half(symmetry: FusionRules = U1) -> Site:
    """The spin-1/2 site, graded by ``U1`` (``2 S^z``) or by ``SU2``.

    Parameters
    ----------
    symmetry : FusionRules, optional
        ``U1`` (the default), whose charge is ``2 S^z`` so the doublet is
        ``{-1, +1}``, or ``SU2``, one ``j = 1/2`` multiplet. Any other provider is
        refused.

    Returns
    -------
    Site
        Under ``U1``: ``Sz``, ``S+``, ``S-`` as rank-3 charge-leg operators, with the
        invariant two-site ``S.S`` reachable as a matrix. Under ``SU2``: ``S.S`` alone,
        as the rank-4 invariant operator (see Notes).

    Raises
    ------
    ValueError
        If ``symmetry`` is neither ``U1`` nor ``SU2``.

    Examples
    --------
    >>> from tenet.models import spin_half
    >>> from tenet.symmetry import SU2
    >>> sorted(spin_half().ops)
    ['S+', 'S-', 'Sz']
    >>> sorted(spin_half(SU2).ops)
    ['S.S']
    >>> spin_half(SU2).ops["S.S"].ndim  # rank 2k, k = 2: one whole term
    4

    Notes
    -----
    **Under SU(2) the set is ``{S.S}`` and that is the whole answer.** ``S+`` is not an
    SU(2) operator and it is not omitted by preference: the charge-leg form needs a
    ``D=1`` sector on the emitted leg, and the only leg a spin-1 tensor operator could
    emit onto is the ``j=1`` multiplet, whose *dense* dimension is 3 -- so
    ``local_op(sz, phys=phys, charge=SU2Sector(2))`` raises on the shape, and no
    irreducible tensor operator of nonzero rank exists in this API to hand back. What
    exists is M13b's invariant *k*-site form, and ``S.S`` is it: one whole Heisenberg
    bond term, whose coupling lives inside its own blocks, which
    [MPO.from_terms][tenet.network.MPO.from_terms] splits with an SVD. The same object
    is invariant under ``U1`` too, where ``S.S`` is what
    [expectation_2site][tenet.network.expectation_2site] measures a bond energy with;
    it sits in ``matrices`` there rather than in ``ops``, because a ``U1`` term list is
    written from the rank-3 three and a rank-4 entry in ``ops`` would be one
    ``from_arrays`` refuses.
    """
    if symmetry is U1:
        return _build(
            GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1}),
            {
                "Sz": (_SZ, U1Sector(0)),
                "S+": (_SP, U1Sector(-2)),
                "S-": (_SP.T, U1Sector(2)),
            },
            {"S.S": _SS},
        )
    if symmetry is SU2:
        return _build(GradedSpace.new(SU2, {SU2Sector(1): 1}), {"S.S": (_SS, None)})
    raise ValueError(
        f"spin_half: the shipped gradings are U1 (charge 2 S^z) and SU2, got {symmetry!r}"
    )


def spinless_fermion() -> Site:
    """The spinless fermion site on ``fZ2``: ``{|0>, |1>}``, even sector first.

    Returns
    -------
    Site
        ``c``, ``c+`` (odd, charge ``FZ2Sector(1)``) and ``n`` (even).

    Examples
    --------
    >>> from tenet.models import spinless_fermion
    >>> site = spinless_fermion()
    >>> sorted(site.ops)
    ['c', 'c+', 'n']
    >>> site.matrices["n"].diagonal().tolist()
    [0.0, 1.0]

    Notes
    -----
    There is no ``JW`` operator to ship and no place to put one: the Jordan-Wigner
    string is the ``fZ2`` braiding an odd MPO bond pays when it crosses a physical line
    (M21/#147), so a term list over these operators is already the fermionic
    Hamiltonian.
    """
    return _build(
        GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1}),
        {
            "c": (_A, FZ2Sector(1)),
            "c+": (_A.T, FZ2Sector(1)),
            "n": (_A.T @ _A, FZ2Sector(0)),
        },
    )


def spinful_fermion() -> Site:
    """The spinful fermion site on ``fZ2``: the ``d=4`` basis ``(|0>, |ud>, |u>, |d>)``.

    Returns
    -------
    Site
        ``c_up``, ``c+_up``, ``c_dn``, ``c+_dn`` (odd), and ``n_up``, ``n_dn``, ``n``,
        ``n_up n_dn`` (even). The last is the Hubbard ``U`` operator, pre-multiplied
        because [MPO.from_terms][tenet.network.MPO.from_terms] places one operator per
        site; under [MPO.from_arrays][tenet.network.MPO.from_arrays] the *same spelling*
        is a two-name block expression on two coincident site indices, which its merge
        multiplies into this very operator.

    Examples
    --------
    >>> from tenet.models import spinful_fermion
    >>> site = spinful_fermion()
    >>> site.phys.dim
    4
    >>> site.matrices["n"].diagonal().tolist()
    [0.0, 2.0, 1.0, 1.0]

    Notes
    -----
    The basis is *graded* -- the even sector ``{|0>, |ud>}`` before the odd
    ``{|u>, |d>}`` -- because a dense array over a ``GradedSpace`` is laid out sector by
    sector in canonical order. Modes run up before down, ``|ud> = c+_up c+_dn |0>``, so
    ``c_up`` carries no intra-site sign and ``c_dn`` pays the Jordan-Wigner ``Z`` on the
    up mode (``c_dn |ud> = -|u>``). Inter-site strings are the braiding's business, not
    a matrix's; this is the convention M21/#147 pinned against a dense oracle.
    """
    n_up, n_dn = _C_UP.T @ _C_UP, _C_DN.T @ _C_DN
    return _build(
        GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2}),
        {
            "c_up": (_C_UP, FZ2Sector(1)),
            "c+_up": (_C_UP.T, FZ2Sector(1)),
            "c_dn": (_C_DN, FZ2Sector(1)),
            "c+_dn": (_C_DN.T, FZ2Sector(1)),
            "n_up": (n_up, FZ2Sector(0)),
            "n_dn": (n_dn, FZ2Sector(0)),
            "n": (n_up + n_dn, FZ2Sector(0)),
            "n_up n_dn": (n_up @ n_dn, FZ2Sector(0)),
        },
    )


def hard_core_boson(symmetry: FusionRules = U1) -> Site:
    """The hard-core boson site, ``{|0>, |1>}``, graded by ``U1`` (the number) or ungraded.

    Parameters
    ----------
    symmetry : FusionRules, optional
        ``U1`` (the default), whose charge is the occupation ``n``, or ``Trivial`` for
        the ungraded ``d=2`` site. Any other provider is refused.

    Returns
    -------
    Site
        ``b``, ``b+`` and ``n``.

    Raises
    ------
    ValueError
        If ``symmetry`` is neither ``U1`` nor ``Trivial``.

    Examples
    --------
    >>> from tenet.models import hard_core_boson
    >>> from tenet.symmetry import Trivial
    >>> sorted(hard_core_boson().ops)
    ['b', 'b+', 'n']
    >>> hard_core_boson(Trivial).phys.dim
    2

    Notes
    -----
    The matrices are the spin-1/2 ladder in disguise (``b = S^-``, ``n = S^z + 1/2``);
    the site exists separately because the *grading* is the difference that matters at a
    call site -- ``U1`` here counts particles from 0, where ``spin_half`` counts
    ``2 S^z`` from ``-1``, and the two are not interchangeable in a term list. Under
    ``Trivial`` nothing is conserved, so every operator still arrives rank 3 on a
    ``D=1`` trivial leg and ``MPO.from_terms`` works unchanged.
    """
    if symmetry is U1:
        phys = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
        # q(p_out) + q(charge) = q(p_in): ``b`` lowers the occupation, so it emits +1.
        return _build(
            phys,
            {"b": (_A, U1Sector(1)), "b+": (_A.T, U1Sector(-1)), "n": (_A.T @ _A, U1Sector(0))},
        )
    if symmetry is Trivial:
        phys = GradedSpace.new(Trivial, {TrivialSector(): 2})
        unit = TrivialSector()
        return _build(phys, {"b": (_A, unit), "b+": (_A.T, unit), "n": (_A.T @ _A, unit)})
    raise ValueError(
        f"hard_core_boson: the shipped gradings are U1 (the occupation) and Trivial, "
        f"got {symmetry!r}"
    )
