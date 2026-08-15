"""``tenet.network.MPS``: the write barrier, the canonical form, and the two exits.

The write barrier is the point of the class (#112): every factorization in
``tenet.linalg`` lowers its input to a *map* first, so a rank-3 factor comes back on the
map's partition, and ``examples/dmrg.py`` used to repair that at each of two call sites
with a private ``_as_site``. Storing the factor is now the repair, and these tests are
what say so.
"""

import dmrg as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, SymmetricTensor
from tenet.network import MPS
from tenet.symmetry import SU2, SU2Sector

# A four-site SU(2) chain of spin-1/2s: the container is provider-generic even though the
# Hamiltonian is not (REPOSITORY_RULES:61). ``SU2Sector(2 j)``, so 1 is the doublet.
SU2_PHYS = GradedSpace.new(SU2, {SU2Sector(1): 1})
SU2_BONDS = [
    GradedSpace.new(SU2, {SU2Sector(0): 1}),
    GradedSpace.new(SU2, {SU2Sector(1): 1}),
    GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(2): 1}),
    GradedSpace.new(SU2, {SU2Sector(1): 1}),
    GradedSpace.new(SU2, {SU2Sector(0): 1}),
]


def u1_mps(n_sites: int = 4, seed: int = 0) -> MPS:
    return MPS.random(example.PHYS, example.bond_spaces(n_sites), seed=seed)


def right_isometry_error(t: SymmetricTensor) -> float:
    """``|| A A^dag - 1 ||`` on the left bond of a right-canonical site tensor."""
    closed = tenet.einsum("apr,Apr->aA", t, tenet.adjoint(t))
    eye = tenet.identity((t.legs[0],))
    return float(tenet.norm(tenet.subtract(closed, eye)))


# --- the write barrier --------------------------------------------------------------


def test_an_lq_factor_stores_without_a_repartition():
    """The whole reason ``_as_site`` no longer exists anywhere.

    ``lq``'s ``q`` comes back on the map partition ``((0,), (1, 2))`` -- its physical leg
    IN-dual in the domain -- and storing it must land it on ``(bond OUT, phys OUT, bond
    IN)`` with the same numbers a caller's own ``tenet.repartition`` would have produced.
    """
    psi = u1_mps()
    _, q = tenet.linalg.lq(psi[2], ((0,), (1, 2)))
    assert tuple(leg.side for leg in q.legs) != (OUT, OUT, IN)  # bent, as delivered

    psi[2] = q
    assert tuple(leg.side for leg in psi[2].legs) == (OUT, OUT, IN)
    assert psi[2].structure == tenet.repartition(q, (0, 1), (2,)).structure
    assert float(tenet.norm(tenet.subtract(psi[2], tenet.repartition(q, (0, 1), (2,))))) == 0.0


def test_the_write_barrier_refuses_the_wrong_rank():
    """Rank 2 and rank 4 are not MPS sites, and silence would be a structure mismatch."""
    psi = u1_mps()
    rank2 = tenet.einsum("apr,Apr->aA", psi[1], tenet.adjoint(psi[1]))
    rank4 = tenet.einsum("apx,xqr->apqr", psi[1], psi[2])
    for bad in (rank2, rank4):
        with pytest.raises(ValueError, match="rank 3"):
            psi[1] = bad


def test_the_write_barrier_refuses_a_non_int_index():
    """YASTN ``_mps_parent.py``:88-105: a slice is not a site."""
    psi = u1_mps()
    with pytest.raises(TypeError, match="int"):
        psi[0:2] = psi[0]


# --- the canonical form -------------------------------------------------------------


def test_canonize_sets_the_centre_it_claims():
    psi = u1_mps(6).canonize_()
    assert psi.center == 0
    assert psi.canonize_(0) is psi  # in place, returns self -- YASTN _mps_obc.py:390


def test_canonize_leaves_every_site_but_the_centre_right_isometric():
    """``A_n A_n^dag = 1`` for ``n > 0``, to 1e-12, against ``tenet.identity``."""
    psi = u1_mps(6, seed=5).canonize_()
    for n in range(1, len(psi)):
        assert right_isometry_error(psi[n]) < 1e-12, n
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)


def test_canonize_refuses_a_centre_it_does_not_implement():
    with pytest.raises(NotImplementedError):
        u1_mps(4).canonize_(3)


# --- the constructors and the exits --------------------------------------------------


def test_random_uses_the_bond_spaces_it_is_given():
    spaces = example.bond_spaces(5)
    psi = MPS.random(example.PHYS, spaces, seed=2)
    assert len(psi) == 5
    for n in range(5):
        assert psi[n].legs[0].space == spaces[n]
        assert psi[n].legs[1].space == example.PHYS
        assert psi[n].legs[2].space == spaces[n + 1]
        assert tuple(leg.side for leg in psi[n].legs) == (OUT, OUT, IN)
    assert psi.center is None  # no claim made until canonize_


def test_from_tensors_round_trips_through_the_barrier():
    psi = u1_mps(4)
    again = MPS.from_tensors(psi.sites)
    assert [t.structure for t in again.sites] == [t.structure for t in psi.sites]


def test_to_dense_matches_a_hand_rolled_einsum_chain_at_n4():
    """The oracle exit against the chain ``tests/integration/test_dmrg.py``:305 writes."""
    psi = u1_mps(4, seed=7).canonize_()
    amplitudes = np.asarray(psi[0].to_dense())
    for site in psi.sites[1:]:
        amplitudes = np.einsum("a...b,bcd->a...cd", amplitudes, np.asarray(site.to_dense()))
    expected = amplitudes[0, ..., 0]
    assert np.abs(np.asarray(psi.to_dense()) - expected).max() < 1e-13
    assert np.linalg.norm(expected) == pytest.approx(psi.norm(), abs=1e-12)


def test_norm_matches_the_dense_norm_before_any_canonization():
    """The transfer pass against the exponential expansion, on a state with no structure.

    After ``canonize_`` the norm is 1 and the check is nearly free; this one runs on a raw
    random MPS, where the two routes share nothing but the answer.
    """
    psi = u1_mps(4, seed=3)
    dense = float(np.linalg.norm(np.asarray(psi.to_dense())))
    assert psi.norm() == pytest.approx(dense, rel=1e-12)


# --- provider-generic: the same container on SU(2) legs ------------------------------


def test_the_container_is_structural_on_su2_legs():
    """The write barrier and the canonical form on SU(2), where the bend is not free.

    U(1) bending coefficients are all 1, so a U(1)-only test cannot tell a correct
    repartition from a missing one. SU(2) carries genuine recoupling and Frobenius-Schur
    data, and ``A_n A_n^dag = 1`` here is a statement about the qdim weights as much as
    about the isometry. The *Hamiltonian* stays U(1) (#112 out of scope); the container
    does not have to.
    """
    psi = MPS.random(SU2_PHYS, SU2_BONDS, seed=1)
    assert len(psi) == 4
    for n in range(4):
        assert tuple(leg.side for leg in psi[n].legs) == (OUT, OUT, IN)

    psi.canonize_()
    assert psi.center == 0
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)
    for n in range(1, len(psi)):
        assert right_isometry_error(psi[n]) < 1e-12, n

    _, q = tenet.linalg.lq(psi[2], ((0,), (1, 2)))
    psi[2] = q
    assert tuple(leg.side for leg in psi[2].legs) == (OUT, OUT, IN)
    assert psi[2].legs[1].space == SU2_PHYS
