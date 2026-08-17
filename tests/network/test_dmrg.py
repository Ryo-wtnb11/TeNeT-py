"""``tenet.network.dmrg``: the Krylov step against ``eigvalsh``, and one short sweep.

Unit scale only -- N=6, chi=8. The N=12/20/32 energies and the ED oracle are
``tests/integration/test_dmrg.py``'s job and stay there.
"""

import dmrg as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

import tenet
from tenet import GradedSpace
from tenet.network import MPS, Env, dmrg_, lanczos, sweep_
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

from . import test_mpo as mpo_test  # the SU(2) Hamiltonian, built once and shared


def test_lanczos_finds_the_lowest_eigenvalue():
    """Against ``np.linalg.eigvalsh`` on a small dense Hermitian, reseeded per iteration.

    The "vector" is a rank-2 tensor on ``(X OUT, BOUNDARY IN)``, so the trivial boundary
    leg restricts it to ``X``'s charge-0 block -- and the eigenvalue to beat is that
    block's. ``ncv=3`` is one Krylov space and no restart, so a single call is not
    expected to converge; the loop is what a sweep does anyway.
    """
    space = GradedSpace.new(U1, {U1Sector(-2): 2, U1Sector(0): 3, U1Sector(2): 2})
    rng = np.random.default_rng(0)
    dense = np.zeros((7, 7))
    for lo, hi in ((0, 2), (2, 5), (5, 7)):
        block = rng.standard_normal((hi - lo, hi - lo))
        dense[lo:hi, lo:hi] = block + block.T
    legs = (tenet.Leg(space, tenet.OUT), tenet.Leg(space, tenet.IN))
    operator = tenet.SymmetricTensor.from_dense(dense, legs)
    expected = np.linalg.eigvalsh(dense[2:5, 2:5])[0]

    v = tenet.SymmetricTensor.random(
        (tenet.Leg(space, tenet.OUT), tenet.Leg(example.BOUNDARY, tenet.IN)), seed=3
    )
    value = None
    for _ in range(10):
        value, v = lanczos(lambda x: tenet.einsum("ab,bc->ac", operator, x), v)
    assert value == pytest.approx(expected, abs=1e-10)


def test_lanczos_leaves_its_output_normalized():
    """A Krylov step returns a unit vector; the sweep's Pythagoras claim depends on it."""
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=4).canonize_()
    h = example.mpo(6)
    env = Env(psi, h).setup_()
    aa = tenet.einsum("apx,xqr->apqr", psi[0], psi[1])
    _, out = lanczos(lambda v: env.heff2(0, v), aa)
    assert float(tenet.norm(out)) == pytest.approx(1.0, abs=1e-12)


def test_one_sweep_lowers_the_energy_monotonically_at_n6():
    """N=6, chi=8: three sweeps, each energy at or below the last. It is variational."""
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    out = dmrg_(psi, example.mpo(6), chi=8, max_sweeps=3)
    energies = [e for e, *_ in out.history]
    assert len(energies) == out.sweeps
    for previous, current in zip(energies, energies[1:], strict=False):
        assert current <= previous + 1e-12, out.history
    assert out.psi is psi  # the driver sweeps the state it was handed, in place


def test_sweep_reports_the_discarded_weight_it_actually_discarded():
    """chi=2 must discard; chi=64 at N=6 cannot, because 2**3 = 8 <= 64."""
    schmidt: dict[int, list[float]] = {}
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0).canonize_()
    h = example.mpo(6)
    env = Env(psi, h).setup_()
    for _ in range(3):  # let the bond grow past chi=2 before judging the truncation
        _, tight = sweep_(psi, h, env, schmidt, chi=2, cutoff=1e-14)
    assert tight > 0.0

    loose_psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    out = dmrg_(loose_psi, h, chi=64)
    assert out.max_discarded_weight < 1e-12


def test_the_schmidt_criterion_is_what_stops_the_loop():
    """Both YASTN criteria, not just the energy (``_dmrg.py``:180-195).

    A run given an impossible Schmidt tolerance must burn every sweep it is allowed even
    though its energy stopped moving long before.
    """
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    out = dmrg_(psi, example.mpo(6), chi=8, schmidt_tol=0.0, max_sweeps=4)
    assert out.sweeps == 4
    assert out.denergy < 1e-12  # the energy criterion alone would have exited earlier


def test_both_mpo_builders_give_the_same_ground_state_energy_at_n6():
    """N=6, chi=16 against ``mpo_from_terms(6)`` reproduces the ``mpo(6)`` run to 1e-10.

    The end-to-end half of #133: a *derived* MPO bond space is not merely the right
    dimension, it carries the right operator all the way through ``Env``, ``lanczos`` and
    ``sweep_``. Both runs are also checked against the exact open-boundary ground state,
    so agreeing on a wrong number is not an available way to pass.
    """
    energies = []
    for h in (example.mpo(6), example.mpo_from_terms(6)):
        psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
        energies.append(dmrg_(psi, h, chi=16).energy)
    assert abs(energies[0] - energies[1]) < 1e-10
    exact = np.linalg.eigvalsh(np.asarray(example.mpo(6).to_dense()))[0]
    assert abs(energies[0] - exact) < 1e-10


def su2_bond_spaces(n_sites):
    """The reachable total-spin sectors of a spin-1/2 chain, degeneracy 1 each.

    ``examples/dmrg.py::bond_spaces``'s SU(2) twin, and the same #112 statement: which
    spaces are reachable is physics and stays out of the library. ``2 j`` on bond ``i``
    runs over ``i % 2, i % 2 + 2, ..., min(i, n - i)`` -- ``i`` spin-1/2s fix the parity,
    and the total spin must still be able to fall back to a singlet by the last site.
    """
    return [
        GradedSpace.new(SU2, {SU2Sector(j): 1 for j in range(i % 2, min(i, n_sites - i) + 1, 2)})
        for i in range(n_sites + 1)
    ]


def test_the_first_su2_dmrg_reproduces_the_u1_ground_state_at_n6():
    """**The first time this repository has run DMRG on a non-Abelian symmetry.**

    Nothing in ``Env``, ``heff2``, ``lanczos`` or ``sweep_`` changed for it -- #135's audit
    said none of them reads a provider and this run is how that claim is falsifiable. The
    Hamiltonian is one ``S.S`` array and a list comprehension, with the ``W``'s recoupling
    derived by ``svd_truncated`` rather than written down (#110 deferred exactly that).

    The state is a total **singlet**: the MPS boundary carries ``SU2Sector(0)``, which is
    the whole "target sector" statement, and the energy is the U(1) run's to 1e-10 as well
    as the exact open-chain value's.
    """
    psi = MPS.random(mpo_test.SU2_PHYS, su2_bond_spaces(6), seed=0)
    assert psi[0].legs[0].space == GradedSpace.new(SU2, {SU2Sector(0): 1})
    energy = dmrg_(psi, mpo_test.su2_heisenberg(6), chi=16).energy

    u1 = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    assert abs(energy - dmrg_(u1, example.mpo(6), chi=16).energy) < 1e-10
    exact = np.linalg.eigvalsh(np.asarray(example.mpo(6).to_dense()))[0]
    assert abs(energy - exact) < 1e-10


def test_the_su2_mps_bond_holds_the_same_state_in_a_third_of_the_multiplets():
    """``max_bond`` bounds the **dense** bond, and for SU(2) that is not the reduced one.

    ``svd_truncated``'s docstring (``linalg.py``:750-757) says the two differ and that it
    will surprise people; this is the first thing in the repository to measure it. At
    ``chi=16``, N=6, both symmetries converge to the same energy, and the SU(2) mid-chain
    bond carries 3 multiplets where U(1) carries 8 states -- the compression is on the MPS
    side, not on the MPO's, whose dense bond is 5 either way.
    """
    psi = MPS.random(mpo_test.SU2_PHYS, su2_bond_spaces(6), seed=0)
    su2 = dmrg_(psi, mpo_test.su2_heisenberg(6), chi=16)
    u1 = dmrg_(MPS.random(example.PHYS, example.bond_spaces(6), seed=0), example.mpo(6), chi=16)
    graded, plain = su2.psi[3].legs[0].space, u1.psi[3].legs[0].space
    assert (graded.reduced_dim, graded.dim) == (3, 8)
    assert (plain.reduced_dim, plain.dim) == (8, 8)
    assert abs(su2.energy - u1.energy) < 1e-10
