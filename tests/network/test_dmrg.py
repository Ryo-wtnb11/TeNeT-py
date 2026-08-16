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
from tenet.symmetry import U1, U1Sector


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
