"""``tenet.network.dmrg``: the Krylov step against ``eigvalsh``, and one short sweep.

Unit scale only -- N=6, chi=8. The N=12/20/32 energies and the ED oracle are
``tests/integration/test_dmrg.py``'s job and stay there.
"""

import dmrg as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

import tenet
from tenet import GradedSpace
from tenet.network import MPO, MPS, Env, Sweep, dmrg_, lanczos, local_op, sweep_
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

    ``examples/toy_codes/dmrg.py::bond_spaces``'s SU(2) twin, and the same #112 statement: which
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


# --- M14: the schedule ---------------------------------------------------------------


def test_schedule_and_flat_kwargs_are_exclusive():
    """Silently letting one spelling win is how a run reports a ``chi`` it did not use."""
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    with pytest.raises(ValueError, match="schedule.*chi"):
        dmrg_(psi, example.mpo(6), schedule=[Sweep(8)], chi=8)
    with pytest.raises(ValueError, match="schedule.*cutoff"):
        dmrg_(psi, example.mpo(6), schedule=[Sweep(8)], cutoff=1e-12)
    with pytest.raises(ValueError, match="empty"):
        dmrg_(psi, example.mpo(6), schedule=[])


def test_the_flat_kwargs_are_a_one_entry_schedule_bit_for_bit():
    """``chi=8`` and ``schedule=[Sweep(chi=8)]`` are the same run: ``==``, no tolerance."""
    flat_psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    flat = dmrg_(flat_psi, example.mpo(6), chi=8, max_sweeps=3)
    sched_psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    sched = dmrg_(sched_psi, example.mpo(6), schedule=[Sweep(chi=8)], max_sweeps=3)
    assert flat.energy == sched.energy
    assert flat.history == sched.history
    assert np.array_equal(np.asarray(flat_psi.to_dense()), np.asarray(sched_psi.to_dense()))
    assert len(flat.schedule) == flat.sweeps == len(flat.history)


def test_the_last_schedule_entry_repeats_until_max_sweeps():
    """A 2-entry schedule with ``max_sweeps=5`` realizes ``[s0, s1, s1, s1, s1]``."""
    s0, s1 = Sweep(4), Sweep(8)
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    out = dmrg_(psi, example.mpo(6), schedule=[s0, s1], schmidt_tol=0.0, max_sweeps=5)
    assert out.schedule == [s0, s1, s1, s1, s1]
    assert len(out.schedule) == out.sweeps == len(out.history) == 5


def test_a_noisy_final_entry_blocks_the_convergence_exit():
    """The block2 guard: no exit while the current sweep's noise is nonzero.

    Both loop tolerances are set absurdly loose, so the *only* thing that can keep the
    loop running is the guard -- the mirror of
    ``test_the_schmidt_criterion_is_what_stops_the_loop``.
    """
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    out = dmrg_(
        psi,
        example.mpo(6),
        schedule=[Sweep(8, noise=1e-3)],
        energy_tol=1e3,
        schmidt_tol=1e3,
        max_sweeps=4,
    )
    assert out.sweeps == 4
    assert out.denergy < 1e3  # the tolerances were met; only the noise guard held


def test_the_exit_waits_for_the_schedules_last_entry():
    """``[Sweep(4)]*2 + [Sweep(16)]`` may not exit inside the chi=4 phase."""
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    schedule = [Sweep(4)] * 2 + [Sweep(16)]
    out = dmrg_(psi, example.mpo(6), schedule=schedule, energy_tol=1e3, schmidt_tol=1e3)
    assert out.sweeps == 3  # the first sweep allowed to exit is the last entry's
    assert out.schedule == schedule


# --- M14: noise ----------------------------------------------------------------------


def test_noise_is_reproducible_by_seed_and_varies_with_it():
    schedule = [Sweep(8, noise=1e-3)] * 2
    energies = []
    for seed in (0, 0, 1):
        psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
        energies.append(
            dmrg_(psi, example.mpo(6), schedule=schedule, max_sweeps=2, seed=seed).energy
        )
    assert energies[0] == energies[1]
    assert energies[0] != energies[2]


def test_a_noisy_sweep_perturbs_and_a_noisy_schedule_recovers():
    """Noise is not variational, and it is transient: the zero-noise tail undoes it."""
    sweep_energies = []
    for noise in (0.0, 1e-2):
        psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0).canonize_()
        env = Env(psi, example.mpo(6)).setup_()
        energy, _ = sweep_(psi, example.mpo(6), env, {}, chi=8, cutoff=1e-14, noise=noise)
        sweep_energies.append(energy)
    assert sweep_energies[1] > sweep_energies[0]

    finals = []
    for schedule in ([Sweep(8)] * 8, [Sweep(8, noise=1e-2)] * 3 + [Sweep(8)] * 5):
        psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
        out = dmrg_(psi, example.mpo(6), schedule=schedule, max_sweeps=8)
        finals.append(out.energy)
    assert finals[1] <= finals[0] + 1e-12


def test_noise_opens_a_sector_the_eigensolver_cannot():
    """The criterion the whole feature exists for, and not an energy comparison.

    ``H = sum_i S^z_i S^z_{i+1}`` is diagonal in the product basis and its MPO bond
    carries only charge 0, so ``heff2`` preserves the coupled-sector content of the
    two-site tensor: a Neel product state is stuck in its own D=1 bonds forever at
    ``noise=0``, which is exactly the local minimum a symmetric DMRG falls into. The
    wavefunction perturbation fills every structurally allowed coupled sector, so one
    noisy sweep opens the bond.
    """
    _, sz, *_ = example._spin_half()
    op_sz = local_op(sz, phys=example.PHYS, charge=U1Sector(0))
    h = MPO.from_terms(6, [(1.0, [(op_sz, i), (op_sz, i + 1)]) for i in range(5)])
    bonds = {}
    for noise in (0.0, 1e-2):
        psi = MPS.product(example.PHYS, [U1Sector(1), U1Sector(-1)] * 3).canonize_()
        env = Env(psi, h).setup_()
        sweep_(psi, h, env, {}, chi=8, cutoff=1e-14, noise=noise, seed=7)
        bonds[noise] = psi[3].legs[0].space
    assert U1Sector(-1) not in bonds[0.0]  # the eigensolver alone cannot create it
    assert U1Sector(-1) in bonds[1e-2]


def test_a_noisy_sweep_leaves_the_mps_normalized():
    """Renormalized after the perturbation, so the Pythagoras discarded weight holds."""
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0).canonize_()
    env = Env(psi, example.mpo(6)).setup_()
    sweep_(psi, example.mpo(6), env, {}, chi=8, cutoff=1e-14, noise=1e-2)
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)


# --- M14: callback and restart -------------------------------------------------------


def test_callback_sees_each_finished_sweep_and_its_exceptions_propagate():
    seen = []
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    out = dmrg_(
        psi,
        example.mpo(6),
        chi=8,
        max_sweeps=4,
        callback=lambda o: seen.append((o.sweeps, len(o.history))),
    )
    assert seen == [(k, k) for k in range(1, out.sweeps + 1)]

    def boom(_):
        raise RuntimeError("from the callback")

    with pytest.raises(RuntimeError, match="from the callback"):
        dmrg_(
            MPS.random(example.PHYS, example.bond_spaces(6), seed=0),
            example.mpo(6),
            chi=8,
            max_sweeps=2,
            callback=boom,
        )


def test_a_saved_run_re_entered_with_a_schedule_slice_matches_uninterrupted(tmp_path):
    """Restart is ``MPS.save``/``load`` plus ``schedule=schedule[k:]`` -- no new argument."""
    h = example.mpo(6)
    schedule = [Sweep(4)] * 2 + [Sweep(8)] * 4
    full_psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    full = dmrg_(full_psi, h, schedule=schedule, energy_tol=0.0, max_sweeps=6)

    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    first = dmrg_(psi, h, schedule=schedule[:2], energy_tol=0.0, max_sweeps=2)
    psi.save(tmp_path / "checkpoint")
    loaded = MPS.load(tmp_path / "checkpoint")
    assert Env(loaded, h).measure() == pytest.approx(first.energy, abs=1e-12)
    resumed = dmrg_(loaded, h, schedule=schedule[2:], energy_tol=0.0, max_sweeps=4)
    assert resumed.energy == pytest.approx(full.energy, abs=1e-10)


# --- M14: the product seed and the SU(2) schedule ------------------------------------


def test_dmrg_from_a_neel_product_state_reaches_the_random_seed_energy():
    """``MPS.product`` end to end: the D=1 seed grows into the same ground state."""
    neel = MPS.product(example.PHYS, [U1Sector(1), U1Sector(-1)] * 3)
    seeded = dmrg_(neel, example.mpo(6), chi=16).energy
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
    assert abs(seeded - dmrg_(psi, example.mpo(6), chi=16).energy) < 1e-10


def test_an_su2_schedule_run_with_noise_completes():
    """#135's Heisenberg MPO under a noisy ramp: the noise path touches no coefficient.

    The proof is the run itself -- a valid MPS whose norm is 1 -- because a wrong
    recoupling in the perturbation would surface as a structure error or a broken norm.
    """
    psi = MPS.random(mpo_test.SU2_PHYS, su2_bond_spaces(6), seed=0)
    schedule = [Sweep(16, noise=1e-4)] * 2 + [Sweep(32)] * 2
    out = dmrg_(psi, mpo_test.su2_heisenberg(6), schedule=schedule, max_sweeps=4)
    assert out.schedule == schedule
    assert len(out.schedule) == out.sweeps == len(out.history) == 4
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)


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


def test_fermionic_dmrg_reaches_the_even_parity_ground_energy():
    """N=4 spinless hopping on fZ2 legs: ``dmrg_`` against even-parity ED to 1e-10.

    #147's gate-1 report measured this run at -1.4142136 against the ED value
    -2.2360680 -- the cap-direction Koszul signs of the environment contractions.
    #160's composition rule (with its explicitly bent wires) is what closes the gap.
    ``cutoff=None`` takes the prepared per-bond path through the Jordan block table;
    the default cutoff takes the dense ``W`` path; both must land on ED.
    """
    ed = -(5.0**0.5)  # even-parity ground energy of the open 4-site hopping chain
    from .test_env import _fermionic_chain, _fermionic_state  # noqa: PLC0415

    for cutoff in (None, 1e-13):
        h = _fermionic_chain(4, cutoff=cutoff)
        out = dmrg_(_fermionic_state(4, seed=3), h, chi=16, cutoff=1e-14)
        assert out.energy == pytest.approx(ed, abs=1e-10)


def test_fermionic_dmrg_on_the_interacting_chain_matches_ed():
    """Gate 3 (#147): N=6 spinless hopping plus ``V n_m n_{m+1}`` against even-parity ED.

    N=6 rather than the issue's N=8: the physics (an interacting fermionic chain whose
    ground energy no free-fermion argument gives) is identical, and N=8 measured ~28 s
    under #160's audit, which does not fit the suite budget; the ED oracle is a 64-dim
    ``eigh`` masked to even parity either way. One route (the compressed dense path);
    the prepared path's agreement is pinned per-bond in
    ``tests/integration/test_dmrg_prepared.py``.
    """
    n, v = 6, 1.5
    from .test_env import _fermionic_chain, _fermionic_state  # noqa: PLC0415

    hop = sum(mpo_test._jw_hop(n, m, m + 1) for m in range(n - 1))
    num = [np.diag([(k >> (n - 1 - s)) & 1 for k in range(2**n)]) for s in range(n)]
    dense = hop + v * sum(num[m] @ num[m + 1] for m in range(n - 1))
    even = [k for k in range(2**n) if bin(k).count("1") % 2 == 0]
    ed = float(np.linalg.eigvalsh(dense[np.ix_(even, even)]).min())  # even-parity block
    h = _fermionic_chain(n, cutoff=1e-13, interaction=v)
    out = dmrg_(_fermionic_state(n, seed=3), h, chi=16, cutoff=1e-14)
    assert out.energy == pytest.approx(ed, abs=1e-10)
