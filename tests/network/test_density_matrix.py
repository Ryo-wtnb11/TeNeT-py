"""M61 Stage C: the density-matrix decimation and the perturbative noise.

``tests/network/test_dmrg.py`` owns the SVD split and the wavefunction noise and is
untouched. This file owns the second decimation: that it *is* the first one when nothing
is folded in, that the family-resolved perturbation is the operator's own action, and
that a converged energy does not move when the mixer is on -- a mixer that changes the
answer is a bug, not a feature.
"""

import heisenberg_walkthrough as example  # noqa: E402  (see conftest.py)
import pytest

import tenet
from tenet.network import MPO, MPS, Env, Sweep, dmrg_, lanczos, sweep_
from tenet.network.common import spectrum
from tenet.network.dmrg import _perturbations, _split_dm
from tenet.symmetry import U1Sector

from . import test_dmrg as dmrg_test
from . import test_hubbard as hubbard_test
from . import test_mpo as mpo_test


def _bond(psi, h, n):
    """``(env, aa)`` at bond ``n`` of a right-canonical ``psi``: the sweep's own state."""
    psi.canonize_(0)
    env = Env(psi, h).setup_()
    for k in range(n):
        env.update_(k, to="last")
    aa = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
    _, aa = lanczos(lambda v: env.heff2(n, v), aa)
    return env, aa


def _spin_case():
    psi = MPS.random(example.PHYS, example.bond_spaces(10), seed=0)
    h = example.mpo_from_terms(10, symbolic=True)
    dmrg_(psi, h, chi=16, max_sweeps=3)
    return psi, h


def _fermionic_case():
    h = MPO.from_terms(6, hubbard_test._rank3_terms(6, 4.0), symbolic=True)
    psi = hubbard_test._state(6, 6, 6, seed=3)
    dmrg_(psi, h, chi=16, max_sweeps=3)
    return psi, h


# --- the split, with nothing folded in ------------------------------------------------


@pytest.mark.parametrize("case", (_spin_case, _fermionic_case), ids=("u1-spin", "fz2-hubbard"))
@pytest.mark.parametrize("chi", (4, 8, 16))
@pytest.mark.parametrize("forward", (True, False))
def test_the_noiseless_density_matrix_split_is_the_svd_split(case, chi, forward):
    """Same decomposition, so: same Schmidt values, same discarded weight, same state.

    ``rho``'s spectrum is ``aa``'s squared, and ``svd_truncated``'s selection is
    monotone in the bare value, so the *kept bond space* is identical by construction --
    the numbers then agree to the accuracy squaring leaves, which is the square root of
    the SVD's. That floor is why a noiseless sweep keeps the SVD split.
    """
    psi, h = case()
    _, aa = _bond(psi, h, 2)
    u, s, vh = tenet.linalg.svd_truncated(aa, ((0, 1), (2, 3)), max_bond=chi, cutoff=1e-14)
    left, right, s2 = _split_dm(aa, forward, chi=chi, cutoff=1e-14)

    assert s2.structure.legs[0].space == s.structure.legs[0].space  # the same bond space
    ref, got = spectrum(s), spectrum(s2)
    assert got == pytest.approx(ref, abs=1e-11)

    carrier = right if forward else left
    ref_dw = 1.0 - float(tenet.norm(s) / tenet.norm(aa)) ** 2
    got_dw = 1.0 - float(tenet.norm(carrier) / tenet.norm(aa)) ** 2
    assert got_dw == pytest.approx(ref_dw, abs=1e-12)

    # The truncated state itself, reassembled the way the sweep's next merge reads it.
    psi[2], psi[3] = left, right
    mine = tenet.einsum("apx,xqr->apqr", psi[2], psi[3])
    psi[2], psi[3] = u, tenet.compose(s, vh)
    theirs = tenet.einsum("apx,xqr->apqr", psi[2], psi[3])
    assert float(tenet.norm(tenet.subtract(mine, theirs))) < 1e-7


# --- the perturbation -----------------------------------------------------------------


@pytest.mark.parametrize("case", (_spin_case, _fermionic_case), ids=("u1-spin", "fz2-hubbard"))
def test_the_term_families_sum_to_the_matvec(case):
    """``heff2_families`` is a read of ``heff2``, not a second operator."""
    psi, h = case()
    env, aa = _bond(psi, h, 2)
    parts = env.heff2_families(2, aa)
    assert len(parts) > 1  # actually resolved, not one lump
    total = parts[0]
    for part in parts[1:]:
        total = tenet.add(total, part)
    assert tenet.allclose(total, env.heff2(2, aa))


def test_the_perturbation_carries_the_noise_as_its_squared_norm():
    """block2's scaling (``moving_environment.hpp``:3698-3713): ``sum_k ||p_k||^2 == noise``.

    That is what makes ``noise`` dimensionless against a unit-norm ``aa``, and what
    transfers block2's 1e-4..1e-5 range.
    """
    psi, h = _spin_case()
    env, aa = _bond(psi, h, 2)
    assert float(tenet.norm(aa)) == pytest.approx(1.0, abs=1e-12)
    for noise in (1e-4, 1e-5):
        vectors = _perturbations(env, 2, aa, noise)
        assert vectors
        total = sum(float(tenet.norm(p)) ** 2 for p in vectors)
        assert total == pytest.approx(noise, rel=1e-10)


def test_an_mpo_with_no_description_still_gets_one_perturbation_vector():
    """The compatibility entry has no families, so the vector is ``H_eff aa`` itself."""
    psi, h = _spin_case()
    bare = MPO(h.sites)
    env, aa = _bond(psi, bare, 2)
    parts = env.heff2_families(2, aa)
    assert len(parts) == 1
    assert tenet.allclose(parts[0], env.heff2(2, aa))


def test_perturbative_noise_fills_a_sector_the_eigensolver_left_empty():
    """The property the current noise's docstring documents, for the new mixer.

    The Neel product state's bonds are ``D=1``. One noiseless sweep at ``chi=8`` grows
    them only where the eigensolver's own Krylov space reaches; the perturbation adds the
    directions the *Hamiltonian couples to*, so the bond comes back strictly wider. It is
    a superset, never a different set: every direction here is one ``H`` can populate.
    """
    h = example.mpo_from_terms(6, symbolic=True)
    spaces = {}
    for noise in (0.0, 1e-2):
        psi = MPS.product(example.PHYS, [U1Sector(1), U1Sector(-1)] * 3).canonize_()
        env = Env(psi, h).setup_()
        sweep_(psi, h, env, {}, chi=8, cutoff=1e-14, noise=noise, noise_type="perturbative")
        spaces[noise] = psi[3].legs[0].space
    plain, mixed = spaces[0.0], spaces[1e-2]
    assert dict(plain.sectors).keys() <= dict(mixed.sectors).keys()
    assert mixed.dim > plain.dim


def test_a_perturbative_sweep_leaves_the_mps_normalized():
    """The carrier factor is divided by its own norm, so Pythagoras still holds."""
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0).canonize_()
    h = example.mpo_from_terms(6, symbolic=True)
    env = Env(psi, h).setup_()
    _, dw = sweep_(psi, h, env, {}, chi=8, cutoff=1e-14, noise=1e-3, noise_type="perturbative")
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= dw < 1.0


# --- selection ------------------------------------------------------------------------


def test_an_unknown_noise_type_is_refused():
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0).canonize_()
    h = example.mpo_from_terms(6, symbolic=True)
    env = Env(psi, h).setup_()
    with pytest.raises(ValueError, match="noise_type"):
        sweep_(psi, h, env, {}, chi=8, cutoff=1e-14, noise=1e-4, noise_type="density_matrix")


def test_the_schedule_carries_the_choice_and_a_noiseless_entry_ignores_it():
    """``noise_type`` on a ``noise=0.0`` entry changes nothing: the SVD split runs."""
    energies = []
    for kind in ("wavefunction", "perturbative"):
        psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=0)
        out = dmrg_(psi, example.mpo(6), schedule=[Sweep(16, noise_type=kind)] * 3, max_sweeps=3)
        energies.append(out.energy)
    assert energies[0] == energies[1]


# --- the oracles, with the mixer on ---------------------------------------------------


def _ramp(chi):
    return [Sweep(chi, noise=1e-4, noise_type="perturbative")] * 3 + [Sweep(chi)] * 3


def test_the_hubbard_ground_state_is_unchanged_with_the_mixer_on():
    """N=4 against even-parity ED, at U/t in {0, 4}: a mixer must not move the answer."""
    for u in (0.0, 4.0):
        h = MPO.from_terms(4, hubbard_test._rank3_terms(4, u), symbolic=True)
        # max_sweeps raised for the same M62 reason as the u-sweep oracle: the budget
        # was calibrated against the twisted pairing's accidentally fast direction.
        out = dmrg_(hubbard_test._state(4, 8, 8, seed=1), h, schedule=_ramp(16), max_sweeps=24)
        assert out.energy == pytest.approx(hubbard_test._ed_even(4, u), abs=1e-10), f"U={u}"


def test_the_heisenberg_ground_state_is_unchanged_with_the_mixer_on():
    psi = MPS.random(example.PHYS, example.bond_spaces(8), seed=0)
    mixed = dmrg_(
        psi, example.mpo_from_terms(8, symbolic=True), schedule=_ramp(16), max_sweeps=6
    ).energy
    plain = dmrg_(
        MPS.random(example.PHYS, example.bond_spaces(8), seed=0),
        example.mpo_from_terms(8, symbolic=True),
        chi=16,
    )
    assert mixed == pytest.approx(plain.energy, abs=1e-10)


def test_the_su2_ground_state_is_unchanged_with_the_mixer_on():
    """Non-Abelian: the perturbation rides the same families, so it touches no coefficient.

    A wrong recoupling in the fold would surface either as a structure error, a broken
    norm, or an energy that misses the U(1) run -- all three are asserted.
    """
    h = mpo_test.su2_heisenberg(6)
    psi = MPS.random(mpo_test.SU2_PHYS, dmrg_test.su2_bond_spaces(6), seed=0)
    out = dmrg_(psi, h, schedule=_ramp(16), max_sweeps=6)
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)
    u1 = dmrg_(MPS.random(example.PHYS, example.bond_spaces(6), seed=0), example.mpo(6), chi=16)
    assert out.energy == pytest.approx(u1.energy, abs=1e-9)
