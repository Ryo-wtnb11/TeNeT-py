"""``tenet.network.Env``: the directed-bond dict, its invalidation, and ``measure``.

The rebuild check is the only test in the repository that catches a stale environment: a
missed ``clear_`` after a site changed gives an energy that is *plausible and wrong*,
which is the worst failure mode a DMRG has.
"""

import dmrg as example  # noqa: E402  (see conftest.py)
import pytest

import tenet
from tenet.network import MPS, Env, dmrg, sweep


def state(n_sites: int = 6, seed: int = 1) -> MPS:
    return MPS.random(example.PHYS, example.bond_spaces(n_sites), seed=seed).canonize_()


def test_setup_builds_every_right_directed_bond_and_nothing_else():
    """``F[(n, n-1)]`` for every ``n`` down to 1, plus the two boundary entries."""
    psi, h = state(6), example.mpo(6)
    env = Env(psi, h).setup_()
    assert set(env.F) == {(-1, 0), (6, 5)} | {(n, n - 1) for n in range(1, 6)}
    assert env.F[-1, 0].ndim == 3 and env.F[6, 5].ndim == 3


def test_swept_environments_match_a_rebuild_from_scratch():
    """One sweep, then every surviving entry must equal a fresh ``setup_`` to 1e-12.

    ``tests/integration/test_dmrg.py``:201-220 promoted. The left-directed entries must be
    *gone* after the return leg of the sweep invalidated them.
    """
    psi, h = state(6, seed=2), example.mpo(6)
    env = Env(psi, h).setup_()
    sweep(psi, h, env, {}, chi=16, cutoff=1e-14)

    rebuilt = Env(psi, h).setup_().F
    assert set(env.F) == set(rebuilt) | {(-1, 0)}
    for key, expected in rebuilt.items():
        assert env.F[key].structure == expected.structure, key
        assert float(tenet.norm(tenet.subtract(env.F[key], expected))) < 1e-12, key


def test_clear_pops_both_directed_bonds_of_every_site():
    """YASTN ``clear_site_`` (``_env.py``:127-133): both directions, before any rewrite."""
    psi, h = state(6), example.mpo(6)
    env = Env(psi, h).setup_()
    env.update_(0, to="last")  # so site 0 has an entry in each direction
    env.update_(1, to="last")
    assert (0, 1) in env.F and (1, 0) in env.F

    env.clear_(1)
    assert (1, 0) not in env.F and (1, 2) not in env.F
    assert (0, 1) in env.F  # a neighbour's entry is not collateral damage
    env.clear_(4, 5)
    assert not {(4, 3), (4, 5), (5, 4), (5, 6)} & set(env.F)
    env.clear_(1)  # popping twice is not an error; a missing key is the normal case


def test_a_missed_update_is_a_key_error_not_a_wrong_number():
    """The point of clearing *before* writing: the failure mode is loud."""
    psi, h = state(6), example.mpo(6)
    env = Env(psi, h).setup_()
    env.clear_(3)
    with pytest.raises(KeyError):
        env.heff2(3, tenet.einsum("apx,xqr->apqr", psi[3], psi[4]))


def test_measure_reproduces_the_dmrg_energy():
    """``Env(psi, h).measure()`` against ``out.energy`` to 1e-12, at N=6, chi=16.

    The first check in this repository that a converged energy survives being measured
    *independently of the eigensolver that produced it*: ``out.energy`` is ``lanczos``'s
    Rayleigh quotient at the last bond of the last sweep, and this is ``<psi|H|psi>`` on a
    private left-to-right pass. ``tests/integration/test_dmrg.py`` runs the same claim at
    N=12 against the ED oracle.
    """
    psi, h = MPS.random(example.PHYS, example.bond_spaces(6), seed=0), example.mpo(6)
    out = dmrg(psi, h, chi=16)
    assert Env(out.psi, h).measure() == pytest.approx(out.energy, abs=1e-12)
