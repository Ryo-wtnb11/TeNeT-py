"""M18 — the usage lane runs in CI, at the same defaults a reader runs.

Each file in ``examples/`` calls ``tenet.network`` as a user would, and its defaults
*are* the tested sizes: ``main()`` here is exactly ``python examples/<file>.py``. The
oracles are independent of the examples — ``tests/integration/test_dmrg.py``'s recorded
N=20 ED energy, the U(1) run ``su2_heisenberg`` computes in the same process, and
``onsager(beta)``. The teaching lane keeps its own CI execution unchanged
(``tests/integration/test_dmrg.py``, ``test_ctmrg.py``, ``test_vmc.py``).
"""

import pathlib
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import heisenberg  # noqa: E402
import ising2d  # noqa: E402
import su2_heisenberg  # noqa: E402

# tests/integration/test_dmrg.py's recorded N=20 open-boundary ED energy (#110).
E_N20 = -8.682473334398956


@pytest.fixture(scope="module")
def heisenberg_run():
    return heisenberg.main()


@pytest.fixture(scope="module")
def su2_run():
    return su2_heisenberg.main()


@pytest.fixture(scope="module")
def ising_run():
    return ising2d.main()


def test_heisenberg_reaches_the_recorded_ed_energy(heisenberg_run):
    out, _ = heisenberg_run
    assert abs(out.energy - E_N20) < 1e-10


def test_heisenberg_profile_sums_to_the_energy(heisenberg_run):
    out, profile = heisenberg_run
    assert abs(sum(profile) - out.energy) < 1e-10


def test_heisenberg_sector_is_structural(heisenberg_run):
    from tenet.network import expectation_1site, local_op

    out, _ = heisenberg_run
    op_sz = local_op(heisenberg.SZ, phys=heisenberg.PHYS)
    assert all(abs(expectation_1site(out.psi, op_sz, n)) < 1e-10 for n in range(len(out.psi)))


def test_su2_agrees_with_the_u1_run_it_computes(su2_run):
    su2, u1, _ = su2_run
    assert abs(su2.energy - u1.energy) < 1e-10


def test_su2_mid_bond_is_multiplet_compressed(su2_run):
    _, _, mid = su2_run
    assert mid.reduced_dim < mid.dim


def test_ising2d_matches_onsager_off_criticality(ising_run):
    results, _ = ising_run
    for beta in (0.3, 0.5):
        _, rel = results[beta]
        assert rel < 1e-12


def test_ising2d_ordered_spectrum_is_pairwise_degenerate(ising_run):
    _, corner = ising_run
    for even, odd in zip(corner[0::2], corner[1::2], strict=False):
        assert abs(even - odd) < 1e-10


def test_lane_basenames_are_disjoint():
    """Both lanes land on sys.path as top-level modules; a shared basename would shadow."""
    flat = {p.name for p in EXAMPLES.glob("*.py")}
    toy = {p.name for p in (EXAMPLES / "toy_codes").glob("*.py")}
    assert not flat & toy
