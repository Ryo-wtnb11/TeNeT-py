"""M61 Stage C's oracle gate: the perturbative mixer must not move a converged answer.

The claim a mixer makes is about the *path*, never about the fixed point. So every
ground-state oracle this repository has is run once more with the mixer on: the dense
``S^z_tot = 0`` restriction of the Heisenberg chain (which is also the MPSKit.jl number),
the MPSKit.jl Hubbard fixture, and the six Hamiltonians ``test_dmrg_prepared.py`` uses to
populate every term family. That last set is the one that matters most here, because the
perturbation is built *from* those families: a family resolved wrongly could not leave
the mixed and unmixed runs on the same number.

``tests/network/test_density_matrix.py`` owns the split itself; this file owns only the
statement that the answer is unchanged.

The three sibling test modules are loaded by explicit path rather than by name: there is
a ``test_dmrg.py`` in both ``tests/integration`` and ``tests/network`` and neither
directory is a package, so a plain import would resolve to whichever ``sys.path`` entry
won.
"""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples" / "toy_codes"))

import dmrg as example  # noqa: E402

from tenet import GradedSpace  # noqa: E402
from tenet.network import MPO, MPS, Sweep, dmrg_, local_op  # noqa: E402
from tenet.symmetry import SU2, FZ2Sector, SU2Sector, U1Sector, fZ2  # noqa: E402

TESTS = pathlib.Path(__file__).parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chain = _load("_oracle_chain", TESTS / "integration" / "test_dmrg.py")
prepared = _load("_oracle_prepared", TESTS / "integration" / "test_dmrg_prepared.py")
hubbard = _load("_oracle_hubbard", TESTS / "network" / "test_hubbard.py")


def ramp(chi):
    """Three mixed sweeps then three cooled ones -- the shape a real schedule has."""
    return [Sweep(chi, noise=1e-4, noise_type="perturbative")] * 3 + [Sweep(chi)] * 3


def heisenberg(n_sites):
    return MPO.from_terms(
        n_sites, prepared._pair_terms([(i, i + 1) for i in range(n_sites - 1)]), cutoff=None
    )


def test_the_ed_and_mpskit_heisenberg_energy_survives_the_mixer():
    """N=12 against the dense ``S^z_tot = 0`` restriction, which is also MPSKit's number."""
    exact = float(np.linalg.eigvalsh(chain.heisenberg_sz0(12))[0])
    assert exact == pytest.approx(chain.E_OBC_12, abs=1e-12)  # the oracle is the oracle
    psi = MPS.random(example.PHYS, example.bond_spaces(12), seed=0)
    out = dmrg_(psi, heisenberg(12), schedule=ramp(64), max_sweeps=6)
    assert out.energy == pytest.approx(chain.E_OBC_12, abs=1e-9)


def test_the_neel_seed_reaches_the_same_energy_through_the_mixer():
    """A ``D=1`` seed is where a mixer is load-bearing; the endpoint is still the oracle's."""
    neel = MPS.product(example.PHYS, [U1Sector(1), U1Sector(-1)] * 6)
    out = dmrg_(neel, heisenberg(12), schedule=ramp(64), max_sweeps=8)
    assert out.energy == pytest.approx(chain.E_OBC_12, abs=1e-9)


def test_the_mpskit_hubbard_fixture_survives_the_mixer():
    """The independent-library oracle for the fermionic model, at N=4 across U/t."""
    fix = json.loads(hubbard.FIXTURE.read_text())
    for u_key, entry in fix["N4"].items():
        u = float(u_key[1:])
        h = MPO.from_terms(4, hubbard._rank3_terms(4, u))
        out = dmrg_(hubbard._state(4, 8, 8, seed=2), h, schedule=ramp(16), max_sweeps=40)
        assert out.energy == pytest.approx(entry["energy"], abs=2e-9), f"U={u}"


def _su2_model():
    phys = GradedSpace.new(SU2, {SU2Sector(1): 1})
    _, sz, sp, sm = example._spin_half()
    ss = local_op(np.kron(sz, sz) + (np.kron(sp, sm) + np.kron(sm, sp)) / 2, phys=phys)
    tri = GradedSpace.new(SU2, {SU2Sector(0): 1})
    mid = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2, SU2Sector(2): 1})
    h = MPO.from_terms(8, [(1.0, [(ss, (i, i + 1))]) for i in range(7)], cutoff=None)
    return h, phys, [tri] + [mid] * 7 + [tri]


def _fermionic_model():
    phys = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    op_cd = local_op(a.T, phys=phys, charge=FZ2Sector(1))
    op_c = local_op(a, phys=phys, charge=FZ2Sector(1))
    op_n = local_op(np.diag([0.0, 1.0]), phys=phys, charge=FZ2Sector(0))
    terms = [(0.8, [(op_n, 2)])]
    for i, j in [(m, m + 1) for m in range(4)] + [(1, 3)]:
        terms += [(1.0, [(op_cd, i), (op_c, j)]), (1.0, [(op_cd, j), (op_c, i)])]
    unit = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
    both = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})
    return MPO.from_terms(5, terms, cutoff=None), phys, [unit] + [both] * 4 + [unit]


def _mixed_model():
    op_sz, op_sp, op_sm = prepared._ops()
    _, sz, _, _ = example._spin_half()
    terms = [
        (0.5, [(op_sp, 1), (op_sm, 4)]),
        (0.7, [(local_op(np.kron(sz, sz), phys=example.PHYS), (0, 3))]),
        (2.5, [(op_sz, 0), (op_sz, 2), (op_sz, 5)]),
        (-1.3, [(op_sz, 3)]),
    ]
    return MPO.from_terms(6, terms, cutoff=None), example.PHYS, example.bond_spaces(6)


def _power_law_model():
    pairs = [(i, j) for i in range(10) for j in range(i + 1, min(i + 5, 10))]
    h = MPO.from_terms(
        10, prepared._pair_terms(pairs, coeff=lambda i, j: (j - i) ** -2.0), cutoff=None
    )
    return h, example.PHYS, example.bond_spaces(10)


def _cylinder_model():
    h = MPO.from_terms(12, prepared._pair_terms(prepared._cylinder_pairs(12, 6)), cutoff=None)
    return h, example.PHYS, example.bond_spaces(12)


def _chain_model():
    return heisenberg(8), example.PHYS, example.bond_spaces(8)


#: ``test_dmrg_prepared.py``'s six, rebuilt here from its own helpers where it has them.
SIX = {
    "nn-heisenberg": _chain_model,
    "r4-power-law": _power_law_model,
    "ly6-cylinder": _cylinder_model,
    "mixed-rank3-and-rank2k": _mixed_model,
    "su2-heisenberg": _su2_model,
    "fermionic-chain": _fermionic_model,
}


@pytest.mark.parametrize("name", sorted(SIX))
def test_the_six_term_family_models_land_on_the_same_energy_with_the_mixer_on(name):
    """Every field pattern the perturbation is resolved over, mixed against unmixed.

    ``test_dmrg_prepared.py`` has already pinned each of these Hamiltonians against the
    dense two-site operator at every bond, so the comparison here is the mixer's alone.
    """
    h, phys, bonds = SIX[name]()
    plain = dmrg_(MPS.random(phys, bonds, seed=5), h, chi=64, max_sweeps=20)
    mixed = dmrg_(MPS.random(phys, bonds, seed=5), h, schedule=ramp(64), max_sweeps=20)
    # The comparison is only about the mixer where the runs are *converged*: two states
    # that each threw away a percent of themselves differ because they were truncated,
    # not because one was mixed. chi=64 puts every model here at machine-zero discarded
    # weight, and this asserts that rather than assuming it -- the Ly=6 cylinder at
    # chi=32 sits at 1e-2 instead, which is where the M61 Stage C entry records what a
    # truncation-limited comparison actually shows.
    assert plain.max_discarded_weight < 1e-12
    assert mixed.max_discarded_weight < 1e-12
    assert mixed.energy == pytest.approx(plain.energy, abs=1e-9)
