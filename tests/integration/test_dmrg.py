"""Issue #110 — finite-chain two-site DMRG against exact diagonalization.

``examples/toy_codes/dmrg.py`` variationally minimizes a U(1)-graded MPS against the open-boundary
spin-1/2 Heisenberg chain, and this module judges it against the *many-body ground state*
rather than against a self-consistency check. Like ``tests/integration/test_ctmrg.py``
(#102) and ``test_vmc.py`` (#69) it adds nothing to ``src/tenet`` and nothing to the
example: it imports ``examples/toy_codes/dmrg.py`` and drives it, so the example cannot rot.

**The N=12 oracle is computed here, not trusted.** The Heisenberg Hamiltonian restricted
to ``S^z_tot = 0`` at N=12 is 924 x 924, which ``np.linalg.eigvalsh`` diagonalizes in
milliseconds with no ``scipy``; :func:`test_the_computed_oracle_is_the_open_chain` pins
that computed value at ``-5.142090632840532`` before anything is judged against it. The
number ``-5.387390917445203`` that a first pass at this issue reached for is the
**periodic** N=12 energy: an open-boundary MPS cannot reproduce it, and a test that asked
it to would fail by 0.245 and look like a DMRG bug. The distinction is recorded here so
nobody re-derives the confusion.

The N=20 value ``-8.682473334398956`` is a hard-coded literal *with its provenance* --
measured for #110 by ``scipy.sparse.linalg.eigsh(k=1, which='SA', tol=0)`` on the same
Hamiltonian -- because ``C(20, 10) = 184 756`` is past what a test should diagonalize.
That is the recorded-baseline pattern ``test_ctmrg.py``:555 already uses.

**The third oracle class (#152) is an independent library**: MPSKit.jl, vendored as
``tests/fixtures/mpskit_heisenberg.json`` and regenerable by its committed generator
``tools/oracles/heisenberg_mpskit.jl`` (Julia is never run in CI or by this module --
the fixture is read by ``json.loads`` and nothing more). The fixture's own validation
entries reproduce ``E_OBC_12``/``E_OBC_20`` to better than 1e-12, which is what licenses
its N=32 numbers as an oracle above ED's ceiling: the ``main_runs`` N=32 chi=64 energy
agrees with MPSKit's chi=256 value to a measured ``2e-10``, and all 31 bond energies
``<S_i . S_{i+1}>`` agree with MPSKit's profile to a measured ``6e-10``.

**Runtime: the budget is 60 s** and the module measures **~44 s** (43.5 s after #183 put
the algorithm back in the example, against 44 s before it: the arithmetic is the same
arithmetic, and the page's committed output did not move a digit), dominated by the
``main_runs`` fixture (~31 s: the N=32 chi=64 run inside ``dmrg.main()`` is the single
most expensive thing here, and it is deliberately *shared* with the MPSKit cross-checks
and the N=12 oracle rather than run three times) and
:func:`test_n20_matches_the_recorded_literal` (5.6 s). Unlike
``test_ctmrg.py``, whose ~47 s floor is one-off XLA compilation, **there is no compile
floor here**: nothing in this module is traced, jitted or differentiated, so the whole
number is arithmetic and it scales with ``N`` and ``chi`` and nothing else. If the N=32
run ever exceeds ~30 s on its own, #110 says the fix is to drop it to N=20 rather than to
raise the budget.

**Two modules, one Hamiltonian (#183).** ``examples/toy_codes/dmrg.py`` writes the
algorithm out on the tensor layer and owns every assertion about *the algorithm* here --
the ground states, the environment cache, the Krylov step, the variational checks.
``examples/heisenberg_walkthrough.py`` is the same physics through ``tenet.network`` and
owns the assertions about *the library* -- the graded ``MPO.from_w``, the schedule
arithmetic, the ``MPS.product`` seed, the SU(2) twin. The recorded literals are shared and
unchanged; the walkthrough's own ``main()`` runs in ``tests/test_examples.py`` with the
rest of the usage lane, so nothing here pays for it twice.

x64 is enabled process-globally in ``tests/conftest.py``; every tolerance here depends on
it.
"""

import contextlib
import io
import json
import pathlib
import sys
import traceback

import numpy as np
import pytest
from helpers import check_example_page

import tenet
from tenet import GradedSpace, network
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

_EXAMPLES = pathlib.Path(__file__).parents[2] / "examples"
sys.path.insert(0, str(_EXAMPLES / "toy_codes"))
sys.path.insert(0, str(_EXAMPLES))
import dmrg  # noqa: E402
import heisenberg_walkthrough as walkthrough  # noqa: E402

# The open-boundary ground-state energies measured for #110 by sparse Lanczos on
# H = Sum_i S_i . S_{i+1}, J = 1, S = 1/2. -5.387390917445203 is the **periodic** N=12
# energy and is not a target for an OBC MPS anywhere in this file.
E_OBC_12 = -5.142090632840532
E_OBC_20 = -8.682473334398956

# The MPSKit.jl oracle (#152): vendored, provenance-pinned, regenerated only by running
# its committed generator. Read here with stdlib json; Julia is never a test dependency.
_MPSKIT = json.loads(
    (pathlib.Path(__file__).parents[1] / "fixtures" / "mpskit_heisenberg.json").read_text()
)

# Simplification: a plain dict of completed runs, as ``test_ctmrg._ENVS`` is. A DMRG run is the
# expensive thing in this module and nothing here mutates one.
_RUNS: dict = {}


def run(n_sites: int, chi: int, **kwargs):
    key = (n_sites, chi, tuple(sorted(kwargs.items())))
    if key not in _RUNS:
        _RUNS[key] = dmrg.dmrg(n_sites, chi, **kwargs)
    return _RUNS[key]


# ``main_runs``'s stdout, for the docs example page (#164).
_MAIN_STDOUT: dict[str, str] = {}


@pytest.fixture(scope="module")
def main_runs():
    """``dmrg.main()``'s own ``(N=12 chi=64, N=32 chi=64)`` pair, run exactly once."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = dmrg.main()
    _MAIN_STDOUT["text"] = buf.getvalue()
    return result


# --- the oracle --------------------------------------------------------------------


def heisenberg_sz0(n_sites: int) -> np.ndarray:
    """The open Heisenberg chain restricted to ``S^z_tot = 0``, dense. No ``scipy``.

    Basis: the ``C(n, n/2)`` bitstrings with ``n/2`` ones (bit ``b`` set = spin up on site
    ``b``), in ascending integer order. ``S^z_i S^z_{i+1}`` is ``+-1/4`` on the diagonal
    and the exchange ``(S^+ S^- + S^- S^+)/2`` flips an antialigned pair with amplitude
    ``1/2``. At N=12 that is 924 x 924 and ``eigvalsh`` costs milliseconds.
    """
    states = [s for s in range(1 << n_sites) if bin(s).count("1") == n_sites // 2]
    index = {s: i for i, s in enumerate(states)}
    h = np.zeros((len(states), len(states)))
    for i, s in enumerate(states):
        for b in range(n_sites - 1):
            up, right = (s >> b) & 1, (s >> (b + 1)) & 1
            if up == right:
                h[i, i] += 0.25
            else:
                h[i, i] -= 0.25
                h[index[s ^ (1 << b) ^ (1 << (b + 1))], i] += 0.5
    return h


def test_the_computed_oracle_is_the_open_chain():
    """The 924 x 924 ``S^z_tot = 0`` block reproduces the recorded OBC value to 1e-12.

    Nothing else in this module compares DMRG to a literal at N=12; it compares to *this*,
    and this is checked first. The periodic value ``-5.387390917445203`` differs by 0.245
    and is the number an OBC test must not reach for.
    """
    h = heisenberg_sz0(12)
    assert h.shape == (924, 924)
    assert np.linalg.eigvalsh(h)[0] == pytest.approx(E_OBC_12, abs=1e-12)
    assert abs(E_OBC_12 - (-5.387390917445203)) > 0.24  # OBC is not PBC


# --- the MPO, checked as an operator before any ground state is computed ------------


def test_mpo_is_the_heisenberg_hamiltonian_at_n8():
    """Contract the MPO to its dense 256 x 256 matrix and compare against explicit ``kron``.

    The MPO is judged as an *operator* first: if this fails, every energy below is
    measuring the wrong Hamiltonian and the ground state would still look plausible.
    """
    n_sites = 8
    tensors = [np.asarray(w.to_dense()) for w in dmrg.mpo(n_sites)]
    acc = tensors[0]
    for w in tensors[1:]:
        acc = np.einsum("a...b,bcde->a...cde", acc, w)
    acc = acc[0, ..., 0]  # the D=1 MPO boundary legs
    order = list(range(0, 2 * n_sites, 2)) + list(range(1, 2 * n_sites, 2))
    from_mpo = acc.transpose(order).reshape(2**n_sites, 2**n_sites)

    eye, sz, sp, sm = dmrg._spin_half()

    def at(op, site):
        out = np.array([[1.0]])
        for k in range(n_sites):
            out = np.kron(out, op if k == site else eye)
        return out

    from_kron = sum(
        at(sz, i) @ at(sz, i + 1) + 0.5 * (at(sp, i) @ at(sm, i + 1) + at(sm, i) @ at(sp, i + 1))
        for i in range(n_sites - 1)
    )
    assert np.abs(from_mpo - from_kron).max() < 1e-12


def test_mpo_refuses_a_perturbed_grading():
    """Move the ``+2`` sector of the MPO bond to ``-2`` and ``from_dense`` must *raise*.

    This is the proof that the grading in :data:`dmrg.MPO_BOND` is right. A passing
    ``allclose`` on the dense reconstruction would not be: the projection onto a wrong
    grading is still *a* tensor, just not this Hamiltonian. Same argument #104 makes for
    the Z2 magnetization, applied to construction instead of to a symmetry claim.
    """
    perturbed = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(-2): 2})
    assert perturbed.dim == dmrg.MPO_BOND.dim
    with pytest.raises(ValueError, match="not symmetric"):
        dmrg.mpo(6, bond=perturbed)


# --- the Krylov solver -------------------------------------------------------------


def test_lanczos_finds_the_lowest_eigenvalue():
    """:func:`dmrg.lanczos` against ``np.linalg.eigvalsh`` on a small dense Hermitian.

    The "vector" is a rank-2 tensor on ``(X OUT, BOUNDARY IN)``, so the trivial boundary
    leg restricts it to ``X``'s charge-0 block exactly as the MPS boundary legs restrict
    the chain to ``S^z_tot = 0`` -- and the eigenvalue to beat is that block's.

    ``ncv=3`` is one Krylov space and no restart, so a single call is not expected to
    converge; the loop below is what a DMRG sweep does anyway (reseed and repeat).
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
        (tenet.Leg(space, tenet.OUT), tenet.Leg(dmrg.BOUNDARY, tenet.IN)), seed=3
    )
    value = None
    for _ in range(10):
        value, v = dmrg.lanczos(lambda x: tenet.einsum("ab,bc->ac", operator, x), v)
    assert value == pytest.approx(expected, abs=1e-10)


# --- the environment cache ---------------------------------------------------------


def test_swept_environments_match_a_rebuild_from_scratch():
    """The only check that catches a missed :func:`dmrg.invalidate`.

    A stale ``F[(n, n+1)]`` after site ``n`` changed gives an energy that is *plausible
    and wrong*, which is the worst failure mode a DMRG has. After one full two-direction
    sweep every right-directed environment must equal the one a fresh
    :func:`dmrg.setup_envs` builds from the final MPS, to 1e-12 -- and the left-directed
    ones must be *gone*, because the return leg of the sweep invalidated every one of them.
    """
    n_sites = 8
    psi = dmrg.canonicalize(dmrg.random_mps(n_sites, seed=1))
    w = dmrg.mpo(n_sites)
    envs = dmrg.setup_envs(psi, w)
    dmrg.sweep(psi, w, envs, {}, chi=16, cutoff=1e-14)

    rebuilt = dmrg.setup_envs(psi, w)
    assert set(envs) == set(rebuilt) | {(-1, 0)}
    for key, expected in rebuilt.items():
        assert envs[key].structure == expected.structure, key
        assert float(tenet.norm(tenet.subtract(envs[key], expected))) < 1e-12, key


def test_every_contraction_in_the_toy_is_a_composition(monkeypatch):
    """M23's rule on the example: operand 1 supplies ``IN`` on every shared wire.

    ``tests/network/test_hygiene.py`` measures this for the library's copy of these same
    contractions, against a dense Jordan-Wigner oracle. The example is U(1), where every
    cap sign is ``+1`` and no arithmetic here can tell a right order from a wrong one --
    which is exactly why it is asserted structurally instead: a reader copying this file
    for a fermionic model would otherwise copy a silent sign error. ``dmrg._composed``
    routes through ``tenet.einsum`` after bending, so one wrapper sees every call.

    ``dmrg.inner`` is the one exception, and it is the *library's* exception too: a
    sesquilinear pairing is not a composition. ``adjoint`` turns every leg around, so the
    domain wires of a rank-3 vector meet OUT-against-IN by construction, and
    ``tenet.inner`` (``ops/contraction.py``) writes the identical einsum. Excluded by
    frame, not by ignoring a violation.
    """
    real = tenet.einsum
    violations = []

    def wrapped(equation, *operands, **kwargs):
        caller = traceback.extract_stack()[-2].name
        if len(operands) == 2 and "->" in equation and caller != "inner":
            terms = equation.split("->")[0].split(",")
            for label in terms[0]:
                if label not in terms[1]:
                    continue
                end1 = operands[0].legs[terms[0].index(label)].side
                end2 = operands[1].legs[terms[1].index(label)].side
                if not (end1 is tenet.IN and end2 is not tenet.IN):
                    violations.append(f"{equation}: wire {label!r} is {end1} against {end2}")
        return real(equation, *operands, **kwargs)

    monkeypatch.setattr(tenet, "einsum", wrapped)
    out = dmrg.dmrg(6, chi=8, max_sweeps=2)
    assert not violations, "\n".join(sorted(set(violations)))
    assert out.sweeps == 2  # the smoke really did sweep


# --- the ground state --------------------------------------------------------------


def test_n12_reaches_the_computed_oracle(main_runs):
    """DMRG at chi=64 against the value :func:`heisenberg_sz0` computes, to 1e-10.

    chi=64 is *exact* for N=12: the middle-cut Schmidt rank is bounded by 2^6 = 64, so
    there is nothing for the truncation to discard and this is a statement about the
    sweep, the eigensolver and the MPO rather than about a bond dimension.
    """
    exact = np.linalg.eigvalsh(heisenberg_sz0(12))[0]
    small, _ = main_runs
    assert small.energy == pytest.approx(exact, abs=1e-10)
    assert small.max_discarded_weight < 1e-12


def test_n20_matches_the_recorded_literal():
    """N=20 at chi=64 against ``-8.682473334398956`` to 1e-8 (provenance in the header)."""
    out = run(20, 64)
    assert out.energy == pytest.approx(E_OBC_20, abs=1e-8)


# --- M22: the MPSKit.jl oracle (#152) ----------------------------------------------


def test_fixture_provenance_pins_the_mpskit_stack():
    """The seven provenance keys and the three pinned versions, asserted.

    The role ``test_fixture_header_names_tensorkitsectors_as_oracle`` plays for the racah
    fixtures: a silent regeneration against a different MPSKit shows up as a red test and
    a version bump in the diff, not as quietly different floats.
    """
    p = _MPSKIT["provenance"]
    assert set(p) == {
        "mpskit",
        "mpskitmodels",
        "tensorkit",
        "julia",
        "generator",
        "generated_utc",
        "script_sha256",
    }
    assert p["mpskit"] == "0.13.13"
    assert p["mpskitmodels"] == "0.4.7"
    assert p["tensorkit"] == "0.17.1"
    assert p["generator"] == "tools/oracles/heisenberg_mpskit.jl"


def test_fixture_validation_entries_match_the_ed_literals():
    """MPSKit's N=12 and N=20 energies equal ``E_OBC_12``/``E_OBC_20`` to 1e-12.

    Pure arithmetic, no DMRG run: this is what fires if the fixture is ever regenerated
    against a different Hamiltonian convention (a rescaled J, a sign flip, periodic
    boundaries). Both #110 literals were single-sourced until #152 cross-checked them.
    """
    v = _MPSKIT["validation"]
    assert v["N12_trivial"]["energy"] == pytest.approx(E_OBC_12, abs=1e-12)
    assert v["N12_su2"]["energy"] == pytest.approx(E_OBC_12, abs=1e-12)
    assert v["N20_su2"]["energy"] == pytest.approx(E_OBC_20, abs=1e-12)


def test_n32_matches_the_mpskit_oracle(main_runs):
    """N=32 at chi=64 against MPSKit's chi=256 value, to 1e-9 (measured gap ~2e-10).

    This replaces ``test_n32_lands_in_the_fitted_band``'s 0.03-wide window, which was the
    best available before #152: fitting ``E(N) = N e_inf + a - c/N`` on #110's own
    N=18/N=20 pair gives ``a = 0.18796``, ``c = 0.1498`` and hence ``E(32) ~ -13.997``,
    and the band ``[-14.01, -13.98]`` asserted exactly that. The fixture's chi ladder
    pins the same quantity to 1e-12, so the band survives only as this record. The
    Bethe-limit line is a different statement (a finite open chain sits above the
    thermodynamic limit) and is kept.
    """
    _, big = main_runs
    assert big.energy == pytest.approx(_MPSKIT["N32"]["chi256"]["energy"], abs=1e-9)
    assert big.energy / 32 > dmrg.E_INF  # a finite open chain sits above the limit


def test_bond_energies_match_the_mpskit_profile(main_runs):
    """All 31 ``<S_i . S_{i+1}>`` of the converged N=32 MPS against MPSKit's, to 1e-8.

    A total energy is one number that averages over 31 bonds and can hide a compensating
    error in a boundary tensor; the profile cannot -- the edge bond is ``-0.652`` and its
    neighbour ``-0.296``, so a swapped or mis-normalized boundary tensor is loud here and
    quiet in the sum. The two-site operator is ``h2_dense()``'s construction from
    ``tests/network/test_mps.py``, rebuilt here because ``tests/`` is not a package.
    Measured max deviation: 5.8e-10, for ~0.6 s on the MPS ``main_runs`` already paid for.
    """
    _, big = main_runs
    psi = network.MPS.from_tensors(big.psi)  # the toy returns a plain list of site tensors
    _, sz, sp, sm = dmrg._spin_half()
    h2 = np.einsum("Pp,Qq->PQpq", sz, sz)
    h2 = h2 + 0.5 * np.einsum("Pp,Qq->PQpq", sp, sm) + 0.5 * np.einsum("Pp,Qq->PQpq", sm, sp)
    legs = (
        tenet.Leg(dmrg.PHYS, tenet.OUT),
        tenet.Leg(dmrg.PHYS, tenet.OUT),
        tenet.Leg(dmrg.PHYS, tenet.IN),
        tenet.Leg(dmrg.PHYS, tenet.IN),
    )
    op = tenet.SymmetricTensor.from_dense(h2, legs)
    profile = _MPSKIT["N32_bond_energies"]
    assert len(profile) == 31
    for n, want in enumerate(profile):  # fixture bond i = 1..31 is tenet bond n = 0..30
        assert network.expectation_2site(psi, op, n) == pytest.approx(want, abs=1e-8), n


# --- the three variational checks --------------------------------------------------


def test_energy_is_monotone_across_sweeps(main_runs):
    """It is variational: a DMRG whose energy rises is broken, full stop.

    The tolerance is 1e-12 and it is float noise, not slack -- once the run is converged
    consecutive energies differ in the last bits of a number near -14.
    """
    for out in (*main_runs, run(20, 64)):
        energies = [e for e, *_ in out.history]
        for previous, current in zip(energies, energies[1:], strict=False):
            assert current <= previous + 1e-12, out.history


@pytest.mark.parametrize("chis", [(4, 8, 16)])
def test_energy_and_discarded_weight_are_monotone_in_chi(chis):
    """Three chi values: the energy falls, and the reported discarded weight falls with it.

    Cheaper than either oracle and it catches a different bug -- a truncation that keeps
    the wrong singular values gets the *ordering* wrong long before it gets the N=12
    energy wrong.
    """
    runs = [run(12, chi) for chi in chis]
    energies = [out.energy for out in runs]
    weights = [out.max_discarded_weight for out in runs]
    assert energies == sorted(energies, reverse=True)
    assert weights == sorted(weights, reverse=True)
    assert energies[-1] > E_OBC_12  # still variational at chi=16


# --- the sector is structural ------------------------------------------------------


def test_the_target_sector_is_structural(main_runs):
    """Every ``S^z_tot != 0`` amplitude of the converged N=12 MPS is **exactly** zero.

    No tolerance, because this is not a numerical statement: the boundary legs are
    ``{U1Sector(0): 1}`` at both ends, so every forbidden entry of every site tensor's
    dense expansion is a structural zero and every amplitude that would need one is a sum
    of products each carrying a zero factor. This is the claim ``vmc_mps.SPACES``'s
    comment makes in passing and that no test in the repository had cashed.
    """
    small, _ = main_runs
    amplitudes = np.asarray(small.psi[0].to_dense())
    for site in small.psi[1:]:
        amplitudes = np.einsum("a...b,bcd->a...cd", amplitudes, np.asarray(site.to_dense()))
    amplitudes = amplitudes[0, ..., 0].reshape(-1)

    n_sites = len(small.psi)
    # dense physical index 0 is charge -1 (down) and 1 is charge +1 (up); the leading axis
    # is site 0, so `s`'s binary expansion read from the top names the configuration.
    forbidden = [s for s in range(2**n_sites) if bin(s).count("1") != n_sites // 2]
    assert np.count_nonzero(amplitudes[forbidden]) == 0
    assert np.linalg.norm(amplitudes) == pytest.approx(1.0, abs=1e-10)


# --- M13b: the same chain, the same number, a different symmetry -------------------


def test_su2_dmrg_reaches_the_same_n12_energy():
    """N=12 SU(2) against ``E_OBC_12``: the same constant, the same Hamiltonian, no U(1).

    The Hamiltonian is one invariant ``S.S`` array split by ``svd_truncated`` (#135); no
    ``W`` was written for it and nothing in ``Env``/``sweep_``/``lanczos`` changed. The
    multiplet numbers are printed rather than asserted -- ``max_bond`` bounds the *dense*
    bond ``Sum_c qdim(c) m_c`` (``linalg.py``:750-757), so at chi=32 the SU(2) mid-chain
    bond holds 12 multiplets against U(1)'s 32 states for the same energy, and no target
    number for that ratio is defensible in advance.

    Runtime: ~0.6 s, against the U(1) chi=32 run's ~1.4 s, so the file's 60 s budget is
    untouched. The seed and the reachable-sector construction are the SU(2) twins of
    ``walkthrough.bond_spaces``, and they live here for #112's reason: which spaces are reachable
    is physics.
    """
    phys = GradedSpace.new(SU2, {SU2Sector(1): 1})
    _, sz, sp, sm = walkthrough._spin_half()
    ss = network.local_op(np.kron(sz, sz) + (np.kron(sp, sm) + np.kron(sm, sp)) / 2, phys=phys)
    h = network.MPO.from_terms(12, [(1.0, [(ss, (i, i + 1))]) for i in range(11)])
    bonds = [
        GradedSpace.new(SU2, {SU2Sector(j): 1 for j in range(i % 2, min(i, 12 - i) + 1, 2)})
        for i in range(13)
    ]
    out = network.dmrg_(network.MPS.random(phys, bonds, seed=0), h, chi=32)
    assert out.energy == pytest.approx(E_OBC_12, abs=1e-10)

    bond = out.psi[6].legs[0].space
    assert (bond.reduced_dim, bond.dim) == (12, 32)
    assert out.psi[0].legs[0].space == GradedSpace.new(SU2, {SU2Sector(0): 1})


# --- M14: the schedule does arithmetic, and the extrapolation recipe ----------------


def test_a_ramp_reaches_below_its_own_first_phase():
    """``[Sweep(8)]*3 + [Sweep(32)]*3`` at N=12: the schedule is not bookkeeping.

    The chi=32 tail must end below the energy the chi=8 phase plateaued at, and with a
    smaller discarded weight than any chi=8 sweep reported.
    """
    psi = network.MPS.random(walkthrough.PHYS, walkthrough.bond_spaces(12), seed=0)
    schedule = [network.Sweep(8)] * 3 + [network.Sweep(32)] * 3
    out = network.dmrg_(psi, walkthrough.mpo(12), schedule=schedule, energy_tol=0.0, max_sweeps=6)
    assert out.schedule == schedule  # M14: the realized schedule, one entry per sweep
    assert len(out.schedule) == out.sweeps == len(out.history)
    assert out.history[-1][0] < out.history[2][0]
    assert out.history[-1][3] < min(h[3] for h in out.history[:3])


def test_the_reverse_schedule_extrapolation_beats_its_cheapest_run(main_runs):
    """block2's energy-extrapolation protocol as a test, not a function.

    A converged chi=64 run is re-swept on a *reverse* schedule -- zero noise throughout,
    ``energy_tol=0.0`` so nothing exits early, an even number of sweeps per chi, the
    first reverse chi below the last forward one -- and the linear fit of energy against
    discarded weight lands closer to the exact N=12 energy than the chi=8 run does. The
    residual check is what makes it a statement about *linearity* rather than luck, and
    the protocol is only valid because every sweep here is two-site (a one-site sweep at
    zero noise reports near-zero discarded weight); the recipe with block2's conventions
    is in ``docs/tutorials/dmrg.md``.
    """
    small, _ = main_runs
    psi = network.MPS.from_tensors(small.psi)  # the toy's converged chi=64 state
    schedule = [network.Sweep(16, noise=0.0)] * 2 + [network.Sweep(12)] * 2 + [network.Sweep(8)] * 2
    out = network.dmrg_(psi, walkthrough.mpo(12), schedule=schedule, energy_tol=0.0, max_sweeps=6)
    dws = [h[3] for h in out.history]
    energies = [h[0] for h in out.history]
    fit = np.polyfit(dws, energies, 1)
    assert abs(fit[1] - E_OBC_12) < abs(run(12, 8).energy - E_OBC_12)
    residuals = np.asarray(energies) - np.polyval(fit, dws)
    assert np.abs(residuals).max() < 0.05 * (max(energies) - min(energies))


def test_a_product_seeded_run_keeps_the_sector_structural():
    """``test_the_target_sector_is_structural`` for an ``MPS.product`` Neel seed at N=12.

    The D=1 product seed carries the ``S^z_tot = 0`` statement on its boundary legs
    exactly as the random seed does, and every forbidden amplitude of the converged
    state is exactly zero -- the sector survives the sweeps structurally, not
    approximately.
    """
    psi = network.MPS.product(walkthrough.PHYS, [U1Sector(1), U1Sector(-1)] * 6)
    assert psi[0].legs[0].space == walkthrough.BOUNDARY
    out = network.dmrg_(psi, walkthrough.mpo(12), chi=32)
    amplitudes = np.asarray(out.psi.to_dense()).reshape(-1)
    forbidden = [s for s in range(2**12) if bin(s).count("1") != 6]
    assert np.count_nonzero(amplitudes[forbidden]) == 0
    assert np.linalg.norm(amplitudes) == pytest.approx(1.0, abs=1e-10)


# --- the example runs standalone ---------------------------------------------------


def test_main_runs(main_runs):
    """``dmrg.main()`` at its own N and chi, as ``test_ctmrg.py``:664 does.

    The pair it returns is the module fixture every large-N assertion above reads, so the
    expensive N=32 run happens exactly once in this file.
    """
    small, big = main_runs
    assert small.sweeps > 1 and big.sweeps > 1
    for out in (small, big):
        assert len(out.history) == out.sweeps
    assert small.denergy < 1e-12 and small.max_dSchmidt < 1e-8


def test_main_page_output_is_current(main_runs):
    check_example_page("toy-dmrg.md", _MAIN_STDOUT["text"])
