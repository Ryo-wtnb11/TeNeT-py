"""M18 — the usage lane runs in CI, at the same defaults a reader runs.

Each file in ``examples/`` calls ``tenet.network`` as a user would, and its defaults
*are* the tested sizes: ``main()`` here is exactly ``python examples/<file>.py``. The
oracles are independent of the examples — ``tests/integration/test_dmrg.py``'s recorded
N=20 ED energy, the U(1) run ``su2_heisenberg`` computes in the same process, and
``onsager(beta)``. The teaching lane keeps its own CI execution unchanged
(``tests/integration/test_dmrg.py``, ``test_ctmrg.py``, ``test_vmc.py``).

The **lane rule itself** is asserted here too (#183): a file in ``examples/toy_codes/``
writes its algorithm on ``tenet``'s tensor layer and imports nothing from
``tenet.network``. There is no exemption.
"""

import ast
import contextlib
import io
import pathlib
import sys

import pytest
from helpers import check_example_page

EXAMPLES = pathlib.Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

import heisenberg  # noqa: E402
import heisenberg_walkthrough  # noqa: E402
import ising2d  # noqa: E402
import su2_heisenberg  # noqa: E402
import su3_heisenberg  # noqa: E402

# Each module fixture's stdout, for the docs example pages (#164).
_STDOUT: dict[str, str] = {}


def _capture(name, call):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = call()
    _STDOUT[name] = buf.getvalue()
    return result


# tests/integration/test_dmrg.py's recorded open-boundary ED energies (#110).
E_N20 = -8.682473334398956
E_OBC_12 = -5.142090632840532


@pytest.fixture(scope="module")
def heisenberg_run():
    return _capture("heisenberg", heisenberg.main)


@pytest.fixture(scope="module")
def walkthrough_run():
    return _capture("heisenberg_walkthrough", heisenberg_walkthrough.main)


@pytest.fixture(scope="module")
def su2_run():
    return _capture("su2_heisenberg", su2_heisenberg.main)


@pytest.fixture(scope="module")
def su3_run():
    return _capture("su3_heisenberg", su3_heisenberg.main)


@pytest.fixture(scope="module")
def ising_run():
    return _capture("ising2d", ising2d.main)


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


def test_walkthrough_routes_agree_on_the_ed_energy(walkthrough_run):
    """The hand-graded ``W`` and the derived term list reach the same N=12 ground state.

    The energy is ``tests/integration/test_dmrg.py``'s computed ED oracle; the point of
    running both routes is that a hand-derived MPO bond grading and one the library
    derives from the operators' own charges must agree as *operators*, and two independent
    DMRG runs landing on the same twelve digits is that statement cashed end to end.
    """
    from_w, from_terms = walkthrough_run
    assert abs(from_w.energy - E_OBC_12) < 1e-10
    assert abs(from_w.energy - from_terms.energy) < 1e-10


def test_walkthrough_bond_spaces_are_the_reachable_ones(walkthrough_run):
    """The seed's bond ``i`` holds exactly the charges ``i`` spins can still bring to zero.

    ``bond_spaces`` is the file's whole symmetry input, so it is asserted rather than
    printed: dimension ``min(i, N - i) + 1`` at bond ``i``, and ``D=1`` at both ends,
    which is the ``S^z_tot = 0`` statement.
    """
    n_sites = 12
    dims = [space.dim for space in heisenberg_walkthrough.bond_spaces(n_sites)]
    assert dims == [min(i, n_sites - i) + 1 for i in range(n_sites + 1)]


def test_su2_agrees_with_the_u1_run_it_computes(su2_run):
    su2, u1, _ = su2_run
    assert abs(su2.energy - u1.energy) < 1e-10


def test_su2_mid_bond_is_multiplet_compressed(su2_run):
    _, _, mid = su2_run
    assert mid.reduced_dim < mid.dim


def test_su3_reproduces_its_own_dense_ed(su3_run):
    """The SU(3) run's oracle is in the same file: numpy-only ED of the permutation chain.

    ``from_blocks`` writing ``+1`` on the ``6`` and ``-1`` on the ``3bar`` is the whole
    statement of the Hamiltonian, so what has to be checked is that the resulting operator
    *is* ``sum_i P_{i,i+1}``: the N=6 DMRG energy against the dense diagonalisation of the
    same chain built from ``numpy.kron`` alone.
    """
    short, _, exact = su3_run
    assert abs(short.energy - exact) < 1e-10


def test_su3_bond_is_multiplet_compressed_and_triality_zero(su3_run):
    """An even cut of a fundamental chain can only carry zero-triality irreps.

    Triality is ``(a_1 + 2 a_2) mod 3``, additive under fusion and ``1`` on the
    fundamental, so twelve sites to the left of the mid bond leave ``0``. DMRG is never
    told this -- the seed offers ``3``, ``3bar``, ``6`` and ``6bar`` too -- and the sweeps
    drop them, which is the symmetry doing the bookkeeping rather than the caller.
    """
    _, long, _ = su3_run
    mid = long.psi[12].legs[0].space
    assert mid.reduced_dim < mid.dim
    assert all((a.dynkin[0] + 2 * a.dynkin[1]) % 3 == 0 for a, _ in mid.sectors)


def test_su3_energy_per_site_brackets_the_bethe_value(su3_run):
    """The N=24 open chain against Sutherland's infinite-chain energy per site.

    An open chain has ``N - 1`` bonds, so ``E/N`` sits above the infinite-chain value and
    ``E/(N-1)`` below it; the two together are the statement that the finite run is
    converging on the Bethe-ansatz number rather than merely near it.
    """
    _, long, _ = su3_run
    assert long.energy / 24 > su3_heisenberg.SUTHERLAND
    assert long.energy / 23 < su3_heisenberg.SUTHERLAND


def test_ising2d_matches_onsager_off_criticality(ising_run):
    results, _ = ising_run
    for beta in (0.3, 0.5):
        _, rel = results[beta]
        assert rel < 1e-12


def test_ising2d_ordered_spectrum_is_pairwise_degenerate(ising_run):
    _, corner = ising_run
    for even, odd in zip(corner[0::2], corner[1::2], strict=False):
        assert abs(even - odd) < 1e-10


def test_heisenberg_page_output_is_current(heisenberg_run):
    check_example_page("heisenberg.md", _STDOUT["heisenberg"])


def test_heisenberg_walkthrough_page_output_is_current(walkthrough_run):
    check_example_page("heisenberg-walkthrough.md", _STDOUT["heisenberg_walkthrough"])


def test_su2_heisenberg_page_output_is_current(su2_run):
    check_example_page("su2-heisenberg.md", _STDOUT["su2_heisenberg"])


def test_su3_heisenberg_page_output_is_current(su3_run):
    check_example_page("su3-heisenberg.md", _STDOUT["su3_heisenberg"])


def test_ising2d_page_output_is_current(ising_run):
    check_example_page("ising2d.md", _STDOUT["ising2d"])


# --- the differentiable lane: examples/ising_thermo.py (needs jax) ------------------


@pytest.fixture(scope="module")
def thermo_run():
    """``ising_thermo.main()`` at its defaults, run once.

    Skipped without JAX, as every AD-facing test in the suite is. It is the slowest
    entry in this module (~20 s: two converged environments, three reverse-mode passes
    each, plus the K scan) and every assertion below reads the same run.
    """
    pytest.importorskip("jax")
    import ising_thermo

    return _capture("ising_thermo", ising_thermo.main)


def test_ising_thermo_traced_bulk_is_the_same_tensor():
    """``from_blocks`` naming the blocks and ``from_dense`` projecting onto them agree.

    The traced builder exists only because ``from_dense`` asks a concrete-value question
    a tracer cannot answer; if the two ever stopped being the same tensor, every number
    on the page would be measuring a different model from ``examples/ising2d.py``.
    """
    pytest.importorskip("jax")
    import ising_thermo
    import numpy as np

    beta = 0.4
    np.testing.assert_allclose(
        np.asarray(ising_thermo.traced_bulk(beta).to_dense()),
        ising2d.ising_bulk(beta).to_dense(),
        rtol=1e-14,
        atol=1e-14,
    )


def test_ising_thermo_free_energy_matches_onsager(thermo_run):
    """The undifferentiated quantity first: nothing downstream means anything without it."""
    results, _ = thermo_run
    for beta, (bf, *_) in results.items():
        assert abs(bf / ising2d.onsager(beta) - 1) < 1e-12


def test_ising_thermo_internal_energy_matches_onsager(thermo_run):
    """``d(beta f)/d beta`` from ``jax.grad`` against Onsager's own derivative.

    ``1e-6`` relative is the level the *oracle* is good to: a central difference of a
    1e-12-accurate quadrature at ``h = 1e-4``. The AD value is not measurably worse than
    that, which is the statement -- the environment is at its fixed point when the traced
    region starts, so the first derivative is already converged in ``K``.
    """
    results, _ = thermo_run
    for _, (_, u, _, u_ref, _) in results.items():
        assert abs(u - u_ref) / max(1.0, abs(u_ref)) < 1e-6


def test_ising_thermo_specific_heat_matches_onsager(thermo_run):
    """``-beta^2 d^2(beta f)/d beta^2`` from ``jax.grad(jax.grad(...))``.

    Looser than the first derivative, and the loosening is physics rather than slop:
    :func:`test_ising_thermo_specific_heat_converges_in_the_unrolling` shows the residual
    is the finite unrolling, not the environment.
    """
    results, _ = thermo_run
    for _, (_, _, cv, _, cv_ref) in results.items():
        assert abs(cv / cv_ref - 1) < 1e-3


def test_ising_thermo_specific_heat_converges_in_the_unrolling(thermo_run):
    """The truncated backprop, measured: ``c_V(K)`` approaches Onsager monotonically.

    This is the assertion that makes "differentiation through ``K`` unrolled moves, not
    an implicit fixed-point derivative" a checkable claim rather than a caveat. The
    environment is *converged* before the traced region starts, so ``beta f`` and its
    first derivative do not move with ``K`` at all; the second derivative does, because
    the ``K`` moves have to carry the environment's second-order response themselves.
    """
    import ising_thermo

    _, scan = thermo_run
    _, cv_ref = ising_thermo.onsager_derivatives(0.5)
    errors = [abs(cv - cv_ref) for _, cv in scan]
    assert errors == sorted(errors, reverse=True), scan
    assert errors[-1] < errors[0] / 10, scan


def test_ising_thermo_page_output_is_current(thermo_run):
    check_example_page("ising-thermo.md", _STDOUT["ising_thermo"])


@pytest.fixture(scope="module")
def toy_chain():
    """``tebd.main()`` and ``exact.main()``, run once, with their stdout kept for the pages.

    The two new teaching modules of #268 print a committed output each and neither has an
    integration module of its own: ``exact.py`` is ~0.2 s of ``numpy.linalg.eigvalsh`` and
    ``tebd.py`` is ~8 s of imaginary time at N=12, which is the usage lane's budget rather
    than ``tests/integration/test_dmrg.py``'s. They share a fixture because ``tebd.main()``
    calls ``exact.ground_energy`` itself, so the reference is computed here either way.
    """
    sys.path.insert(0, str(EXAMPLES / "toy_codes"))
    import exact
    import tebd

    exact_out = _capture("exact", exact.main)
    _, history, reference = _capture("tebd", lambda: tebd.main())
    return exact_out, history, reference


def test_toy_exact_reproduces_the_recorded_open_chain_energies(toy_chain):
    """``exact.py`` is an oracle, so it is pinned before anything is judged against it.

    ``-5.142090632840532`` is ``tests/integration/test_dmrg.py``'s independently computed
    N=12 open-boundary value; that module builds its matrix from bit flips written out by
    hand, this one builds it from ``model.h_bonds()`` read out dense, and the two agree.
    """
    energies, _, _ = toy_chain
    assert energies[12] == pytest.approx(E_OBC_12, abs=1e-12)
    assert all(e / n > -0.4431471805599453 for n, e in energies.items())


def test_toy_tebd_reaches_the_exact_energy_from_above(toy_chain):
    """#268's acceptance for the new algorithm: variational, and converged.

    Imaginary time can only lower the energy and the truncation can only raise it, so a
    stage that came out *below* ``exact.py`` would be a bug rather than a lucky run. The
    last stage lands within 1e-9 of it -- the page's committed output shows 1.2e-11 -- and
    that is the same ground state ``dmrg.py`` reaches from the MPO form of the same model.
    """
    _, history, reference = toy_chain
    energies = [e for _, _, e, _ in history]
    assert all(e >= reference for e in energies), "TEBD went below the exact energy"
    assert energies == sorted(energies, reverse=True), "imaginary time raised the energy"
    assert energies[-1] - reference < 1e-9


def test_toy_tebd_page_output_is_current(toy_chain):
    check_example_page("toy-tebd.md", _STDOUT["tebd"])


def test_toy_exact_page_output_is_current(toy_chain):
    check_example_page("toy-exact.md", _STDOUT["exact"])


def test_toy_ctmrg_reproduces_the_library_environment():
    """The teaching lane is the library's CTMRG, not a lookalike (#183/#187).

    ``examples/toy_codes/ctmrg.py`` writes the corner, the edge, the two absorbers, the
    projector and the sweep out on the tensor layer, so what has to be checked is that it
    lands where ``EnvCTMc4v`` lands. Two statements at ``beta=0.4``, ``chi=8``, on the same
    physical Boltzmann tensor written in the two lanes' own leg conventions: the converged
    corner *spectra* agree, and so does the free energy.

    Both are gauge-invariant, and that is why the tolerance is ``1e-10`` rather than
    float64 round-off: the teaching lane keeps only ``U`` from the projector SVD and takes
    the renormalized corner to be ``S``, while ``EnvCTMc4v`` keeps ``V`` as well and
    renormalizes to ``V^dagger U S``. The two environments are the same fixed point in
    different gauges, so a quantity that depends on the gauge would not be comparable at
    all.
    """
    from tenet.network import EnvCTMc4v, Peps, SquareLattice

    sys.path.insert(0, str(EXAMPLES / "toy_codes"))
    import ctmrg as toy

    beta, chi = 0.4, 8
    toy_env = toy.converge(*toy.single_layer_ctm(toy.ising_bulk(beta)), chi=chi, tol=1e-12)

    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising2d.ising_bulk(beta)))
    assert env.iterate_(max_bond=chi, max_sweeps=300, corner_tol=1e-12).converged

    # the usage lane carries its own Onsager quadrature, so that it runs on a core
    # install; the teaching lane's is the second source it is judged against
    assert ising2d.onsager(beta) == pytest.approx(toy.onsager(beta), abs=1e-12)
    assert toy.spectrum(toy_env[0]) == pytest.approx(ising2d.corner_spectrum(env), abs=1e-10)
    assert float(toy.beta_free_energy(beta, toy_env)) == pytest.approx(
        -float(ising2d.log_kappa(env)), abs=1e-10
    )


def _imports(path):
    """Every module root ``path`` imports, by AST -- ``tests/network/test_hygiene.py``'s scan."""
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots |= {f"{node.module}.{alias.name}" for alias in node.names}
            roots.add(node.module)
    return roots


@pytest.mark.parametrize(
    "path", sorted((EXAMPLES / "toy_codes").glob("*.py")), ids=lambda p: p.name
)
def test_a_toy_code_imports_nothing_from_the_network_layer(path):
    """The teaching lane's rule, enforced (#183): the algorithm is written, not called.

    A toy code may use ``tenet``'s tensor layer -- ``SymmetricTensor``, ``tenet.einsum``,
    ``tenet.linalg`` -- because that is the library's *subject matter*, not the algorithm
    being taught. ``tenet.network`` is the algorithm, so a file that imports it is a usage
    example under the wrong name, which is what ``examples/toy_codes/dmrg.py`` had become
    before #183.
    """
    offenders = sorted(name for name in _imports(path) if name.startswith("tenet.network"))
    assert not offenders, f"{path.name} imports {offenders} from the algorithm layer"


# The public way into a ``SymmetricTensor``'s numbers is ``tenet.to_matrices``. These are
# the ways around it: ``t.blocks``, ``t.apply_blocks(...)``, and building the tensor from a
# dense carrier-basis array with ``from_dense`` instead of naming its blocks.
_BLOCK_BACKDOORS = ("blocks", "apply_blocks", "from_dense")


@pytest.mark.parametrize(
    "path", sorted((EXAMPLES / "toy_codes").glob("*.py")), ids=lambda p: p.name
)
def test_a_toy_code_reads_block_values_only_through_to_matrices(path):
    """The teaching lane's second rule: a toy code stays on the *public* tensor layer.

    A symmetric-tensor algorithm states its input in the symmetric form it is in --
    ``SymmetricTensor.from_blocks`` with the block values written out, or ``zeros`` /
    ``random`` -- and reads block values back through ``tenet.to_matrices``. Reaching for
    ``.blocks``, ``apply_blocks`` or ``autoray`` reimplements the library in the file that
    is supposed to be demonstrating it: ``tenet.inner`` and ``tenet.full_trace`` were both
    hand-written over ``.items()`` here, one of them with the sign bug the library had
    already fixed.

    ``.items()`` is allowed on the ``tenet.to_matrices`` result and nowhere else, which is
    what makes "block values are read through ``to_matrices``" mechanical rather than a
    matter of reading the receiver's type.
    """
    offenders = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr in _BLOCK_BACKDOORS:
            offenders.append(f"line {node.lineno}: .{node.attr}")
        elif node.attr == "items" and not ast.unparse(node.value).startswith("tenet.to_matrices("):
            offenders.append(f"line {node.lineno}: .items() on {ast.unparse(node.value)}")
    offenders += [f"imports {name}" for name in sorted(_imports(path)) if name == "autoray"]
    assert not offenders, (
        f"{path.name} goes around the public block API: {'; '.join(offenders)}. Build "
        "inputs with SymmetricTensor.from_blocks and read values with tenet.to_matrices."
    )


def test_lane_basenames_are_disjoint():
    """Both lanes land on sys.path as top-level modules; a shared basename would shadow."""
    flat = {p.name for p in EXAMPLES.glob("*.py")}
    toy = {p.name for p in (EXAMPLES / "toy_codes").glob("*.py")}
    assert not flat & toy
