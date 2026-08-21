"""M18 — the usage lane runs in CI, at the same defaults a reader runs.

Each file in ``examples/`` calls ``tenet.network`` as a user would, and its defaults
*are* the tested sizes: ``main()`` here is exactly ``python examples/<file>.py``. The
oracles are independent of the examples — ``tests/integration/test_dmrg.py``'s recorded
N=20 ED energy, the U(1) run ``su2_heisenberg`` computes in the same process, and
``onsager(beta)``. The teaching lane keeps its own CI execution unchanged
(``tests/integration/test_dmrg.py``, ``test_ctmrg.py``, ``test_vmc.py``).

The **lane rule itself** is asserted here too (#183): a file in ``examples/toy_codes/``
writes its algorithm on ``tenet``'s tensor layer and imports nothing from
``tenet.network``. There is no exemption -- #187 rewrote ``ctmrg.py``, the last file that
had one.
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


def test_ising2d_page_output_is_current(ising_run):
    check_example_page("ising2d.md", _STDOUT["ising2d"])


def test_toy_ctmrg_reproduces_the_library_environment():
    """#187's acceptance: the rewritten toy code is the library's CTMRG, not a lookalike.

    ``examples/toy_codes/ctmrg.py`` now writes the corner, the edge, the two absorbers, the
    projector and the sweep out on the tensor layer, so the thing that has to be checked is
    that it still lands where ``tenet.network.ctmrg`` lands. Two statements at ``beta=0.4``,
    ``chi=8``: the converged environments carry the same graded bond and the same corner
    spectrum, and the toy's own free energy reads the same number off either environment.
    The contractions are the same arithmetic in the same order, so the tolerance is
    ``1e-12`` -- float64 round-off, not an algorithmic allowance.
    """
    import tenet.network as net

    sys.path.insert(0, str(EXAMPLES / "toy_codes"))
    import ctmrg as toy

    beta, chi = 0.4, 8
    bulk = toy.ising_bulk(beta)
    toy_env = toy.converge(*toy.single_layer_ctm(bulk), chi=chi, tol=1e-12)
    lib = net.ctmrg(*net.single_layer_ctm(bulk), chi=chi, tol=1e-12).env

    assert toy_env[2] == lib.bond
    assert toy.spectrum(toy_env[0]) == pytest.approx(net.spectrum(lib.c), abs=1e-12)
    assert float(toy.beta_free_energy(beta, toy_env)) == pytest.approx(
        float(toy.beta_free_energy(beta, lib)), abs=1e-12
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


def test_lane_basenames_are_disjoint():
    """Both lanes land on sys.path as top-level modules; a shared basename would shadow."""
    flat = {p.name for p in EXAMPLES.glob("*.py")}
    toy = {p.name for p in (EXAMPLES / "toy_codes").glob("*.py")}
    assert not flat & toy
