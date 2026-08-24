"""M79d: gates, the bond metric, the truncation, and what the fermionic metric costs.

The oracle every measurement here rests on is
[to_dense_state][tests.network.test_evolution.to_dense_state]: an ``obc``
[Peps][tenet.network.Peps] contracted into its dense state vector, one physical index
per site in the lattice's own fermionic (column-major) order. Nothing about the
evolution is checked against another part of the evolution -- a gate is checked against
``scipy``-free matrix exponentials of Jordan-Wigner strings, a metric against the norm it
claims to measure, and an imaginary-time trajectory against exact diagonalization of the
same Hamiltonian.

**The bond metric is checked by what it is for.** ``g`` is the Gram form of the linear
map ``M -> |psi(M)>`` that a bond matrix induces, so the test builds that map's columns
one dense basis element at a time and compares the Gram matrix entry by entry. That is
independent of every contraction in ``network/evolution.py`` and of the twelve primitives
underneath it.

**The fermionic gap this stage found, and did not paper over.**
``test_the_fermionic_metric_on_a_loop_is_a_recorded_gap`` measures a disagreement between
the ``fZ2`` bond metric and that same oracle whenever the environment cluster contains a
*loop*. It is committed as a measurement rather than an ``xfail``, because the number is
the finding: on every cluster without a loop -- a row, a column, any 1D chain -- the
fermionic metric is exact to machine precision, and the gate path (which needs no metric
at all) is exact in 2D as well. ``docs/design.md``'s M79d entry says where the gap sits.
"""

import string

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import (
    EnvCTM,
    EnvNTU,
    Evolution_out,
    Peps,
    SquareLattice,
    accumulated_truncation_error,
    apply_gate,
    evolution_step_,
    gate_nn,
    gates_nn,
    local_op,
    truncate_,
)
from tenet.network.common import composed
from tenet.network.evolution import _bond_matrix, _dagger, _split
from tenet.symmetry import U1, FZ2Sector, Trivial, TrivialSector, U1Sector, fZ2

EVEN, ODD = FZ2Sector(0), FZ2Sector(1)
SPACES = {
    "trivial": GradedSpace.new(Trivial, {TrivialSector(): 2}),
    "u1": GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1}),
    "fz2": GradedSpace.new(fZ2, {EVEN: 1, ODD: 1}),
}
#: The one-mode annihilation matrix on ``(|0>, |1>)``; the hopping model every provider
#: below carries, so that the same numbers can be compared across gradings.
LOWER = np.array([[0.0, 1.0], [0.0, 0.0]])
Z = np.diag([1.0, -1.0])
HOPPING = -(np.kron(LOWER.T, LOWER) + np.kron(LOWER, LOWER.T))
LETTERS = string.ascii_lowercase + string.ascii_uppercase


# --- the dense closer -----------------------------------------------------------------


def to_dense_state(psi: Peps) -> np.ndarray:
    """An ``obc`` [Peps][tenet.network.Peps] as ``(d,) * N``, sites in fermionic order.

    Every boundary leg must be one-dimensional. The sites are absorbed in the geometry's
    own order -- column-major, which is what
    [f_ordered][tenet.network.SquareLattice.f_ordered] calls the fermionic order -- and
    each absorption is one [composed][tenet.network.composed], so the operand order and
    the bends are the layer's own and not this file's.
    """
    geometry = psi.geometry
    pool = iter(LETTERS)
    seen: dict[tuple, str] = {}

    def letter(key: tuple) -> str:
        if key not in seen:
            seen[key] = next(pool)
        return seen[key]

    acc, labels, phys = None, [], []
    for site in geometry.sites():
        x, y = site
        s = letter(("s", x, y))
        phys.append(s)
        here = [
            letter(("v", x - 1, y)),
            letter(("h", x, y - 1)),
            letter(("v", x, y)),
            letter(("h", x, y)),
            s,
        ]
        if acc is None:
            acc, labels = psi[site], list(here)
            continue
        shared = [c for c in labels if c in here]
        out = [c for c in labels if c not in shared] + [c for c in here if c not in shared]
        acc = composed(f"{''.join(labels)},{''.join(here)}->{''.join(out)}", acc, psi[site])
        labels = out
    dense = np.asarray(acc.to_dense())
    order = [labels.index(c) for c in phys] + [i for i, c in enumerate(labels) if c not in phys]
    dense = np.transpose(dense, order)
    return dense.reshape(dense.shape[: len(phys)])


def peps(provider: str, dims: tuple[int, int] = (2, 2), bond: int = 2, seed: int = 1) -> Peps:
    """A random ``obc`` PEPS whose boundary legs are one-dimensional."""
    phys = SPACES[provider]
    unit = GradedSpace.new(phys.provider, {phys.provider.unit: 1})
    inner = (
        GradedSpace.new(Trivial, {TrivialSector(): bond})
        if provider == "trivial"
        else GradedSpace.new(
            phys.provider,
            dict(zip(sorted(c for c, _ in phys.sectors), [bond] * 2, strict=False)),
        )
    )
    geometry = SquareLattice(dims=dims, boundary="obc")
    tensors = {}
    for x, y in geometry.sites():
        legs = (
            Leg(inner if x > 0 else unit, IN),
            Leg(inner if y > 0 else unit, OUT),
            Leg(inner if x < dims[0] - 1 else unit, OUT),
            Leg(inner if y < dims[1] - 1 else unit, IN),
            Leg(phys, OUT),
        )
        tensors[x, y] = SymmetricTensor.random(legs, seed=seed + 3 * x + 7 * y)
    return Peps(geometry, tensors)


def hamiltonian(provider: str) -> SymmetricTensor:
    """The nearest-neighbour hopping term as a rank-4 invariant operator."""
    return local_op(HOPPING, phys=SPACES[provider])


# --- the gate -------------------------------------------------------------------------


def test_the_gate_is_the_matrix_exponential_of_the_bond_term():
    """``g0 . g1`` recombines into ``exp(-step h)``, checked against a dense ``eigh``."""
    h = hamiltonian("u1")
    gate = gate_nn(h, 0.37, ((0, 0), (0, 1)))
    joined = composed("Ppc,cQq->PQpq", gate.g0, gate.g1)
    w, v = np.linalg.eigh(np.asarray(h.to_dense()).reshape(4, 4))
    assert np.asarray(joined.to_dense()).reshape(4, 4) == pytest.approx(
        v @ np.diag(np.exp(-0.37 * w)) @ v.T, abs=1e-12
    )


def test_the_gate_refuses_a_term_that_is_not_a_bond():
    with pytest.raises(ValueError, match="rank 4"):
        gate_nn(
            SymmetricTensor.random((Leg(SPACES["u1"], OUT), Leg(SPACES["u1"], IN))),
            0.1,
            ((0, 0), (0, 1)),
        )


def test_the_auxiliary_leg_may_sit_anywhere():
    """Where the gate's auxiliary wire lands in the enlarged tensor does not move it.

    A leg permutation is an isomorphism the next contraction undoes, so the two spellings
    of ``apply_gate`` differ by a transpose and by nothing else -- which is why the module
    states its output order and does not argue for it. The check is the *closed* state,
    the only place a wrong braid would show.
    """
    psi = peps("fz2")
    gate = gate_nn(hamiltonian("fz2"), 0.29, ((0, 0), (0, 1)))
    a0, a1 = psi[0, 0], psi[0, 1]
    default = apply_gate(a0, a1, gate)
    moved = (
        composed("Ppc,tlbrp->ctlbrP", gate.g0, a0),
        composed("cQq,tlbrq->tlcbrQ", gate.g1, a1),
    )
    for standard, other, order in (
        (default[0], moved[0], (1, 2, 3, 4, 0, 5)),
        (default[1], moved[1], (0, 1, 3, 4, 2, 5)),
    ):
        assert bool(tenet.allclose(standard, tenet.transpose(other, order)))


def jordan_wigner_hopping(i, j, n=4):
    """``-(c_i^dagger c_j + h.c.)`` on ``n`` modes, with the string on every earlier mode."""

    def mode(op, k):
        m = np.array([[1.0]])
        for q in range(n):
            m = np.kron(m, Z if q < k else (op if q == k else np.eye(2)))
        return m

    ci, cj = mode(LOWER, i), mode(LOWER, j)
    return -(ci.T @ cj + cj.T @ ci)


MODES = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


def evolved_against_jordan_wigner(bond, step=0.31):
    """``(fidelity)`` of one exact gate on a 2x2 ``fZ2`` patch against the dense oracle."""
    h = jordan_wigner_hopping(MODES[bond[0]], MODES[bond[1]])
    w, v = np.linalg.eigh(h)
    psi = peps("fz2")
    before = to_dense_state(psi).reshape(-1)
    expected = v @ (np.exp(-step * w) * (v.T @ before))
    gate = gate_nn(hamiltonian("fz2"), step, bond)
    truncate_(
        EnvNTU(psi), gate.bond, gate=gate, max_bond=None, cutoff=None, max_iter=0, fix_metric=None
    )
    got = to_dense_state(psi).reshape(-1)
    return abs(got @ expected) / np.linalg.norm(got) / np.linalg.norm(expected)


@pytest.mark.parametrize("bond", [((0, 0), (0, 1)), ((1, 0), (1, 1)), ((0, 1), (1, 1))])
def test_the_gate_carries_the_jordan_wigner_string_the_lattice_implies(bond):
    """fZ2, 2x2: an exact hopping gate against ``exp(-step H)`` on four modes.

    The dense oracle numbers the modes in the lattice's fermionic order --
    ``(0,0), (1,0), (0,1), (1,1)``, column-major -- and puts the Jordan-Wigner ``Z``
    string on every mode before the one it acts on. The horizontal bonds are the ones
    that matter: their two sites are **not** adjacent in that order, so the string runs
    through a third site, and it is carried by nothing but the braiding the composition
    rule already pays. No truncation is involved -- ``max_bond=None, cutoff=None`` is the
    exact split -- so this measures the gate and the ``qr`` and nothing else.

    The bond this list leaves out is the fourth one, and it is the subject of
    ``test_the_fermionic_gaps_this_stage_measured``.
    """
    assert evolved_against_jordan_wigner(bond) == pytest.approx(1.0, abs=1e-12)


# --- the bond metric ------------------------------------------------------------------


def metric_from_the_norm(env, psi, bond, dirn, gate):
    """The bond metric read off [to_dense_state][tests.network.test_evolution.to_dense_state].

    ``g`` is by definition the Gram form of ``M -> |psi(M)>``, so this evaluates that map
    on one dense basis element of the bond-matrix space at a time and takes the inner
    products. It shares no contraction with ``network/evolution.py``.

    Returns the keys it could build, the oracle's matrix, and the environment's own.
    """
    a0, a1 = apply_gate(psi[bond[0]], psi[bond[1]], gate)
    q0, r0 = _split(a0, dirn, 0)
    q1, r1 = _split(a1, dirn, 1)
    rr = _bond_matrix(composed("Axc,Bxc->AB", r0, r1))
    absorb = "tlbAs,AB->tlbBs" if dirn == "lr" else "tlAbs,AB->tlBbs"
    columns = {}
    for a in range(rr.shape[0]):
        for b in range(rr.shape[1]):
            entry = np.zeros(rr.shape)
            entry[a, b] = 1.0
            try:
                probe = SymmetricTensor.from_dense(entry, rr.legs)
            except ValueError:
                continue  # the entry is not a symmetry-allowed direction
            if float(tenet.norm(probe)) < 1e-12:
                continue
            trial = Peps(psi.geometry, dict(psi.items()))
            trial[bond[0]] = composed(absorb, q0, probe)
            trial[bond[1]] = q1
            columns[a, b] = to_dense_state(trial).reshape(-1)
    keys = sorted(columns)
    oracle = np.array([[columns[i] @ columns[j] for j in keys] for i in keys])
    g = np.asarray(env.bond_metric(q0, q1, bond[0], bond[1], dirn).to_dense())
    mine = np.array([[g[i[0], i[1], j[0], j[1]] for j in keys] for i in keys])
    return keys, oracle, mine


def relative(a, b):
    a, b = a / np.abs(a).max(), b / np.abs(b).max()
    return float(min(np.abs(a - b).max(), np.abs(a + b).max()))


LOOPLESS = [
    ((1, 2), ((0, 0), (0, 1)), "lr"),
    ((2, 1), ((0, 0), (1, 0)), "tb"),
    ((1, 3), ((0, 0), (0, 1)), "lr"),
    ((1, 3), ((0, 1), (0, 2)), "lr"),
    ((3, 1), ((0, 0), (1, 0)), "tb"),
    ((3, 1), ((1, 0), (2, 0)), "tb"),
]


@pytest.mark.parametrize("dims,bond,dirn", LOOPLESS, ids=lambda x: str(x))
@pytest.mark.parametrize("provider", sorted(SPACES))
def test_the_ntu_metric_is_the_norm_it_claims_to_measure(provider, dims, bond, dirn):
    """On a cluster with no loop the ``'NN'`` metric is *exact*, for every grading.

    A one-dimensional lattice's ``'NN'`` cluster is the two hairs and nothing else, so
    this pins the hair construction -- including the two hairs whose operand order the
    composition rule leaves tied, which the oracle decides.
    """
    psi = peps(provider, dims=dims)
    gate = gate_nn(hamiltonian(provider), 0.4, bond)
    _, oracle, mine = metric_from_the_norm(EnvNTU(psi), psi, bond, dirn, gate)
    assert relative(oracle, mine) < 1e-12


@pytest.mark.parametrize("provider", ["trivial", "u1"])
@pytest.mark.parametrize("bond,dirn", [(((0, 0), (0, 1)), "lr"), (((0, 0), (1, 0)), "tb")])
def test_the_ntu_metric_on_a_2x2_patch_is_exact_where_the_grading_does_not_braid(
    provider, bond, dirn
):
    """On a 2x2 ``obc`` lattice the ``'NN'`` cluster *is* the rest of the lattice, so the
    metric is the exact one and the oracle can demand equality rather than closeness."""
    psi = peps(provider, dims=(2, 2))
    gate = gate_nn(hamiltonian(provider), 0.4, bond)
    _, oracle, mine = metric_from_the_norm(EnvNTU(psi), psi, bond, dirn, gate)
    assert relative(oracle, mine) < 1e-12


@pytest.mark.parametrize("bond,dirn", [(((0, 0), (0, 1)), "lr"), (((0, 0), (1, 0)), "tb")])
def test_the_fermionic_gaps_this_stage_measured(bond, dirn):
    """The measurements M79d records rather than hides: ``fZ2`` plus a loop is wrong.

    A 2x2 ``obc`` cluster closes a loop around the bond, and there the double layer's
    per-site pieces stop composing: for ``fZ2`` the metric disagrees with the Gram form
    the dense closer builds, while ``Trivial`` and U(1) agree to machine precision on the
    *same* contraction. The disagreement is not a phase -- no permutation of the legs
    involved reproduces the oracle -- and it survives replacing every piece by its own
    exact Gram, so it is a property of joining double-layer objects over a doubled wire
    and not of any one primitive.

    Two consequences are asserted here so that neither can be forgotten: the number is
    large (this is not numerical noise), and the same metric fails the Hermiticity that a
    Gram form cannot fail. ``truncate_`` reports both, which is why the evolution does not
    silently produce a wrong state without saying so.
    """
    psi = peps("fz2", dims=(2, 2))
    gate = gate_nn(hamiltonian("fz2"), 0.4, bond)
    env = EnvNTU(psi)
    _, oracle, mine = metric_from_the_norm(env, psi, bond, dirn, gate)
    assert relative(oracle, mine) > 0.1
    a0, a1 = apply_gate(psi[bond[0]], psi[bond[1]], gate)
    q0, _ = _split(a0, dirn, 0)
    q1, _ = _split(a1, dirn, 1)
    g = env.bond_metric(q0, q1, bond[0], bond[1], dirn)
    assert float(tenet.norm(g - _dagger(g)) / tenet.norm(g)) > 0.1
    # The second gap, and the reason it is filed next to the first: the one bond of the
    # 2x2 patch whose exact gate does *not* reproduce the Jordan-Wigner oracle is the
    # vertical one in the first column, and the three others do. A gate involves no
    # environment at all, so this is the same fermionic bookkeeping seen from the other
    # side -- the extra wire the gate opens between the two sites is one more parallel
    # edge, which is one more loop.
    assert evolved_against_jordan_wigner(((0, 0), (1, 0))) == pytest.approx(0.2538, abs=1e-3)


@pytest.mark.parametrize("provider", ["trivial", "u1"])
def test_the_ctm_metric_agrees_with_the_ntu_one_on_a_lattice_where_both_are_exact(provider):
    """Oracle (iii): the two environments, the same bond, the same reduced tensors.

    On a 2x2 ``obc`` lattice the ``'NN'`` cluster is the exact environment, and a CTM
    environment seeded on the same open patch is the same object seen through its own
    corners -- so a disagreement here would be about the two constructions and not about
    the physics. They agree to the CTM's own convergence.
    """
    psi = peps(provider, dims=(2, 2))
    bond, dirn = ((0, 0), (0, 1)), "lr"
    gate = gate_nn(hamiltonian(provider), 0.4, bond)
    a0, a1 = apply_gate(psi[bond[0]], psi[bond[1]], gate)
    q0, _ = _split(a0, dirn, 0)
    q1, _ = _split(a1, dirn, 1)
    ntu = EnvNTU(psi).bond_metric(q0, q1, bond[0], bond[1], dirn)
    ctm = EnvCTM(psi, init="eye")
    ctm.update_(max_bond=16, moves="hv")
    got = ctm.bond_metric(q0, q1, bond[0], bond[1], dirn)
    assert relative(np.asarray(ntu.to_dense()), np.asarray(got.to_dense())) < 1e-8


def test_the_ctm_metric_refuses_what_it_cannot_do():
    psi = peps("u1", dims=(2, 2))
    env = EnvCTM(psi, init="eye")
    a = psi[0, 0]
    with pytest.raises(ValueError, match="not a lr bond"):
        env.bond_metric(a, a, (0, 0), (1, 1), "lr")
    classical = Peps(SquareLattice(dims=(1, 1)), SymmetricTensor.random(a.legs[:4], seed=2))
    with pytest.raises(ValueError, match="no bond to truncate"):
        EnvCTM(classical, init="eye").bond_metric(a, a, (0, 0), (0, 1), "lr")


# --- the truncation and the step ------------------------------------------------------


def test_the_truncation_reports_what_it_did_to_the_metric():
    """``nonhermitian_part``, ``min_eigenvalue`` and ``wrong_eigenvalues`` are measured.

    On an exact cluster the metric is a Gram form: Hermitian to machine precision and
    positive. That is asserted rather than assumed, because the fields exist precisely so
    that an environment which is *not* one says so.
    """
    psi = peps("u1", dims=(2, 2))
    gate = gate_nn(hamiltonian("u1"), 0.2, ((0, 0), (0, 1)))
    out = truncate_(EnvNTU(psi), gate.bond, gate=gate, max_bond=2)
    assert out.nonhermitian_part < 1e-12
    assert out.min_eigenvalue is not None and out.min_eigenvalue > -1e-12
    assert out.wrong_eigenvalues is not None
    plain = truncate_(EnvNTU(psi), gate.bond, gate=gate, max_bond=2, fix_metric=None)
    assert plain.min_eigenvalue is None and plain.wrong_eigenvalues is None


def test_the_least_squares_sweep_never_raises_the_error_it_was_given():
    """The optimization is an optimization: its error is at most the SVD's."""
    psi = peps("trivial", dims=(2, 2), bond=3)
    gate = gate_nn(hamiltonian("trivial"), 0.5, ((0, 0), (0, 1)))
    plain = truncate_(
        EnvNTU(Peps(psi.geometry, dict(psi.items()))), gate.bond, gate=gate, max_bond=2, max_iter=0
    )
    tuned = truncate_(
        EnvNTU(Peps(psi.geometry, dict(psi.items()))), gate.bond, gate=gate, max_bond=2, max_iter=40
    )
    assert tuned.truncation_error <= plain.truncation_error + 1e-12
    assert tuned.iterations > 0 and tuned.pinv_cutoff is not None


def test_the_reported_truncation_error_is_the_one_the_dense_state_shows():
    """The error the record carries is the distance the closed state actually moved."""
    psi = peps("trivial", dims=(2, 2), bond=3)
    gate = gate_nn(hamiltonian("trivial"), 0.5, ((0, 0), (0, 1)))
    exact = Peps(psi.geometry, dict(psi.items()))
    truncate_(EnvNTU(exact), gate.bond, gate=gate, max_bond=None, cutoff=None, max_iter=0)
    cut = Peps(psi.geometry, dict(psi.items()))
    out = truncate_(EnvNTU(cut), gate.bond, gate=gate, max_bond=2, max_iter=40)
    ve = to_dense_state(exact).reshape(-1)
    vt = to_dense_state(cut).reshape(-1)
    overlap = abs(ve @ vt) / np.linalg.norm(ve) / np.linalg.norm(vt)
    assert out.truncation_error == pytest.approx((1 - overlap**2) ** 0.5, rel=1e-3)


def heisenberg(phys: GradedSpace) -> SymmetricTensor:
    """The spin-1/2 Heisenberg bond term on the ungraded physical space."""
    sz = np.diag([0.5, -0.5])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    dense = np.kron(sz, sz) + 0.5 * (np.kron(sp, sp.T) + np.kron(sp.T, sp))
    return local_op(dense, phys=phys)


def exact_energy(bonds, n=4):
    """The 2x2 Heisenberg ground state energy, by dense diagonalization."""
    sz = np.diag([0.5, -0.5])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    eye = np.eye(2)

    def op(m, i):
        out = np.array([[1.0]])
        for k in range(n):
            out = np.kron(out, m if k == i else eye)
        return out

    h = np.zeros((2**n, 2**n))
    for i, j in bonds:
        h = h + op(sz, i) @ op(sz, j)
        h = h + 0.5 * (op(sp, i) @ op(sp.T, j) + op(sp.T, i) @ op(sp, j))
    w, v = np.linalg.eigh(h)
    return h, w[0]


def test_imaginary_time_evolution_reaches_the_exact_ground_state():
    """Oracle (i): 2x2 ``obc`` Heisenberg, evolved, against exact diagonalization.

    The lattice is small enough that the ``'NN'`` cluster is the exact environment and
    that ``D = 4`` represents the state exactly, so the only errors left are the Trotter
    step and the imaginary time not yet run -- both of which the schedule removes. The
    energy is read off the *dense* state, so the comparison never routes through the
    library's own measurement machinery.
    """
    phys = SPACES["trivial"]
    modes = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    geometry = SquareLattice(dims=(2, 2), boundary="obc")
    dense_bonds = [(modes[tuple(b[0])], modes[tuple(b[1])]) for b in geometry.bonds()]
    h_dense, reference = exact_energy(dense_bonds)

    psi = peps("trivial", dims=(2, 2), bond=4, seed=5)
    h = heisenberg(phys)
    for step, sweeps in ((0.2, 40), (0.05, 60), (0.01, 80)):
        gates = gates_nn(geometry, h, step)
        for _ in range(sweeps):
            evolution_step_(EnvNTU(psi), gates, max_bond=4, cutoff=1e-14, max_iter=20)
    v = to_dense_state(psi).reshape(-1)
    v = v / np.linalg.norm(v)
    assert float(v @ h_dense @ v) == pytest.approx(reference, abs=1e-6)


def test_a_smaller_bond_is_variational_from_above_and_says_what_it_cost():
    """``D = 2`` cannot hold the state, so the energy is higher and the error is not zero."""
    phys = SPACES["trivial"]
    modes = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    geometry = SquareLattice(dims=(2, 2), boundary="obc")
    dense_bonds = [(modes[tuple(b[0])], modes[tuple(b[1])]) for b in geometry.bonds()]
    h_dense, reference = exact_energy(dense_bonds)
    psi = peps("trivial", dims=(2, 2), bond=2, seed=5)
    gates = gates_nn(geometry, heisenberg(phys), 0.1)
    infoss = [evolution_step_(EnvNTU(psi), gates, max_bond=2, max_iter=20) for _ in range(60)]
    v = to_dense_state(psi).reshape(-1)
    v = v / np.linalg.norm(v)
    energy = float(v @ h_dense @ v)
    assert energy > reference - 1e-9
    assert accumulated_truncation_error(infoss) > 0.0


def test_accumulated_truncation_error_adds_the_bonds_up_the_way_it_says():
    bond = tuple(SquareLattice(dims=(2, 2)).bonds()[0])
    step = [Evolution_out(bond=bond, truncation_error=0.1)] * 2
    assert accumulated_truncation_error([step, step]) == pytest.approx(0.4)
    assert accumulated_truncation_error([step, step], "max") == pytest.approx(0.4)
    with pytest.raises(ValueError, match="'mean' or 'max'"):
        accumulated_truncation_error([step], "median")


def test_the_environments_refuse_what_they_cannot_do():
    psi = peps("u1", dims=(2, 2))
    with pytest.raises(ValueError, match="only 'NN'"):
        EnvNTU(psi, which="NNN")
    classical = Peps(SquareLattice(dims=(1, 1)), SymmetricTensor.random(psi[0, 0].legs[:4], seed=3))
    with pytest.raises(ValueError, match="physical leg"):
        EnvNTU(classical)
    assert "NN" in repr(EnvNTU(psi))


def test_every_two_operand_step_of_the_evolution_is_a_composition(monkeypatch):
    """The composition rule (#160) on the new module, and the coverage claim asserted.

    ``evolution.py`` reaches ``tenet.einsum`` only through ``common.composed`` and through
    ``peps.py``'s primitives, both of which are ``einsum_chain`` steps carrying their bent
    wires in the step's own field. Every recorded step must have operand 1 supplying the
    ``IN`` end of every shared wire after its bend; and every ``composed`` call site the
    ``ast`` finds in the module must be reached.
    """
    import ast
    import pathlib
    import traceback

    from tenet.network import evolution

    module = pathlib.Path(evolution.__file__)
    seen: set[int] = set()
    real = tenet.einsum_chain

    def spy(steps):
        for frame in traceback.extract_stack():
            if pathlib.Path(frame.filename) == module:
                seen.add(frame.lineno)
        steps = list(steps)
        for equation, a, b, bend in steps:
            left, right = equation.split("->")[0].split(",")
            if a is None or b is None:
                continue
            turned = set(bend)
            for label in left:
                if label not in right:
                    continue
                leg = a.legs[left.index(label)]
                assert ((leg.side is IN) != leg.dual) != (label in turned), equation
        return real(steps)

    monkeypatch.setattr(tenet, "einsum_chain", spy)
    for provider in ("u1", "fz2"):
        psi = peps(provider, dims=(2, 2))
        gates = gates_nn(psi.geometry, hamiltonian(provider), 0.1, symmetrize=False)
        evolution_step_(EnvNTU(psi), gates, max_bond=2, max_iter=3)
    # a CTM environment is stale the moment a bond changes, so each truncation gets a
    # fresh one -- which is also what puts both branches of ``bond_metric`` on the smoke
    psi = peps("u1", dims=(2, 2))
    gate = gate_nn(hamiltonian("u1"), 0.1, ((0, 0), (0, 1)))
    truncate_(EnvCTM(psi, init="eye"), gate.bond, gate=gate, max_bond=2, max_iter=2)
    truncate_(EnvCTM(psi, init="eye"), ((0, 0), (1, 0)), max_bond=2, max_iter=0, fix_metric=None)
    truncate_(EnvCTM(psi, init="eye"), ((0, 1), (0, 0)), max_bond=2, max_iter=0)

    wanted = {
        node.lineno
        for node in ast.walk(ast.parse(module.read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "composed"
    }
    assert len(wanted) > 20, wanted
    assert not wanted - seen, f"the smoke never reached these lines: {sorted(wanted - seen)}"
