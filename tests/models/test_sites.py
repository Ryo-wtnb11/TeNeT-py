"""``tenet.models``: the shipped sites, held to the algebra they claim (M36, #198).

Every assertion below reads the operators back out of the **built** tensors
(``to_dense``), never out of the module's private matrices, so what is tested is the
object a caller gets rather than the array it was made from.
"""

import ast
import pathlib

import numpy as np
import pytest

from tenet import GradedSpace
from tenet.models import (
    Site,
    hard_core_boson,
    hubbard,
    spin_half,
    spinful_fermion,
    spinless_fermion,
)
from tenet.network import MPO, MPS, dmrg_, local_op
from tenet.symmetry import (
    SU2,
    U1,
    Z2,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    U1Sector,
    fZ2,
)

#: The spin-SU(2) grading of the spinful site: parity, particle number, total spin.
FUS = ProductProvider((fZ2, U1, SU2))


def fus(parity: int, charge: int, two_j: int) -> ProductSector:
    return ProductSector((FZ2Sector(parity), U1Sector(charge), SU2Sector(two_j)))


SITES = {
    "spin_half u1": spin_half(),
    "spin_half su2": spin_half(SU2),
    "spinless_fermion": spinless_fermion(),
    "spinful_fermion": spinful_fermion(),
    "spinful_fermion su2": spinful_fermion(FUS),
    "hard_core_boson u1": hard_core_boson(),
    "hard_core_boson trivial": hard_core_boson(Trivial),
}


def matrix(site: Site, name: str) -> np.ndarray:
    """The operator's dense matrix, read back off the built tensor."""
    op = site.ops[name]
    dense = np.asarray(op.to_dense())
    if op.ndim == 3:
        return dense[:, :, 0]
    return np.reshape(dense, (site.phys.dim ** (op.ndim // 2),) * 2)


# --- the acceptance criterion: built through local_op, refusals attached ---------------


def test_the_module_builds_nothing_except_through_local_op():
    """``from_dense`` is never called here, so no operator can dodge its validation."""
    source = (
        pathlib.Path(__file__).parents[2] / "src" / "tenet" / "models" / "sites.py"
    ).read_text()
    calls = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "from_dense" not in calls
    names = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "local_op" in names


@pytest.mark.parametrize("key", sorted(SITES), ids=sorted(SITES))
def test_every_operator_round_trips_the_matrix_it_claims(key):
    """A tensor that survived ``from_dense``'s default atol *is* the declared matrix."""
    site = SITES[key]
    for name, op in site.ops.items():
        assert np.allclose(matrix(site, name), site.matrices[name])
        assert op.legs[0].space == site.phys
    assert set(site.ops) <= set(site.matrices)


@pytest.mark.parametrize(
    ("site", "name", "wrong"),
    [
        (spin_half(), "S+", U1Sector(2)),  # S+ emits -2, not +2
        (spin_half(), "Sz", U1Sector(2)),  # Sz is neutral
        (spinless_fermion(), "c", FZ2Sector(0)),  # c is odd
        (spinful_fermion(), "n_up", FZ2Sector(1)),  # n_up is even
        (hard_core_boson(), "b", U1Sector(-1)),  # b lowers the occupation
    ],
    ids=lambda x: str(x)[:24],
)
def test_a_shipped_matrix_under_a_wrong_charge_still_raises(site, name, wrong):
    """The refusal is carried, not spent: ``from_dense``'s default atol is still the gate."""
    with pytest.raises(ValueError):
        local_op(site.matrices[name], phys=site.phys, charge=wrong)


# --- the algebra oracles ---------------------------------------------------------------


def test_the_u1_spin_operators_obey_the_su2_commutators():
    """``[Sz, S+-] = +-S+-`` and ``[S+, S-] = 2 Sz``, off the built rank-3 tensors."""
    site = spin_half()
    sz, sp, sm = (matrix(site, n) for n in ("Sz", "S+", "S-"))
    assert np.allclose(sz @ sp - sp @ sz, sp)
    assert np.allclose(sz @ sm - sm @ sz, -sm)
    assert np.allclose(sp @ sm - sm @ sp, 2 * sz)
    assert np.allclose(sm, sp.T)


@pytest.mark.parametrize(
    ("key", "pairs"),
    [
        ("spinless", [("c", "c+")]),
        ("spinful", [("c_up", "c+_up"), ("c_dn", "c+_dn")]),
    ],
)
def test_the_fermion_operators_obey_the_canonical_anticommutators(key, pairs):
    """``{c, c+} = 1``, ``{c, c} = 0``, and every cross-flavour pair anticommutes to 0."""
    site = spinless_fermion() if key == "spinless" else spinful_fermion()
    eye = np.eye(site.phys.dim)
    for c_name, cd_name in pairs:
        c, cd = matrix(site, c_name), matrix(site, cd_name)
        assert np.allclose(c @ cd + cd @ c, eye)
        assert np.allclose(c @ c, 0.0)
        assert np.allclose(cd, c.T)
        assert np.allclose(cd @ c, matrix(site, "n" if key == "spinless" else f"n{c_name[1:]}"))
    if key == "spinful":
        up, dn = matrix(site, "c_up"), matrix(site, "c_dn")
        for a, b in ((up, dn), (up, dn.T), (up.T, dn), (up.T, dn.T)):
            assert np.allclose(a @ b + b @ a, 0.0)
        assert np.allclose(matrix(site, "n"), matrix(site, "n_up") + matrix(site, "n_dn"))
        assert np.allclose(matrix(site, "n_up n_dn"), matrix(site, "n_up") @ matrix(site, "n_dn"))


def test_the_hard_core_boson_operators_are_the_hard_core_algebra():
    """``b b = 0``, ``[b, b+] = 1 - 2n`` (hard core), ``n = b+ b``, under both gradings."""
    for site in (hard_core_boson(), hard_core_boson(Trivial)):
        b, bd, n = (matrix(site, name) for name in ("b", "b+", "n"))
        assert np.allclose(b @ b, 0.0)
        assert np.allclose(bd @ b, n)
        assert np.allclose(b @ bd - bd @ b, np.eye(2) - 2 * n)


# --- the SU(2) answer, concretely ------------------------------------------------------


def test_su2_ships_the_invariant_two_site_operator_and_no_ladder():
    """``S+`` is absent because SU(2) has no such operator, not by preference (#198)."""
    site = spin_half(SU2)
    assert set(site.ops) == {"S.S"}
    assert site.ops["S.S"].ndim == 4  # rank 2k, k = 2: one whole term
    # The spin-1 irreducible tensor operator has nowhere to go: the charge-leg form puts
    # the emitted sector on a D=1 leg, and the dense dimension of j=1 is 3.
    with pytest.raises(ValueError):
        local_op(site.matrices["S.S"][:2, :2], phys=site.phys, charge=SU2Sector(2))


def test_su2_s_dot_s_is_invariant_and_u1_s_dot_s_is_the_same_matrix():
    """Invariance is what ``from_dense`` checked; ``Sz (x) Sz`` alone fails it."""
    su2, u1 = spin_half(SU2), spin_half()
    assert np.array_equal(su2.matrices["S.S"], u1.matrices["S.S"])
    sz = u1.matrices["Sz"]
    with pytest.raises(ValueError):
        local_op(np.kron(sz, sz), phys=su2.phys)  # not an SU(2) scalar
    ss = np.reshape(np.asarray(su2.ops["S.S"].to_dense()), (4, 4))
    assert np.allclose(ss, su2.matrices["S.S"])
    assert np.allclose(np.linalg.eigvalsh(ss), [-0.75, 0.25, 0.25, 0.25])  # singlet, triplet


# --- the refusals ----------------------------------------------------------------------


def test_a_grading_that_is_not_shipped_is_refused_by_name():
    with pytest.raises(ValueError, match="U1"):
        spin_half(Z2)
    with pytest.raises(ValueError, match="Trivial"):
        hard_core_boson(SU2)


# --- end to end: the call shape #197 populates -----------------------------------------


def _kron_chain(n_sites, d, factors):
    """``factors`` as ``{site: matrix}``, identity elsewhere, as one dense array."""
    full = np.array([[1.0]])
    for m in range(n_sites):
        full = np.kron(full, factors.get(m, np.eye(d)))
    return full


def test_the_u1_heisenberg_call_shape_matches_a_dense_kron_oracle():
    """``MPO.from_arrays(n, site.ops, blocks)`` -- the whole Hamiltonian, no numpy."""
    site, n = spin_half(), 4
    bond = [(i, i + 1) for i in range(n - 1)]
    h = MPO.from_arrays(
        n,
        site.ops,
        [
            ("Sz Sz", bond, [1.0] * (n - 1)),
            ("S+ S-", bond, [0.5] * (n - 1)),
            ("S- S+", bond, [0.5] * (n - 1)),
        ],
    )
    ss = site.matrices["S.S"]  # the invariant two-site matrix, in np.kron layout
    want = np.zeros((2**n, 2**n))
    for i in range(n - 1):
        want = want + np.kron(np.kron(np.eye(2**i), ss), np.eye(2 ** (n - i - 2)))
    assert np.allclose(np.asarray(h.to_dense()), want)


def test_the_su2_heisenberg_call_shape_is_from_terms_on_the_invariant_operator():
    """The same chain under SU(2): one invariant term per bond, the same dense operator."""
    n = 4
    su2 = spin_half(SU2)
    h = MPO.from_terms(n, [(1.0, [(su2.ops["S.S"], (i, i + 1))]) for i in range(n - 1)])
    u1 = spin_half()
    bond = [(i, i + 1) for i in range(n - 1)]
    ref = MPO.from_arrays(
        n,
        u1.ops,
        [
            ("Sz Sz", bond, [1.0] * (n - 1)),
            ("S+ S-", bond, [0.5] * (n - 1)),
            ("S- S+", bond, [0.5] * (n - 1)),
        ],
    )
    assert np.allclose(np.asarray(h.to_dense()), np.asarray(ref.to_dense()))


def _dense_hubbard(n_sites, u, t=1.0):
    """``-t sum (c+ c + h.c.) + U sum n_up n_dn`` with the Jordan-Wigner string, dense."""
    site = spinful_fermion()
    parity = np.diag([1.0, 1.0, -1.0, -1.0])

    def chain(m, local):
        return _kron_chain(n_sites, 4, {i: parity for i in range(m)} | {m: local})

    h = np.zeros((4**n_sites, 4**n_sites))
    for m in range(n_sites - 1):
        for flavour in ("up", "dn"):
            cd = chain(m, site.matrices[f"c+_{flavour}"])
            c = chain(m + 1, site.matrices[f"c_{flavour}"])
            h = h - t * (cd @ c + (cd @ c).T)
    for m in range(n_sites):
        h = h + u * _kron_chain(n_sites, 4, {m: site.matrices["n_up n_dn"]})
    return h


def test_the_spinful_hubbard_call_shape_matches_the_jordan_wigner_dense_oracle():
    """The fermionic end to end: ``ops`` straight into ``from_arrays``, signs and all."""
    site, n, u = spinful_fermion(), 3, 3.0
    fwd = [(m, m + 1) for m in range(n - 1)]
    bwd = [(m + 1, m) for m in range(n - 1)]
    blocks = []
    for flavour in ("up", "dn"):
        expr = f"c+_{flavour} c_{flavour}"
        blocks += [(expr, fwd, [-1.0] * (n - 1)), (expr, bwd, [-1.0] * (n - 1))]
    # coincident indices: from_arrays' merge multiplies the pair into ``n_up n_dn``
    blocks.append(("n_up n_dn", [(m, m) for m in range(n)], [u] * n))
    h = MPO.from_arrays(n, site.ops, blocks)
    assert np.allclose(np.asarray(h.to_dense()), _dense_hubbard(n, u))


def test_the_spinful_hubbard_term_list_route_agrees():
    """``from_terms`` over the same ``ops``, with ``n_up n_dn`` as the one on-site operator."""
    site, n, u = spinful_fermion(), 3, 3.0
    terms = []
    for m in range(n - 1):
        for flavour in ("up", "dn"):
            cd, c = site.ops[f"c+_{flavour}"], site.ops[f"c_{flavour}"]
            terms += [(-1.0, [(cd, m), (c, m + 1)]), (-1.0, [(cd, m + 1), (c, m)])]
    terms += [(u, [(site.ops["n_up n_dn"], m)]) for m in range(n)]
    h = MPO.from_terms(n, terms)
    assert np.allclose(np.asarray(h.to_dense()), _dense_hubbard(n, u))


def test_the_spinless_fermion_call_shape_matches_its_dense_oracle():
    site, n = spinless_fermion(), 4
    fwd = [(m, m + 1) for m in range(n - 1)]
    bwd = [(m + 1, m) for m in range(n - 1)]
    h = MPO.from_arrays(
        n, site.ops, [("c+ c", fwd, [-1.0] * (n - 1)), ("c+ c", bwd, [-1.0] * (n - 1))]
    )
    parity = np.diag([1.0, -1.0])
    want = np.zeros((2**n, 2**n))
    for m in range(n - 1):
        cd = _kron_chain(n, 2, {i: parity for i in range(m)} | {m: site.matrices["c+"]})
        c = _kron_chain(n, 2, {i: parity for i in range(m + 1)} | {m + 1: site.matrices["c"]})
        want = want - (cd @ c + (cd @ c).T)
    assert np.allclose(np.asarray(h.to_dense()), want)


# --- the spin-SU(2) spinful site -------------------------------------------------------


def test_the_su2_spinful_space_is_four_states_in_three_multiplets():
    """``dim`` counts states, ``reduced_dim`` multiplets, and the order is the graded one."""
    phys = spinful_fermion(FUS).phys
    assert (phys.dim, phys.reduced_dim) == (4, 3)
    # Empty and doubly occupied are even spin singlets, the singly occupied states one
    # odd j = 1/2 doublet -- the same even-before-odd basis (|0>, |ud>, |u>, |d>) the
    # fZ2 site is written in, which is why its matrices carry over.
    assert phys.sectors == ((fus(0, 0, 0), 1), (fus(0, 2, 0), 1), (fus(1, 1, 1), 1))


def test_the_su2_spinful_site_ships_only_invariant_operators():
    """No ``c_up``, and not by preference: neither of ``local_op``'s forms can hold it."""
    site, fz2 = spinful_fermion(FUS), spinful_fermion()
    assert sorted(site.ops) == ["S.S", "hop", "n", "n_up n_dn"]
    assert sorted(site.matrices) == ["S.S", "hop", "n", "n_up n_dn"]
    assert not {"c_up", "c+_up", "c_dn", "c+_dn", "n_up", "n_dn"} & set(site.ops)
    # The charge-leg form puts the emitted sector on a D=1 *dense* leg, and the sector
    # c_up emits is the j = 1/2 doublet, of dense dimension 2.
    with pytest.raises(ValueError):
        local_op(fz2.matrices["c_up"], phys=site.phys, charge=fus(1, -1, 1))
    # ... and c_up is no scalar either, so the invariant form refuses it too.
    with pytest.raises(ValueError):
        local_op(fz2.matrices["c_up"], phys=site.phys)
    # n_up alone is not invariant; only the sum and the product of the two are.
    with pytest.raises(ValueError):
        local_op(fz2.matrices["n_up"], phys=site.phys)


@pytest.mark.parametrize(
    "provider",
    [U1, ProductProvider((U1, fZ2)), ProductProvider((fZ2, SU2, U1))],
    ids=["u1", "u1 x fZ2", "wrong order"],
)
def test_a_spinful_grading_that_is_not_shipped_is_refused_by_name(provider):
    """The factors must be exactly ``(fZ2, U1, SU2)`` in that order."""
    with pytest.raises(ValueError, match="fZ2"):
        spinful_fermion(provider)
    with pytest.raises(ValueError, match=type(provider).__name__):
        spinful_fermion(provider)


def test_the_diagonal_matrices_are_the_fz2_ones_unchanged():
    """Both bases are even before odd, so ``n`` and ``n_up n_dn`` are the same arrays."""
    su2, fz2 = spinful_fermion(FUS), spinful_fermion()
    for name in ("n", "n_up n_dn"):
        assert np.array_equal(su2.matrices[name], fz2.matrices[name])
        assert np.allclose(matrix(su2, name), fz2.matrices[name])


def test_the_hop_operator_densifies_to_the_jordan_wigner_kron_expression():
    """The dense oracle: ``sum_sigma (c+_i Z (x) c_j + h.c.)``, written out here."""
    su2, fz2 = spinful_fermion(FUS), spinful_fermion()
    parity = np.diag([1.0, 1.0, -1.0, -1.0])
    fwd = sum(
        np.kron(fz2.matrices[f"c+_{s}"] @ parity, fz2.matrices[f"c_{s}"]) for s in ("up", "dn")
    )
    assert np.allclose(matrix(su2, "hop"), fwd + fwd.T)


def test_the_spin_exchange_is_the_spin_half_one_on_the_singly_occupied_block():
    """``S.S`` acts on the doublet and annihilates the singlets, so its 1-1 block is spin-1/2."""
    su2 = spinful_fermion(FUS)
    ss = matrix(su2, "S.S")
    singly = [4 * a + b for a in (2, 3) for b in (2, 3)]  # |u>, |d> of the graded basis
    block = ss[np.ix_(singly, singly)]
    assert np.allclose(np.linalg.eigvalsh(block), [-0.75, 0.25, 0.25, 0.25])
    assert np.allclose(ss[:, 0], 0.0)  # |0>|0> carries no spin


# --- the decisive one: the Hubbard chain, both gradings --------------------------------

HUBBARD_N, HUBBARD_T, HUBBARD_U = 4, 1.0, 4.0


def _su2_hubbard(n: int, t: float, u: float) -> MPO:
    """The Hubbard chain on the SU(2) site: a hopping bond and an on-site ``U``.

    Written the way the model reads -- ``-t`` on each bond, ``u`` on each site. The
    on-site term is ``local_op``'s invariant form on one site, the rank-2 term operator,
    which is the only form this physical space admits: ``irrep_dim > 1`` here, so no
    operator of it has a rank-3 charge-leg form.
    """
    site = spinful_fermion(FUS)
    hop = local_op(site.matrices["hop"], phys=site.phys)
    nn = local_op(site.matrices["n_up n_dn"], phys=site.phys)
    terms = [(-t, [(hop, (i, i + 1))]) for i in range(n - 1)]
    return MPO.from_terms(n, terms + [(u, [(nn, i)]) for i in range(n)])


def _half_filled_singlet_energy(n: int, t: float, u: float) -> float:
    """The same chain on the ``fZ2`` site, diagonalized densely at half filling, ``S^z = 0``.

    The oracle is a dense diagonalization rather than a second DMRG run because ``fZ2``
    conserves the parity alone: a run on that grading cannot be held to ``n`` electrons,
    and its ground state is at some other filling. Restricting the dense Hamiltonian by
    the two diagonal charges is what fixes the sector the SU(2) seed picks structurally.
    """
    fz2 = spinful_fermion()
    dense = np.asarray(hubbard(n, t=t, U=u).to_dense())
    n_tot = sum(_kron_chain(n, 4, {m: fz2.matrices["n"]}) for m in range(n))
    sz = (fz2.matrices["n_up"] - fz2.matrices["n_dn"]) / 2
    sz_tot = sum(_kron_chain(n, 4, {m: sz}) for m in range(n))
    keep = (np.abs(np.diag(n_tot) - n) < 1e-9) & (np.abs(np.diag(sz_tot)) < 1e-9)
    return float(np.linalg.eigvalsh(dense[np.ix_(keep, keep)])[0])


@pytest.fixture(scope="module")
def su2_hubbard_run():
    """One DMRG run on the product-graded chain, shared by the two tests that read it."""
    n = HUBBARD_N
    site = spinful_fermion(FUS)
    # D=1 boundaries carry the target sector: the left bond is the vacuum and the right
    # one the total charge, so the seed is half filling in the total-spin-zero channel
    # and the sweeps never leave it. The interior offers a couple of copies of every
    # sector a chain of this length can reach; multiplicities the ground state needs grow
    # under the sweeps and ones it does not are truncated away.
    empty = GradedSpace.new(FUS, {fus(0, 0, 0): 1})
    full = GradedSpace.new(FUS, {fus(0, n, 0): 1})
    mid = GradedSpace.new(
        FUS,
        {fus(0, q, 0): 2 for q in range(0, 2 * n + 1, 2)}
        | {fus(0, q, 2): 1 for q in range(2, 2 * n, 2)}
        | {fus(1, q, 1): 2 for q in range(1, 2 * n, 2)},
    )
    psi = MPS.random(site.phys, [empty] + [mid] * (n - 1) + [full], seed=0)
    return dmrg_(psi, _su2_hubbard(n, HUBBARD_T, HUBBARD_U), chi=32)


def test_the_su2_hubbard_chain_reaches_the_fz2_chains_ground_state_energy(su2_hubbard_run):
    """The acceptance criterion: same model, same number, one grading finer than the other."""
    want = _half_filled_singlet_energy(HUBBARD_N, HUBBARD_T, HUBBARD_U)
    assert abs(su2_hubbard_run.energy - want) < 1e-8


def test_the_su2_hubbard_bond_is_multiplet_compressed(su2_hubbard_run):
    """``dim`` counts dense states, ``reduced_dim`` multiplets; the gap is what SU(2) buys."""
    mid = su2_hubbard_run.psi[HUBBARD_N // 2].legs[0].space
    assert mid.reduced_dim < mid.dim


def test_the_su2_hubbard_mpo_is_the_fz2_one_operator_for_operator():
    """The on-site ``U`` as its own term is the same operator the ``fZ2`` model builds.

    The ``fZ2`` grading has a rank-3 form of ``n_up n_dn`` and this one does not, so the
    two spellings of the Hubbard chain are independent -- one charge-leg operator per
    site against one invariant operator per bond plus one per site -- and ``to_dense``
    is the common exit where they can be compared.
    """
    n = HUBBARD_N
    want = np.asarray(hubbard(n, t=HUBBARD_T, U=HUBBARD_U).to_dense())
    got = np.asarray(_su2_hubbard(n, HUBBARD_T, HUBBARD_U).to_dense())
    assert np.abs(got - want).max() < 1e-12
