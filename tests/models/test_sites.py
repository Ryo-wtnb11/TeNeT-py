"""``tenet.models``: the shipped sites, held to the algebra they claim (M36, #198).

Every assertion below reads the operators back out of the **built** tensors
(``to_dense``), never out of the module's private matrices, so what is tested is the
object a caller gets rather than the array it was made from.
"""

import ast
import pathlib

import numpy as np
import pytest

from tenet.models import Site, hard_core_boson, spin_half, spinful_fermion, spinless_fermion
from tenet.network import MPO, local_op
from tenet.symmetry import SU2, Z2, FZ2Sector, SU2Sector, Trivial, U1Sector

SITES = {
    "spin_half u1": spin_half(),
    "spin_half su2": spin_half(SU2),
    "spinless_fermion": spinless_fermion(),
    "spinful_fermion": spinful_fermion(),
    "hard_core_boson u1": hard_core_boson(),
    "hard_core_boson trivial": hard_core_boson(Trivial),
}


def matrix(site: Site, name: str) -> np.ndarray:
    """The operator's dense matrix, read back off the built tensor."""
    op = site.ops[name]
    dense = np.asarray(op.to_dense())
    return dense[:, :, 0] if op.ndim == 3 else np.reshape(dense, (site.phys.dim**2,) * 2)


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
