"""M61 Stage D (#232): ``Env`` with ``bra`` not ``psi``, and what it does and refuses.

The engine half of #213. Every claim here is pinned against an explicit dense oracle --
``MPS.to_dense`` and ``MPO.to_dense`` contract through ``tenet.einsum``, so the graded
providers pay their braiding on the reference side too -- and each runs on U(1), fZ2 and
SU(2), because a mixed ``<phi|...|psi>`` transfer is exactly where a cap direction that
is right for a chain against itself can be wrong for two different chains.
"""

import numpy as np
import pytest

import tenet
from tenet import GradedSpace
from tenet.network import MPO, MPS, Env, local_op
from tenet.symmetry import SU2, U1, FZ2Sector, SU2Sector, U1Sector, fZ2


def _spin_half():
    sz = np.diag([0.5, -0.5])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    return sz, sp


def _u1(n_sites):
    """U(1) Heisenberg: the physical space, a Hamiltonian, and reachable bond spaces."""
    phys = GradedSpace.new(U1, {U1Sector(1): 1, U1Sector(-1): 1})
    sz, sp = _spin_half()
    op = {
        0: local_op(sz, phys=phys, charge=U1Sector(0)),
        2: local_op(sp, phys=phys, charge=U1Sector(2)),
        -2: local_op(sp.T, phys=phys, charge=U1Sector(-2)),
    }
    terms = []
    for i in range(n_sites - 1):
        terms += [
            (1.0, [(op[0], i), (op[0], i + 1)]),
            (0.5, [(op[2], i), (op[-2], i + 1)]),
            (0.5, [(op[-2], i), (op[2], i + 1)]),
        ]
    bonds = [
        GradedSpace.new(U1, {U1Sector(q): 1 for q in range(-w, w + 1, 2)})
        for w in (min(i, n_sites - i) for i in range(n_sites + 1))
    ]
    return phys, MPO.from_terms(n_sites, terms), bonds


def _fz2(n_sites):
    """Spinless fermions: fZ2, so the Koszul string is live on every bond."""
    phys = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = local_op(a.T, phys=phys, charge=FZ2Sector(1))
    c = local_op(a, phys=phys, charge=FZ2Sector(1))
    terms = []
    for i in range(n_sites - 1):
        terms += [(-1.0, [(cd, i), (c, i + 1)]), (-1.0, [(cd, i + 1), (c, i)])]
    unit = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
    both = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})
    return phys, MPO.from_terms(n_sites, terms), [unit] + [both] * (n_sites - 1) + [unit]


def _su2(n_sites):
    """SU(2) Heisenberg: one invariant ``S.S``, so recoupling is exercised."""
    phys = GradedSpace.new(SU2, {SU2Sector(1): 1})
    sz, sp = _spin_half()
    ss = local_op(np.kron(sz, sz) + (np.kron(sp, sp.T) + np.kron(sp.T, sp)) / 2, phys=phys)
    h = MPO.from_terms(n_sites, [(1.0, [(ss, (i, i + 1))]) for i in range(n_sites - 1)])
    bonds = [
        GradedSpace.new(SU2, {SU2Sector(j): 1 for j in range(i % 2, min(i, n_sites - i) + 1, 2)})
        for i in range(n_sites + 1)
    ]
    return phys, h, bonds


MODELS = {"u1": _u1, "fz2": _fz2, "su2": _su2}
N = 6


def _pair(model, seeds=(1, 5)):
    phys, h, bonds = MODELS[model](N)
    return (
        phys,
        h,
        MPS.random(phys, bonds, seed=seeds[0]).canonize_(),
        MPS.random(phys, bonds, seed=seeds[1]).canonize_(),
    )


def _hs(a, b):
    """``<a|b>`` in the pairing ``norm`` induces -- see ``dmrg._dot`` for why not ``inner``."""
    m = tenet.repartition(a, (0, 1), (2, 3))
    return float(
        tenet.full_trace(tenet.compose(tenet.adjoint(m), tenet.repartition(b, (0, 1), (2, 3))))
    )


@pytest.mark.parametrize("model", sorted(MODELS))
def test_a_two_state_env_measures_the_mixed_matrix_element(model):
    """``Env(psi, h, bra=phi).measure()`` is ``<phi|H|psi>``, against the dense oracle."""
    _phys, h, psi, phi = _pair(model)
    dense_h = np.asarray(h.to_dense())
    v_psi = np.asarray(psi.to_dense()).reshape(-1)
    v_phi = np.asarray(phi.to_dense()).reshape(-1)
    got = Env(psi, h, bra=phi).measure()
    assert got == pytest.approx(float(v_phi @ dense_h @ v_psi), abs=1e-11)
    # and it is symmetric in the two states for a Hermitian H, which a wrong cap
    # direction on one side of the transfer would break
    assert Env(phi, h, bra=psi).measure() == pytest.approx(got, abs=1e-11)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_the_identity_operator_turns_the_two_state_env_into_an_overlap(model):
    """With [MPO.identity][] the same object is ``<phi|psi>``, and ``<psi|psi>`` is the norm."""
    phys, _h, psi, phi = _pair(model)
    ident = MPO.identity(N, phys)
    v_psi = np.asarray(psi.to_dense()).reshape(-1)
    v_phi = np.asarray(phi.to_dense()).reshape(-1)
    assert Env(psi, ident, bra=phi).measure() == pytest.approx(float(v_phi @ v_psi), abs=1e-12)
    assert Env(psi, ident).measure() == pytest.approx(psi.norm() ** 2, abs=1e-12)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_bra_is_psi_is_exactly_the_one_state_env(model):
    """Passing ``bra=psi`` explicitly changes nothing -- the default is not a second path."""
    _phys, h, psi, _phi = _pair(model)
    assert Env(psi, h, bra=psi).measure() == pytest.approx(Env(psi, h).measure(), abs=0.0)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_project2_is_the_overlap_read_at_one_bond(model):
    """``<project2(n, phi_pair), psi_pair>`` is ``<phi|psi>`` at every bond, both directions.

    The projection vector is what the excited-state sweep hands to ``lanczos``, and this
    is the statement it stands on: pairing it with the sweeping state's own two-site
    tensor reproduces the overlap the whole-chain contraction gives.
    """
    phys, _h, psi, phi = _pair(model)
    env = Env(phi, MPO.identity(N, phys), bra=psi).setup_()
    ref = env.measure()
    for n in range(N - 1):
        if n:  # walk psi's centre across the bond, exactly as the sweep does
            aa = tenet.einsum("apx,xqr->apqr", psi[n - 1], psi[n])
            u, s, vh = tenet.linalg.svd_truncated(aa, ((0, 1), (2, 3)), max_bond=64, cutoff=1e-14)
            psi[n - 1] = u
            psi[n] = vh
            psi[n] = tenet.einsum("xy,yqr->xqr", s, psi[n])
            env.clear_(n - 1, n)
            env.update_(n - 1, to="last")
        p = env.project2(n, tenet.einsum("apx,xqr->apqr", phi[n], phi[n + 1]))
        aa = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
        assert _hs(p, aa) == pytest.approx(ref, abs=1e-11)


def test_a_two_state_env_refuses_the_prepared_matvec():
    """``heff2`` reads the IdL/IdR channels as gauge identities; a mixed transfer is not one."""
    _phys, h, psi, phi = _pair("u1")
    env = Env(psi, h, bra=phi).setup_()
    aa = tenet.einsum("apx,xqr->apqr", psi[0], psi[1])
    with pytest.raises(ValueError, match="one-state operation"):
        env.heff2(0, aa)
    with pytest.raises(ValueError, match="one-state operation"):
        env.heff2_families(0, aa)


def test_two_chains_must_share_a_bond_sector_to_be_contracted():
    """The standing limitation, asserted rather than described.

    Two chains whose bond spaces share no sector at some cut have an identically zero
    transfer there, and ``tenet.compose`` has no block to take its backend reference
    from -- so the two-state ``Env`` raises instead of returning the structural zero.
    It never bites the excited-state workflow, whose states are seeded on one set of
    bond spaces and swept together, and it is why
    [dmrg_][tenet.network.dmrg_] skips an ``orthogonal_to`` state whose boundary leg
    puts it in another sector: that state is orthogonal already.
    """
    phys = GradedSpace.new(U1, {U1Sector(1): 1, U1Sector(-1): 1})
    up, down = U1Sector(1), U1Sector(-1)
    psi = MPS.product(phys, [up, down, up, down])
    phi = MPS.product(phys, [up, up, down, down])  # same sector, disjoint bond spaces
    with pytest.raises(IndexError):
        Env(psi, MPO.identity(4, phys), bra=phi).measure()
