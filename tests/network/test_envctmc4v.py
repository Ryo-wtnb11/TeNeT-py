"""M79c (#277): the C4v environment, against Onsager and against the ansatz it needs.

Three things are settled here.

* **The flip is the sublattice partner.** ``tenet.conj(tenet.adjoint(t))`` reverses every
  ``side`` and keeps every block, so ``flip(A)``'s left leg contracts with ``A``'s right
  one; ``tenet.flip_dual`` does not, and the difference is measured rather than asserted.
* **Onsager, through one corner and one edge.** The classical Ising bulk carries the full
  point group, so the C4v environment is exact for it, and the free energy per site lands
  on the closed form in both phases -- graded and ungraded, ``'eye'`` and ``'dl'``.
* **#243's instrument, on this lane.** The enlarged corner is measured, found *not*
  Hermitian on an iPEPS double layer, and the sweep converges anyway: the projector is an
  SVD of the enlarged corner whose two index groups leave as two different factors, and
  the correction ``V^dagger U`` between them is kept rather than assumed to be one.

The iPEPS fixture of ``tests/integration/test_ctmrg.py`` cannot come here, and
``test_the_toy_u1_virtual_space_carries_no_c4v_ansatz`` is that statement with a number
behind it: its U(1) virtual space is not self-conjugate, so no rotation acts on any
tensor over it and the C4v ansatz does not exist. The lane refuses it.
"""

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import EnvCTMc4v, Peps, PepsFlip, SquareLattice, corner2x2, flip
from tenet.structure import TensorStructure
from tenet.symmetry import SU2, U1, Z2, SU2Sector, Trivial, TrivialSector, U1Sector, Z2Sector

CHI = 16

# --- the fixtures ---------------------------------------------------------------------


def ising(beta: float, graded: bool = True) -> SymmetricTensor:
    """The Boltzmann tensor on four *identical* legs -- the C4v ansatz's signature.

    ``a[t,l,b,r] = sum_s W[s,t] W[s,l] W[s,b] W[s,r]``, the same numbers
    ``tests/network/test_envctm.py`` builds on the alternating ``Peps`` signature. It is
    symmetric under every permutation of its four legs, so it carries the whole point
    group and one corner and one edge describe its environment exactly.
    """
    c, s = np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))
    w = np.array([[c, s], [c, -s]])
    block = np.einsum("st,sl,sb,sr->tlbr", w, w, w, w)
    space = (
        GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
        if graded
        else GradedSpace.new(Trivial, {TrivialSector(): 2})
    )
    return SymmetricTensor.from_dense(block, (Leg(space, OUT),) * 4)


def onsager(beta: float, points: int = 200_001) -> float:
    """``beta f`` by quadrature -- the same form ``examples/toy_codes/ising.py`` uses."""
    kk = 1.0 / np.sinh(2.0 * beta) ** 2
    theta = np.linspace(0.0, np.pi, points)
    integrand = np.log(
        np.cosh(2.0 * beta) ** 2 + np.sqrt(1.0 + kk**2 - 2.0 * kk * np.cos(2.0 * theta)) / kk
    )
    return -(np.log(2.0) / 2.0 + np.trapezoid(integrand, theta) / (2.0 * np.pi))


#: The eight elements of C4v as permutations of ``(t, l, b, r)``, physical leg last.
_ROT, _MIRROR = (1, 2, 3, 0, 4), (1, 0, 3, 2, 4)


def _group(rank: int) -> list[tuple[int, ...]]:
    compose = lambda p, q: tuple(p[i] for i in q)  # noqa: E731
    elements, current = [], tuple(range(5))
    for _ in range(4):
        elements.append(current[:rank])
        elements.append(compose(current, _MIRROR)[:rank])
        current = compose(current, _ROT)
    return elements


def c4v_ipeps(virtual: GradedSpace, physical: GradedSpace, seed: int = 1) -> SymmetricTensor:
    """A random iPEPS averaged over the point group: the ansatz this lane needs.

    Four identical virtual legs, so the 90-degree rotation is the cyclic transpose of the
    first four axes and the average over the eight elements is invariant under it.
    """
    legs = (Leg(virtual, OUT),) * 4 + (Leg(physical, OUT),)
    a = SymmetricTensor.random(legs, seed=seed)
    out = None
    for permutation in _group(5):
        turned = tenet.transpose(a, permutation)
        out = turned if out is None else out + turned
    return out / 8


def log_kappa(env: EnvCTMc4v, site=(0, 0)):
    """``ln`` of the partition function per site, Baxter's corner-transfer telescoping.

    ``kappa = z_a z_c / z_h**2``: four corners cover an ``L x L`` patch, four corners with
    four edges and the bulk tensor cover ``(L + 1) x (L + 1)``, and four corners with two
    edges cover ``L x (L + 1)``. Every contracted pair meets ``IN`` against ``OUT``, which
    for the four-corner and four-corner-two-edge objects means one of each pair crosses a
    sublattice boundary and enters flipped; the eight-tensor ring around a site does not,
    because corners and edges already alternate around it.
    """
    e, a = env[site], env.psi[site]
    c, cf, t, tf = e.tl, flip(e.tl), e.t, flip(e.t)
    z_c = tenet.full_trace(tenet.einsum("ab,ac,dc,eb->de", c, cf, c, cf))
    z_h = tenet.full_trace(tenet.einsum("ab,ac,dfc,ed,eg,gfh->hb", c, cf, tf, cf, c, t))
    z_a = tenet.full_trace(
        tenet.einsum("ab,apc,cd,eqd,fe,grf,gh,hsk,spqr->kb", c, t, c, t, c, t, c, t, a)
    )
    return ar.do("log", z_a * z_c / z_h**2)


# --- the flip -------------------------------------------------------------------------


def test_flip_reverses_every_side_and_moves_no_element():
    a = ising(0.4)
    b = flip(a)
    assert [leg.side for leg in b.legs] == [IN] * 4
    assert [(leg.space, leg.dual) for leg in b.legs] == [(leg.space, leg.dual) for leg in a.legs]
    assert np.allclose(np.asarray(b.to_dense()), np.asarray(a.to_dense()))
    assert tenet.allclose(flip(b), a)


def test_flip_is_what_makes_the_checkerboard_contract_and_flip_dual_is_not():
    """The two candidate primitives, told apart by what each one does to a leg.

    ``A``'s right leg meets its neighbour's left leg, and both are ``OUT``: the partner is
    whatever reverses ``side``. ``flip`` does, and the contraction goes through.
    ``tenet.flip_dual`` leaves every ``side`` alone -- it toggles ``dual`` and relabels the
    space, keeping the *same morphism*, so it is the wrong end of the wire whatever the
    grading lets through.
    """
    a = ising(0.4)
    assert tenet.einsum("tlbx,TxBR->tlbTBR", a, flip(a)).ndim == 6
    turned = tenet.flip_dual(a, (0, 1, 2, 3))
    assert [leg.side for leg in turned.legs] == [OUT] * 4
    assert [leg.dual for leg in turned.legs] == [True] * 4
    assert [leg.side for leg in flip(a).legs] == [IN] * 4
    assert [leg.dual for leg in flip(a).legs] == [False] * 4


def test_peps_flip_flips_the_odd_sublattice_only():
    psi = PepsFlip(Peps(SquareLattice(dims=(1, 1)), ising(0.4)))
    assert [leg.side for leg in psi[0, 0].legs] == [OUT] * 4
    assert [leg.side for leg in psi[0, 1].legs] == [IN] * 4
    assert [leg.side for leg in psi[1, 1].legs] == [OUT] * 4


# --- what the lane refuses ------------------------------------------------------------


def test_a_multi_site_geometry_is_refused():
    psi = Peps(SquareLattice(dims=(2, 2)), ising(0.4))
    with pytest.raises(ValueError, match="one unique site"):
        EnvCTMc4v(psi)


def test_an_ansatz_whose_four_virtual_legs_differ_is_refused():
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    legs = (Leg(space, IN), Leg(space, OUT), Leg(space, OUT), Leg(space, IN))
    psi = Peps(SquareLattice(dims=(1, 1)), SymmetricTensor.random(legs, seed=0))
    with pytest.raises(ValueError, match="must be identical"):
        EnvCTMc4v(psi)


def test_the_toy_u1_virtual_space_carries_no_c4v_ansatz():
    """``examples/toy_codes/ctmrg.py``'s U(1) fixture, and why it stays off this lane.

    Its virtual space is ``{q = 0: 1, q = +1: 1}``, whose conjugate is
    ``{q = 0: 1, q = -1: 1}``. The 90-degree rotation identifies a virtual space with its
    dual, so a rotation-covariant tensor exists only over a self-conjugate space -- and
    with four legs that are not one leg, this lane refuses the tensor outright.
    """
    virtual = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    assert virtual.sectors != GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(-1): 1}).sectors
    physical = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    legs = (Leg(virtual, OUT), Leg(virtual, OUT), Leg(virtual, IN), Leg(virtual, IN))
    psi = Peps(SquareLattice(dims=(1, 1)), SymmetricTensor.random((*legs, Leg(physical, OUT))))
    with pytest.raises(ValueError, match="must be identical"):
        EnvCTMc4v(psi)


def test_a_move_other_than_d_is_refused():
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising(0.4)))
    with pytest.raises(ValueError, match="only 'd'"):
        env.update_(max_bond=4, moves="hv")


def test_an_unknown_init_is_refused():
    psi = Peps(SquareLattice(dims=(1, 1)), ising(0.4))
    with pytest.raises(ValueError, match="should be 'eye', 'dl' or None"):
        EnvCTMc4v(psi, init="rand")


# --- Onsager --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.3, 0.4, 0.5])
def test_free_energy_matches_onsager_in_both_phases(beta):
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising(beta)))
    out = env.iterate_(max_bond=CHI, max_sweeps=300, corner_tol=1e-12)
    assert out.converged
    assert float(-log_kappa(env)) == pytest.approx(onsager(beta), rel=1e-9)


def test_the_ungraded_bulk_lands_on_the_same_number():
    graded = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising(0.4)))
    graded.iterate_(max_bond=CHI, max_sweeps=300, corner_tol=1e-12)
    plain = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising(0.4, graded=False)))
    plain.iterate_(max_bond=CHI, max_sweeps=300, corner_tol=1e-12)
    assert float(-log_kappa(plain)) == pytest.approx(float(-log_kappa(graded)), rel=1e-9)


def test_the_dl_seed_reaches_the_same_environment():
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising(0.4)), init="dl")
    env.iterate_(max_bond=CHI, max_sweeps=300, corner_tol=1e-12)
    assert float(-log_kappa(env)) == pytest.approx(onsager(0.4), rel=1e-9)


def test_the_environment_is_one_corner_and_one_edge_read_eight_ways():
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising(0.4)))
    env.update_(max_bond=CHI)
    local = env[0, 0]
    assert local.tr is local.tl and local.bl is local.tl and local.br is local.tl
    assert local.l is local.t and local.b is local.t and local.r is local.t
    assert env[0, 1].tl.legs[0].side is IN  # the odd sublattice, flipped on the way out


# --- the double layer -----------------------------------------------------------------

DOUBLE = {
    "su2": (
        GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 1}),
        GradedSpace.new(SU2, {SU2Sector(1): 1}),
    ),
    "u1": (
        GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(0): 1, U1Sector(1): 1}),
        GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1}),
    ),
}


@pytest.mark.parametrize("provider", ["su2", "u1"])
def test_a_double_layer_c4v_sweep_converges(provider):
    """A point-group-averaged iPEPS over a *self-conjugate* virtual space.

    SU(2)'s spaces are all self-conjugate; the U(1) one here is ``{-1, 0, +1}``, chosen
    self-conjugate for the same reason -- without it there is no rotation to average over.
    """
    a = c4v_ipeps(*DOUBLE[provider])
    assert tenet.allclose(a, tenet.transpose(a, _ROT))
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), a))
    out = env.iterate_(max_bond=8, max_sweeps=200, corner_tol=1e-10)
    assert out.converged
    assert env[0, 0].t.ndim == 4  # the ket bond and the bra bond, adjacent and separate


# --- #243, on this lane ---------------------------------------------------------------


def _hermiticity(t: SymmetricTensor) -> float:
    """``||B - B^H|| / ||B||`` with the enlarged corner's two mirror groups as its two
    matrix indices. Dense, because the two groups sit on the *same* side -- the corner is
    a bilinear form, not a map -- so the question is about the array and not about a
    morphism the leg metadata would have to agree to."""
    dense = np.asarray(t.to_dense())
    side = int(np.prod(dense.shape[: t.ndim // 2]))
    m = dense.reshape(side, side)
    return float(np.linalg.norm(m - m.conj().T) / np.linalg.norm(m))


def test_the_full_point_group_is_what_a_hermitian_corner_needs():
    """M63/#243's finding, stated forwards: Hermiticity is a property of the *ansatz*.

    Both fixtures here carry the whole point group -- the Ising bulk by construction, the
    iPEPS by the eight-element average -- and both enlarged corners come back Hermitian to
    machine precision at every sweep. That is the only circumstance under which they do.
    """
    for psi in (
        Peps(SquareLattice(dims=(1, 1)), ising(0.4)),
        Peps(SquareLattice(dims=(1, 1)), c4v_ipeps(*DOUBLE["u1"])),
    ):
        env = EnvCTMc4v(psi)
        for _ in range(6):
            env.update_(max_bond=8)
            assert _hermiticity(corner2x2(env, "tl", (0, 0))) < 1e-12


def test_a_broken_rotation_leaves_the_corner_non_hermitian_and_the_sweep_converges():
    """And stated backwards, which is the projector's half of #243.

    Drop the average and keep the signature: four identical virtual legs, so the lane
    accepts the tensor, but no rotation acts on it. The enlarged corner is then nowhere
    near Hermitian, and the sweep converges regardless -- because the projector is an SVD
    whose two factors stay two factors, and the correction ``V^dagger U`` the renormalized
    corner carries is kept rather than assumed to be one. A single isometry on both index
    groups, or an eigendecomposition, would need the number below to be zero.
    """
    virtual, physical = DOUBLE["u1"]
    legs = (Leg(virtual, OUT),) * 4 + (Leg(physical, OUT),)
    a = SymmetricTensor.random(legs, seed=5)
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), a))
    distances = []
    for _ in range(6):
        env.update_(max_bond=8)
        distances.append(_hermiticity(corner2x2(env, "tl", (0, 0))))
    assert min(distances) > 1e-2, distances
    assert _hermiticity(env[0, 0].tl) > 1e-6
    assert env.iterate_(max_bond=8, max_sweeps=300, corner_tol=1e-10).converged


# --- the trace boundary: JAX from here down -------------------------------------------

jax = pytest.importorskip("jax")


def traced_ising(beta):
    """``ising(beta)`` with a *traced* ``beta``: blocks through ``jax.numpy``, no
    ``from_dense``, exactly as ``examples/toy_codes/ising.py`` builds its own."""
    c, s = jax.numpy.sqrt(jax.numpy.cosh(beta)), jax.numpy.sqrt(jax.numpy.sinh(beta))
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    legs = (Leg(space, OUT),) * 4
    blocks = {}
    for key in TensorStructure(legs).block_order:
        w = [c if sector.parity == 0 else s for sector in key.output_tree.uncoupled]
        w += [c if sector.parity == 0 else s for sector in key.input_tree.uncoupled]
        blocks[key] = jax.numpy.full((1, 1, 1, 1), 2.0 * (w[0] * w[1] * w[2] * w[3]))
    return SymmetricTensor.from_blocks(legs, blocks)


def test_grad_through_unrolled_fixed_bond_moves_matches_onsager():
    """Decide the bond outside, project inside -- and differentiate the inside.

    ``update_()`` reads singular *values* to decide which sectors survive, so it cannot
    be traced; ``update_(bond=B)`` reuses a bond decided out here and is shape-static, so
    ``k`` of them are one trace. The oracle is Onsager's internal energy, and the
    central difference of the closed form is the second one.
    """
    beta, k, delta = 0.4, 4, 1e-5
    warm = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_ising(beta)))
    assert warm.iterate_(max_bond=CHI, max_sweeps=300, corner_tol=1e-11).converged
    bond = warm[0, 0].tl.legs[0].space
    seed = (warm[0, 0].tl, warm[0, 0].t)

    def beta_free_energy(b):
        env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_ising(b)), init=None)
        env.env[0, 0].tl, env.env[0, 0].t = seed
        for _ in range(k):
            env.update_(bond=bond)
        return -log_kappa(env)

    got = float(jax.grad(beta_free_energy)(beta))
    oracle = (onsager(beta + delta) - onsager(beta - delta)) / (2 * delta)
    assert got == pytest.approx(oracle, rel=1e-6)


def test_deciding_a_bond_is_refused_under_jit():
    """The other half of the boundary: ``update_()`` without a bond reads singular values
    to decide which sectors survive, which no trace allows."""
    with pytest.raises(tenet.StructureChangingError):
        jax.jit(lambda b: log_kappa(_swept(b)))(0.4)


def _swept(beta):
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), traced_ising(beta)))
    env.update_(max_bond=CHI)
    return env
