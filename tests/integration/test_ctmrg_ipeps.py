"""The variational half of the differentiable-CTMRG lane: a U(1) and an SU(2) iPEPS.

Its sibling ``test_ctmrg.py`` judges a *classical* partition function against Onsager.
This module has no closed form to answer to, so it asks the questions that do not need
one: that the energy is real and reproduces its own recorded baseline, that the open-leg
normalization agrees with the closed-leg contraction it must, that the gradient of a
double-layer environment matches central differences block by block, and that nothing in
the traced region rewrites a ``TensorStructure``.

The ansatz is a point-group average over a **self-conjugate** virtual space, the only kind
a rotation acts on at all: SU(2) ``{0, 1}`` and U(1) ``{-1, 0, +1}``. That is what puts a
C4v ansatz on this lane, and its two-site energy is :data:`ENERGY_BASELINE`.

**Runtime** is the two cold gradients -- the fusion-tree enumeration and per-block plan
build behind the first backward pass through each distinct bond structure, plus one-off
XLA compiles -- and nothing else. Measured at ``CHI_IPEPS = 6``: 62 s (U(1)) and 11 s
(SU(2)) against 8.9 s and 2.6 s warm. Neither ``k`` nor ``chi`` moves those numbers --
``K_IPEPS`` 1 and 2 cost the same 62 s, and ``chi`` 6 and 8 differ by 6 % -- because they
count distinct block shapes and not arithmetic. The environment and the gradient are
therefore built once per provider and shared by every test below. This module is split
from ``test_ctmrg.py`` for that reason as well: the two halves share no fixture, and
``--dist loadfile`` can only put whole files on separate workers.

x64 is enabled process-globally in ``tests/conftest.py``; every tolerance here depends on
it.
"""

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import EnvCTMc4v, Peps, SquareLattice, flip
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

jax = pytest.importorskip("jax")

import tenet.ad  # noqa: E402
import tenet.pytree  # noqa: E402, F401  # registration is the import's side effect

# ``opt_einsum``'s greedy path, for the patch contractions only. Simplification: greedy,
# not "auto" -- at seven operands "auto"'s dynamic-programming *search* costs an order of
# magnitude more than the contraction it plans, and it is re-run on every call.
PATH = "greedy"

CHI_IPEPS, K_IPEPS = 6, 2
PROVIDERS = ["u1", "su2"]

# Converging an environment is the expensive part, so each is built once per module.
# Simplification: a plain dict, not a fixture per provider -- nothing below mutates a
# stored environment (every traced run seeds a fresh one).
_ENVS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def broadened_svd():
    """``tenet.ad`` is process-global, so this module installs it and puts it back.

    Every gradient here runs through a factorization of a symmetric fixed point, which is
    where degenerate singular values live.
    """
    tenet.ad.install()
    yield
    tenet.ad.uninstall()


#: Virtual and physical space per provider. Both virtual spaces are **self-conjugate**: a
#: rotation identifies a virtual space with its dual, so no other kind carries a point
#: group at all.
SPACES = {
    "su2": (
        GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 1}),
        GradedSpace.new(SU2, {SU2Sector(1): 1}),
    ),
    "u1": (
        GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(0): 1, U1Sector(1): 1}),
        GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1}),
    ),
}

#: The eight elements of C4v as permutations of ``(t, l, b, r)``, physical leg last.
_ROT, _MIRROR = (1, 2, 3, 0, 4), (1, 0, 3, 2, 4)


def _group() -> list[tuple[int, ...]]:
    compose = lambda p, q: tuple(p[i] for i in q)  # noqa: E731
    elements, current = [], (0, 1, 2, 3, 4)
    for _ in range(4):
        elements.append(current)
        elements.append(compose(current, _MIRROR))
        current = compose(current, _ROT)
    return elements


def build_ipeps(provider: str, seed: int = 1) -> SymmetricTensor:
    """A random single-site iPEPS averaged over the point group: four identical virtual
    legs, so the 90-degree rotation is the cyclic transpose of the first four axes and the
    average over the eight elements is invariant under it."""
    virtual, physical = SPACES[provider]
    legs = (Leg(virtual, OUT),) * 4 + (Leg(physical, OUT),)
    a = SymmetricTensor.random(legs, seed=seed)
    out = None
    for permutation in _group():
        turned = tenet.transpose(a, permutation)
        out = turned if out is None else out + turned
    return (out / 8).to_backend("jax")


def build_h(provider: str, seed: int = 100) -> SymmetricTensor:
    """A random self-adjoint two-site term on the checkerboard's *alternating* signature.

    The odd sublattice's tensor is ``flip(a)`` -- every leg's ``side`` reversed, the
    physical one included -- so an operator that meets both sites has legs
    ``(bra_even OUT, bra_odd IN, ket_even IN, ket_odd OUT)``. ``transpose(adjoint(g),
    (2, 3, 0, 1))`` lands back on that same signature, which is what makes the symmetrized
    combination well-formed and ``<h>`` real.

    A plumbing operator, exactly as ``examples/toy_codes/vmc_mps.py``'s is: this half
    exercises graded truncation, ``svd(bond=)`` across sectors and multiplet degeneracies,
    and makes no benchmark-energy claim.
    """
    physical = SPACES[provider][1]
    legs = (Leg(physical, OUT), Leg(physical, IN), Leg(physical, IN), Leg(physical, OUT))
    g = SymmetricTensor.random(legs, seed=seed)
    return ((g + tenet.transpose(tenet.adjoint(g), (2, 3, 0, 1))) / 2).to_backend("jax")


#: The 2x1 patch, split down the middle. ``left`` is the top-left corner, the bottom-left
#: corner, the left edge and the first site's own top and bottom edges, with the site
#: absorbed; ``right`` is its mirror image. Each half is built environment-first, then ket,
#: then bra, so no double layer is ever formed. ``True`` leaves the physical legs open --
#: the numerator, which ``h`` closes -- and ``False`` closes them against each other.
_LEFT = {
    True: "pu,utTX,qlLp,sq,YbBs,tlbRw,TLBSW->XRSYwW",
    False: "pu,utTX,qlLp,sq,YbBs,tlbRw,TLBSw->XRSY",
}
_RIGHT = {
    True: "AtTv,vm,mcCn,no,obBE,tRbcx,TSBCz->ARSExz",
    False: "AtTv,vm,mcCn,no,obBE,tRbcx,TSBCx->ARSE",
}


def halves(env, open_phys: bool, s0=(0, 0), s1=(0, 1)):
    """The two halves of the 2x1 patch on ``(s0, s1)``, for either environment lane.

    The ring is ``C_tl, T_t(0), T_t(1), C_tr, T_r, C_br, T_b(1), T_b(0), C_bl, T_l``, and
    each tensor is read off the environment of the site it belongs to -- which is what
    makes one helper serve [EnvCTM][tenet.network.EnvCTM], where those are ten different
    tensors, and [EnvCTMc4v][tenet.network.EnvCTMc4v], where they are one corner and one
    edge and their flips.
    """
    le, re = env[s0], env[s1]
    a0, a1 = env.psi[s0], env.psi[s1]
    left = tenet.einsum(
        _LEFT[open_phys], le.tl, le.t, le.l, le.bl, le.b, a0.ket, a0.bra, optimize=PATH
    )
    right = tenet.einsum(
        _RIGHT[open_phys], re.t, re.tr, re.r, re.br, re.b, a1.ket, a1.bra, optimize=PATH
    )
    return left, right


def energy(env, h, s0=(0, 0), s1=(0, 1)):
    """``<h> / <1>`` on a 2x1 patch. With :func:`halves` this is a reduced-density-matrix
    API at one geometry, which is why the library's environment module stops short of it."""
    left, right = halves(env, False, s0, s1)
    denominator = tenet.full_trace(tenet.einsum("XRSY,ARSY->AX", left, right))
    left, right = halves(env, True, s0, s1)
    numerator = tenet.full_trace(tenet.einsum("XRSYwW,ARSYxz,Wzwx->AX", left, right, h))
    return numerator / denominator


def ipeps(provider: str):
    """``(a, h, seed, bond)``: the ansatz, the term, and the converged C4v environment as
    the traced region's initial condition."""
    key = f"ipeps-{provider}"
    if key not in _ENVS:
        a, h = build_ipeps(provider), build_h(provider)
        env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), a))
        assert env.iterate_(max_bond=CHI_IPEPS, max_sweeps=200, corner_tol=1e-10).converged
        local = env[0, 0]
        _ENVS[key] = (a, h, (local.tl, local.t), local.tl.legs[0].space)
    return _ENVS[key]


def ipeps_energy(a, h, seed, bond, k=K_IPEPS):
    """``<h>`` after ``k`` fixed-bond moves at the current ``a`` -- the function
    differentiated."""
    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), a), init=None)
    env.env[0, 0].tl, env.env[0, 0].t = seed
    for _ in range(k):
        env.update_(bond=bond)
    return energy(env, h)


def ipeps_grad(provider: str):
    """One gradient per provider, reused. The backward pass through a double-layer
    environment is the most expensive thing in this module; three tests want the same one."""
    key = f"grad-{provider}"
    if key not in _ENVS:
        _ENVS[key] = jax.value_and_grad(ipeps_energy)(*ipeps(provider))
    return _ENVS[key]


@pytest.fixture(params=PROVIDERS)
def provider(request):
    return request.param


def test_ipeps_energy_is_real(provider):
    value, _ = ipeps_grad(provider)
    assert abs(complex(value).imag) < 1e-12


#: ``<h>`` on the point-group-averaged ansatz of :func:`build_ipeps`, at ``CHI_IPEPS`` and
#: ``K_IPEPS`` moves. Twelve significant figures, which is what the directional lane
#: measuring the same state to ``5.3e-11`` relative entitles us to -- that cross-check is a
#: 200 s sweep and is recorded in the M80 design entry rather than run here.
ENERGY_BASELINE = {"su2": 0.288093877946, "u1": 0.151081276175}


def test_ipeps_energy_matches_the_baseline(provider):
    value, _ = ipeps_grad(provider)
    assert float(value) == pytest.approx(ENERGY_BASELINE[provider], rel=1e-10)


def test_the_normalization_is_the_same_contraction(provider):
    """``<1> = 1`` with the physical legs held open and closed by an identity, which is the
    open-leg route checked against the closed one it must agree with."""
    a, _, seed, bond = ipeps(provider)
    physical = SPACES[provider][1]
    even = tenet.identity((Leg(physical, OUT),))
    unit = tenet.einsum("ab,cd->acbd", even, flip(even))
    assert float(ipeps_energy(a, unit, seed, bond)) == pytest.approx(1.0, abs=1e-12)


def test_ipeps_grad_matches_central_differences_on_one_block(provider):
    a, h, seed, bond = ipeps(provider)
    grads = ipeps_grad(provider)[1]
    blk, delta = 0, 1e-5

    def shifted(idx, d):
        blocks = list(a.blocks)
        blocks[blk] = blocks[blk].at[idx].add(d)
        return float(ipeps_energy(SymmetricTensor(a.structure, tuple(blocks)), h, seed, bond))

    block = a.blocks[blk]
    want = np.zeros(block.shape)
    for idx in np.ndindex(block.shape):
        want[idx] = (shifted(idx, delta) - shifted(idx, -delta)) / (2 * delta)
    np.testing.assert_allclose(np.asarray(grads.blocks[blk]), want, rtol=1e-5, atol=1e-8)


def test_structures_survive_the_traced_region(provider):
    """The trust boundary, re-run: nothing in the differentiated region rewrites a
    ``TensorStructure``, so the originals are still ``is``-identical afterwards."""
    a, h, seed, bond = ipeps(provider)
    corner, edge = seed
    before = (a.structure, corner.structure, edge.structure)
    ipeps_grad(provider)  # the traced region has run by now

    env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), a), init=None)
    env.env[0, 0].tl, env.env[0, 0].t = seed
    env.update_(bond=bond)
    assert all(
        x is y for x, y in zip((a.structure, corner.structure, edge.structure), before, strict=True)
    )
    # the traced region is structure-preserving: the move comes back on the frozen bond
    assert env[0, 0].tl.structure == corner.structure
    assert env[0, 0].t.structure == edge.structure
    for t in (a, h, corner, edge, env[0, 0].tl, env[0, 0].t):
        SymmetricTensor(t.structure, t.blocks)  # the trust boundary, re-run
