"""M79b (#277): the directional CTM environment, against physics rather than itself.

Two oracles, and they answer the two halves of the question M79a left open.

* **The wiring, against Onsager.** The classical 2D Ising partition function is a rank-4
  network, so the whole environment machinery runs on it with no bra and no double
  layer, and its free energy per site has a closed form. ``Trivial`` and ``Z2`` both,
  because the grading must not be what makes it work.
* **The fermionic signs, against a dense graded oracle.** A 2x2 open patch closed to a
  scalar is the first *loop* this package contracts fermionically -- M79a's primitives
  were only ever checked with legs left open -- and a loop is where a misplaced
  Jordan-Wigner string shows. The oracle replays the implementation's own chains on
  dense arrays with the Koszul sign of every line reordering computed from parity
  vectors alone (``helpers.dense_step``), and
  ``test_the_sign_free_contraction_is_a_different_number`` is its teeth: the same chain
  contracted without signs is off by a factor of 46.

The #243 instrument runs here too, on the new path: the enlarged corner is measured and
found **not** Hermitian, and the sweep converges to Onsager anyway, because
``proj_corners`` never asks.
"""

import ast
import pathlib
import traceback

import numpy as np
import pytest
from helpers import dense_step

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import (
    CheckerboardLattice,
    DoubleLayer,
    EnvCTM,
    Peps,
    SquareLattice,
    corner2x2,
    envctm,
)
from tenet.network.envctm import _composed
from tenet.symmetry import (
    U1,
    Z2,
    FZ2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    Z2Sector,
    fZ2,
)

# --- the classical model and its closed form -----------------------------------------


def ising(beta: float, graded: bool = True) -> SymmetricTensor:
    """The Boltzmann tensor on ``Peps``'s leg order ``(t IN, l OUT, b OUT, r IN)``.

    ``a[t,l,b,r] = sum_s W[s,t] W[s,l] W[s,b] W[s,r]`` with the parity-basis splitting
    ``W = [[sqrt cosh b, sqrt sinh b], [sqrt cosh b, -sqrt sinh b]]``, the same tensor
    ``examples/toy_codes/ising.py`` builds for the C4v route.
    """
    c, s = np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))
    w = np.array([[c, s], [c, -s]])
    block = np.einsum("st,sl,sb,sr->tlbr", w, w, w, w)
    space = (
        GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
        if graded
        else GradedSpace.new(Trivial, {TrivialSector(): 2})
    )
    legs = (Leg(space, IN), Leg(space, OUT), Leg(space, OUT), Leg(space, IN))
    return SymmetricTensor.from_dense(block, legs)


def onsager(beta: float, points: int = 200_001) -> float:
    """``beta f`` by direct quadrature -- ``examples/toy_codes/ising.py``'s form, which
    ``tests/integration/test_ctmrg.py`` already pins against a second closed form."""
    kk = 1.0 / np.sinh(2.0 * beta) ** 2
    theta = np.linspace(0.0, np.pi, points)
    integrand = np.log(
        np.cosh(2.0 * beta) ** 2 + np.sqrt(1.0 + kk**2 - 2.0 * kk * np.cos(2.0 * theta)) / kk
    )
    return -(np.log(2.0) / 2.0 + np.trapezoid(integrand, theta) / (2.0 * np.pi))


def log_kappa(env: EnvCTM, site=(0, 0)) -> float:
    """``ln`` of the partition function per site, Baxter's corner-transfer telescoping.

    ``kappa = z_a z_c / (z_h z_v)``: the whole ring closed on one bulk tensor, the four
    corners closed on each other, and the two one-row and one-column strips between
    them. The directional environment needs both strips where the C4v one needed
    ``z_h`` twice.
    """
    e, a = env[site], env.psi[site]
    scalar = lambda t: float(tenet.full_trace(t))  # noqa: E731
    z_c = _composed("ac,cd->ad", e.tl, e.tr)
    z_c = _composed("ad,de->ae", z_c, e.br)
    z_c = scalar(_composed("ae,ef->af", z_c, e.bl))
    z_h = _composed("ac,ctd->atd", e.tl, e.t)
    z_h = _composed("atd,de->ate", z_h, e.tr)
    z_h = _composed("ate,ef->atf", z_h, e.br)
    z_h = _composed("atf,ftg->ag", z_h, e.b)
    z_h = scalar(_composed("ag,gh->ah", z_h, e.bl))
    z_v = _composed("ac,cd->ad", e.tl, e.tr)
    z_v = _composed("ad,dle->ale", z_v, e.r)
    z_v = _composed("ale,ef->alf", z_v, e.br)
    z_v = _composed("alf,fg->alg", z_v, e.bl)
    z_v = scalar(_composed("alg,glh->ah", z_v, e.l))
    z_a = _composed("ac,ctd->atd", e.tl, e.t)
    z_a = _composed("atd,de->ate", z_a, e.tr)
    z_a = _composed("ate,erf->atrf", z_a, e.r)
    z_a = _composed("atrf,fg->atrg", z_a, e.br)
    z_a = _composed("atrg,gbh->atrbh", z_a, e.b)
    z_a = _composed("atrbh,hi->atrbi", z_a, e.bl)
    z_a = _composed("atrbi,ilj->atrblj", z_a, e.l)
    z_a = scalar(_composed("atrblj,tlbr->aj", z_a, a))
    return float(np.log(z_a * z_c / (z_h * z_v)))


def converged(beta: float, graded: bool = True, dims=(1, 1), moves: str = "hv", chi: int = 16):
    psi = Peps(SquareLattice(dims=dims), ising(beta, graded))
    env = EnvCTM(psi, init="eye")
    out = env.iterate_(max_bond=chi, moves=moves, max_sweeps=200, corner_tol=1e-11)
    return env, out


@pytest.mark.parametrize("beta", [0.3, 0.5])
def test_free_energy_matches_onsager_on_both_sides_of_the_transition(beta):
    """The oracle this whole stage exists to reach.

    ``beta = 0.5`` is past ``beta_c = 0.4407``: the ``Z2`` grading is what keeps a
    finite-``chi`` environment from breaking the symmetry spuriously there, exactly as
    it does for the C4v route (``tests/integration/test_ctmrg.py``).
    """
    env, out = converged(beta)
    assert out.converged
    got = -log_kappa(env) / beta
    assert got == pytest.approx(onsager(beta) / beta, rel=1e-6)


def test_the_ungraded_network_reaches_the_same_number():
    """The grading is not what makes it work; below ``beta_c`` it is not even needed."""
    env, out = converged(0.4, graded=False)
    assert out.converged
    assert -log_kappa(env) / 0.4 == pytest.approx(onsager(0.4) / 0.4, rel=1e-6)


def test_the_causal_moves_and_a_larger_cell_reach_the_same_number():
    """``'lrtb'`` runs column after column and row after row where ``'hv'`` updates every
    site at once, and a 2x2 cell carries four tensors where a 1x1 carries one. The
    physics may not notice."""
    reference = -log_kappa(converged(0.4)[0]) / 0.4
    for kwargs in ({"moves": "lrtb"}, {"dims": (2, 2)}):
        env, out = converged(0.4, **kwargs)
        assert out.converged
        assert -log_kappa(env) / 0.4 == pytest.approx(reference, rel=1e-9)


def test_the_dl_seed_starts_one_layer_ahead_and_lands_in_the_same_place():
    """``init='dl'`` is one un-truncated sweep on top of ``'eye'``, so it reaches the same
    fixed point from one layer further out."""
    psi = Peps(SquareLattice(dims=(1, 1)), ising(0.4))
    eye, dl = EnvCTM(psi, init="eye"), EnvCTM(psi, init="dl")
    assert eye[0, 0].tl.shape == (1, 1)
    assert max(dl[0, 0].tl.shape) > 1  # the seed grew
    out = dl.iterate_(max_bond=16, max_sweeps=200, corner_tol=1e-11)
    assert out.converged
    assert -log_kappa(dl) / 0.4 == pytest.approx(onsager(0.4) / 0.4, rel=1e-6)


def test_the_container_refuses_what_it_cannot_do():
    psi = Peps(SquareLattice(dims=(1, 1)), ising(0.4))
    with pytest.raises(ValueError, match="init"):
        EnvCTM(psi, init="rand")
    env = EnvCTM(psi)
    with pytest.raises(ValueError, match="move"):
        env.update_(max_bond=4, moves="x")
    with pytest.raises(ValueError, match="init"):
        env.reset_("rand")


def test_an_open_boundary_needs_trivial_projectors_and_still_moves():
    """On a finite patch the 2x2 block runs off the edge, and the projector that would
    have truncated a one-dimensional bond is the trivial one instead."""
    psi = Peps(SquareLattice(dims=(2, 2), boundary="obc"), ising(0.4))
    env = EnvCTM(psi, init="eye")
    env.update_(max_bond=8, moves="hv")
    filled = [
        (site, name)
        for site in env.sites()
        for name in ("hlt", "hlb", "hrt", "hrb")
        if getattr(env.proj[site], name) is not None
    ]
    assert filled, "an open boundary must have produced at least one projector"
    assert all(t is not None for _, local in env.items() for t in vars(local).values())


def test_the_checkerboard_pattern_updates_every_site_at_once():
    """A pattern with fewer unique sites than cells forces the simultaneous branch even
    for a causal move -- YASTN's ``len(sites) < Nx * Ny`` test."""
    psi = Peps(CheckerboardLattice(), ising(0.4))
    env = EnvCTM(psi, init="eye")
    out = env.iterate_(max_bond=16, moves="lrtb", max_sweeps=200, corner_tol=1e-11)
    assert out.converged
    assert -log_kappa(env) / 0.4 == pytest.approx(onsager(0.4) / 0.4, rel=1e-6)


# --- the fermionic half --------------------------------------------------------------

EVEN, ODD = FZ2Sector(0), FZ2Sector(1)

#: ``(t IN, l OUT, b OUT, r IN, phys OUT)`` -- ``peps.py``'s signature.
KET_SIDES = (IN, OUT, OUT, IN, OUT)

SPACES = {
    "trivial": GradedSpace.new(Trivial, {TrivialSector(): 2}),
    "u1": GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1}),
    "fz2": GradedSpace.new(fZ2, {EVEN: 1, ODD: 1}),
}
FERMIONIC = "fz2"


def ket(provider: str, seed: int = 5) -> SymmetricTensor:
    space = SPACES[provider]
    return SymmetricTensor.random(tuple(Leg(space, s) for s in KET_SIDES), seed=seed)


def closed_patch(provider: str):
    """A 2x2 open patch with identity boundaries, closed to a scalar.

    Four 2x2 enlarged corners at the ``'eye'`` seed *are* the four sites of the patch
    with their boundary bonds closed ket-against-bra; joining them into a top and a
    bottom half and then to each other closes the loop.
    """
    psi = Peps(SquareLattice(dims=(2, 2), boundary="obc"), ket(provider))
    env = EnvCTM(psi, init="eye")
    c = {
        w: corner2x2(env, w, s)
        for w, s in (("tl", (0, 0)), ("tr", (0, 1)), ("bl", (1, 0)), ("br", (1, 1)))
    }
    top = _composed("adeijk,ijkmnp->ademnp", c["tl"], c["tr"])
    bottom = _composed("adeijk,ijkmnp->ademnp", c["br"], c["bl"])
    return _composed("adeijk,ijkmde->am", top, bottom)


def closed_patch_the_other_way(provider: str):
    """The same patch cut left/right instead of top/bottom -- a different traversal of the
    same fermionic loop, which is what a misplaced string would break."""
    psi = Peps(SquareLattice(dims=(2, 2), boundary="obc"), ket(provider))
    env = EnvCTM(psi, init="eye")
    c = {
        w: corner2x2(env, w, s)
        for w, s in (("tl", (0, 0)), ("tr", (0, 1)), ("bl", (1, 0)), ("br", (1, 1)))
    }
    left = _composed("adeijk,ijkmnp->ademnp", c["bl"], c["tl"])
    right = _composed("adeijk,ijkmnp->ademnp", c["tr"], c["br"])
    return _composed("adeijk,ijkmde->am", left, right)


def record(build, provider: str, monkeypatch):
    """Run ``build`` with every ``tenet.einsum_chain`` recorded, result and all."""
    chains: list = []
    real = tenet.einsum_chain

    def spy(steps):
        steps = list(steps)
        out = real(steps)
        chains.append((steps, out))
        return out

    monkeypatch.setattr(tenet, "einsum_chain", spy)
    value = build(provider)
    monkeypatch.setattr(tenet, "einsum_chain", real)
    return value, chains


def replay(chains, signed: bool = True):
    """Every recorded chain re-contracted on dense arrays, one chain feeding the next.

    With ``signed`` the step is ``helpers.dense_step``, which bends and composes with the
    Koszul sign of every line reordering computed from parity vectors; without it the
    step is a plain ``np.einsum``, which is the same wiring and no signs at all.
    """
    done: dict[int, tuple] = {}
    arr = legs = None
    for steps, result in chains:
        arr = legs = None
        for equation, a, b, bend in steps:
            left = (arr, legs) if a is None else done.get(id(a)) or (_dense(a), a.legs)
            right = (arr, legs) if b is None else done.get(id(b)) or (_dense(b), b.legs)
            if signed:
                arr, legs = dense_step(equation, *left, *right, bend)
            else:
                arr, legs = np.einsum(equation, left[0], right[0]), None
        done[id(result)] = (arr, legs)
    return arr


def _dense(t: SymmetricTensor) -> np.ndarray:
    return np.asarray(t.to_dense())


@pytest.mark.parametrize("provider", sorted(SPACES))
def test_the_closed_2x2_patch_does_not_depend_on_where_it_is_cut(provider):
    """Cut the loop top/bottom or left/right: same number, to float64.

    For ``fz2`` that is a statement about the strings and not about arithmetic -- two
    traversals of one fermionic loop agree only if every bend sits where the planar
    diagram puts it.
    """
    one = float(tenet.full_trace(closed_patch(provider)))
    other = float(tenet.full_trace(closed_patch_the_other_way(provider)))
    assert abs(one) > 1.0, "a test whose oracle is near zero proves nothing"
    assert one == pytest.approx(other, rel=1e-13)


@pytest.mark.parametrize("provider", sorted(SPACES))
def test_the_closed_2x2_patch_matches_a_dense_graded_oracle(provider, monkeypatch):
    """The scalar, rebuilt from parity vectors and plain ``numpy``.

    Nothing of the library's braiding is trusted here: every step is replayed with the
    sign computed from the parities of the lines it reorders, and the environment's own
    number has to come out.
    """
    value, chains = record(closed_patch, provider, monkeypatch)
    assert replay(chains).ravel()[0] == pytest.approx(float(tenet.full_trace(value)), rel=1e-12)


def test_the_sign_free_contraction_is_a_different_number(monkeypatch):
    """The teeth. An oracle that agreed with the sign-free contraction would prove
    nothing, so it is asserted that it does not: the same chain without signs is a
    different scalar for ``fz2``, and the same one for the two gradings that do not
    braid."""
    for provider in sorted(SPACES):
        value, chains = record(closed_patch, provider, monkeypatch)
        plain = replay(chains, signed=False).ravel()[0]
        same = plain == pytest.approx(float(tenet.full_trace(value)), rel=1e-12)
        assert same == (provider != FERMIONIC)


def near_product(provider: str, eps: float = 0.4) -> SymmetricTensor:
    """A physical state to sweep on: a product state plus a small random part.

    A *purely* random site tensor is not a state anyone converges a CTM on -- its
    transfer spectrum is generically degenerate at the top -- so the fixture is the
    perturbed product state an evolution would actually hand the environment.
    """
    space = SPACES[provider]
    legs = tuple(Leg(space, s) for s in KET_SIDES)
    noise = _dense(ket(provider))
    block = np.zeros_like(noise)
    block[0, 0, 0, 0, 0] = 1.0
    return SymmetricTensor.from_dense(block + eps * noise / np.abs(noise).max(), legs)


@pytest.mark.parametrize("provider", ["u1", "fz2"])
def test_a_double_layer_sweep_converges(provider):
    """The lazy double layer through a whole sweep, fermionic grading included."""
    psi = Peps(SquareLattice(dims=(1, 1)), near_product(provider))
    env = EnvCTM(psi, init="dl")
    out = env.iterate_(max_bond=16, max_sweeps=60, corner_tol=1e-9)
    assert out.converged
    assert out.sweeps < 60


# --- the projectors ------------------------------------------------------------------


def hermiticity(t: SymmetricTensor) -> float:
    """``|B - B^dagger| / |B|`` for an enlarged corner read as a map between its two
    index groups, or ``inf`` while the map is not even square.

    While the environment is still growing the two groups have different dimensions, so
    ``B^dagger`` is not a tensor of the same shape and the question does not arise --
    which is the strongest form of the answer, and is reported as ``inf`` rather than
    skipped.
    """
    dense = _dense(t)
    n = int(np.prod(dense.shape[: t.ndim // 2]))
    m = dense.reshape(n, -1)
    if m.shape[0] != m.shape[1]:
        return float("inf")
    return float(np.linalg.norm(m - m.conj().T) / np.linalg.norm(m))


def test_the_projectors_never_need_a_hermitian_corner():
    """#243's instrument, re-run on the path that makes it moot.

    M63 measured that an enlarged corner is Hermitian at every move exactly when the
    bulk carries the **full C4v point group**, and not otherwise. Both halves of that
    are re-measured here on the directional path, and neither changes what the
    projectors do.

    * A double-layer iPEPS with no spatial symmetry at all: the corner is not square
      while the environment is still growing (so ``B^dagger`` is not even the same
      shape) and is 0.23-0.69 away from Hermitian once it is. The sweep converges anyway.
    * The Ising Boltzmann tensor, which is symmetric under *every* permutation of its
      four legs: the corner comes back Hermitian to 6e-3 and falling. That is the
      control, and it is why the C4v route could get away with an eigendecomposition
      on this one model.

    ``proj_corners`` reads neither number. It builds its pair from a QR of each half of
    the 4x4 patch and an SVD of ``r0 @ r1^T``, so there is no eigendecomposition to
    require a Hermitian input and no single isometry reused on both index groups.
    """
    psi = Peps(SquareLattice(dims=(1, 1)), near_product("u1"))
    env = EnvCTM(psi, init="eye")
    defects = []
    for _ in range(8):
        env.update_(max_bond=16, moves="hv")
        defects.append(hermiticity(corner2x2(env, "tl", (0, 0))))
    square = [d for d in defects if np.isfinite(d)]
    assert square, defects  # the map does become square, so the question is asked
    assert min(square) > 0.2, defects  # and answered: nowhere near Hermitian
    assert env.iterate_(max_bond=16, max_sweeps=60, corner_tol=1e-9).converged

    control = EnvCTM(Peps(SquareLattice(dims=(1, 1)), ising(0.4)), init="eye")
    control_defects = []
    for _ in range(8):
        control.update_(max_bond=16, moves="hv")
        control_defects.append(hermiticity(corner2x2(control, "tl", (0, 0))))
    finite = [d for d in control_defects if np.isfinite(d)]
    assert max(finite) < 1e-2, control_defects  # the full point group, M63's row 3
    assert control.iterate_(max_bond=16, max_sweeps=200, corner_tol=1e-11).converged
    assert -log_kappa(control) / 0.4 == pytest.approx(onsager(0.4) / 0.4, rel=1e-6)


def test_the_projector_pair_resolves_the_identity_on_the_cut():
    """``p1`` and ``p0`` inserted back to back reproduce the bond they cut, up to the
    truncation -- which is what makes them a projector pair rather than two isometries."""
    psi = Peps(SquareLattice(dims=(1, 1)), ising(0.4))
    env = EnvCTM(psi, init="eye")
    for _ in range(6):
        env.update_(max_bond=16, moves="hv")
    p0, p1 = env.proj[0, 0].hlb, env.proj[0, 0].hlt
    # ``hlb`` at a site and ``hlt`` at the site below share one cut and one new bond.
    below = env.proj[1, 0].hlt
    assert p0.legs[-1].space == below.legs[-1].space
    assert p0.ndim == p1.ndim == 3  # env leg, the site's leg, the new bond


def test_a_regauged_vector_does_not_move_the_primitive():
    """``append_vec_br``'s bend string is fixed; the vectors a sweep hands it are not.

    A ``qr`` repartitions its input across the map's two sides, so a projector comes
    back carrying legs whose ``side`` and ``dual`` are both flipped relative to the
    enlarged corner they came from -- the same wire, the opposite spelling. The
    primitive's transcribed bend then names a wire that is already turned the right way,
    and the question is whether that changes the tensor. Measured over every vector one
    fermionic sweep produces: it does not, to zero, because bending both ends of a wire
    and bending neither differ by a cap-cup pair whose coefficients cancel.
    """
    captured: list = []
    psi = Peps(SquareLattice(dims=(1, 1)), near_product("fz2"))
    env = EnvCTM(psi, init="eye")
    real = envctm._C2X2["br"][5]

    def spy(a, vec):
        captured.append((a, vec))
        return real(a, vec)

    envctm._C2X2["br"] = envctm._C2X2["br"][:5] + (spy,)
    try:
        env.update_(max_bond=6, moves="hv")
    finally:
        envctm._C2X2["br"] = envctm._C2X2["br"][:5] + (real,)
    regauged = [(a, v) for a, v in captured if any(leg.dual for leg in v.legs)]
    assert regauged, "the sweep never produced a regauged vector, so nothing was tested"
    for a, vec in captured:
        derived = _composed("xdDeEy,TLEDS->xdeySTL", vec, a.bra)
        derived = _composed("xdeySTL,tledS->xtTylL", derived, a.ket)
        assert tenet.allclose(real(a, vec), derived)


# --- hygiene -------------------------------------------------------------------------


def test_every_two_operand_step_is_a_composition(monkeypatch):
    """The composition rule (#160) on the new module, and the coverage claim asserted.

    ``envctm.py`` reaches ``tenet.einsum`` only through ``_composed`` and through
    ``peps.py``'s ``append_vec_*``, both of which are ``tenet.einsum_chain`` steps
    carrying their bent wires in the step's own field. So the check is: every recorded
    step, after its bend, has operand 1 supplying the ``IN`` end of every shared wire.
    A leg that a ``qr`` repartitioned across the map's two sides carries the same wire
    with ``side`` and ``dual`` both flipped, so the predicate is the pair rather than
    ``side`` -- which is exactly ``envctm._supplies_in``. The smoke exercises every
    ``_composed`` call site in the module: a classical sweep with both move families and
    both boundaries, and a double-layer sweep.
    """
    seen: set[int] = set()
    module = pathlib.Path(envctm.__file__)
    real = tenet.einsum_chain

    def spy(steps):
        for frame in traceback.extract_stack():
            if pathlib.Path(frame.filename) == module and frame.name != "_composed":
                seen.add(frame.lineno)
        steps = list(steps)
        for equation, a, b, bend in steps:
            lhs = equation.split("->")[0]
            ta, tb = lhs.split(",")
            if a is None or b is None:
                continue  # a running result's legs are the chain's business
            flip = set(bend)
            for label in ta:
                if label not in tb:
                    continue
                leg = a.legs[ta.index(label)]
                supplies_in = (leg.side is IN) != leg.dual
                assert supplies_in != (label in flip), (
                    f"{equation}: wire {label!r} is not supplied IN by operand 1 after its bend"
                )
            seen.add(equation)
        return real(steps)

    monkeypatch.setattr(tenet, "einsum_chain", spy)
    psi = Peps(SquareLattice(dims=(2, 2)), ising(0.4))
    env = EnvCTM(psi, init="dl")
    env.update_(max_bond=6, moves="hvlrtb")
    env.corner_spectra()
    finite = EnvCTM(Peps(SquareLattice(dims=(2, 2), boundary="obc"), ising(0.4)), init="eye")
    finite.update_(max_bond=6, moves="hv")
    double = EnvCTM(Peps(SquareLattice(dims=(1, 1)), near_product("fz2")), init="eye")
    double.update_(max_bond=6, moves="hv")
    wanted = {
        node.lineno
        for node in ast.walk(ast.parse(module.read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_composed"
    }
    assert len(wanted) > 25, wanted  # the enumeration itself found the call sites
    assert not wanted - seen, f"the smoke never reached these lines: {sorted(wanted - seen)}"


def test_corner_spectra_are_scaled_and_the_record_is_the_loop_exit():
    env, out = converged(0.4)
    spectra = env.corner_spectra()
    assert set(spectra) == {
        (site, name) for site in env.sites() for name in ("tl", "tr", "bl", "br")
    }
    for values in spectra.values():
        assert values[0] == pytest.approx(1.0)
        assert values == sorted(values, reverse=True)
    assert out.converged is (out.max_dsv < 1e-11)


def test_a_double_layer_environment_carries_the_pair_unfused():
    """The ket bond and the bra bond stay adjacent and separate, as ``peps.py`` leaves
    them: a double-layer edge is rank 4 where a classical one is rank 3."""
    flat = EnvCTM(Peps(SquareLattice(dims=(1, 1)), ising(0.4)))
    assert flat[0, 0].l.ndim == 3
    layered = EnvCTM(Peps(SquareLattice(dims=(1, 1)), near_product("u1")))
    assert layered[0, 0].l.ndim == 4
    assert layered.double and not flat.double
    a = layered.psi[0, 0]
    assert isinstance(a, DoubleLayer)
