"""M79a (#277): the geometry, the lazy double layer and the twelve contraction primitives.

Two oracles, and every primitive meets both.

* **Wiring.** For a provider that does not braid with signs -- Trivial, U(1), SU(2) --
  the primitive's chain is one ``np.einsum`` over the dense expansions with the *same*
  index letters the implementation writes. That pins which legs are closed and in which
  order the survivors come back, with no reference to anything fermionic.
* **Signs.** For fZ2 that plain einsum is *wrong*, which is the point, and the oracle
  becomes ``helpers.dense_step``: the chain replayed on dense arrays with the Koszul
  sign of every line reordering computed from parity vectors alone. The steps are not
  transcribed here -- they are recorded off the implementation's own
  ``tenet.einsum_chain`` calls, so the oracle cannot drift from the code it checks --
  and ``dense_compose`` asserts the composition rule on every step as it goes, which is
  how the operand orders in ``peps.py``'s docstrings are pinned rather than asserted.

``test_the_plain_einsum_is_wrong_for_fz2`` is the teeth: an oracle that agreed with the
sign-free contraction would prove nothing.
"""

import numpy as np
import pytest
from helpers import dense_compose, dense_repartition, dense_step

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import (
    Bond,
    CheckerboardLattice,
    DoublePepsTensor,
    Lattice,
    Peps,
    Peps2Layers,
    RectangularUnitcell,
    Site,
    SquareLattice,
    append_vec_bl,
    append_vec_br,
    append_vec_tl,
    append_vec_tr,
    cor_bl,
    cor_br,
    cor_tl,
    cor_tr,
    edge_b,
    edge_l,
    edge_r,
    edge_t,
)
from tenet.symmetry import (
    SU2,
    U1,
    FZ2Sector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

# --- geometry ----------------------------------------------------------------------


def test_infinite_nn_site_shifts_and_site2index_folds():
    lat = SquareLattice(dims=(2, 3))
    assert lat.nn_site((0, 2), "r") == Site(0, 3)  # never None on an infinite lattice
    assert lat.nn_site((0, 0), "tl") == Site(-1, -1)
    assert lat.site2index((0, 3)) == (0, 0)
    assert lat.site2index((-1, -1)) == (1, 2)


def test_obc_returns_none_outside_and_cylinder_wraps_rows_only():
    obc = SquareLattice(dims=(2, 3), boundary="obc")
    assert obc.nn_site((0, 2), "r") is None
    assert obc.nn_site((0, 0), "t") is None
    assert obc.nn_site((0, 0), "br") == Site(1, 1)
    cyl = SquareLattice(dims=(2, 3), boundary="cylinder")
    assert cyl.nn_site((1, 0), "b") == Site(0, 0)  # rows are periodic
    assert cyl.nn_site((0, 2), "r") is None  # columns are open
    assert SquareLattice(dims=(2, 3), boundary="obc").nn_site(None, "r") is None


def test_bonds_are_emitted_in_the_fermionic_order():
    """Left before right, top before bottom -- ``bonds`` orients, callers do not."""
    lat = SquareLattice(dims=(2, 2))
    for bond in lat.bonds("h"):
        assert bond.site1 == lat.nn_site(bond.site0, "r")
        assert lat.f_ordered(*bond)
    for bond in lat.bonds("v"):
        assert bond.site1 == lat.nn_site(bond.site0, "b")
        assert lat.f_ordered(*bond)
    assert lat.bonds() == lat.bonds("h") + lat.bonds("v")
    assert lat.bonds(reverse=True) == lat.bonds("v")[::-1] + lat.bonds("h")[::-1]
    obc = SquareLattice(dims=(2, 2), boundary="obc")
    assert len(obc.bonds("h")) == 2 and len(obc.bonds("v")) == 2


def test_f_ordered_is_column_major():
    lat = SquareLattice(dims=(3, 3))
    assert lat.f_ordered((2, 0), (0, 1))  # left column entirely before the next
    assert lat.f_ordered((0, 1), (1, 1))  # within a column, top before bottom
    assert not lat.f_ordered((1, 1), (0, 1))
    assert lat.f_ordered((1, 1), (1, 1))  # identical sites count as ordered
    assert lat.sites() == tuple(Site(nx, ny) for ny in range(3) for nx in range(3))


def test_nn_bond_dirn_names_the_four_orientations_and_refuses_the_rest():
    lat = SquareLattice(dims=(3, 3), boundary="obc")
    assert lat.nn_bond_dirn((0, 0), (0, 1)) == "lr"
    assert lat.nn_bond_dirn((0, 0), (1, 0)) == "tb"
    assert lat.nn_bond_dirn((0, 1), (0, 0)) == "rl"
    assert lat.nn_bond_dirn(Bond(Site(1, 0), Site(0, 0))) == "bt"
    with pytest.raises(ValueError, match="nearest-neighbour"):
        lat.nn_bond_dirn((0, 0), (1, 1))


def test_checkerboard_is_two_sublattices():
    lat = CheckerboardLattice()
    assert lat.sites() == (Site(0, 0), Site(0, 1))
    assert [lat.site2index((nx, ny)) for nx in range(2) for ny in range(2)] == [0, 1, 1, 0]
    assert lat == CheckerboardLattice()
    assert lat != SquareLattice(dims=(2, 2))


@pytest.mark.parametrize(
    ("pattern", "labels"),
    [
        ([[0]], [[0]]),
        ([[0, 1]], [[0, 1]]),
        ([[0, 1], [1, 0]], [[0, 1], [1, 0]]),
        ([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
        ({(0, 0): "a", (0, 1): "b"}, [["a", "b"]]),
    ],
)
def test_rectangular_unitcell_accepts_single_momentum_patterns(pattern, labels):
    lat = RectangularUnitcell(pattern)
    got = [[lat.site2index((nx, ny)) for ny in range(lat.Ny)] for nx in range(lat.Nx)]
    assert got == labels
    assert len(lat.sites()) == len({x for row in labels for x in row})
    # the tiling really is periodic in both directions
    assert lat.site2index((lat.Nx, lat.Ny)) == lat.site2index((0, 0))


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        ([[0, 1], [1, 1]], "same neighbours"),
        ([[0, 1], [1]], "rectangular"),
        ({(0, 1): 0}, "cover"),
        ({(0, 0): 0, (1, 1): 1}, "cover"),
    ],
)
def test_rectangular_unitcell_refuses_what_one_environment_cannot_describe(pattern, message):
    with pytest.raises(ValueError, match=message):
        RectangularUnitcell(pattern)


def test_lattice_stores_one_object_per_unique_site():
    lat = Lattice(CheckerboardLattice(), {(0, 0): "A", (0, 1): "B"})
    assert (lat[0, 0], lat[1, 1], lat[0, 3], lat[1, 0]) == ("A", "A", "B", "B")
    lat[2, 2] = "C"  # (2, 2) folds onto sublattice 0
    assert lat[0, 0] == "C"
    assert [s for s, _ in lat.items()] == list(lat.sites())
    with pytest.raises(ValueError, match="not assigned"):
        Lattice(CheckerboardLattice(), {(0, 0): "A"})
    with pytest.raises(ValueError, match="two different objects"):
        Lattice(CheckerboardLattice(), {(0, 0): "A", (1, 1): "B"})
    spread = Lattice(SquareLattice(dims=(1, 2)), "x")
    assert spread[0, 0] == spread[0, 1] == "x"


# --- providers, spaces and a random site --------------------------------------------

EVEN, ODD = FZ2Sector(0), FZ2Sector(1)

#: Five *different* spaces, one per axis, so a transposed axis is a failure rather than
#: a coincidence. ``x`` and ``y`` are the environment vector's spectators.
SPACES = {
    "trivial": {
        k: GradedSpace.new(Trivial, {TrivialSector(): d})
        for k, d in zip("tlbrsxy", (2, 3, 2, 3, 2, 2, 3), strict=True)
    },
    "u1": {
        "t": GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1}),
        "l": GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1}),
        "b": GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(-1): 1}),
        "r": GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(-1): 1}),
        "s": GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1}),
        "x": GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1}),
        "y": GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(-1): 1}),
    },
    "fz2": {
        "t": GradedSpace.new(fZ2, {EVEN: 1, ODD: 1}),
        "l": GradedSpace.new(fZ2, {EVEN: 2, ODD: 1}),
        "b": GradedSpace.new(fZ2, {EVEN: 1, ODD: 2}),
        "r": GradedSpace.new(fZ2, {EVEN: 1, ODD: 1}),
        "s": GradedSpace.new(fZ2, {EVEN: 1, ODD: 1}),
        "x": GradedSpace.new(fZ2, {EVEN: 1, ODD: 1}),
        "y": GradedSpace.new(fZ2, {EVEN: 2, ODD: 1}),
    },
    "su2": {
        "t": GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 1}),
        "l": GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 1}),
        "b": GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 1}),
        "r": GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(2): 1}),
        "s": GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 1}),
        "x": GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 1}),
        "y": GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 1}),
    },
}
PROVIDERS = tuple(SPACES)
FERMIONIC = "fz2"

#: ``(t IN, l OUT, b OUT, r IN, phys OUT)`` -- ``peps.py``'s stated signature.
KET_SIDES = (IN, OUT, OUT, IN, OUT)


def site(provider: str, seed: int = 5) -> DoublePepsTensor:
    sp = SPACES[provider]
    legs = tuple(Leg(sp[k], s) for k, s in zip("tlbrs", KET_SIDES, strict=True))
    ket = SymmetricTensor.random(legs, seed=seed)
    return DoublePepsTensor(ket=ket, bra=tenet.adjoint(ket))


def vector(provider: str, pairs: str, seed: int = 9) -> SymmetricTensor:
    """``(x, <first pair>, <second pair>, y)``; each pair is ket leg then bra leg."""
    sp = SPACES[provider]
    ket_side = dict(zip("tlbrs", KET_SIDES, strict=True))
    legs = [Leg(sp["x"], OUT)]
    for axis in pairs:
        opposite = IN if ket_side[axis] is OUT else OUT
        legs.append(Leg(sp[axis], opposite))  # meets the ket leg
        legs.append(Leg(sp[axis], ket_side[axis]))  # meets the bra leg
    legs.append(Leg(sp["y"], IN))
    return SymmetricTensor.random(tuple(legs), seed=seed)


#: Every primitive, with the arguments it takes and the index letters its chain uses.
#: The letters are only for the wiring oracle; the sign oracle records the real chain.
CORNERS = {
    "cor_tl": (cor_tl, "tlBRs,tlbrs->bBrR"),
    "cor_bl": (cor_bl, "TlbRs,tlbrs->rRtT"),
    "cor_br": (cor_br, "TLbrs,tlbrs->tTlL"),
    "cor_tr": (cor_tr, "tlbrs,tLBrs->lLbB"),
    "edge_t": (edge_t, "tLBRs,tlbrs->lLbBrR"),
    "edge_l": (edge_l, "TlBRs,tlbrs->bBrRtT"),
    "edge_b": (edge_b, "TLbRs,tlbrs->rRtTlL"),
    "edge_r": (edge_r, "TLBrs,tlbrs->tTlLbB"),
}

#: ``name -> (function, pair order of the vector, dense einsum of the whole chain)``.
VECTORS = {
    "append_vec_tl": (append_vec_tl, "lt", "xcCaAy,ACBRS,acbrS->xbByrR"),
    "append_vec_br": (append_vec_br, "rb", "xdDeEy,TLEDS,tledS->xtTylL"),
    "append_vec_tr": (append_vec_tr, "tr", "xaAdDy,ALBDS,albdS->xlLybB"),
    "append_vec_bl": (append_vec_bl, "bl", "xeEcCy,TCERS,tcerS->xrRytT"),
}
PRIMITIVES = tuple(CORNERS) + tuple(VECTORS)


def _minimal_bends(equation: str, legs_a, legs_b) -> int:
    """How many wires a step *must* bend: the smaller of the two operand orders.

    Exactly one end of every contractible wire is ``IN`` here, so the two counts sum to
    the number of shared wires and the minimum is what the planar diagram forces.
    """
    lhs = equation.split("->")[0]
    ta, tb = lhs.split(",")
    shared = [c for c in ta if c in tb]
    a_first = sum(legs_a[ta.index(c)].side is not IN for c in shared)
    return min(a_first, len(shared) - a_first)


def call(name: str, provider: str):
    """Run a primitive, and return it together with its operands' dense expansions."""
    a = site(provider)
    if name in CORNERS:
        fn, _ = CORNERS[name]
        return fn(a), (a.bra, a.ket) if name != "cor_tr" else (a.ket, a.bra)
    fn, pairs, _ = VECTORS[name]
    vec = vector(provider, pairs)
    return fn(a, vec), (vec, a.bra, a.ket)


# --- the oracle's own check ---------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_oracle_reproduces_repartition_and_einsum(provider):
    """``dense_repartition``/``dense_compose`` against the library, before they judge it.

    The fermionic line order the helper assumes -- OUT axes forward, IN axes *reversed*
    -- was measured, not derived from the source, so it is re-measured here for every
    provider the suite runs.
    """
    sp = SPACES[provider]
    legs = (Leg(sp["t"], OUT), Leg(sp["l"], IN), Leg(sp["b"], OUT), Leg(sp["r"], IN))
    t = SymmetricTensor.random(legs, seed=4)
    dense = np.asarray(t.to_dense())
    for outs in [(0, 1, 2), (0,), (3, 1), (), (0, 1, 2, 3)]:
        ins = tuple(i for i in range(4) if i not in outs)
        want, _ = dense_repartition(dense, legs, outs, ins)
        got = np.asarray(tenet.repartition(t, outs, ins).to_dense())
        np.testing.assert_allclose(got, want, atol=1e-11)
    a = SymmetricTensor.random((Leg(sp["x"], OUT), Leg(sp["t"], IN), Leg(sp["l"], IN)), seed=6)
    b = SymmetricTensor.random((Leg(sp["t"], OUT), Leg(sp["l"], OUT), Leg(sp["y"], IN)), seed=7)
    for equation in ("xmn,mny->xy", "xmn,mny->yx"):
        want, _ = dense_compose(
            equation, np.asarray(a.to_dense()), a.legs, np.asarray(b.to_dense()), b.legs
        )
        np.testing.assert_allclose(
            np.asarray(tenet.einsum(equation, a, b).to_dense()), want, atol=1e-11
        )


# --- the primitives -----------------------------------------------------------------


@pytest.mark.parametrize("name", PRIMITIVES)
@pytest.mark.parametrize("provider", [p for p in PROVIDERS if p != FERMIONIC])
def test_wiring_against_a_plain_dense_einsum(name, provider):
    """Which legs are closed, and in which order the survivors return.

    No provider here braids with a sign, so the whole chain collapses to one
    ``np.einsum`` over the dense expansions with the implementation's own letters.
    """
    got, operands = call(name, provider)
    equation = CORNERS[name][1] if name in CORNERS else VECTORS[name][2]
    want = np.einsum(equation, *(np.asarray(t.to_dense()) for t in operands))
    assert np.abs(want).max() > 1e-8, "a test whose oracle is all zeros proves nothing"
    np.testing.assert_allclose(np.asarray(got.to_dense()), want, atol=1e-10)


@pytest.mark.parametrize("name", PRIMITIVES)
@pytest.mark.parametrize("provider", PROVIDERS)
def test_signs_against_the_graded_dense_oracle(name, provider, monkeypatch):
    """Every step of every primitive's chain, replayed on dense arrays.

    The steps are recorded off the implementation, so the oracle contracts exactly what
    ``peps.py`` asks for -- and ``dense_compose`` refuses any step whose operand 1 does
    not supply ``IN``, which is what turns the docstrings' operand-order arguments into
    an assertion.
    """
    recorded: list = []
    real = tenet.einsum_chain

    def spy(steps):
        recorded.append(list(steps))
        return real(steps)

    monkeypatch.setattr(tenet, "einsum_chain", spy)
    got, _ = call(name, provider)
    assert len(recorded) == 1, "a primitive is one chain"

    arr, legs = None, None
    for equation, a, b, bend in recorded[0]:
        left = (arr, legs) if a is None else (np.asarray(a.to_dense()), a.legs)
        right = (arr, legs) if b is None else (np.asarray(b.to_dense()), b.legs)
        assert len(bend) == _minimal_bends(equation, left[1], right[1]), (
            f"{name}: step {equation!r} bends {bend!r}, more than the diagram forces"
        )
        arr, legs = dense_step(equation, *left, *right, bend)
    assert np.abs(arr).max() > 1e-8, "a test whose oracle is all zeros proves nothing"
    assert tuple(leg.side for leg in legs) == tuple(leg.side for leg in got.legs)
    np.testing.assert_allclose(np.asarray(got.to_dense()), arr, atol=1e-10)


#: Every primitive's fZ2 result differs from the sign-free contraction, and the list is
#: kept (empty) because it used to hold ``cor_tr``: that corner's one bend lands on the
#: physical wire, so before M82 phase 3 it paid nothing the plain einsum did not. Now the
#: bend also carries the ribbon twist of the cycle the step closes, and it differs too.
SIGN_FREE: tuple[str, ...] = ()


@pytest.mark.parametrize("name", PRIMITIVES)
def test_the_plain_einsum_is_wrong_for_fz2(name):
    """The teeth: for fZ2 the sign-free contraction is a *different* tensor."""
    got, operands = call(name, FERMIONIC)
    equation = CORNERS[name][1] if name in CORNERS else VECTORS[name][2]
    plain = np.einsum(equation, *(np.asarray(t.to_dense()) for t in operands))
    same = np.allclose(np.asarray(got.to_dense()), plain, atol=1e-8)
    assert same == (name in SIGN_FREE)


def test_a_second_bend_is_a_second_tensor():
    """Why bend-minimality is a criterion and not a taste.

    ``cor_tl`` also has a composition-rule-respecting spelling with the ket first and
    *two* bends. It contracts the same legs into the same leg order and is a different
    tensor, so "operand 1 supplies IN" alone does not determine a contraction -- the
    number of wires that turn around does.
    """
    a = site(FERMIONIC)
    two_bends = tenet.einsum_chain([("tlbrs,tlBRs->bBrR", a.ket, a.bra, "ls")])
    assert not tenet.allclose(cor_tl(a), two_bends)


@pytest.mark.parametrize("name", PRIMITIVES)
def test_every_primitive_returns_the_legs_its_docstring_promises(name):
    """Spaces and sides, read back off the result -- five distinct spaces make it sharp."""
    got, _ = call(name, "u1")
    sp = SPACES["u1"]
    ket_side = dict(zip("tlbrs", KET_SIDES, strict=True))
    order = {
        "cor_tl": "bbrr",
        "cor_bl": "rrtt",
        "cor_br": "ttll",
        "cor_tr": "llbb",
        "edge_t": "llbbrr",
        "edge_l": "bbrrtt",
        "edge_b": "rrttll",
        "edge_r": "ttllbb",
        "append_vec_tl": "xbbyrr",
        "append_vec_br": "xttyll",
        "append_vec_tr": "xllybb",
        "append_vec_bl": "xrrytt",
    }[name]
    assert got.ndim == len(order)
    seen: dict[str, int] = {}
    for leg, axis in zip(got.legs, order, strict=True):
        assert leg.space == sp[axis], f"{name}: axis {axis} has the wrong space"
        if axis in "xy":
            continue
        # each pair is the ket leg then the bra leg, so the two sides are opposite
        n = seen.get(axis, 0)
        seen[axis] = n + 1
        want = ket_side[axis] if n == 0 else (IN if ket_side[axis] is OUT else OUT)
        assert leg.side is want, f"{name}: axis {axis} copy {n} has side {leg.side}"


# --- containers ---------------------------------------------------------------------


def peps(provider: str = "u1") -> Peps:
    sp = SPACES[provider]
    legs = tuple(Leg(sp[k], s) for k, s in zip("tlbrs", KET_SIDES, strict=True))
    return Peps(CheckerboardLattice(), SymmetricTensor.random(legs, seed=2))


def test_peps_carries_rank_five_and_rank_four_and_refuses_the_rest():
    psi = peps()
    assert psi.has_physical()
    assert psi[0, 0] is psi[1, 1] and psi[0, 1] is psi[1, 0]
    sp = SPACES["u1"]
    flat = SymmetricTensor.random(
        tuple(Leg(sp[k], s) for k, s in zip("tlbr", KET_SIDES[:4], strict=True)), seed=2
    )
    assert not Peps(CheckerboardLattice(), flat).has_physical()
    with pytest.raises(ValueError, match="rank 4 or rank 5"):
        Peps(CheckerboardLattice(), SymmetricTensor.random((Leg(sp["t"], OUT),), seed=1))


def test_peps2layers_is_a_view_that_never_materializes_the_product():
    psi = peps()
    net = Peps2Layers(psi)
    a = net[0, 1]
    assert a.ndim == 4 and len(a.legs) == 4
    assert a.ket is psi[0, 1]
    assert tenet.allclose(a.bra, tenet.adjoint(psi[0, 1]))
    assert net[2, 3].ket is net[0, 1].ket  # the geometry folds, the view does not copy
    with pytest.raises(TypeError, match="view"):
        net[0, 0] = a
    sp = SPACES["u1"]
    flat = Peps(
        CheckerboardLattice(),
        SymmetricTensor.random(
            tuple(Leg(sp[k], s) for k, s in zip("tlbr", KET_SIDES[:4], strict=True)), seed=2
        ),
    )
    with pytest.raises(ValueError, match="physical leg"):
        Peps2Layers(flat)


def test_peps2layers_takes_an_independent_bra():
    psi, phi = peps(), peps()
    net = Peps2Layers(psi, phi)
    assert net[0, 0].bra is phi[0, 0]
    with pytest.raises(ValueError, match="geometry"):
        Peps2Layers(psi, Peps(SquareLattice(dims=(2, 2)), psi[0, 0]))
