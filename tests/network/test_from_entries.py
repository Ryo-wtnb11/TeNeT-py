"""``MPO.from_entries``: the sparse ``W`` named entry by entry, against ``from_terms``.

M52 (#217). The claim is not that a third builder exists; it is that a **hand-built** MPO
now produces the same [EdgeTable][tenet.network.EdgeTable] a term list does, so it is
indistinguishable to [Env][tenet.network.Env] and takes the one prepared engine path
(#218). Every test here is therefore one of two shapes:

* an oracle against [MPO.from_terms][tenet.network.MPO.from_terms] on the same
  Hamiltonian -- ``to_dense`` element-wise and the DMRG ground-state energy, on a U(1)
  spin chain and a fermionic ``fZ2`` chain, because the fermionic one is the only one
  that can fail on a Koszul sign the hand-written ``W`` never mentions;
* a structural assertion about what comes back -- the description is present, the
  block table is real, ``Env.heff2`` prepares against it, and both corner channels are
  exact unit slabs (``tests/network/test_pinned.py``'s #204 property, on an operator
  written by hand instead of one a sweep compressed).

The two models are written so that all four entry spellings and both identity routes
occur: the spin chain carries a constant ``Sz Sz`` tail through an ``(k, k): None``
spectator ride and an exact-distance-2 coupling through an ``(k, k+1): None`` identity
*move* between two different states, and the fermion chain carries an exponentially
decaying ``n n`` tail through a bare number entry, which is ``lambda`` times the identity.
"""

import numpy as np
import pytest

import tenet
from tenet import GradedSpace
from tenet.models import spin_half, spinless_fermion
from tenet.network import MPO, MPS, Env, dmrg_, local_op
from tenet.network.mps import _IDL, _IDR, _corner_slots

# --- the two models, each as a sparse W and as the term list it must equal ------------

J2, J3 = 0.35, 0.11  # the distance-2 coupling and the constant Sz Sz tail
T_HOP, V_NN, LAM = 1.0, 0.7, 0.4  # the hop, the n n amplitude and its decay


def _spin_w():
    """U(1) Heisenberg, plus ``J2 Sz_i Sz_{i+2}``, plus ``J3 Sz_i Sz_j`` for every ``j > i``.

    Channel 1 and 2 close the ``S-+``/``S+-`` hops, channel 3 the nearest-neighbour
    ``Sz Sz``; channels 4 and 5 are the distance-2 pair (an identity **move** carries the
    string across exactly one site) and channel 6 is the constant tail (an identity
    **ride** on one state carries it across any number of them).
    """
    site = spin_half()
    sz, sp, sm = site.ops["Sz"], site.ops["S+"], site.ops["S-"]
    return site.phys, {
        (0, 0): None,
        (0, 1): (0.5, sm),
        (1, -1): sp,
        (0, 2): (0.5, sp),
        (2, -1): sm,
        (0, 3): sz,
        (3, -1): sz,
        (0, 4): (J2, sz),
        (4, 5): None,
        (5, -1): sz,
        (0, 6): (J3, sz),
        (6, 6): None,
        (6, -1): sz,
        (-1, -1): None,
    }


def _spin_terms(n):
    site = spin_half()
    sz, sp, sm = site.ops["Sz"], site.ops["S+"], site.ops["S-"]
    terms = []
    for i in range(n - 1):
        terms += [
            (1.0, [(sz, i), (sz, i + 1)]),
            (0.5, [(sm, i), (sp, i + 1)]),
            (0.5, [(sp, i), (sm, i + 1)]),
        ]
    for i in range(n - 2):
        terms.append((J2, [(sz, i), (sz, i + 2)]))
    for i in range(n):
        for j in range(i + 1, n):
            terms.append((J3, [(sz, i), (sz, j)]))
    return terms


def _fermion_w():
    """Spinless ``fZ2`` hopping plus ``V lambda^(r-1) n_i n_{i+r}``, the exponential tail.

    The self-loop ``(3, 3): LAM`` is a bare number, so it is ``lambda`` times the
    identity -- the standard way an exponentially decaying interaction is written into a
    ``W``, and the reason a number is one of the four entry spellings.
    """
    site = spinless_fermion()
    c, cd, nop = site.ops["c"], site.ops["c+"], site.ops["n"]
    return site.phys, {
        (0, 0): None,
        (0, 1): (-T_HOP, cd),
        (1, -1): c,
        (0, 2): (T_HOP, c),
        (2, -1): cd,
        (0, 3): (V_NN, nop),
        (3, 3): LAM,
        (3, -1): nop,
        (-1, -1): None,
    }


def _fermion_terms(n):
    site = spinless_fermion()
    c, cd, nop = site.ops["c"], site.ops["c+"], site.ops["n"]
    terms = []
    for i in range(n - 1):
        terms += [(-T_HOP, [(cd, i), (c, i + 1)]), (-T_HOP, [(cd, i + 1), (c, i)])]
    for i in range(n):
        for j in range(i + 1, n):
            terms.append((V_NN * LAM ** (j - i - 1), [(nop, i), (nop, j)]))
    return terms


MODELS = {"spin U(1)": (_spin_w, _spin_terms), "spinless fZ2": (_fermion_w, _fermion_terms)}


def _both(model, n):
    """The same Hamiltonian through both builders."""
    make_w, make_terms = MODELS[model]
    phys, w = make_w()
    return phys, MPO.from_entries([w] * n), MPO.from_terms(n, make_terms(n), cutoff=None)


# --- the oracle: the same operator, both routes ---------------------------------------


@pytest.mark.parametrize("model", sorted(MODELS), ids=lambda k: k)
def test_the_hand_written_w_is_the_operator_the_term_list_builds(model):
    """``to_dense`` element-wise at 1e-13, on a bosonic chain and a fermionic one.

    The fermionic half is the one with something to say: the hand-written ``W`` names no
    Jordan-Wigner operator anywhere, and it does not have to -- the string is the odd
    ``fZ2`` MPO bond crossing a physical line, paid by the braiding inside
    ``_Walk.transition``, which this builder reaches through exactly as ``from_terms``
    does (M21/#147).
    """
    _, built, ref = _both(model, 6)
    a, b = np.asarray(built.to_dense()), np.asarray(ref.to_dense())
    assert a.shape == b.shape
    assert np.abs(a - b).max() < 1e-13 * max(np.abs(b).max(), 1.0)


@pytest.mark.parametrize("model", sorted(MODELS), ids=lambda k: k)
def test_both_routes_reach_the_same_dmrg_ground_state_energy(model):
    """One seed, two operators, one energy -- the oracle one level above ``to_dense``."""
    n, chi = 8, 24
    phys, built, ref = _both(model, n)
    bonds = _bond_spaces(phys, n)
    energies = [
        dmrg_(MPS.random(phys, bonds, seed=3), h, chi=chi, max_sweeps=12).energy
        for h in (built, ref)
    ]
    assert abs(energies[0] - energies[1]) < 1e-9


def _bond_spaces(phys, n):
    """A seed bond profile for ``phys``: every sector reachable in ``i`` steps, degeneracy 1."""
    sym = phys.provider
    spaces = [GradedSpace.new(sym, {sym.unit: 1})]
    for _ in range(n - 1):
        merged: dict = {}
        for a, _m in spaces[-1].sectors:
            for b, _n in phys.sectors:
                for csec in sym.fusion(a, b):
                    merged[csec] = 1
        spaces.append(GradedSpace.new(sym, merged))
    spaces.append(GradedSpace.new(sym, {sym.unit: 1}))
    return spaces


# --- what comes back is a description, on the one engine path -------------------------


@pytest.mark.parametrize("model", sorted(MODELS), ids=lambda k: k)
def test_the_built_operator_carries_a_real_block_table(model):
    """``edges`` is present and every site answers ``edge_blocks`` -- what ``from_w`` cannot.

    ``tests/network/test_heff2.py`` asserts the other half of this: a ``from_w`` operator
    hands out ``None`` throughout and routes onto the compatibility entry.
    """
    n = 6
    _, built, _ref = _both(model, n)
    assert built.edges is not None
    assert all(built.edge_blocks(m) is not None for m in range(n))
    assert MPO(built.sites).edge_blocks(0) is None  # the container alone keeps no symbols


@pytest.mark.parametrize("model", sorted(MODELS), ids=lambda k: k)
def test_heff2_takes_the_prepared_path_on_a_hand_built_operator(model):
    """``Env.heff2`` builds a prepared operator for the bond, which is the single path.

    Asserted on ``Env._prepared``, the cache the prepared path fills and the site-tensor
    compatibility entry never touches, rather than by reading the operator's type.
    """
    n = 6
    phys, built, _ref = _both(model, n)
    psi = MPS.random(phys, _bond_spaces(phys, n), seed=1).canonize_()
    env = Env(psi, built).setup_()
    aa = tenet.einsum("apx,xqr->apqr", psi[0], psi[1])  # the two-site tensor heff2 eats
    out = env.heff2(0, aa)
    assert 0 in env._prepared
    assert out.legs == aa.legs


@pytest.mark.parametrize("model", sorted(MODELS), ids=lambda k: k)
def test_every_cut_keeps_its_corner_channels_exact(model):
    """#204's corner-exactness property, on an operator written by hand.

    ``W[IdL, :, :, IdL]`` and ``W[IdR, :, :, IdR]`` are the identity on the physical
    space, nothing else in the ``IdL`` column is non-zero and nothing else in the ``IdR``
    row is -- the four zeros of MPSKit's ``(1 C D; . A B; . . 1)``. Here they are what the
    ``0``/``-1`` index convention *promises*, so this is the test the convention owes:
    ``tests/network/test_pinned.py`` asserts the same three things about a compressed
    operator, with the same mechanism.
    """
    n = 6
    _, built, _ref = _both(model, n)
    tab = built.edges
    assert tab is not None
    sym, d = tab.phys.provider, tab.phys.dim
    for site in range(n):
        w = np.asarray(built[site].to_dense())
        lo, hi = _corner_slots(tab.bonds[site], sym), _corner_slots(tab.bonds[site + 1], sym)
        if _IDL in tab.ordered[site] and _IDL in tab.ordered[site + 1]:
            assert np.abs(w[lo["idl"], :, :, hi["idl"]] - np.eye(d)).max() < 1e-13
            column = np.delete(w[:, :, :, hi["idl"]], lo["idl"], axis=0)
            assert column.size == 0 or np.abs(column).max() < 1e-13
        if _IDR in tab.ordered[site] and _IDR in tab.ordered[site + 1]:
            assert np.abs(w[lo["idr"], :, :, hi["idr"]] - np.eye(d)).max() < 1e-13
            row = np.delete(w[lo["idr"]], hi["idr"], axis=2)
            assert row.size == 0 or np.abs(row).max() < 1e-13


def test_one_bulk_w_serves_the_first_and_last_site_too():
    """The two boundary bonds are ``D=1``: bond 0 keeps ``IdL``, bond N keeps ``IdR``.

    That is ``from_w``'s ``start`` row and ``end`` column, done by the finite-state
    machine's own pruning rather than by slicing an array -- which is what lets the caller
    hand over the same bulk mapping for every site, including the two ends.
    """
    _phys, w = _spin_w()
    h = MPO.from_entries([w] * 5)
    assert h[0].legs[0].space.dim == 1
    assert h[4].legs[3].space.dim == 1


# --- the refusals ---------------------------------------------------------------------


def _op():
    return spin_half().ops["Sz"]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        pytest.param([], "chain length", id="no sites"),
        pytest.param([{(0, 0): None}], "every entry is an identity", id="no operator"),
        pytest.param([{0: _op()}], "keyed by", id="key is not a pair"),
        pytest.param([{(0, "a"): _op()}], "pair of ints", id="index is not an int"),
        pytest.param([{(0, -2): _op()}], "below -1", id="index below -1"),
        pytest.param([{(1, 0): _op()}], "enters the IdL channel", id="into IdL"),
        pytest.param([{(-1, 1): _op()}], "leaves the IdR channel", id="out of IdR"),
        pytest.param([{(0, 0): _op()}], "identities by definition", id="operator on IdL"),
        pytest.param([{(-1, -1): 2.0}], "identities by definition", id="scaled IdR"),
        pytest.param([{(0, -1): (1.0,)}], "(coefficient, operator)", id="short pair"),
        pytest.param([{(0, -1): "Sz"}], "a rank-3 operator from local_op", id="a string"),
        pytest.param([{(0, -1): np.eye(2)}], "a rank-3 operator from local_op", id="an array"),
    ],
)
def test_a_malformed_entry_names_what_is_wrong(entries, message):
    with pytest.raises(ValueError, match=message):
        MPO.from_entries(entries)


def test_an_invariant_k_site_operator_is_refused_with_a_pointer_to_from_terms():
    """One ``W`` entry sits on one site; ``local_op``'s rank-2k form spans ``k`` of them."""
    op = local_op(spin_half().matrices["S.S"], phys=spin_half().phys)
    with pytest.raises(ValueError, match="rank-4 operator.*MPO.from_terms"):
        MPO.from_entries([{(0, -1): op}])


def test_operators_on_two_physical_spaces_are_refused():
    other = spinless_fermion().ops["n"]
    with pytest.raises(ValueError, match="disagree about the physical space"):
        MPO.from_entries([{(0, 1): _op(), (1, -1): other}, {(0, -1): _op()}])


def test_a_charged_channel_cannot_close_into_idr():
    """Both MPO boundaries are the trivial ``D=1`` leg, so a channel closes at unit charge."""
    site = spin_half()
    with pytest.raises(ValueError, match="closes into IdR"):
        MPO.from_entries([{(0, -1): site.ops["S+"]}, {(0, -1): None}])


def test_two_entries_reaching_one_state_with_different_charges_are_refused():
    """A bond state carries one ``GradedSpace``, so the entries writing into it must agree."""
    sz, sp = spin_half().ops["Sz"], spin_half().ops["S+"]
    with pytest.raises(ValueError, match="two different charges"):
        MPO.from_entries([{(0, 1): sz, (0, 2): sp}, {(1, 3): sz, (2, 3): sz}, {(3, -1): sz}])


def test_a_state_that_carries_nothing_names_itself():
    """A channel that opens and never closes carries nothing, and saying so is the point.

    The bar is "dead at *every* bond", not "dead at some bond": a range-2 coupling's
    channel is legitimately dead at the last bond, because a term opening there would run
    off the end, and the two boundary bonds drop everything but ``IdL`` / ``IdR`` by
    design -- which is how one bulk mapping serves every site.
    """
    sz = spin_half().ops["Sz"]
    w = {(0, 0): None, (0, 3): sz, (0, 1): sz, (1, -1): sz}
    with pytest.raises(ValueError, match="state 3 carries nothing"):
        MPO.from_entries([w, {(1, -1): sz, (3, 3): None}, {(0, -1): None}])
