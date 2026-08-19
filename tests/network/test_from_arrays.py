"""``MPO.from_arrays``: block2's three arrays as the input shape, against ``from_terms``.

Every test here is an oracle against [MPO.from_terms][tenet.network.MPO.from_terms] on the
same Hamiltonian, because that is the whole claim: ``from_arrays`` is a second *input
layer* onto the same finite-state-machine assembler, not a second assembler. The models
are the ones the layer already has fixtures for -- the U(1) Heisenberg chain, the
spinless fZ2 hop chain, and the spinful Hubbard chain whose on-site ``n_up n_dn`` is
where coincident indices arrive -- and the tolerance is 1e-14 on ``to_dense()``, with the
bond profile compared cut by cut wherever no coefficient cancels exactly.
"""

import numpy as np
import pytest

from tenet import GradedSpace
from tenet.network import MPO, local_op
from tenet.symmetry import U1, FZ2Sector, U1Sector, fZ2

# --- the three models, each as a term list and as blocks -----------------------------

SPIN = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
_SZ = np.diag([-0.5, 0.5])
_SP = np.array([[0.0, 0.0], [1.0, 0.0]])

FZ2_PHYS = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
_A = np.array([[0.0, 1.0], [0.0, 0.0]])  # annihilation, |1> -> |0>

PHYS4 = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})  # test_hubbard.py's site
C_UP = np.zeros((4, 4))
C_UP[0, 2] = 1.0
C_UP[3, 1] = 1.0
C_DN = np.zeros((4, 4))
C_DN[0, 3] = 1.0
C_DN[2, 1] = -1.0
N_UP, N_DN = C_UP.T @ C_UP, C_DN.T @ C_DN


def _pairs(n_sites):
    return np.array([[m, m + 1] for m in range(n_sites - 1)])


def heisenberg(n_sites):
    """``sum_i Sz Sz + (S+ S- + S- S+)/2``: U(1), bosonic, three blocks."""
    ops = {
        "z": local_op(_SZ, phys=SPIN, charge=U1Sector(0)),
        "+": local_op(_SP, phys=SPIN, charge=U1Sector(-2)),
        "-": local_op(_SP.T, phys=SPIN, charge=U1Sector(2)),
    }
    terms = []
    for m in range(n_sites - 1):
        terms += [
            (1.0, [(ops["z"], m), (ops["z"], m + 1)]),
            (0.5, [(ops["+"], m), (ops["-"], m + 1)]),
            (0.5, [(ops["-"], m), (ops["+"], m + 1)]),
        ]
    bond = _pairs(n_sites)
    blocks = [
        ("zz", bond, np.ones(n_sites - 1)),
        ("+-", bond, np.full(n_sites - 1, 0.5)),
        ("-+", bond, np.full(n_sites - 1, 0.5)),
    ]
    return ops, terms, blocks


def spinless(n_sites):
    """``-sum_m (c+_m c_m+1 + h.c.) + sum_m n_m n_m+1``: fZ2, the backward hop out of order."""
    ops = {
        "C": local_op(_A.T, phys=FZ2_PHYS, charge=FZ2Sector(1)),
        "c": local_op(_A, phys=FZ2_PHYS, charge=FZ2Sector(1)),
        "n": local_op(_A.T @ _A, phys=FZ2_PHYS, charge=FZ2Sector(0)),
    }
    terms = []
    for m in range(n_sites - 1):
        terms += [
            (-1.0, [(ops["C"], m), (ops["c"], m + 1)]),
            (-1.0, [(ops["C"], m + 1), (ops["c"], m)]),
            (0.7, [(ops["n"], m), (ops["n"], m + 1)]),
        ]
    fwd, bwd = _pairs(n_sites), _pairs(n_sites)[:, ::-1]
    blocks = [
        ("Cc", fwd, np.full(n_sites - 1, -1.0)),
        ("Cc", bwd, np.full(n_sites - 1, -1.0)),  # indices out of site order: a Koszul sign
        ("nn", fwd, np.full(n_sites - 1, 0.7)),
    ]
    return ops, terms, blocks


def hubbard(n_sites, u=3.0):
    """``-t sum (c+ c + h.c.) + U sum n_up n_dn``: the on-site term is two coincident indices."""
    ops = {
        "U": local_op(C_UP.T, phys=PHYS4, charge=FZ2Sector(1)),
        "u": local_op(C_UP, phys=PHYS4, charge=FZ2Sector(1)),
        "D": local_op(C_DN.T, phys=PHYS4, charge=FZ2Sector(1)),
        "d": local_op(C_DN, phys=PHYS4, charge=FZ2Sector(1)),
        "a": local_op(N_UP, phys=PHYS4, charge=FZ2Sector(0)),
        "b": local_op(N_DN, phys=PHYS4, charge=FZ2Sector(0)),
    }
    nn = local_op(N_UP @ N_DN, phys=PHYS4, charge=FZ2Sector(0))
    terms = []
    for m in range(n_sites - 1):
        for cd, c in (("U", "u"), ("D", "d")):
            terms += [
                (-1.0, [(ops[cd], m), (ops[c], m + 1)]),
                (-1.0, [(ops[cd], m + 1), (ops[c], m)]),
            ]
    terms += [(u, [(nn, m)]) for m in range(n_sites)]
    fwd, bwd = _pairs(n_sites), _pairs(n_sites)[:, ::-1]
    blocks = [
        ("Uu", fwd, np.full(n_sites - 1, -1.0)),
        ("Uu", bwd, np.full(n_sites - 1, -1.0)),
        ("Dd", fwd, np.full(n_sites - 1, -1.0)),
        ("Dd", bwd, np.full(n_sites - 1, -1.0)),
        ("ab", np.array([[m, m] for m in range(n_sites)]), np.full(n_sites, u)),
    ]
    return ops, terms, blocks


MODELS = {"heisenberg": (heisenberg, 6), "spinless": (spinless, 6), "hubbard": (hubbard, 4)}


def _profile(mpo):
    return [mpo[n].legs[0].space.dim for n in range(len(mpo))] + [mpo[-1].legs[3].space.dim]


def _deviation(a, b):
    da, db = np.asarray(a.to_dense()), np.asarray(b.to_dense())
    return float(np.abs(da - db).max() / np.abs(db).max())


# --- the acceptance criteria ---------------------------------------------------------


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("cutoff", [1e-13, 0.0, None])
def test_from_arrays_is_the_operator_from_terms_builds(model, cutoff):
    """The gate: same Hamiltonian, both input layers, at all three cutoffs.

    Spin and fermion, bosonic and sign-braiding, coincident and not -- and the bond
    profile compared cut by cut, because these fixtures have no exactly cancelling
    coefficient for the merge to remove.
    """
    build, n_sites = MODELS[model]
    ops, terms, blocks = build(n_sites)
    array = MPO.from_arrays(n_sites, ops, blocks, cutoff=cutoff)
    listed = MPO.from_terms(n_sites, terms, cutoff=cutoff)
    assert _deviation(array, listed) <= 1e-14
    assert _profile(array) == _profile(listed)


def test_the_index_array_may_arrive_flat():
    """block2 stores ``nn`` indices per term concatenated; a 1-D buffer is reshaped."""
    ops, _terms, blocks = heisenberg(5)
    flat = [(expr, np.asarray(idx).reshape(-1), data) for expr, idx, data in blocks]
    assert _deviation(MPO.from_arrays(5, ops, flat), MPO.from_arrays(5, ops, blocks)) == 0.0


def test_a_flat_index_buffer_of_the_wrong_length_raises_naming_the_block():
    """``len(expr) * len(data) == indices.size``, block2's own assert as a ValueError."""
    ops, _terms, _blocks = heisenberg(4)
    bad = [("zz", np.array([0, 1, 2]), np.ones(1))]
    with pytest.raises(ValueError, match=r"'zz'.*2 operator.*1 coefficient.*got 3"):
        MPO.from_arrays(4, ops, bad)
    with pytest.raises(ValueError, match=r"'zz'.*got 4"):
        MPO.from_arrays(4, ops, [("zz", np.array([[0, 1], [1, 2]]), np.ones(1))])


def test_an_expr_naming_an_operator_the_table_does_not_define_raises():
    """The table is the caller's and it is the only vocabulary; an unknown name says so."""
    ops, _terms, _blocks = heisenberg(4)
    with pytest.raises(ValueError, match=r"'zx' names \['x'\]"):
        MPO.from_arrays(4, ops, [("zx", np.array([[0, 1]]), np.ones(1))])


def test_a_site_index_outside_the_chain_raises():
    ops, _terms, _blocks = heisenberg(4)
    with pytest.raises(ValueError, match=r"outside range\(4\)"):
        MPO.from_arrays(4, ops, [("zz", np.array([[0, 4]]), np.ones(1))])


def test_a_k_site_operator_in_the_table_is_refused_with_a_pointer_to_from_terms():
    """One site index per name, so ``local_op``'s invariant k-site form has nowhere to go."""
    ops, _terms, _blocks = heisenberg(4)
    ops = dict(ops, k=local_op(np.kron(_SZ, _SZ), phys=SPIN))
    with pytest.raises(ValueError, match="from_terms"):
        MPO.from_arrays(4, ops, [("zz", np.array([[0, 1]]), np.ones(1))])


def test_no_blocks_at_all_raises_rather_than_building_a_nameless_zero():
    """An MPO is read off its terms; with none there is nothing to read."""
    ops, _terms, _blocks = heisenberg(4)
    with pytest.raises(ValueError, match="no term survived"):
        MPO.from_arrays(4, ops, [])
    with pytest.raises(ValueError, match="no term survived"):
        MPO.from_arrays(4, ops, [("zz", np.zeros((0, 2), dtype=int), np.zeros(0))])


def test_coincident_sites_are_pre_multiplied_rather_than_refused():
    """``from_terms``' "multiply them first" burden, discharged by the input layer.

    ``n_up n_dn`` on one site, spelled as two coincident indices, must build the same
    operator as the single pre-multiplied ``local_op(N_UP @ N_DN)`` handed to
    ``from_terms`` -- which is the refusal message's own instruction, now unnecessary.
    """
    ops = {"a": local_op(N_UP, phys=PHYS4, charge=FZ2Sector(0))}
    ops["b"] = local_op(N_DN, phys=PHYS4, charge=FZ2Sector(0))
    nn = local_op(N_UP @ N_DN, phys=PHYS4, charge=FZ2Sector(0))
    array = MPO.from_arrays(3, ops, [("ab", np.array([[1, 1]]), np.array([2.0]))])
    listed = MPO.from_terms(3, [(2.0, [(nn, 1)])])
    assert _deviation(array, listed) <= 1e-14
    with pytest.raises(ValueError, match="sit on site 1; multiply them first"):
        MPO.from_terms(3, [(2.0, [(ops["a"], 1), (ops["b"], 1)])])


def test_a_term_whose_on_site_product_vanishes_is_dropped():
    """``c c`` on one site is zero, and a zero term must not allocate an FSM state."""
    ops, _terms, _blocks = spinless(4)
    both = [("cc", np.array([[1, 1]]), np.array([5.0])), ("nn", _pairs(4), np.full(3, 0.7))]
    only = [("nn", _pairs(4), np.full(3, 0.7))]
    dropped, plain = MPO.from_arrays(4, ops, both), MPO.from_arrays(4, ops, only)
    assert _deviation(dropped, plain) == 0.0
    assert _profile(dropped) == _profile(plain)
    with pytest.raises(ValueError, match="no term survived"):
        MPO.from_arrays(4, ops, [("cc", np.array([[1, 1]]), np.array([5.0]))])


def test_terms_that_agree_are_merged_with_their_coefficients_summed():
    """The same string twice at half weight is the string once at full weight.

    And an exact cancellation is *visible* to the merge, so the state it would have
    needed is never allocated -- the one place ``from_arrays`` gives a narrower bond
    than ``from_terms``, recorded here rather than hidden.
    """
    ops, terms, _blocks = heisenberg(5)
    halves = [("zz", _pairs(5), np.full(4, 0.5))] * 2 + [
        ("+-", _pairs(5), np.full(4, 0.5)),
        ("-+", _pairs(5), np.full(4, 0.5)),
    ]
    assert _deviation(MPO.from_arrays(5, ops, halves), MPO.from_terms(5, terms)) <= 1e-14

    cancelling = [
        ("zz", _pairs(5), np.full(4, 1.0)),
        ("+-", np.array([[0, 2]]), np.array([1.0])),
        ("-+", np.array([[2, 0]]), np.array([-1.0])),  # the same string, opposite sign
    ]
    merged = MPO.from_arrays(5, ops, cancelling, cutoff=None)
    kept = MPO.from_terms(
        5,
        [(1.0, [(ops["z"], m), (ops["z"], m + 1)]) for m in range(4)]
        + [(1.0, [(ops["+"], 0), (ops["-"], 2)]), (-1.0, [(ops["-"], 2), (ops["+"], 0)])],
        cutoff=None,
    )
    assert _deviation(merged, kept) <= 1e-14
    assert sum(_profile(merged)) < sum(_profile(kept))


def test_the_screen_threshold_drops_terms_after_the_merge():
    """``screen`` is applied to the merged coefficient, not to the raw block entries."""
    ops, _terms, _blocks = heisenberg(5)
    blocks = [
        ("zz", _pairs(5), np.full(4, 1.0)),
        ("+-", np.array([[0, 2]]), np.array([1e-14])),  # under the default 1e-12
    ]
    screened = MPO.from_arrays(5, ops, blocks, cutoff=None)
    kept = MPO.from_arrays(5, ops, blocks, cutoff=None, screen=1e-16)
    assert sum(_profile(screened)) < sum(_profile(kept))
    assert _deviation(kept, screened) > 0.0

    # The merge runs first, so two entries that only clear the threshold together do.
    split = [
        ("zz", _pairs(5), np.full(4, 1.0)),
        ("+-", np.array([[0, 2]] * 2), np.full(2, 1e-9)),
    ]
    assert sum(_profile(MPO.from_arrays(5, ops, split, cutoff=None))) == sum(_profile(kept))


def test_a_complex_data_array_decides_the_mpo_dtype():
    """``data``'s dtype is the coefficient's dtype, exactly as in ``from_terms``."""
    ops, _terms, _blocks = heisenberg(4)
    blocks = [("+-", _pairs(4), np.full(3, 0.5j)), ("-+", _pairs(4), np.full(3, -0.5j))]
    array = MPO.from_arrays(4, ops, blocks)
    listed = MPO.from_terms(
        4,
        [(0.5j, [(ops["+"], m), (ops["-"], m + 1)]) for m in range(3)]
        + [(-0.5j, [(ops["-"], m), (ops["+"], m + 1)]) for m in range(3)],
    )
    assert np.iscomplexobj(np.asarray(array.to_dense()))
    assert _deviation(array, listed) <= 1e-14


def test_a_four_fermion_string_agrees_however_its_indices_are_ordered():
    """The Koszul sign of the sort into site order, on every permutation of one string.

    ``from_terms`` pays one sign per inversion of two sign-braiding operators, strictly
    ``>``; ``from_arrays`` pays the same sign vectorized. The oracle is that every
    spelling of ``c+_i c+_j c_k c_l`` builds what the term list builds from the same
    spelling -- so the two agree term by term and not merely in total.
    """
    ops, _terms, _blocks = spinless(4)
    names = ("C", "C", "c", "c")
    for idx in ((0, 1, 2, 3), (1, 0, 3, 2), (3, 2, 1, 0), (2, 0, 3, 1), (0, 3, 1, 2)):
        blocks = [("".join(names), np.array([idx]), np.array([0.9]))]
        listed = [(0.9, [(ops[nm], site) for nm, site in zip(names, idx, strict=True)])]
        assert _deviation(MPO.from_arrays(4, ops, blocks), MPO.from_terms(4, listed)) <= 1e-14


def test_a_long_operator_name_needs_a_separator_and_a_bare_pattern_does_not():
    """block2's spelling rule: whitespace separates, otherwise one name per character."""
    ops, _terms, _blocks = heisenberg(4)
    ops = {"sz": ops["z"], "z": ops["z"], "+": ops["+"], "-": ops["-"]}
    spaced = MPO.from_arrays(4, ops, [("sz sz", _pairs(4), np.ones(3))])
    bare = MPO.from_arrays(4, ops, [("zz", _pairs(4), np.ones(3))])
    assert _deviation(spaced, bare) == 0.0
    with pytest.raises(ValueError, match=r"names \['s'\]"):
        MPO.from_arrays(4, ops, [("szsz", _pairs(4), np.ones(3))])
