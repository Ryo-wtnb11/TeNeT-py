"""Symmetry restriction — ``to_symmetry``, one test per criterion of #92.

The oracle throughout is the *defining dense identity*: ``to_symmetry`` is a permutation
of the dense basis and nothing else, so

    to_dense(to_symmetry(t, U1)) == take(to_dense(t), take_i, axis=i)      exactly, 0.0

with ``take_i`` written independently below (:func:`expected_take`) rather than
read back out of the implementation. Everything a caller cares about — norm
preservation, functoriality under ``tensordot`` — follows from it.
"""

import math
import re
from collections import Counter

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.cast import _cast_leg, to_symmetry
from tenet.ops.dense import from_dense
from tenet.symmetry import (
    SU2,
    U1,
    BranchingRules,
    CapabilityError,
    ProductProvider,
    SU2Sector,
    Trivial,
    U1Sector,
    fZ2,
)

# --- fixtures -------------------------------------------------------------------

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {SINGLET: 2, HALF: 1})  # the prototype's {j=0: 2, j=1: 1}
W = GradedSpace.new(SU2, {HALF: 2, ONE: 1})  # the prototype's {j=1: 2, j=2: 1}

STRUCTURES = {
    "su2_2": (Leg(V, OUT), Leg(V, IN)),
    "su2_2w": (Leg(W, OUT), Leg(W, IN)),
    # the prototype fixture: 14 -> 38 parameters, 4 -> 13 blocks
    "su2_3": (Leg(V, OUT), Leg(W, OUT), Leg(W, IN)),
    # fusion multiplicity K > 1: two block keys land in one dense cell
    "su2_4": (Leg(V, OUT), Leg(W, OUT), Leg(V, IN), Leg(W, IN)),
    "su2_dual": (Leg(V, OUT, dual=True), Leg(V, IN)),
    "su2_dual_3": (Leg(V, OUT), Leg(W, OUT, dual=True), Leg(W, IN)),
}
ALL = tuple(STRUCTURES)


def tensor(name: str, seed: int = 0, dtype=np.float64) -> SymmetricTensor:
    return SymmetricTensor.random(STRUCTURES[name], seed=seed, dtype=dtype)


def expected_take(leg: Leg) -> tuple[list[int], list[U1Sector]]:
    """The per-axis gather and the target label of each source dense index.

    Written from the layout contract alone, independently of ``_cast_leg``: the
    source dense index enumerates ``(a, alpha, k)`` in that nesting, and index
    ``k`` of ``V_j`` carries ``2 S_z = two_j - 2k``.
    """
    labels: list[U1Sector] = []
    for a, m in leg.space.sectors:
        for _alpha in range(m):
            for k in range(a.two_j + 1):
                labels.append(U1Sector(a.two_j - 2 * k))
    take = sorted(range(len(labels)), key=labels.__getitem__)  # stable
    return take, labels


def gather(dense: np.ndarray, takes) -> np.ndarray:
    for axis, take in enumerate(takes):
        dense = np.take(dense, take, axis=axis)
    return dense


# --- the capability -------------------------------------------------------------


def test_branch_is_the_doubled_magnetic_numbers_descending():
    # froSTspin spells the same vector `np.arange(irr - 1, -irr - 1, -2)`, with
    # `irr = two_j + 1` its dimension label.
    for two_j in range(7):
        a = SU2Sector(two_j)
        got = SU2.branch(U1, a)
        assert got == tuple(U1Sector(two_j - 2 * k) for k in range(two_j + 1))
        assert len(got) == SU2.irrep_dim(a)
        assert list(got) == sorted(got, reverse=True)
        assert np.array_equal([q.charge for q in got], np.arange(two_j + 1 - 1, -(two_j + 1), -2))


@pytest.mark.parametrize("target", [SU2, Trivial, fZ2, ProductProvider((U1, U1))])
def test_branch_refuses_a_target_it_cannot_reach(target):
    with pytest.raises(CapabilityError, match="cannot branch to"):
        SU2.branch(target, HALF)


def test_only_su2_advertises_the_capability():
    assert isinstance(SU2, BranchingRules)
    for p in (U1, fZ2, Trivial, ProductProvider((U1, U1))):
        assert not isinstance(p, BranchingRules)


def test_branching_rules_is_exported_and_listed():
    import tenet.symmetry as sym
    from tenet.symmetry import base

    assert "BranchingRules" in base.__all__
    assert "BranchingRules" in sym.__all__
    assert sym.BranchingRules is base.BranchingRules


# --- the defining dense identity ------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_to_symmetry_is_exactly_the_per_axis_gather_of_the_dense_array(name):
    t = tensor(name, seed=1)
    takes = [expected_take(leg)[0] for leg in t.legs]
    got = to_symmetry(t, U1).to_dense()
    assert np.max(np.abs(got - gather(t.to_dense(), takes))) == 0.0


@pytest.mark.parametrize("name", ALL)
def test_the_default_symmetry_check_passes(name):
    # atol=None: `from_dense` proves the U(1) blocks reproduce the permuted dense
    # array rather than approximate it. A wrong branch or a wrong sort is refused.
    to_symmetry(tensor(name, seed=2), U1)


def test_function_method_and_ops_export_agree():
    t = tensor("su2_3", seed=3)
    a, b, c = tenet.to_symmetry(t, U1), t.to_symmetry(U1), tenet.ops.to_symmetry(t, U1)
    assert tenet.allclose(a, b) and tenet.allclose(a, c)


# --- the dual convention, pinned by a refusal -----------------------------------


def test_the_negated_dual_convention_is_refused_and_the_shipped_one_is_exact():
    """``q = two_j - 2k`` on a ``dual`` leg; ``q = -(two_j - 2k)`` is *wrong*.

    ``to_dense`` applies the Z-isomorphism to a dual leg, and SU(2)'s ``z_matrix``
    is an antidiagonal signed permutation (see the sibling test), so reversing the
    magnetic order and negating the weight are the same operation and cancel. The
    wrong convention is therefore caught by ``from_dense``'s existing check, with
    a residual — not by a comment.
    """
    t = tensor("su2_dual", seed=4)
    dense = t.to_dense()

    shipped_legs, shipped_takes = zip(*(_cast_leg(leg, U1) for leg in t.legs), strict=True)
    ok = from_dense(gather(dense, shipped_takes), shipped_legs)
    assert np.max(np.abs(ok.to_dense() - gather(dense, shipped_takes))) == 0.0

    # the obvious-but-wrong convention: negate the charge on the *dual* leg only.
    # (Negating every leg at once is a global relabelling and stays symmetric, so
    # it would prove nothing — the claim is specifically about `dual`.)
    negated_legs, negated_takes = [], []
    for leg in t.legs:
        _, labels = expected_take(leg)
        if leg.dual:
            labels = [U1Sector(-q.charge) for q in labels]
        take = sorted(range(len(labels)), key=labels.__getitem__)
        negated_legs.append(Leg(GradedSpace.new(U1, Counter(labels)), leg.side, leg.dual, leg.name))
        negated_takes.append(take)

    with pytest.raises(ValueError, match="is not symmetric") as exc:
        from_dense(gather(dense, negated_takes), negated_legs)
    residual = float(re.search(r"residual ([0-9.e+-]+)", str(exc.value)).group(1))
    assert residual > 0.1


@pytest.mark.parametrize("two_j", range(5))
def test_z_matrix_is_an_antidiagonal_signed_permutation(two_j):
    """Why the negation cancels: ``Z`` reverses the magnetic order, up to signs.

    ``Z[k, d-1-k] = ±1`` and every other entry is exactly ``0.0``, so a dual leg's
    dense slab is the direct one read backwards — which is the same reordering a
    charge negation would ask for. The two cancel, and ``branch`` needs no sign.
    """
    z = SU2.z_matrix(SU2Sector(two_j))
    d = two_j + 1
    assert z.shape == (d, d)
    anti = np.array([z[k, d - 1 - k] for k in range(d)])
    np.testing.assert_allclose(anti, [(-1.0) ** k for k in range(d)], atol=1e-15)
    off = z.copy()
    off[np.arange(d), d - 1 - np.arange(d)] = 0.0
    assert np.array_equal(off, np.zeros((d, d)))  # exactly zero, not merely small


# --- the target space and its layout --------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_the_target_space_is_exactly_the_branching_multiset(name):
    t = tensor(name, seed=5)
    out = to_symmetry(t, U1)
    for axis, leg in enumerate(t.legs):
        _, labels = expected_take(leg)
        space = out.legs[axis].space
        for q in set(labels):
            assert space.degeneracy(q) == labels.count(q)
        assert len(space) == len(set(labels))


def test_the_prototype_spaces_branch_to_the_stated_charges():
    t = tensor("su2_3", seed=6)
    out = to_symmetry(t, U1)
    assert dict(out.legs[0].space.sectors) == {
        U1Sector(-1): 1,
        U1Sector(0): 2,
        U1Sector(1): 1,
    }
    assert dict(out.legs[1].space.sectors) == {
        U1Sector(-2): 1,
        U1Sector(-1): 2,
        U1Sector(0): 1,
        U1Sector(1): 2,
        U1Sector(2): 1,
    }


@pytest.mark.parametrize("name", ALL)
def test_the_degeneracy_index_layout_is_source_enumeration_order(name):
    """Within a target sector the degeneracy index runs ``(a, alpha, k)`` ascending.

    ``sorted`` is stable, so no tie-break rule is invented and the layout is
    reproducible across processes — which is what makes two independently cast
    tensors contractable against each other.
    """
    for leg in STRUCTURES[name]:
        got_leg, got_take = _cast_leg(leg, U1)
        want_take, labels = expected_take(leg)
        assert got_take == want_take
        # the gathered labels are the target space read out slab by slab
        gathered = [labels[i] for i in got_take]
        assert gathered == sorted(labels)
        assert [q for q, m in got_leg.space.sectors for _ in range(m)] == gathered


# --- what is preserved, and what is not -----------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_norm_is_preserved(name):
    """#20's ``qdim`` weight on the SU(2) side, plain sums on the U(1) side; #82
    pinned both against the dense Frobenius norm, which a relabelling cannot move.
    Not bit-exact: the two are accumulated over different block decompositions.
    """
    t = tensor(name, seed=7)
    assert float(tenet.norm(to_symmetry(t, U1))) == pytest.approx(float(tenet.norm(t)), abs=1e-14)


def test_to_symmetry_forgets_measurably():
    """More parameters, more blocks — SU(2) also constrains the ``m``-dependence
    through the CG coefficients that ``to_dense`` just spent. That is why there is
    no inverse: recovering SU(2) would be a projection with a tolerance.
    """
    t = tensor("su2_3", seed=8)
    out = to_symmetry(t, U1)
    assert (sum(b.size for b in t.blocks), len(t.blocks)) == (14, 4)
    assert (sum(b.size for b in out.blocks), len(out.blocks)) == (38, 13)


@pytest.mark.parametrize("name", ALL)
def test_side_dual_and_name_are_carried_through(name):
    legs = tuple(leg.renamed(f"leg{i}") for i, leg in enumerate(STRUCTURES[name]))
    t = SymmetricTensor.random(legs, seed=9)
    for old, new in zip(t.legs, to_symmetry(t, U1).legs, strict=True):
        assert (new.side, new.dual, new.name) == (old.side, old.dual, old.name)
        assert new.provider is U1
        assert new.space != old.space


# --- functoriality --------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "axes"),
    [
        ("su2_3", "su2_2w", ((2,), (0,))),
        ("su2_4", "su2_3", ((2, 3), (0, 1))),
    ],
)
def test_to_symmetry_commutes_with_tensordot(a, b, axes):
    """The load-bearing criterion: it is what lets a caller cast a whole network
    tensor by tensor. It works because a composable pair of legs shares a space
    and a ``dual`` flag, hence the *same* per-axis gather.
    """
    x, y = tensor(a, seed=10), tensor(b, seed=11)
    assert tenet.allclose(
        to_symmetry(tenet.tensordot(x, y, axes), U1),
        tenet.tensordot(to_symmetry(x, U1), to_symmetry(y, U1), axes),
    )


def test_to_symmetry_commutes_with_compose_on_a_map_view():
    x = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN)), seed=12)
    y = SymmetricTensor.random((Leg(W, OUT), Leg(V, IN)), seed=13)
    assert tenet.allclose(
        to_symmetry(tenet.as_map(x).compose(y), U1),
        tenet.as_map(to_symmetry(x, U1)).compose(to_symmetry(y, U1)),
    )


def test_to_symmetry_commutes_with_add_multiply_and_conj():
    x, y = tensor("su2_3", seed=14), tensor("su2_3", seed=15)
    assert tenet.allclose(
        to_symmetry(tenet.add(x, y), U1), tenet.add(to_symmetry(x, U1), to_symmetry(y, U1))
    )
    assert tenet.allclose(
        to_symmetry(tenet.multiply(x, 2.5), U1), tenet.multiply(to_symmetry(x, U1), 2.5)
    )
    assert tenet.allclose(to_symmetry(tenet.conj(x), U1), tenet.conj(to_symmetry(x, U1)))


def test_complex_blocks_stay_complex_and_satisfy_both_criteria():
    t = tensor("su2_3", seed=16)
    t = SymmetricTensor(t.structure, tuple(b + 0.5j * b[::-1] for b in t.blocks))
    out = to_symmetry(t, U1)
    assert out.dtype == np.complex128
    takes = [expected_take(leg)[0] for leg in t.legs]
    assert np.max(np.abs(out.to_dense() - gather(t.to_dense(), takes))) == 0.0
    u = tensor("su2_2w", seed=17)
    assert tenet.allclose(
        to_symmetry(tenet.tensordot(t, u, ((2,), (0,))), U1),
        tenet.tensordot(out, to_symmetry(u, U1), ((2,), (0,))),
    )


# --- refusals -------------------------------------------------------------------


class _U1Like:
    """A U(1)-shaped provider for the refusal tests; hashable by identity."""

    name = "Fake"

    @property
    def unit(self) -> U1Sector:
        return U1Sector(0)

    def dual(self, a):
        return U1Sector(-a.charge)

    def fusion(self, a, b):
        return (U1Sector(a.charge + b.charge),)

    def n_symbol(self, a, b, c):
        return int(c.charge == a.charge + b.charge)

    def qdim(self, a):
        return 1.0

    def branch(self, target, a):
        return (a,)


class _NoCGC(_U1Like):
    name = "NoCGC"


class _NoDualBasis(_U1Like):
    name = "NoDualBasis"

    def irrep_dim(self, a):
        return 1

    def cgc(self, a, b, c):
        return np.ones((1, 1, 1, 1))


class _BranchesTooBig(_NoDualBasis):
    name = "BranchesTooBig"

    def branch(self, target, a):
        return (ONE,)  # irrep_dim 3: no single label per basis vector


def _fake_tensor(provider, *, dual=False):
    space = GradedSpace.new(provider, {U1Sector(0): 2, U1Sector(1): 1})
    return SymmetricTensor.random((Leg(space, OUT, dual=dual), Leg(space, IN)), seed=0)


def test_a_provider_without_the_capability_is_refused():
    t = SymmetricTensor.random(
        (
            Leg(GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1}), OUT),
            Leg(GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1}), IN),
        ),
        seed=18,
    )
    with pytest.raises(
        CapabilityError, match="U1Provider does not provide capability BranchingRules"
    ):
        to_symmetry(t, U1)


def test_a_target_the_provider_cannot_reach_is_refused():
    with pytest.raises(CapabilityError, match="SU2: cannot branch to"):
        to_symmetry(tensor("su2_2", seed=19), Trivial)


def test_a_target_sector_with_irrep_dim_above_one_is_refused():
    with pytest.raises(CapabilityError, match="irrep_dim 3 > 1"):
        to_symmetry(_fake_tensor(_BranchesTooBig()), SU2)


def test_a_provider_without_clebsch_gordan_is_refused_by_to_dense():
    with pytest.raises(CapabilityError, match="does not provide capability ClebschGordanData"):
        to_symmetry(_fake_tensor(_NoCGC()), U1)


def test_a_dual_leg_without_dual_basis_is_refused_by_to_dense():
    with pytest.raises(CapabilityError, match="does not implement DualBasis"):
        to_symmetry(_fake_tensor(_NoDualBasis(), dual=True), U1)


# --- plumbing -------------------------------------------------------------------


def test_to_symmetry_owns_no_plan_and_no_cache_and_records_the_shortcut():
    import inspect
    import sys

    src = inspect.getsource(sys.modules["tenet.ops.cast"])
    assert "@cache" not in src and "lru_cache" not in src
    assert "class " not in src  # no plan object of its own; it reuses dense_plan's
    assert "Simplification: " in src


def test_dispatch_list_is_untouched():
    from tenet.array import dispatch

    assert "to_symmetry" not in getattr(dispatch, "__all__", [])


# --- backends -------------------------------------------------------------------


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  # registration is the import's side effect

    return jax


@pytest.mark.parametrize("name", ["su2_3", "su2_dual"])
def test_to_symmetry_on_jax_blocks_returns_jax_blocks_with_the_same_values(name):
    import autoray as ar

    use_jax()
    t = tensor(name, seed=20)
    out = to_symmetry(t.to_backend("jax"), U1)
    assert ar.infer_backend(out.blocks[0]) == "jax"
    np.testing.assert_allclose(
        np.asarray(out.to_dense()), to_symmetry(t, U1).to_dense(), atol=1e-13
    )


def test_to_symmetry_traces_only_with_atol_inf():
    """The boundary, both sides on one screen, exactly as #82 pins ``from_dense``."""
    jax = use_jax()
    t = tensor("su2_3", seed=21).to_backend("jax")

    @jax.jit
    def go(x):
        return to_symmetry(x, U1, atol=math.inf)

    out = go(t)
    np.testing.assert_allclose(
        np.asarray(out.to_dense()), to_symmetry(tensor("su2_3", seed=21), U1).to_dense(), atol=1e-13
    )

    @jax.jit
    def checked(x):
        return to_symmetry(x, U1)

    with pytest.raises(jax.errors.ConcretizationTypeError):
        checked(t)
