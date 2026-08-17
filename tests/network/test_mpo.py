"""``tenet.network.MPO``: the leg structure, the refusal, and the operator it builds."""

import pathlib
import sys

import dmrg as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import MPO, local_op
from tenet.symmetry import SU2, U1, FZ2Sector, SU2Sector, U1Sector, fZ2


def _ops():
    """``(Sz, S+, S-)`` as :func:`local_op` rank-3 tensors on this example's physical leg."""
    _, sz, sp, sm = example._spin_half()
    return (
        local_op(sz, phys=example.PHYS, charge=U1Sector(0)),
        local_op(sp, phys=example.PHYS, charge=U1Sector(-2)),
        local_op(sm, phys=example.PHYS, charge=U1Sector(2)),
    )


def _heisenberg_terms(n_sites):
    op_sz, op_sp, op_sm = _ops()
    terms = []
    for i in range(n_sites - 1):
        terms.append((1.0, [(op_sz, i), (op_sz, i + 1)]))
        terms.append((0.5, [(op_sp, i), (op_sm, i + 1)]))
        terms.append((0.5, [(op_sm, i), (op_sp, i + 1)]))
    return terms


def _kron_oracle(n_sites, terms):
    """``sum_t coeff * prod_j op_j`` by explicit ``np.kron`` -- the #110 oracle pattern."""
    eye = np.eye(2)

    def at(op, site):
        out = np.array([[1.0]])
        for k in range(n_sites):
            out = np.kron(out, op if k == site else eye)
        return out

    total = np.zeros((2**n_sites, 2**n_sites))
    for coeff, ops in terms:
        acc = np.eye(2**n_sites)
        for op, site in ops:
            acc = acc @ at(np.asarray(op.to_dense())[:, :, 0], site)
        total = total + coeff * acc
    return total


def test_from_w_gives_every_site_the_same_rank_and_leg_pattern():
    """``(wl IN, p OUT, p IN, wr OUT)`` at every site, with ``D=1`` MPO bonds at the ends.

    The boundary vectors are spelled as ``D=1`` legs rather than as rank-3 end tensors,
    which is what removes the boundary special case from every contraction downstream.
    """
    h = MPO.from_w(
        example.mpo_array(),
        5,
        phys=example.PHYS,
        bond=example.MPO_BOND,
        boundary=example.BOUNDARY,
        start=example._START,
        end=example._END,
    )
    assert len(h) == 5
    for w in h:
        assert w.ndim == 4
        assert tuple(leg.side for leg in w.legs) == (IN, OUT, IN, OUT)
        assert w.legs[1].space == example.PHYS and w.legs[2].space == example.PHYS
    assert h[0].legs[0].space == example.BOUNDARY
    assert h[4].legs[3].space == example.BOUNDARY
    assert h[1].legs[0].space == example.MPO_BOND
    assert h[1].structure == h[2].structure  # one bulk tensor, shared


def test_from_w_refuses_a_perturbed_grading():
    """The refusal is the proof the grading is right; a passing ``allclose`` would not be.

    ``from_dense`` runs at its **default** relative ``atol`` (``ops/dense.py``:301), so
    moving the ``+2`` sector of the MPO bond to ``-2`` -- same total dimension -- makes
    construction *raise* rather than projecting onto a different Hamiltonian. Asserted
    here directly against ``MPO.from_w``, and end to end in
    ``tests/integration/test_dmrg.py``:146-157.
    """
    perturbed = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(-2): 2})
    assert perturbed.dim == example.MPO_BOND.dim
    with pytest.raises(ValueError, match="not symmetric"):
        MPO.from_w(
            example.mpo_array(),
            6,
            phys=example.PHYS,
            bond=perturbed,
            boundary=example.BOUNDARY,
            start=example._START,
            end=example._END,
        )


def test_the_mpo_is_the_heisenberg_hamiltonian_at_n4():
    """Contract to the dense 16 x 16 operator and compare against explicit ``kron``.

    ``tests/integration/test_dmrg.py``:116-143 does this at N=8; at N=4 it is a unit test
    of ``from_w``'s row/column slicing rather than of the physics.
    """
    n_sites = 4
    from_mpo = np.asarray(example.mpo(n_sites).to_dense())

    eye, sz, sp, sm = example._spin_half()

    def at(op, site):
        out = np.array([[1.0]])
        for k in range(n_sites):
            out = np.kron(out, op if k == site else eye)
        return out

    from_kron = sum(
        at(sz, i) @ at(sz, i + 1) + 0.5 * (at(sp, i) @ at(sm, i + 1) + at(sm, i) @ at(sp, i + 1))
        for i in range(n_sites - 1)
    )
    assert np.abs(from_mpo - from_kron).max() < 1e-12


def test_mpo_is_not_an_mps():
    """Two classes, no shape flag: an :class:`MPO` has no ``center`` and no write barrier.

    The claim is that nothing in ``tenet.network`` branches on ``ndim`` or an
    ``nr_phys``-style attribute to decide what it is operating on (YASTN
    ``_mps_obc.py``:284, :291, :438, :443, rejected).
    """
    h = example.mpo(4)
    assert not hasattr(h, "center")
    assert not hasattr(h, "canonize_")
    assert not hasattr(MPO, "__setitem__")


# --- M13: local_op and MPO.from_terms (#133) -----------------------------------------


@pytest.mark.parametrize(
    ("which", "flipped"), [("sz", 2), ("sp", 2), ("sm", -2)], ids=["Sz", "Sp", "Sm"]
)
def test_local_op_refuses_the_wrong_charge(which, flipped):
    """All three channels of the Heisenberg ``W``, each with the sign of its charge flipped.

    ``local_op`` builds through ``from_dense`` at its **default** relative ``atol``, so a
    declared charge that does not match the array *raises*. This is
    ``test_from_w_refuses_a_perturbed_grading``'s refusal moved one level down -- per
    operator instead of per Hamiltonian, which is a strictly sharper place for it.
    """
    _, sz, sp, sm = example._spin_half()
    dense = {"sz": sz, "sp": sp, "sm": sm}[which]
    with pytest.raises(ValueError, match="not symmetric"):
        local_op(dense, phys=example.PHYS, charge=U1Sector(flipped))


def test_local_op_round_trips_every_matrix_of_the_example():
    """``local_op(op).to_dense()[:, :, 0]`` is the input array; rank 3, ``D=1`` charge leg."""
    eye, sz, sp, sm = example._spin_half()
    for dense, charge in ((eye, 0), (sz, 0), (sp, -2), (sm, 2)):
        op = local_op(dense, phys=example.PHYS, charge=U1Sector(charge))
        assert op.ndim == 3
        assert tuple(leg.side for leg in op.legs) == (OUT, IN, OUT)
        assert op.legs[2].space.reduced_dim == 1
        assert np.abs(np.asarray(op.to_dense())[:, :, 0] - dense).max() < 1e-14


def test_local_op_refuses_an_array_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"expected a \(2, 2\) array"):
        local_op(np.eye(3), phys=example.PHYS, charge=U1Sector(0))


def test_from_terms_gives_every_site_the_same_rank_and_leg_pattern():
    """``from_w``'s structural claims, run against the derived builder."""
    h = MPO.from_terms(5, _heisenberg_terms(5))
    assert len(h) == 5
    for w in h:
        assert w.ndim == 4
        assert tuple(leg.side for leg in w.legs) == (IN, OUT, IN, OUT)
        assert w.legs[1].space == example.PHYS and w.legs[2].space == example.PHYS
    assert h[0].legs[0].space == example.BOUNDARY
    assert h[4].legs[3].space == example.BOUNDARY


def test_from_terms_matches_the_kron_oracle_at_n4():
    """The #110 dense oracle, against the term-list builder rather than against ``W``."""
    terms = _heisenberg_terms(4)
    from_mpo = np.asarray(MPO.from_terms(4, terms).to_dense())
    assert np.abs(from_mpo - _kron_oracle(4, terms)).max() < 1e-12


def test_from_terms_matches_the_kron_oracle_for_long_range_and_three_site_strings():
    """Non-uniform terms at N=6: the implicit-identity and k-site paths have no other test.

    A next-nearest-neighbour ``Sz_i Sz_{i+2}``, a single-site field ``Sz_3``, and a 3-site
    string ``Sz_0 Sz_2 Sz_5`` with a non-unit coefficient.
    """
    op_sz, op_sp, op_sm = _ops()
    terms = [(0.7, [(op_sz, i), (op_sz, i + 2)]) for i in range(4)]
    terms.append((-1.3, [(op_sz, 3)]))
    terms.append((2.5, [(op_sz, 0), (op_sz, 2), (op_sz, 5)]))
    terms.append((0.5, [(op_sp, 1), (op_sm, 4)]))
    from_mpo = np.asarray(MPO.from_terms(6, terms).to_dense())
    assert np.abs(from_mpo - _kron_oracle(6, terms)).max() < 1e-12


@pytest.mark.parametrize("n_sites", [4, 6])
def test_the_two_builders_agree_as_operators(n_sites):
    """A hand-derived grading and a derived one, cross-checked -- the point of #133.

    As **operators**, not tensor by tensor: the MPO bond is fixed only up to a gauge and
    the sector order inside it is ``GradedSpace``'s, not the ``_START``/``_END`` channel
    table's, so a per-site comparison would fail on a correct implementation.
    """
    from_terms = np.asarray(example.mpo_from_terms(n_sites).to_dense())
    assert np.abs(from_terms - np.asarray(example.mpo(n_sites).to_dense())).max() < 1e-12


@pytest.mark.parametrize("n_sites", [6, 12])
def test_from_terms_derives_the_hand_written_mpo_bond(n_sites):
    """Bulk bond dimension exactly 5, and the bond *space* equal to ``MPO_BOND``.

    ``M = 3 (N - 1)`` before compression -- 15 at N=6 and 33 at N=12 -- so the sweep is
    doing real work. The sharpest available form of "derived, not declared": the sectors
    and their degeneracies come out ``{0: 3, +2: 1, -2: 1}`` with nothing written down.
    The two bonds nearest the ends come out **4**, not 5, because only four channels are
    reachable there; the derived grading is tighter than the hand-written one.
    """
    h = MPO.from_terms(n_sites, _heisenberg_terms(n_sites))
    bulk = [h[n].legs[0].space for n in range(2, n_sites - 1)]
    assert [space.dim for space in bulk] == [5] * len(bulk)
    assert all(space == example.MPO_BOND for space in bulk)
    assert h[1].legs[0].space.dim == 4


def test_the_default_cutoff_is_lossless():
    """``cutoff=0.0`` and the default ``cutoff=1e-13`` give the same operator.

    If they did not, the default would be wrong and the tolerance would not be the thing
    to widen.
    """
    terms = _heisenberg_terms(6)
    tight = np.asarray(MPO.from_terms(6, terms, cutoff=0.0).to_dense())
    assert np.abs(tight - np.asarray(MPO.from_terms(6, terms).to_dense())).max() < 1e-12


def test_from_terms_refuses_an_empty_term_list():
    with pytest.raises(ValueError, match="no terms"):
        MPO.from_terms(4, [])


def test_from_terms_refuses_a_site_outside_the_chain():
    op_sz, _, _ = _ops()
    with pytest.raises(ValueError, match=r"outside range\(4\)"):
        MPO.from_terms(4, [(1.0, [(op_sz, 0), (op_sz, 4)])])


def test_from_terms_refuses_two_operators_on_one_site():
    op_sz, _, _ = _ops()
    with pytest.raises(ValueError, match="two operators of one term sit on site 2"):
        MPO.from_terms(4, [(1.0, [(op_sz, 2), (op_sz, 2)])])


def test_from_terms_refuses_operators_on_different_physical_spaces():
    op_sz, _, _ = _ops()
    other = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1, U1Sector(3): 1})
    wide = local_op(np.eye(3), phys=other, charge=U1Sector(0))
    with pytest.raises(ValueError, match="disagree about the physical space"):
        MPO.from_terms(4, [(1.0, [(op_sz, 0), (wide, 1)])])


def test_from_terms_refuses_a_rank_2_operator():
    """The mistake a user arriving from tenpy makes first: a bare matrix, no charge leg."""
    legs = (Leg(example.PHYS, OUT), Leg(example.PHYS, IN))
    bare = SymmetricTensor.from_dense(np.diag([-0.5, 0.5]), legs)
    with pytest.raises(ValueError, match="rank 3.*got rank 2"):
        MPO.from_terms(4, [(1.0, [(bare, 0)])])


def test_from_terms_refuses_a_wide_charge_leg():
    """The charge leg names *the* MPO bond the operator emits: degeneracy 1 or nothing."""
    charge = GradedSpace.new(U1, {U1Sector(0): 2})
    legs = (Leg(example.PHYS, OUT), Leg(example.PHYS, IN), Leg(charge, OUT))
    op = SymmetricTensor.random(legs, seed=0)
    with pytest.raises(ValueError, match="one sector at degeneracy 1"):
        MPO.from_terms(4, [(1.0, [(op, 0)])])


def test_from_terms_refuses_a_non_abelian_operator():
    """SU(2) coverage per REPOSITORY_RULES:61, on ``tests/network/test_mps.py``:22-23's rule.

    The container is provider-generic even though the Hamiltonian is not: a *list* of
    operators does not determine a non-Abelian term, because three spin-1 tensor operators
    fuse through three channels and the DSL has no slot for a coupling tree. The 2-site
    case is tractable and is the named follow-up; ``tenet.ops.fusion.fused_leg`` is the
    mechanism and "``fuse_spaces`` returned more than one sector" is the ambiguity to
    detect.

    #135 did **not** overturn that argument, it routed around the premise: a declared
    chain of charges is still ambiguous and this refusal still fires for it, and the route
    that works is next to it -- hand the whole term over as one invariant *k*-site
    operator (``local_op(dense, phys=...)`` with no ``charge``) on a tuple of sites, where
    the coupling lives in the operator's own blocks and the bond comes out of an SVD.
    """
    phys = GradedSpace.new(SU2, {SU2Sector(1): 1})
    charge = GradedSpace.new(SU2, {SU2Sector(2): 1})
    legs = (Leg(phys, OUT), Leg(phys, IN), Leg(charge, OUT))
    op = SymmetricTensor.random(legs, seed=0)
    with pytest.raises(ValueError, match="Abelian-only"):
        MPO.from_terms(3, [(1.0, [(op, 0), (op, 1)])])


def test_from_terms_refuses_fermionic_braiding():
    """fZ2 legs are refused, not silently accepted, and the message names the gap.

    Jordan-Wigner needs a swap gate between an odd MPO bond and a physical line -- a line
    crossing, not a permutation of one tensor's legs, so tenet's Koszul machinery does not
    supply it for free. ``Env``/``sweep_`` have never contracted an odd-parity MPO bond and
    contain no swap gate, so a fermionic MPO built here would be silently wrong rather than
    refused, which is worse than not shipping it.

    **The one experiment that decides the follow-up**, recorded so it need not be
    rediscovered: build ``c+_0 c_1`` on fZ2 legs, take ``MPO.to_dense()`` and compare
    against the dense Jordan-Wigner oracle. One test file, no ``src/`` change; if it
    passes, the fermionic scope is exactly the swap-gate question and nothing else.
    """
    phys = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
    charge = GradedSpace.new(fZ2, {FZ2Sector(1): 1})
    legs = (Leg(phys, OUT), Leg(phys, IN), Leg(charge, OUT))
    op = SymmetricTensor.random(legs, seed=0)
    with pytest.raises(ValueError, match="fermionic braiding"):
        MPO.from_terms(3, [(1.0, [(op, 0), (op, 1)])])


def test_from_terms_refuses_a_charged_term():
    """Both MPO boundaries are the trivial ``D=1`` leg, so a term must close on the unit."""
    _, op_sp, _ = _ops()
    with pytest.raises(ValueError, match="sum to the unit sector"):
        MPO.from_terms(4, [(1.0, [(op_sp, 0)])])


def test_mpo_to_dense_is_the_oracle_exit_both_builders_share():
    """One spelling, three callers -- ``MPS.to_dense``'s twin, ``D=1`` boundaries dropped."""
    dense = np.asarray(example.mpo(3).to_dense())
    assert dense.shape == (8, 8)
    assert np.abs(dense - dense.T).max() < 1e-12  # the Heisenberg chain is real symmetric


# --- M13b: the invariant k-site form, on SU(2) and SU(3) legs ------------------------

SU2_PHYS = GradedSpace.new(SU2, {SU2Sector(1): 1})


def _ss_dense():
    """``S.S`` for two spin-1/2s as a ``(2, 2, 2, 2)`` array indexed ``(o0, o1, i0, i1)``.

    ``np.kron(a, b)`` reshaped to ``(d,)*4`` already carries that layout, which is why the
    ``kron`` oracle this file has used since #110 is now also the *input* format. The
    example's ascending basis and SU(2)'s descending one give the same ``S.S``: the flip
    is a rotation, under which ``S^z -> -S^z`` and ``S^+ <-> S^-``, and both terms of
    ``S.S`` are even in that.
    """
    _, sz, sp, sm = example._spin_half()
    return np.kron(sz, sz) + (np.kron(sp, sm) + np.kron(sm, sp)) / 2


def su2_heisenberg(n_sites, **kwargs):
    """The whole SU(2) Hamiltonian: one invariant operator and a list comprehension."""
    ss = local_op(_ss_dense(), phys=SU2_PHYS)
    terms = [(1.0, [(ss, (i, i + 1))]) for i in range(n_sites - 1)]
    return MPO.from_terms(n_sites, terms, **kwargs)


def _heisenberg_kron(n_sites):
    eye, sz, sp, sm = example._spin_half()

    def at(op, site):
        out = np.array([[1.0]])
        for k in range(n_sites):
            out = np.kron(out, op if k == site else eye)
        return out

    return sum(
        at(sz, i) @ at(sz, i + 1) + 0.5 * (at(sp, i) @ at(sm, i + 1) + at(sm, i) @ at(sp, i + 1))
        for i in range(n_sites - 1)
    )


def test_local_op_cannot_express_a_symmetry_breaking_term_on_su2_legs():
    """``Sz (x) Sz`` alone raises; ``S.S`` builds. The consequence is the point.

    A term is a scalar under the symmetry, so ``from_dense`` at its default ``atol``
    (``ops/dense.py``:301) refuses any array that is not invariant -- which makes the DSL
    *incapable* of expressing a symmetry-breaking term on non-Abelian legs. That is #133's
    "a wrong grading is unreachable" promoted from the bond to the physics, and it is a
    sharper refusal than the ``irrep_dim > 1`` check it replaces for this path.
    """
    _, sz, _, _ = example._spin_half()
    with pytest.raises(ValueError, match="not symmetric"):
        local_op(np.kron(sz, sz), phys=SU2_PHYS)
    ss = local_op(_ss_dense(), phys=SU2_PHYS)
    assert ss.ndim == 4
    assert tuple(leg.side for leg in ss.legs) == (OUT, OUT, IN, IN)


def test_local_op_without_a_charge_round_trips_on_abelian_legs():
    """The k-site form is verified on U(1) too, where ``Sz (x) Sz`` *is* invariant.

    ``(d**k, d**k)`` and ``(d,)*2k`` are the same input; ``k`` is inferred from ``d``.
    """
    _, sz, _, _ = example._spin_half()
    want = np.reshape(np.kron(sz, sz), (2,) * 4)
    flat = local_op(np.kron(sz, sz), phys=example.PHYS)
    nested = local_op(want, phys=example.PHYS)
    assert flat.structure == nested.structure
    assert np.abs(np.asarray(flat.to_dense()) - want).max() < 1e-14


def test_local_op_refuses_a_shape_no_integer_k_follows_from():
    with pytest.raises(ValueError, match=r"got \(2, 3\), from which no integer k follows"):
        local_op(np.zeros((2, 3)), phys=SU2_PHYS)


def test_from_terms_refuses_a_site_tuple_of_the_wrong_length():
    ss = local_op(_ss_dense(), phys=SU2_PHYS)
    with pytest.raises(ValueError, match="spans 2 site"):
        MPO.from_terms(4, [(1.0, [(ss, 0)])])


def test_from_terms_refuses_an_mpo_site_tensor_as_a_term_operator():
    """The check that catches a user handing in a ``W``: rank 4, but not (OUT, OUT, IN, IN)."""
    w = su2_heisenberg(4)[1]
    with pytest.raises(ValueError, match="is not a term operator"):
        MPO.from_terms(4, [(1.0, [(w, (0, 1))])])


@pytest.mark.parametrize("n_sites", [4, 6])
def test_the_su2_heisenberg_mpo_is_the_heisenberg_hamiltonian(n_sites):
    """The criterion #110 said could not be got right by hand -- and nothing was.

    Same ``np.kron`` oracle as the U(1) case, because ``to_dense`` erases the symmetry.
    The recoupling in the ``W`` was never written down: it came out of ``svd_truncated``.
    """
    h = su2_heisenberg(n_sites)
    assert len(h) == n_sites
    for w in h:
        assert w.ndim == 4
        assert tuple(leg.side for leg in w.legs) == (IN, OUT, IN, OUT)
        assert w.legs[1].space == SU2_PHYS and w.legs[2].space == SU2_PHYS
    assert h[0].legs[0].space.dim == 1 and h[-1].legs[3].space.dim == 1
    assert np.abs(np.asarray(h.to_dense()) - _heisenberg_kron(n_sites)).max() < 1e-12


def test_the_su2_bond_spaces_are_derived_and_are_the_predicted_ones():
    """Asserted as ``GradedSpace`` equality, not as a dimension. #135's three predictions.

    One ``S.S`` splits on a single adjoint multiplet, and the compressed bulk bond is
    ``{0: 2, 2: 1}`` -- **3 blocks**, ``reduced_dim`` 3, ``dim`` 5 -- against U(1)'s
    ``MPO_BOND`` at 5 blocks and ``dim`` 5. The block count is MPSKit's finite Jordan form
    ``(1 C D; . A B; . . 1)`` with no on-site ``D``: unit (+) adjoint (+) unit. The MPO's
    *dense* bond is 5 either way; the multiplet compression is on the MPS side, and
    ``tests/network/test_dmrg.py`` is where that is measured.
    """
    ss = local_op(_ss_dense(), phys=SU2_PHYS)
    one_term = MPO.from_terms(2, [(1.0, [(ss, (0, 1))])])
    assert one_term[0].legs[3].space == GradedSpace.new(SU2, {SU2Sector(2): 1})

    bulk = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(2): 1})
    assert (bulk.reduced_dim, bulk.dim) == (3, 5)
    for n_sites in (6, 12):
        h = su2_heisenberg(n_sites)
        assert all(h[n].legs[0].space == bulk for n in range(2, n_sites - 1))


def test_the_su2_default_cutoff_is_lossless():
    tight = np.asarray(su2_heisenberg(6, cutoff=0.0).to_dense())
    assert np.abs(tight - np.asarray(su2_heisenberg(6).to_dense())).max() < 1e-12


def test_a_non_adjacent_su2_term_carries_its_graded_bond_through_the_spectators():
    """``S.S`` on sites ``(0, 3)``: the adjoint bond runs through the identities between.

    The spectator sites are ``identity(aux) (x) identity(phys)`` and ``aux`` is graded
    here, which the old ``np.eye`` spectator could not have been.
    """
    ss = local_op(_ss_dense(), phys=SU2_PHYS)
    h = MPO.from_terms(4, [(1.0, [(ss, (0, 3))])])
    eye, sz, sp, sm = example._spin_half()

    def at(op, site):
        out = np.array([[1.0]])
        for k in range(4):
            out = np.kron(out, op if k == site else eye)
        return out

    oracle = at(sz, 0) @ at(sz, 3) + 0.5 * (at(sp, 0) @ at(sm, 3) + at(sm, 0) @ at(sp, 3))
    assert np.abs(np.asarray(h.to_dense()) - oracle).max() < 1e-12


def test_casting_the_su2_sites_to_u1_gives_the_same_operator():
    """``branch``'s basis ordering against ``from_dense``'s, at MPO scale.

    **#135 predicted this as a per-tensor identity and that prediction was wrong**:
    :func:`tenet.cast` *reorders the dense basis* by construction (``ops/cast.py``:69-77
    gathers each axis into the target's sector order), so ``cast(w, U1).to_dense()``
    differs from ``w.to_dense()`` by a per-leg permutation even when everything is right.
    Contracting the whole chain is the honest form of the same check: the bond gathers
    cancel between neighbours, the physical one is the spin flip that leaves ``H``
    invariant, and what is left is exactly the statement that ``branch``'s basis order and
    ``from_dense``'s agree.

    Still not a cross-builder comparison: an MPO bond is fixed only up to a gauge and the
    sector order inside it is ``GradedSpace``'s, so comparing ``cast(SU2_W, U1)`` against
    the hand-graded ``W`` of ``examples/dmrg.py`` per site would fail on a *correct*
    implementation -- which is why ``MPO.cast`` does not exist. ``MPO`` over a list
    comprehension is all the container-level cast anyone needed.
    """
    h = su2_heisenberg(6)
    cast = MPO([tenet.cast(w, U1) for w in h])
    assert cast[3].legs[1].space == example.PHYS
    assert np.abs(np.asarray(cast.to_dense()) - _heisenberg_kron(6)).max() < 1e-12


def test_an_su3_two_site_term_derives_a_bond_with_multiplicity_two():
    """The SU(3) probe #135 handed to the implementation: it passes, so nothing is refused.

    The swap operator on two adjoint (``8``) sites is invariant, and its split bond comes
    back carrying ``8`` at **degeneracy 2** -- the ``N^8_88 = 2`` multiplicity arriving as
    an ordinary ``GradedSpace`` degeneracy out of ``svd_truncated``, with no ``mu`` slot
    anywhere in the DSL. That is the issue's central design claim, tested rather than
    asserted.

    Two sites, not three: ``tests/symmetry/_su3_fixture.py`` is a *truncated* sector set
    (eleven sectors up to ``27``), and three adjoint sites need ``27 (x) 8`` channels it
    does not vendor -- ``from_dense`` refuses even the plain identity there. That is the
    fixture's documented boundary, not a limit of this path.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "symmetry"))
    from _su3_fixture import EIGHT, SU3  # noqa: PLC0415

    d = SU3.irrep_dim(EIGHT)
    swap = np.einsum("ad,bc->abcd", np.eye(d), np.eye(d))
    op = local_op(swap, phys=GradedSpace.new(SU3, {EIGHT: 1}))
    h = MPO.from_terms(2, [(1.0, [(op, (0, 1))])])
    assert h[0].legs[3].space.degeneracy(EIGHT) == 2
    assert np.abs(np.asarray(h.to_dense()) - np.reshape(swap, (d * d, d * d))).max() < 1e-12


# --- M15: the symbolic assembly (#138) -----------------------------------------------


def test_the_su3_multiplicity_two_bond_stands_without_the_sweep():
    """The SU(3) probe of the test above, under ``cutoff=None``.

    The ``N^8_88 = 2`` multiplicity is carried by the state's own space — the ``Leg``
    the operator's internal SVD derived — so skipping the compressing sweeps changes
    nothing about it.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "symmetry"))
    from _su3_fixture import EIGHT, SU3  # noqa: PLC0415

    d = SU3.irrep_dim(EIGHT)
    swap = np.einsum("ad,bc->abcd", np.eye(d), np.eye(d))
    op = local_op(swap, phys=GradedSpace.new(SU3, {EIGHT: 1}))
    h = MPO.from_terms(2, [(1.0, [(op, (0, 1))])], cutoff=None)
    assert h[0].legs[3].space.degeneracy(EIGHT) == 2
    assert np.abs(np.asarray(h.to_dense()) - np.reshape(swap, (d * d, d * d))).max() < 1e-12


def _ising_terms(n_sites, r=None):
    """The truncated ``1/r^2`` Ising sum ``Σ_{i<j, j-i<=r} (j-i)^-2 Sz_i Sz_j``."""
    op_sz, _, _ = _ops()
    return [
        ((j - i) ** -2.0, [(op_sz, i), (op_sz, j)])
        for i in range(n_sites)
        for j in range(i + 1, n_sites)
        if r is None or j - i <= r
    ]


def _widest_bond(h):
    return max(w.legs[3].space.dim for w in h)


def test_cutoff_none_derives_the_heisenberg_bond_with_no_svd_anywhere():
    """The finite-state machine *is* the MPO: bulk bond ``MPO_BOND``, end bonds pruned to 4.

    ``cutoff=None`` skips both compressing sweeps, so the ``{0: 3, +2: 1, -2: 1}`` bulk
    bond and the dim-4 bonds nearest the ends are properties of the graph — reachable ∩
    co-reachable states, each carrying its own space — not of any SVD. This is
    ``test_from_terms_derives_the_hand_written_mpo_bond`` with the compression removed
    from the explanation.
    """
    n_sites = 12
    h = MPO.from_terms(n_sites, _heisenberg_terms(n_sites), cutoff=None)
    bulk = [h[n].legs[0].space for n in range(2, n_sites - 1)]
    assert all(space == example.MPO_BOND for space in bulk)
    assert h[1].legs[0].space.dim == 4
    assert h[0].legs[0].space == example.BOUNDARY
    assert h[-1].legs[3].space == example.BOUNDARY


def test_cutoff_none_and_the_default_agree_as_operators():
    """The uncompressed FSM and the swept MPO are the same operator, at N=6.

    On the Heisenberg set and on the long-range/3-site set of the ``kron`` oracle test —
    both branches of the ``cutoff`` keyword, one Hamiltonian each way.
    """
    op_sz, op_sp, op_sm = _ops()
    long_range = [(0.7, [(op_sz, i), (op_sz, i + 2)]) for i in range(4)]
    long_range.append((-1.3, [(op_sz, 3)]))
    long_range.append((2.5, [(op_sz, 0), (op_sz, 2), (op_sz, 5)]))
    long_range.append((0.5, [(op_sp, 1), (op_sm, 4)]))
    for terms in (_heisenberg_terms(6), long_range):
        swept = np.asarray(MPO.from_terms(6, terms).to_dense())
        exact = np.asarray(MPO.from_terms(6, terms, cutoff=None).to_dense())
        assert np.abs(swept - exact).max() < 1e-12


def test_finite_range_needs_no_compression():
    """R=4 ``1/r^2`` Ising at N=32: the FSM bond equals the compressed bond, both 6.

    One state per open coupling makes the pre-compression bond 6 where the old assembly
    handed the sweep an M-wide channel bond (118 here, one per term); at finite range the
    sweep then finds nothing left to compress. All-pairs is the counter-case: the FSM
    bond stays combinatorial (≤ 40 against M=496) and only the SVD sees the numerical
    low rank of the power law.
    """
    exact = MPO.from_terms(32, _ising_terms(32, 4), cutoff=None)
    swept = MPO.from_terms(32, _ising_terms(32, 4))
    assert _widest_bond(exact) == _widest_bond(swept) == 6
    assert _widest_bond(MPO.from_terms(32, _ising_terms(32), cutoff=None)) <= 40


def test_a_mixed_form_term_set_builds_exactly_and_matches_the_kron_oracle():
    """Rank-3 charged, non-adjacent rank-4 invariant, and a 3-site string, in one MPO.

    Built under ``cutoff=None`` and by default: the two operator forms coexist on one
    bond because each FSM state carries its own space, whatever derived it. The k-site
    operator's *internal* SVD still runs under ``cutoff=None`` — it decides the
    operator's own factorization, not the assembly.
    """
    _, sz, _, _ = example._spin_half()
    op_sz, op_sp, op_sm = _ops()
    op_zz = local_op(np.kron(sz, sz), phys=example.PHYS)
    terms = [
        (0.5, [(op_sp, 1), (op_sm, 4)]),
        (0.7, [(op_zz, (0, 3))]),
        (2.5, [(op_sz, 0), (op_sz, 2), (op_sz, 5)]),
        (-1.3, [(op_sz, 3)]),
    ]
    oracle = _kron_oracle(6, [t for t in terms if t[1][0][0].ndim == 3])
    eye, *_ = example._spin_half()

    def at(op, site):
        out = np.array([[1.0]])
        for k in range(6):
            out = np.kron(out, op if k == site else eye)
        return out

    oracle = oracle + 0.7 * at(sz, 0) @ at(sz, 3)
    for cutoff in ({}, {"cutoff": None}):
        h = MPO.from_terms(6, terms, **cutoff)
        assert np.abs(np.asarray(h.to_dense()) - oracle).max() < 1e-12


def test_the_su2_bond_keeps_its_three_blocks_without_the_sweep():
    """``{0: 2, 2: 1}`` — unit ⊕ adjoint ⊕ unit — is the graph's bond, not the SVD's.

    The mid-split state's space is the graded, multiplicity-carrying ``Leg`` that
    :func:`tenet.linalg.svd_truncated` derived inside the operator, embedded whole.
    """
    bulk = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(2): 1})
    h = su2_heisenberg(12, cutoff=None)
    assert all(h[n].legs[0].space == bulk for n in range(2, 11))
    assert (
        np.abs(np.asarray(su2_heisenberg(6, cutoff=None).to_dense()) - _heisenberg_kron(6)).max()
        < 1e-12
    )


def test_a_state_space_that_disagrees_with_its_bond_slot_raises_inside_from_dense():
    """The embedding isometries are built at ``from_dense``'s default ``atol``.

    A hand-built inconsistency — a state claiming sector ``-2`` placed into the slot the
    bond keeps for ``+2`` — puts the embedding's ones into a symmetry-forbidden cell, so
    construction *raises* instead of projecting the state away. Production code derives
    the slots from the states, so this is the refusal that proves the derivation right.
    """
    from tenet.network import mps as _mps  # noqa: PLC0415

    bond = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(2): 1})
    state = GradedSpace.new(U1, {U1Sector(-2): 1})
    starts = {U1Sector(-2): bond.sector_offset(U1Sector(2))}
    with pytest.raises(ValueError, match="not symmetric"):
        _mps._embedding(bond, state, starts, left=True, dual=False)
