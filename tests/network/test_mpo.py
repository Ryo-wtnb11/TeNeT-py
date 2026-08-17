"""``tenet.network.MPO``: the leg structure, the refusal, and the operator it builds."""

import dmrg as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

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
