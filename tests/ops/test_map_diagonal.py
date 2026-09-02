"""The diagonal of a square map, in the reduced basis — M61 Stage A (#232, absorbing #231).

Issue #230 measured that the diagonal of the two-site effective Hamiltonian cannot be
manufactured by contracting per-leg diagonals of the operator's factors: exact on U(1),
sign-wrong on fZ2, and *shape*-wrong on SU(2), where one external sector tuple carries two
inner lines with unrelated entries. This file tests the operation that is right instead,
and it tests it the way that measurement defined the truth:

* the oracle is an **explicitly formed dense** ``H_eff`` — ``to_dense`` on the assembled
  map, then ``<e|H|e> / <e|e>`` on the dense image of each reduced-basis unit vector, a
  quantity that is basis-free and therefore not a restatement of the implementation;
* a second oracle probes the *public* map application, ``compose(m, e)``, and reads the
  same entry back, which is the definition a solver iterates on;
* the SU(2) two-inner-line case is **constructed**, at ``(j=1, 1/2, 1/2, 1)`` with both
  entries pinned, not a fixture that happens to produce multiplicity;
* the allocation claim is instrumented on ``autoray.do``'s outputs, with a positive
  control that trips it.
"""

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.map_view import assemble, map_layout
from tenet.symmetry import SU2, U1, FZ2Sector, SU2Sector, U1Sector, fZ2

# --- the four providers, each as an assembled two-site H_eff ---------------------------


def heff(bond, phys, x1, x2, x3, seed=0):
    """``GL W1 W2 GR`` contracted into ``(a p q r | a' p' q' r')``, the square partition.

    The MPO wires ``x``, ``m``, ``y`` are summed over inside ``einsum``, so the result is
    not a tensor product of per-leg operators — which is exactly what #230's candidate
    assumed it could be.
    """
    gl = SymmetricTensor.random((Leg(bond, OUT), Leg(x1, IN), Leg(bond, IN)), seed=seed)
    w1 = SymmetricTensor.random(
        (Leg(x1, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(x2, IN)), seed=seed + 1
    )
    w2 = SymmetricTensor.random(
        (Leg(x2, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(x3, IN)), seed=seed + 2
    )
    gr = SymmetricTensor.random((Leg(bond, OUT), Leg(x3, OUT), Leg(bond, IN)), seed=seed + 3)
    return tenet.einsum("axA,xpPm,mqQy,ryR->apqrAPQR", gl, w1, w2, gr)


QB = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(0): 2, U1Sector(1): 1})
QP = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
QX = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(0): 1, U1Sector(1): 1})

FB = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})
FP = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
# The spinful-Hubbard grading: d = 4, both parities twice (tests/network/test_hubbard.py).
HUB = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})

SB = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(1): 2, SU2Sector(2): 1})
SP = GradedSpace.new(SU2, {SU2Sector(1): 1})
SX = GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(2): 1})

CASES = {
    "u1": (QB, QP, QX, QX, QX),
    "fz2": (FB, FP, FP, FP, FP),
    "hubbard": (FB, HUB, FP, FP, FP),
    "su2": (SB, SP, SX, SX, SX),
}
PROVIDERS = list(CASES)


def unit_vectors(structure):
    """Every reduced-basis unit vector of ``structure``, with its ``(block, entry)`` slot."""
    zeros = [np.zeros(structure.block_shape(k)) for k in structure.block_order]
    for i, blk in enumerate(zeros):
        for j in range(blk.size):
            z = [np.zeros_like(b) for b in zeros]
            z[i].flat[j] = 1.0
            yield i, j, SymmetricTensor(structure, tuple(z))


def dense_reference(m, structure):
    """``<e|H|e> / <e|e>`` on the dense image of each unit vector: the formed-dense oracle."""
    dense = np.asarray(m.to_dense())
    n = int(np.prod(dense.shape[: len(structure.legs)]))
    mat = dense.reshape(n, n)
    out = [np.zeros(structure.block_shape(k)) for k in structure.block_order]
    for i, j, e in unit_vectors(structure):
        v = np.asarray(e.to_dense()).reshape(n)
        out[i].flat[j] = v @ mat @ v / (v @ v)
    return out


def probe_reference(m, structure):
    """``compose(m, e)`` read back at ``e``'s own slot: the matrix a solver iterates on."""
    out = [np.zeros(structure.block_shape(k)) for k in structure.block_order]
    for i, j, e in unit_vectors(structure):
        out[i].flat[j] = np.asarray(tenet.compose(m, e).blocks[i]).flat[j]
    return out


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_diagonal_agrees_with_a_formed_dense_heff(name):
    m = heff(*CASES[name])
    d = tenet.map_diagonal(m)
    assert d.structure == TensorStructure(tuple(m.codomain))
    assert d.structure.num_blocks > 0
    ref = dense_reference(m, d.structure)
    scale = max(float(np.abs(r).max(initial=0.0)) for r in ref)
    assert scale > 1e-6, "the fixture's diagonal is numerically empty; it tests nothing"
    for got, want in zip(d.blocks, ref, strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-12 * max(scale, 1.0))


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_diagonal_agrees_with_probing_the_public_map_application(name):
    m = heff(*CASES[name])
    d = tenet.map_diagonal(m)
    for got, want in zip(d.blocks, probe_reference(m, d.structure), strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-13)


# --- the SU(2) case #230 made the decisive one, constructed ----------------------------

ONE = GradedSpace.new(SU2, {SU2Sector(2): 1})  # j = 1
HALF = GradedSpace.new(SU2, {SU2Sector(1): 1})  # j = 1/2
KET = (Leg(ONE, OUT), Leg(HALF, OUT), Leg(HALF, OUT), Leg(ONE, OUT))


def test_su2_external_tuple_1_half_half_1_carries_two_inner_lines():
    """The premise, asserted before the values: multiplicity is exercised, not hoped for."""
    structure = TensorStructure(KET)
    assert structure.num_blocks == 2
    inner = sorted(tuple(s.two_j for s in k.output_tree.inner) for k in structure.block_order)
    assert inner == [(1, 2), (3, 2)]  # the two inner lines are j = 1/2 and j = 3/2
    assert len({tuple(structure.axis_sectors(k)) for k in structure.block_order}) == 1


def test_the_two_inner_lines_get_two_unrelated_diagonal_entries():
    """Both entries pinned, and both checked against the formed-dense oracle.

    One external sector tuple, two blocks, two values that are not equal and not related:
    the object #230 showed no leg-factorized contraction can even have the shape of.
    """
    m = heff(ONE, HALF, SX, SX, SX)
    d = tenet.map_diagonal(m)
    assert d.structure == TensorStructure(KET)
    entries = {
        tuple(s.two_j for s in k.output_tree.inner): float(np.asarray(b).ravel()[0])
        for k, b in zip(d.structure.block_order, d.blocks, strict=True)
    }
    assert entries[(1, 2)] == pytest.approx(0.2917652786409168, abs=1e-13)
    assert entries[(3, 2)] == pytest.approx(0.8465778161839884, abs=1e-13)
    assert abs(entries[(1, 2)] - entries[(3, 2)]) > 0.5

    for got, want in zip(d.blocks, dense_reference(m, d.structure), strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-13)


def test_a_random_square_map_on_the_same_legs_also_splits_the_two_lines():
    """Not an artefact of the assembled fixture: any square map on those legs splits."""
    bra = tuple(Leg(leg.space, IN, leg.dual, leg.name) for leg in KET)
    m = SymmetricTensor.random((*KET, *bra), seed=7)
    d = tenet.map_diagonal(m)
    values = [float(np.asarray(b).ravel()[0]) for b in d.blocks]
    assert values[0] == pytest.approx(-0.45467078517172255, abs=1e-13)
    assert values[1] == pytest.approx(-0.02925182246327349, abs=1e-13)
    for got, want in zip(d.blocks, dense_reference(m, d.structure), strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=1e-13)


# --- refusals --------------------------------------------------------------------------


def test_a_map_that_is_not_square_is_refused_naming_the_mismatch():
    legs = (Leg(QB, OUT), Leg(QP, OUT), Leg(QB, IN), Leg(QX, IN))
    with pytest.raises(ValueError, match="map_diagonal: the map is not square at position 1"):
        tenet.map_diagonal(SymmetricTensor.random(legs, seed=0))

    lopsided = (Leg(QB, OUT), Leg(QP, OUT), Leg(QB, IN))
    with pytest.raises(ValueError, match="map_diagonal: the map is not square"):
        tenet.map_diagonal(SymmetricTensor.random(lopsided, seed=0))

    with pytest.raises(ValueError, match="map_diagonal: the map is not square"):
        tenet.map_diagonal(SymmetricTensor.random((Leg(QB, OUT), Leg(QB, OUT)), seed=0))


def test_the_identity_map_has_an_all_ones_diagonal():
    """The one value that can be checked without any oracle at all."""
    for legs in ((Leg(QB, OUT), Leg(QP, OUT, dual=True)), (Leg(SB, OUT), Leg(SP, OUT))):
        d = tenet.map_diagonal(tenet.identity(legs))
        assert d.legs == legs
        for b in d.blocks:
            np.testing.assert_allclose(np.asarray(b), np.ones(b.shape))


# --- the allocation claim ---------------------------------------------------------------


def _recording_do(monkeypatch):
    """Record the element count of every array ``autoray.do`` hands back."""
    seen = []
    original = ar.do

    def spy(fn, *args, **kwargs):
        out = original(fn, *args, **kwargs)
        shape = getattr(out, "shape", None)
        if shape is not None:
            seen.append(int(np.prod(shape)) if len(shape) else 1)
        return out

    monkeypatch.setattr(ar, "do", spy)
    return seen


def _via_matrices(m):
    """The lazy implementation the instrument must be able to tell apart.

    Same numbers, reached by gathering the map into its coupled-sector matrices and
    reading the diagonal of one — i.e. by allocating the full width of the map.

    ``assemble``, not ``to_matrices``: the map already *holds* its coupled-sector
    matrices, so asking for them allocates nothing and would make a control that
    measures nothing. Re-gathering them from the blocks is the full-width allocation
    this route exists to avoid, spelled out.
    """
    structure = TensorStructure(tuple(m.codomain))
    layout = map_layout(m.structure)
    unit = m.structure.provider.unit
    mats = assemble(m.structure, m.blocks)
    flat = np.diagonal(np.asarray(mats[layout.sectors.index(unit)]))
    assert len(flat) == map_layout(m.structure).shape(unit)[0]
    blocks, off = [], 0
    for key in structure.block_order:
        shape = structure.block_shape(key)
        size = int(np.prod(shape))
        blocks.append(flat[off : off + size].reshape(shape))
        off += size
    return blocks


def test_no_full_width_intermediate_is_allocated(monkeypatch):
    """Instrumented on the allocation itself, with the full-width route as the control."""
    m = heff(*CASES["hubbard"])
    want = tenet.map_diagonal(m)
    # the budget is the *result's* own storage -- one coupled-sector matrix of the
    # diagonal -- not the map's, which is the full width this route exists to avoid
    budget = max(int(np.prod(mat.shape)) for mat in want.data)
    widest = max(int(np.prod(mat.shape)) for mat in m.data)
    assert widest > budget  # narrower is a real statement on this fixture

    seen = _recording_do(monkeypatch)
    got = tenet.map_diagonal(m)
    assert seen, "the instrument never fired: map_diagonal allocated nothing at all"
    assert max(seen) <= budget
    for a, b in zip(got.blocks, want.blocks, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    seen.clear()
    m.blocks  # noqa: B018  # cutting the blocks out is not the control's allocation
    control = _via_matrices(m)  # the positive control: one full-width matrix per sector
    assert max(seen) > budget
    for a, b in zip(control, want.blocks, strict=True):
        np.testing.assert_allclose(a, np.asarray(b))


def test_map_diagonal_never_densifies(monkeypatch):
    """``to_dense`` and ``to_matrices`` are both off the path, asserted by breaking them."""

    def refuse(*args, **kwargs):
        raise AssertionError("map_diagonal reached a dense or full-width route")

    m = heff(*CASES["su2"])
    monkeypatch.setattr(SymmetricTensor, "to_dense", refuse)
    # broken at its source, not at ``ops.map``'s re-export: ``compose`` reads ``data``
    # positionally now, so ``ops.map`` no longer names ``to_matrices`` at all
    monkeypatch.setattr(tenet.map_view, "to_matrices", refuse)
    assert tenet.map_diagonal(m).structure.num_blocks > 0


# --- the consumer -----------------------------------------------------------------------


@pytest.mark.parametrize("name", PROVIDERS)
def test_the_jacobi_quotient_the_two_primitives_exist_for(name):
    """``q / (lambda - diag)`` type-checks and is entrywise, on every provider."""
    m = heff(*CASES[name])
    d = tenet.map_diagonal(m)
    q = SymmetricTensor.random(d.legs, seed=11)
    out = tenet.zip_blocks(q, d, lambda x, y: x / (10.0 - y))
    assert out.structure == q.structure
    for got, x, y in zip(out.blocks, q.blocks, d.blocks, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(x) / (10.0 - np.asarray(y)))


@pytest.mark.parametrize("name", ["u1", "su2"])
def test_bending_the_solvers_vector_is_one_scalar_per_block(name):
    """The claim the docstring rests on: reaching another partition is a *diagonal* change.

    A two-site MPS tensor keeps its right bond IN, so a solver's vector lives one bend
    away from the all-OUT legs ``map_diagonal`` returns. If that bend were anything but a
    scalar per block, the diagonal — and the quotient above — would not survive it.
    """
    legs = tenet.map_diagonal(heff(*CASES[name])).legs
    a, b = (SymmetricTensor.random(legs, seed=s) for s in (3, 4))
    ba, bb = tenet.bend(a, a.ndim - 1), tenet.bend(b, b.ndim - 1)
    assert ba.structure == bb.structure
    for pa, pb, qa, qb in zip(ba.blocks, bb.blocks, a.blocks, b.blocks, strict=True):
        ratio_a = np.asarray(pa).ravel() / np.asarray(qa).ravel()
        ratio_b = np.asarray(pb).ravel() / np.asarray(qb).ravel()
        np.testing.assert_allclose(ratio_a, ratio_b)  # independent of the vector
        np.testing.assert_allclose(ratio_a, np.full_like(ratio_a, ratio_a[0]))  # one scalar
