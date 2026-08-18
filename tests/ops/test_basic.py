"""Tests for linear arithmetic, conj and norm — issue #20."""

import dataclasses
import pathlib

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, U1, CapabilityError, SU2Sector, Trivial, TrivialSector, U1Sector

HALF, ONE = SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 4, ONE: 3})
W = GradedSpace.new(SU2, {HALF: 2})
TRIV = GradedSpace.new(Trivial, {TrivialSector(): 3})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})

SU2_LEGS = (Leg(V, OUT), Leg(W, IN), Leg(V, OUT))
U1_LEGS = (Leg(Q, OUT), Leg(Q, IN), Leg(Q, OUT))
TRIV_LEGS = (Leg(TRIV, OUT), Leg(TRIV, IN))
# two IN legs, so the coupled sector is not pinned by a single leg: coupled
# spans several irreps of differing qdim, which is what makes the weight visible
NORM_LEGS = (Leg(V, OUT), Leg(W, IN), Leg(V, OUT), Leg(W, IN))
BLOCK_FREE_LEGS = (
    Leg(GradedSpace.new(SU2, {HALF: 2}), OUT),
    Leg(GradedSpace.new(SU2, {ONE: 2}), IN),
)


def rand_complex(legs, seed):
    """A random complex tensor: real draw + i * independent real draw."""
    return (
        SymmetricTensor.random(legs, seed=seed) + SymmetricTensor.random(legs, seed=seed + 1) * 1j
    )


def use_jax():
    jax = pytest.importorskip("jax")
    # Simplification: global x64 so float64 survives the move and the 1e-12 tolerances
    # in these tests mean something. Drop it if a test ever needs f32 semantics.
    jax.config.update("jax_enable_x64", True)
    return jax


@dataclasses.dataclass(frozen=True, slots=True)
class QdimOnly:
    """Fusion + quantum dimension, but no ClebschGordanData (no dense expansion)."""

    name: str = "QdimOnly"

    @property
    def unit(self):
        return TrivialSector()

    def dual(self, a):
        return a

    def fusion(self, a, b):
        return (TrivialSector(),)

    def n_symbol(self, a, b, c):
        return 1

    def qdim(self, a):
        return 2.5


@dataclasses.dataclass(frozen=True, slots=True)
class Bare:
    """Neither QuantumDimensionData nor ClebschGordanData."""

    name: str = "Bare"

    @property
    def unit(self):
        return TrivialSector()

    def dual(self, a):
        return a

    def fusion(self, a, b):
        return (TrivialSector(),)

    def n_symbol(self, a, b, c):
        return 1


# --- shape of the results -------------------------------------------------------


@pytest.mark.parametrize("legs", [TRIV_LEGS, U1_LEGS, SU2_LEGS])
def test_results_keep_structure_and_block_count(legs):
    a = SymmetricTensor.random(legs, seed=0)
    b = SymmetricTensor.random(legs, seed=1)
    for t in (a + b, a - b, -a, 2.0 * a, a * 2.0, a / 2.0, a.conj()):
        assert isinstance(t, SymmetricTensor)
        assert t.structure is a.structure
        assert isinstance(t.blocks, tuple)
        assert len(t.blocks) == len(a.blocks)


def test_arithmetic_values_are_the_obvious_ones():
    a = SymmetricTensor.random(SU2_LEGS, seed=2)
    b = SymmetricTensor.random(SU2_LEGS, seed=3)
    for got, want in (
        (a + b, [x + y for x, y in zip(a.blocks, b.blocks, strict=True)]),
        (a - b, [x - y for x, y in zip(a.blocks, b.blocks, strict=True)]),
        (-a, [-x for x in a.blocks]),
        (3.0 * a, [3.0 * x for x in a.blocks]),
        (a / 4.0, [x / 4.0 for x in a.blocks]),
    ):
        for x, y in zip(got.blocks, want, strict=True):
            np.testing.assert_allclose(x, y)


def test_functions_and_dunders_agree():
    a = SymmetricTensor.random(SU2_LEGS, seed=4)
    b = SymmetricTensor.random(SU2_LEGS, seed=5)
    assert tenet.add(a, b) == a + b
    assert tenet.subtract(a, b) == a - b
    assert tenet.negative(a) == -a
    assert tenet.multiply(a, 2.0) == a * 2.0
    assert tenet.divide(a, 2.0) == a / 2.0
    assert tenet.conj(a) == a.conj()
    assert tenet.norm(a) == a.norm()


# --- structure mismatches fail loudly -------------------------------------------


@pytest.mark.parametrize(
    "other_legs",
    [
        pytest.param(
            (Leg(GradedSpace.new(SU2, {HALF: 4}), OUT), Leg(W, IN), Leg(V, OUT)), id="sectors"
        ),
        pytest.param(
            (Leg(GradedSpace.new(SU2, {HALF: 5, ONE: 3}), OUT), Leg(W, IN), Leg(V, OUT)),
            id="degeneracy",
        ),
        pytest.param((Leg(V, IN), Leg(W, IN), Leg(V, OUT)), id="side"),
        pytest.param((Leg(V, OUT, dual=True), Leg(W, IN), Leg(V, OUT)), id="dual"),
        pytest.param((Leg(V, OUT, name="x"), Leg(W, IN), Leg(V, OUT)), id="name"),
    ],
)
def test_add_rejects_differing_legs_naming_the_axis(other_legs):
    a = SymmetricTensor.random(SU2_LEGS, seed=6)
    b = SymmetricTensor.zeros(other_legs)
    with pytest.raises(ValueError, match="axis 0"):
        _ = a + b
    with pytest.raises(ValueError, match="axis 0"):
        _ = a - b


def test_add_error_message_shows_both_legs():
    a = SymmetricTensor.zeros(SU2_LEGS)
    b = SymmetricTensor.zeros((Leg(V, OUT), Leg(W, IN), Leg(W, OUT)))
    with pytest.raises(ValueError) as e:
        _ = a + b
    assert "axis 2" in str(e.value)
    assert repr(a.legs[2]) in str(e.value) and repr(b.legs[2]) in str(e.value)


def test_add_across_providers_names_both_providers():
    a = SymmetricTensor.zeros(SU2_LEGS)
    b = SymmetricTensor.zeros(U1_LEGS)
    with pytest.raises(ValueError, match="SU2.*U1"):
        _ = a + b


def test_add_with_different_ndim():
    a = SymmetricTensor.zeros(SU2_LEGS)
    b = SymmetricTensor.zeros((Leg(V, OUT), Leg(W, IN)))
    with pytest.raises(ValueError, match="ndim"):
        _ = a + b


def test_tensor_times_tensor_points_at_tensordot():
    a = SymmetricTensor.random(SU2_LEGS, seed=7)
    with pytest.raises(TypeError, match="tensordot"):
        _ = a * a
    with pytest.raises(TypeError, match="tensordot"):
        _ = a / a


def test_scalar_addition_is_a_type_error():
    a = SymmetricTensor.random(SU2_LEGS, seed=8)
    with pytest.raises(TypeError):
        _ = a + 1.0
    with pytest.raises(TypeError):
        _ = a - 1.0
    with pytest.raises(TypeError):
        _ = a * "two"


def test_zero_d_array_counts_as_a_scalar():
    a = SymmetricTensor.random(SU2_LEGS, seed=9)
    assert tenet.allclose(a * np.float64(2.0), a * 2.0)
    assert tenet.allclose(a * np.array(2.0), a * 2.0)


# --- dense oracle ---------------------------------------------------------------


def test_su2_linearity_against_the_dense_oracle():
    a = SymmetricTensor.random(SU2_LEGS, seed=10)
    b = SymmetricTensor.random(SU2_LEGS, seed=11)
    got = (2.5 * a + b * (-0.75)).to_dense()
    want = 2.5 * a.to_dense() - 0.75 * b.to_dense()
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_su2_negation_and_subtraction_against_the_dense_oracle():
    a = SymmetricTensor.random(SU2_LEGS, seed=12)
    b = SymmetricTensor.random(SU2_LEGS, seed=13)
    np.testing.assert_allclose((-a).to_dense(), -a.to_dense(), atol=1e-12)
    np.testing.assert_allclose((a - b).to_dense(), a.to_dense() - b.to_dense(), atol=1e-12)


def test_su2_conjugation_against_the_dense_oracle():
    """Blockwise conj == dense conj, which pins the real-CG (Condon-Shortley) gauge."""
    a = rand_complex(SU2_LEGS, seed=14)
    assert np.iscomplexobj(a.blocks[0])
    np.testing.assert_allclose(a.conj().to_dense(), a.to_dense().conj(), atol=1e-12)


def test_conj_leaves_legs_untouched_and_is_an_involution():
    a = rand_complex(SU2_LEGS, seed=16)
    assert a.conj().legs == a.legs
    for x, y in zip(a.conj().legs, a.legs, strict=True):
        assert (x.side, x.dual, x.name) == (y.side, y.dual, y.name)
    assert a.conj().conj() == a


def test_conj_on_real_blocks_does_not_upcast():
    a = SymmetricTensor.random(SU2_LEGS, seed=18)
    c = a.conj()
    assert c.dtype == a.dtype == np.float64
    assert c == a


# --- norm -----------------------------------------------------------------------


def test_su2_norm_identity_and_the_necessity_of_the_qdim_weight():
    a = SymmetricTensor.random(NORM_LEGS, seed=20)
    coupled = {k.coupled for k in a.structure.block_order}
    assert len({SU2.qdim(c) for c in coupled}) > 1  # several qdims in play

    assert tenet.norm(a) == pytest.approx(np.linalg.norm(a.to_dense()), abs=1e-10)

    unweighted = np.sqrt(sum(float(np.sum(np.abs(b) ** 2)) for b in a.blocks))
    assert abs(tenet.norm(a) - unweighted) > 0.1 * tenet.norm(a)


@pytest.mark.parametrize("legs", [TRIV_LEGS, U1_LEGS])
def test_abelian_norm_is_the_plain_frobenius_norm(legs):
    a = SymmetricTensor.random(legs, seed=21)
    unweighted = np.sqrt(sum(float(np.sum(np.abs(b) ** 2)) for b in a.blocks))
    assert tenet.norm(a) == pytest.approx(unweighted, rel=0, abs=1e-14)
    assert tenet.norm(a) == pytest.approx(np.linalg.norm(a.to_dense()), abs=1e-10)


def test_norm_needs_only_quantum_dimension():
    space = GradedSpace.new(QdimOnly(), {TrivialSector(): 2})
    a = SymmetricTensor.random((Leg(space, OUT), Leg(space, IN)), seed=22)
    unweighted = np.sqrt(sum(float(np.sum(np.abs(b) ** 2)) for b in a.blocks))
    assert tenet.norm(a) == pytest.approx(np.sqrt(2.5) * unweighted)
    with pytest.raises(CapabilityError):
        _ = a.shape  # no ClebschGordanData, so no dense expansion at all


def test_norm_without_quantum_dimension_raises():
    space = GradedSpace.new(Bare(), {TrivialSector(): 2})
    a = SymmetricTensor.zeros((Leg(space, OUT), Leg(space, IN)))
    with pytest.raises(CapabilityError, match="QuantumDimensionData"):
        _ = tenet.norm(a)


def test_norm_is_homogeneous_and_subadditive():
    a = SymmetricTensor.random(SU2_LEGS, seed=23)
    b = SymmetricTensor.random(SU2_LEGS, seed=24)
    for s in (-3.0, 0.5, 2.0 - 1.5j):
        assert tenet.norm(a * s) == pytest.approx(abs(s) * tenet.norm(a), abs=1e-12)
    assert tenet.norm(a + b) <= tenet.norm(a) + tenet.norm(b) + 1e-12


def test_norm_of_a_block_free_tensor_is_zero():
    t = SymmetricTensor.zeros(BLOCK_FREE_LEGS)
    assert t.blocks == ()
    assert tenet.norm(t) == 0.0


# --- allclose -------------------------------------------------------------------


def test_allclose_tolerance_and_exactness():
    a = SymmetricTensor.random(SU2_LEGS, seed=25)
    b = SymmetricTensor.random(SU2_LEGS, seed=26)
    perturbed = a + b * 1e-12
    assert tenet.allclose(a, perturbed)
    assert not tenet.allclose(a, perturbed, rtol=0, atol=0)
    assert tenet.allclose(a, a, rtol=0, atol=0)


def test_allclose_on_different_structures_is_false_not_an_error():
    a = SymmetricTensor.zeros(SU2_LEGS)
    assert tenet.allclose(a, SymmetricTensor.zeros(U1_LEGS)) is False
    assert tenet.allclose(a, SymmetricTensor.zeros(TRIV_LEGS)) is False


# --- backend agnosticism --------------------------------------------------------


def test_jax_arithmetic_matches_numpy():
    use_jax()
    a = SymmetricTensor.random(SU2_LEGS, seed=27)
    b = SymmetricTensor.random(SU2_LEGS, seed=28)
    ja, jb = a.to_backend("jax"), b.to_backend("jax")
    for got, want in (
        (ja + jb, a + b),
        (ja - jb, a - b),
        (-ja, -a),
        (2.0 * ja, 2.0 * a),
        (ja * 2.0, a * 2.0),
        (ja / 2.0, a / 2.0),
    ):
        assert got.backend == "jax"
        assert got.structure == want.structure
        assert tenet.allclose(got.to_backend("numpy"), want, rtol=0, atol=1e-12)


def test_jax_conj_matches_numpy():
    use_jax()
    a = rand_complex(SU2_LEGS, seed=29)
    ja = a.to_backend("jax").conj()
    assert ja.backend == "jax"
    assert ja.legs == a.legs
    assert tenet.allclose(ja.to_backend("numpy"), a.conj(), rtol=0, atol=1e-12)


def test_jax_norm_matches_numpy():
    use_jax()
    a = SymmetricTensor.random(SU2_LEGS, seed=31)
    j = a.to_backend("jax")
    # #41 dropped norm's float(): a JAX tensor gives a 0-d jax.Array (traceable),
    # not a Python float. float(v) == v still holds, which is all callers needed.
    v = tenet.norm(j)
    assert getattr(v, "ndim", 0) == 0 and float(v) == v
    assert float(v) == pytest.approx(float(tenet.norm(a)), abs=1e-12)


def test_jax_allclose_and_structure_errors():
    use_jax()
    a = SymmetricTensor.random(SU2_LEGS, seed=32).to_backend("jax")
    assert tenet.allclose(a, a)
    with pytest.raises(ValueError, match="axis 0"):
        _ = a + SymmetricTensor.zeros((Leg(W, OUT), Leg(W, IN), Leg(V, OUT))).to_backend("jax")


# --- no NumPy in ops ------------------------------------------------------------


def test_ops_basic_contains_no_numpy_call():
    src = pathlib.Path(tenet.ops.basic.__file__).read_text()
    assert "np." not in src
    assert "import numpy" not in src
