"""The M24a acceptance exercise (#158): the capability lattice against Fibonacci.

Every check here is one the pre-decomposition design would have failed —
crashed obscurely (``AttributeError`` from a missing ``cgc``), answered wrongly
(a silently mis-braided ``transpose``), or been unable to express (a provider
with no dense expansion at all).
"""

import numpy as np
import pytest
from _fibonacci_fixture import ONE, PHI, TAU, Fib

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import (
    CapabilityError,
    ClebschGordanData,
    DualBasis,
    QuantumDimensionData,
    RecouplingData,
    TwistData,
    supports,
)
from tenet.symmetry.coherence import (
    properties,
    validate_hexagon,
    validate_non_degenerate_braiding,
    validate_pentagon,
    validate_snake,
    validate_spherical,
)

SECTORS = (ONE, TAU)


def tau_tau() -> SymmetricTensor:
    v = GradedSpace.new(Fib, {ONE: 1, TAU: 1})
    return SymmetricTensor.random((Leg(v, OUT), Leg(v, IN)), seed=7)


# --- the capability profile: what Fibonacci has and pointedly lacks -----------


def test_capability_profile():
    assert supports(Fib, RecouplingData)
    assert supports(Fib, QuantumDimensionData)
    assert supports(Fib, TwistData)
    assert not supports(Fib, ClebschGordanData)
    assert not supports(Fib, DualBasis)


# --- coherence: the data is a category, not a lookup table --------------------


def test_pentagon():
    assert validate_pentagon(Fib, SECTORS) > 0


def test_hexagon():
    assert validate_hexagon(Fib, SECTORS) > 0


def test_snake():
    assert validate_snake(Fib, SECTORS) > 0


def test_spherical():
    assert validate_spherical(Fib, SECTORS) == 2


def test_non_degenerate():
    assert validate_non_degenerate_braiding(Fib, SECTORS) == 2


def test_properties_not_symmetric_but_modular():
    p = properties(Fib)
    assert p.braided
    assert not p.symmetric  # chiral: R != R**-1
    assert p.spherical
    assert p.modular
    assert p.unitary


# --- the tensor exercise: build, bend, trace ----------------------------------


def test_bend_with_irrational_qdim_round_trips():
    t = tau_tau()
    bent = tenet.repartition(t, (0, 1), ())  # bends the IN leg: sqrt(phi) coefficients
    assert [leg.side for leg in bent.legs] == [OUT, OUT]
    back = tenet.repartition(bent, (0,), (1,))
    assert tenet.allclose(back, t)


def test_full_trace_is_the_qdim_weighted_spherical_trace():
    t = tau_tau()
    blocks = {key.output_tree.coupled: block for key, block in t.items()}
    expected = complex(blocks[ONE][0, 0]) + PHI * complex(blocks[TAU][0, 0])
    assert np.isclose(complex(tenet.full_trace(t)), expected)


# --- the refusals: wrong answers are unreachable, and by name -----------------


def test_transpose_refuses_the_chiral_braid_naming_braid():
    bent = tenet.repartition(tau_tau(), (0, 1), ())
    with pytest.raises(CapabilityError, match=r"braid\(t, i, over=") as err:
        bent.transpose((1, 0))
    assert "chiral" in str(err.value)


def test_to_dense_raises_capability_error_not_attribute_error():
    with pytest.raises(CapabilityError, match="ClebschGordanData"):
        tau_tau().to_dense()
