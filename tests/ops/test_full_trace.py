"""Tests for ``tenet.full_trace`` and ``tenet.inner`` — the scalar exits of #126.

The oracle is dense: ``np.trace`` of ``to_dense()`` for a rank-2 map and
``np.einsum("abab->")`` for a rank-4 one, the second pinning that the pair closed is the
**map view** (codomain against domain, in order) and not an axis-adjacent one. The qdim
weight is what makes both identities hold, so each is paired with the unweighted sum in
the shape ``test_su2_norm_identity_and_the_necessity_of_the_qdim_weight`` uses for
``norm``.
"""

import dataclasses

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, U1, CapabilityError, SU2Sector, TrivialSector, U1Sector

HALF, ONE = SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 4, ONE: 3})
W = GradedSpace.new(SU2, {HALF: 2, ONE: 1})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
EMPTY = GradedSpace.new(SU2, {})  # square (it matches itself) and carries no sector
BLOCK_FREE = (Leg(EMPTY, OUT), Leg(EMPTY, IN))


@dataclasses.dataclass(frozen=True, slots=True)
class Bare:
    """Neither QuantumDimension nor ClebschGordan — ``test_basic.py``'s provider."""

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


def use_jax():
    jax = pytest.importorskip("jax")
    import tenet.pytree  # noqa: F401  # registration is the import's side effect

    jax.config.update("jax_enable_x64", True)
    return jax


# --- the dense oracle, and the necessity of the qdim weight ---------------------


def test_su2_full_trace_is_the_dense_trace_and_needs_the_qdim_weight():
    t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=1)
    mats = tenet.to_matrices(t)
    assert len({SU2.qdim(c) for c in mats}) > 1  # several qdims in play

    assert tenet.full_trace(t) == pytest.approx(np.trace(t.to_dense()), abs=1e-12)

    unweighted = sum(float(np.trace(m)) for m in mats.values())  # -1.2440709725867443
    assert abs(unweighted - float(tenet.full_trace(t))) > 0.1 * abs(tenet.full_trace(t))


def test_rank_4_full_trace_closes_the_map_view_not_an_adjacent_pair():
    """``"abab->"``: OUT axis 0 against IN axis 2, OUT axis 1 against IN axis 3."""
    t = SymmetricTensor.random((Leg(V, OUT), Leg(W, OUT), Leg(V, IN), Leg(W, IN)), seed=2)
    assert tenet.full_trace(t) == pytest.approx(np.einsum("abab->", t.to_dense()), abs=1e-12)


def test_abelian_full_trace_is_the_plain_trace():
    t = SymmetricTensor.random((Leg(Q, OUT), Leg(Q, IN)), seed=3)
    unweighted = sum(float(np.trace(m)) for m in tenet.to_matrices(t).values())
    assert tenet.full_trace(t) == pytest.approx(unweighted, rel=0, abs=1e-14)
    assert tenet.full_trace(t) == pytest.approx(np.trace(t.to_dense()), abs=1e-12)


# --- the refusals ---------------------------------------------------------------


def test_a_non_square_map_is_refused_where_it_used_to_return_a_number():
    """``network.common.scalar`` returned ``0.35065980168968025`` here — a rectangular
    ``np.trace`` down the leading diagonal. #126 turned that into a refusal."""
    t = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN)), seed=3)
    with pytest.raises(ValueError, match="full_trace: the map is not square at position 0"):
        tenet.full_trace(t)


def test_a_same_side_pair_is_refused_where_it_used_to_return_a_number():
    """``network.common.scalar`` returned ``-0.6517911526116896`` here: an empty domain,
    so ``B_c`` is one column wide and its ``trace`` is the top-left entry."""
    t = SymmetricTensor.random((Leg(V, OUT), Leg(V, OUT)), seed=4)
    with pytest.raises(ValueError, match="full_trace: the map is not square at position 0"):
        tenet.full_trace(t)


def test_full_trace_without_quantum_dimension_raises():
    space = GradedSpace.new(Bare(), {TrivialSector(): 2})
    t = SymmetricTensor.zeros((Leg(space, OUT), Leg(space, IN)))
    with pytest.raises(CapabilityError, match="QuantumDimension"):
        tenet.full_trace(t)


def test_full_trace_of_a_block_free_tensor_is_zero():
    t = SymmetricTensor.zeros(BLOCK_FREE)
    assert t.blocks == ()
    assert tenet.full_trace(t) == 0.0


# --- inner ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "legs",
    [
        (Leg(V, OUT), Leg(W, IN)),
        (Leg(V, OUT), Leg(W, IN), Leg(V, OUT), Leg(W, IN)),
        (Leg(Q, OUT), Leg(Q, IN), Leg(Q, OUT)),
    ],
)
def test_inner_with_itself_is_the_norm_squared(legs):
    """Moved out of ``tests/integration/test_dmrg.py`` (#126): it is a vector-space
    identity, not a DMRG one, and ``lanczos`` is only its first caller."""
    a = SymmetricTensor.random(legs, seed=5)
    assert float(tenet.inner(a, a)) == pytest.approx(float(tenet.norm(a)) ** 2, rel=1e-12)


def test_inner_is_sesquilinear_in_the_first_argument():
    a = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN)), seed=6)
    b = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN)), seed=7)
    s = 2.0 - 1.5j
    ab = complex(tenet.inner(a, b))
    assert complex(tenet.inner(a * s, b)) == pytest.approx(s.conjugate() * ab)
    assert complex(tenet.inner(a, b * s)) == pytest.approx(s * ab)
    assert complex(tenet.inner(b, a)) == pytest.approx(ab.conjugate())


# --- backend genericity ---------------------------------------------------------


def test_full_trace_under_jit_and_grad_stays_a_traced_scalar():
    jax = use_jax()
    t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=8)
    want = float(tenet.full_trace(t))
    seen = []

    @jax.jit
    def f(x):
        out = tenet.full_trace(x)
        seen.append(type(out))  # never a Python float: float() would break grad
        return out

    jt = t.to_backend("jax")
    assert float(f(jt)) == pytest.approx(want, abs=1e-12)
    assert not any(issubclass(k, float) for k in seen)

    grad = jax.grad(lambda x: tenet.full_trace(x).real)(jt)
    assert grad.backend == "jax"
    assert tenet.norm(grad) > 0  # the identity's blocks, not zeros


def test_inner_under_jit_matches_numpy():
    jax = use_jax()
    legs = (Leg(V, OUT), Leg(W, IN))
    a = SymmetricTensor.random(legs, seed=9)
    b = SymmetricTensor.random(legs, seed=10)
    got = jax.jit(tenet.inner)(a.to_backend("jax"), b.to_backend("jax"))
    assert float(got) == pytest.approx(float(tenet.inner(a, b)), abs=1e-12)
