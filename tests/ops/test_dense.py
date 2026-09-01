"""The dense boundary — ``to_dense`` / ``from_dense``, one test per criterion of #82.

The oracle for "values are unchanged" is ``legacy_to_dense`` below: the pre-#82
body of ``SymmetricTensor.to_dense``, transcribed once and never edited, scatter-add
and ``block.any()`` skip included. Comparing against it is stricter than comparing
against captured fixtures — it runs on every structure the parametrization names —
and it is what makes ``np.array_equal`` (not ``allclose``) the assertion.
"""

import math
import pathlib
import re

import numpy as np
import pytest
from helpers import NoBendProvider

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.dense import _tree_cgt, dense_plan, from_dense
from tenet.structure import TensorStructure
from tenet.symmetry import (
    SU2,
    U1,
    CapabilityError,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

# --- fixture structures ---------------------------------------------------------

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 2, ONE: 1})
W = GradedSpace.new(SU2, {SINGLET: 1, HALF: 2})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 1, U1Sector(1): 2})
T = GradedSpace.new(Trivial, {TrivialSector(): 3})
F = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 3})
UU = ProductProvider((U1, U1))
P = GradedSpace.new(
    UU,
    {
        ProductSector((U1Sector(0), U1Sector(0))): 2,
        ProductSector((U1Sector(1), U1Sector(0))): 1,
        ProductSector((U1Sector(0), U1Sector(1))): 2,
    },
)

STRUCTURES = {
    "trivial_1": (Leg(T, OUT),),
    "trivial_2": (Leg(T, OUT), Leg(T, IN)),
    "u1_2": (Leg(Q, OUT), Leg(Q, IN)),
    "u1_4": (Leg(Q, OUT), Leg(Q, OUT), Leg(Q, IN), Leg(Q, IN)),
    "su2_3": (Leg(V, OUT), Leg(W, OUT), Leg(V, IN)),
    # fusion multiplicity K > 1: two keys land in one dense cell
    "su2_4": (Leg(V, OUT), Leg(W, OUT), Leg(V, IN), Leg(W, IN)),
    "su2_dual": (Leg(V, OUT, dual=True), Leg(V, IN)),
    "su2_dual_3": (Leg(V, OUT), Leg(W, OUT, dual=True), Leg(V, IN)),
    "fz2_3": (Leg(F, OUT), Leg(F, OUT), Leg(F, IN)),
    "product_3": (Leg(P, OUT), Leg(P, OUT), Leg(P, IN)),
}
ALL = tuple(STRUCTURES)
JAX_CASES = ("u1_4", "su2_3", "su2_dual", "fz2_3", "product_3")


def tensor(name: str, seed: int = 0, dtype=np.float64) -> SymmetricTensor:
    return SymmetricTensor.random(STRUCTURES[name], seed=seed, dtype=dtype)


def legacy_to_dense(self: SymmetricTensor) -> np.ndarray:
    """``SymmetricTensor.to_dense`` exactly as it stood before #82."""
    provider = self.provider
    duals = tuple(leg.dual for leg in self.legs)
    n = self.ndim
    out_axes, in_axes = self.structure.out_axes, self.structure.in_axes
    order = (*out_axes, *in_axes)
    dtype = np.result_type(self.blocks[0].dtype, np.float64) if self.blocks else np.float64
    dense = np.zeros(tuple(leg.space.dim for leg in self.legs), dtype=dtype)
    a_sub = list(range(n))
    c_sub = [n + ax for ax in order]
    out_sub = [x for i in range(n) for x in (i, n + i)]
    for key, block in self.items():
        if not block.any():
            continue
        xout = _tree_cgt(provider, key.output_tree, tuple(duals[a] for a in out_axes))
        xin = _tree_cgt(provider, key.input_tree, tuple(duals[a] for a in in_axes))
        cgt = np.tensordot(xout, xin.conj(), axes=([-1], [-1]))
        full = np.einsum(block, a_sub, cgt, c_sub, out_sub)
        sectors = self.structure.axis_sectors(key)
        slabs, shape = [], []
        for leg, a in zip(self.legs, sectors, strict=True):
            size = leg.degeneracy(a) * provider.irrep_dim(a)
            slabs.append(slice(leg.space.sector_offset(a), leg.space.sector_offset(a) + size))
            shape.append(size)
        dense[tuple(slabs)] += full.reshape(shape)
    return dense


# --- part 1: values are unchanged, exactly --------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_to_dense_is_byte_identical_to_the_pre_change_implementation(name):
    t = tensor(name, seed=5)
    assert np.array_equal(t.to_dense(), legacy_to_dense(t))


@pytest.mark.parametrize("name", ["u1_4", "su2_3", "su2_dual"])
def test_complex_blocks_are_byte_identical_and_stay_complex(name):
    t = tensor(name, seed=6)
    t = SymmetricTensor(t.structure, tuple(b + 0.5j * b[::-1] for b in t.blocks))
    dense = t.to_dense()
    assert dense.dtype == np.complex128
    assert np.array_equal(dense, legacy_to_dense(t))


@pytest.mark.parametrize("name", ALL)
def test_shape_and_layout_contract(name):
    t = tensor(name, seed=7)
    assert t.to_dense().shape == t.shape


def test_no_data_dependent_branch_over_block_data():
    """A zero block and a ``1e-300`` block densify to the same array.

    The pre-#82 ``if not block.any(): continue`` made the *code path* depend on
    the values; invariant 9 rules that out, and its saving is now structural.
    """
    t = tensor("su2_3", seed=8)
    zeroed = SymmetricTensor(t.structure, (t.blocks[0] * 0.0, *t.blocks[1:]))
    tiny = SymmetricTensor(t.structure, (t.blocks[0] * 0.0 + 1e-300, *t.blocks[1:]))
    a, b = zeroed.to_dense(), tiny.to_dense()
    # same array up to the perturbation itself: the zero block's cell is still
    # assembled, so nothing about the layout or the dtype turned on the values
    assert (a.shape, a.dtype) == (b.shape, b.dtype)
    np.testing.assert_allclose(a, b, atol=1e-299)
    # and the skip it replaces changed nothing: byte-identical to the old path,
    # which *did* skip that block
    assert np.array_equal(a, legacy_to_dense(zeroed))


def test_source_has_no_value_dependent_branch():
    src = (pathlib.Path(tenet.__file__).parent / "ops" / "dense.py").read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[-1]  # drop the module docstring
    for forbidden in (".any()", ".all()", "bool("):
        assert forbidden not in body, forbidden


# --- the plan: cached, array-free, out of TensorStructure -----------------------


def test_dense_plan_is_cached_on_the_structure_alone():
    dense_plan.cache_clear()
    legs = STRUCTURES["su2_4"]
    a, b = SymmetricTensor.random(legs, seed=1), SymmetricTensor.random(legs, seed=2)
    a.to_dense()
    misses = dense_plan.cache_info().misses
    hits = dense_plan.cache_info().hits
    b.to_dense()  # an *equal* structure, not the same object
    assert dense_plan.cache_info().misses == misses
    assert dense_plan.cache_info().hits == hits + 1
    assert dense_plan(TensorStructure(tuple(legs))) is dense_plan(a.structure)


def test_cg_coefficients_are_built_once_for_n_calls():
    dense_plan.cache_clear()
    t = tensor("su2_4", seed=3)
    for _ in range(5):
        t.to_dense()
    assert dense_plan.cache_info().misses == 1
    assert dense_plan.cache_info().hits == 4


def test_the_plan_is_not_reachable_from_the_structure():
    """Invariant 8: no array-valued field, and no back-reference to one."""
    s = tensor("su2_3").structure
    assert hash(s) == hash(TensorStructure(s.legs))
    for name in dir(s):
        if name.startswith("__"):
            continue
        value = getattr(s, name, None)
        assert not isinstance(value, np.ndarray)
        assert type(value).__name__ != "DensePlan"


# --- the adjoint is pinned, not assumed -----------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_pinv_equals_conjugate_transpose_over_qdim(name):
    """``C Cᴴ = diag(qdim)``: the fusion basis is orthogonal, not orthonormal.

    That weight is the *same* one ``ops.basic.norm`` carries — it is what makes
    ``‖T‖`` equal the dense Frobenius norm — surfacing on the other side of the
    identity. So the adjoint is ``Cᴴ diag(1/qdim)``, spelled ``pinv`` in the
    implementation; this test is what keeps the two spellings in step.
    """
    structure = TensorStructure(STRUCTURES[name])
    plan = dense_plan(structure)
    provider = structure.provider
    for cell in plan.cells:
        C = cell.matrix
        qdim = np.array(
            [provider.qdim(structure.block_order[i].coupled) for i in cell.block_indices]
        )
        np.testing.assert_allclose(C @ C.conj().T, np.diag(qdim), atol=1e-12)
        np.testing.assert_allclose(np.linalg.pinv(C), C.conj().T @ np.diag(1 / qdim), atol=1e-12)
        np.testing.assert_allclose(cell.adjoint, np.linalg.pinv(C), atol=0.0, rtol=0.0)


def test_backend_calls_do_not_scale_with_the_grid(monkeypatch):
    """The walk is batched: the call count follows the distinct fusion multiplicities.

    The regression this guards is not a wrong value but a compile storm — one
    ``ar.do`` per cell per axis level is free on NumPy and cost 420 XLA executables
    on JAX for exactly this structure, because a symmetric tensor's cells are all
    different shapes and every distinct shape is a fresh module.
    """
    import autoray as ar

    space = GradedSpace.new(SU2, {SINGLET: 2, HALF: 3, ONE: 2})
    legs = (Leg(space, OUT),) * 3 + (Leg(space, IN),) * 2
    t = SymmetricTensor.random(legs, seed=11)
    plan = dense_plan(t.structure)
    assert len(plan.cells) > 50  # a grid worth batching in the first place

    calls = 0
    real = ar.do

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ar, "do", counting)
    dense = t.to_dense()
    assert calls < len(plan.cells)
    assert np.array_equal(dense, legacy_to_dense(t))


def test_a_cell_with_fusion_multiplicity_exists_in_the_fixtures():
    """The K > 1 case the adjoint exists for is actually exercised above."""
    plan = dense_plan(TensorStructure(STRUCTURES["su2_4"]))
    assert max(len(c.block_indices) for c in plan.cells) > 1


# --- part 2: round trips ---------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_from_dense_inverts_to_dense(name):
    t = tensor(name, seed=9)
    back = SymmetricTensor.from_dense(t.to_dense(), t.legs)
    assert back.structure == t.structure
    for a, b in zip(t.blocks, back.blocks, strict=True):
        np.testing.assert_allclose(b, a, atol=1e-14)


@pytest.mark.parametrize("name", ALL)
def test_to_dense_inverts_from_dense(name):
    d = tensor(name, seed=10).to_dense()
    np.testing.assert_allclose(
        SymmetricTensor.from_dense(d, STRUCTURES[name]).to_dense(), d, atol=1e-14
    )


def test_from_dense_round_trips_complex_data():
    t = tensor("su2_3", seed=11)
    t = SymmetricTensor(t.structure, tuple(b + 0.5j for b in t.blocks))
    back = SymmetricTensor.from_dense(t.to_dense(), t.legs)
    for a, b in zip(t.blocks, back.blocks, strict=True):
        np.testing.assert_allclose(b, a, atol=1e-14)


# --- refusals --------------------------------------------------------------------


def forbidden_cell_slabs(name):
    """Slices of one symmetry-forbidden cell of ``name``'s grid, or ``None``."""
    plan = dense_plan(TensorStructure(STRUCTURES[name]))
    occupied = {c.sectors for c in plan.cells}
    for index in np.ndindex(*(len(s) for s in plan.axis_sectors)):
        sectors = tuple(plan.axis_sectors[ax][i] for ax, i in enumerate(index))
        if sectors in occupied:
            continue
        return tuple(
            slice(sum(plan.axis_sizes[ax][:i]), sum(plan.axis_sizes[ax][: i + 1]))
            for ax, i in enumerate(index)
        ), sectors
    return None


def residual_of(exc: Exception) -> float:
    return float(re.search(r"residual ([0-9.eE+-]+) exceeds", str(exc)).group(1))


@pytest.mark.parametrize("name", ["u1_4", "su2_3", "fz2_3", "product_3"])
def test_a_unit_perturbation_in_a_forbidden_cell_is_refused_with_residual_one(name):
    slabs, sectors = forbidden_cell_slabs(name)
    d = tensor(name, seed=12).to_dense()
    d[tuple(s.start for s in slabs)] = 1.0
    with pytest.raises(ValueError, match="not symmetric") as exc:
        SymmetricTensor.from_dense(d, STRUCTURES[name])
    assert residual_of(exc.value) == pytest.approx(1.0, abs=1e-12)
    assert str(sectors) in str(exc.value)
    assert "atol" in str(exc.value)


def test_a_perturbation_orthogonal_to_an_allowed_cell_is_refused():
    """Inside an allowed cell, but outside the span of that cell's CG tensors."""
    name = "su2_3"
    legs = STRUCTURES[name]
    plan = dense_plan(TensorStructure(legs))
    cell = next(c for c in plan.cells if math.prod(c.dims) > len(c.block_indices))
    C = cell.matrix
    v = np.arange(1.0, math.prod(cell.dims) + 1.0)
    v -= v @ (np.linalg.pinv(C) @ C)  # project out the reproducible part
    v /= np.linalg.norm(v)
    d = tensor(name, seed=13).to_dense()
    # place `v` on the first degeneracy row of the cell, in the dense layout
    mat = np.zeros((math.prod(cell.degens), math.prod(cell.dims)))
    mat[0] = v
    n = len(cell.degens)
    piece = mat.reshape((*cell.degens, *cell.dims))
    piece = piece.transpose(
        [x for pair in zip(range(n), range(n, 2 * n), strict=True) for x in pair]
    )
    d[cell.slabs] += piece.reshape(cell.shape)
    with pytest.raises(ValueError, match="not symmetric") as exc:
        SymmetricTensor.from_dense(d, legs)
    assert residual_of(exc.value) == pytest.approx(1.0, abs=1e-10)
    assert str(cell.sectors) in str(exc.value)


def test_a_perturbation_below_atol_is_accepted_silently():
    name = "su2_3"
    slabs, _ = forbidden_cell_slabs(name)
    t = tensor(name, seed=14)
    d = t.to_dense()
    d[tuple(s.start for s in slabs)] = 1e-12 * np.linalg.norm(d)
    back = SymmetricTensor.from_dense(d, STRUCTURES[name])
    for a, b in zip(t.blocks, back.blocks, strict=True):
        np.testing.assert_allclose(b, a, atol=1e-10)


def test_the_default_atol_is_relative_and_scales_with_the_input():
    """The same *relative* perturbation is refused identically on ``d`` and ``1e6 d``."""
    name = "su2_3"
    slabs, _ = forbidden_cell_slabs(name)
    d = tensor(name, seed=15).to_dense()
    scale = np.linalg.norm(d)
    for factor in (1.0, 1e6):
        big = d * factor
        big[tuple(s.start for s in slabs)] = 1e-3 * scale * factor
        with pytest.raises(ValueError, match="not symmetric") as exc:
            SymmetricTensor.from_dense(big, STRUCTURES[name])
        assert residual_of(exc.value) == pytest.approx(1e-3 * scale * factor, rel=1e-6)
        small = d * factor
        small[tuple(s.start for s in slabs)] = 1e-12 * scale * factor
        SymmetricTensor.from_dense(small, STRUCTURES[name])


def test_atol_inf_projects_without_checking():
    name = "su2_3"
    slabs, _ = forbidden_cell_slabs(name)
    t = tensor(name, seed=16)
    d = t.to_dense()
    d[tuple(s.start for s in slabs)] = 100.0
    back = SymmetricTensor.from_dense(d, t.legs, atol=math.inf)
    for a, b in zip(t.blocks, back.blocks, strict=True):
        np.testing.assert_allclose(b, a, atol=1e-12)


# --- PROJECT, the named spelling of the sentinel (#210) ---------------------------


def test_project_is_exported_and_is_exactly_math_inf():
    """The whole point of the option chosen in #210: a name, not a second value."""
    assert "PROJECT" in tenet.__all__
    assert tenet.PROJECT is math.inf


@pytest.mark.parametrize("name", ["u1_4", "su2_3", "fz2_3", "product_3"])
def test_project_and_atol_inf_are_the_same_call_on_an_asymmetric_array(name):
    """No behaviour change: the same asymmetric array down both spellings, compared."""
    slabs, _ = forbidden_cell_slabs(name)
    d = tensor(name, seed=17).to_dense()
    d[tuple(s.start for s in slabs)] = 100.0
    old = SymmetricTensor.from_dense(d, STRUCTURES[name], atol=math.inf)
    new = SymmetricTensor.from_dense(d, STRUCTURES[name], atol=tenet.PROJECT)
    assert old.structure == new.structure
    for a, b in zip(old.blocks, new.blocks, strict=True):
        np.testing.assert_array_equal(b, a)


def test_the_from_dense_refusal_names_the_project_spelling():
    name = "su2_3"
    slabs, _ = forbidden_cell_slabs(name)
    d = tensor(name, seed=18).to_dense()
    d[tuple(s.start for s in slabs)] = 1.0
    with pytest.raises(ValueError, match="not symmetric") as exc:
        SymmetricTensor.from_dense(d, STRUCTURES[name])
    assert "tenet.PROJECT" in str(exc.value)


# --- structural refusals, before any block is read -------------------------------


def test_from_dense_refuses_a_shape_that_disagrees_with_the_legs():
    legs = STRUCTURES["su2_3"]
    with pytest.raises(ValueError, match="does not match"):
        SymmetricTensor.from_dense(np.zeros((2, 2, 2)), legs)


def test_both_directions_refuse_a_dual_leg_without_dual_basis():
    """Same message, same capability, both ways round.

    The vehicle used to be ``ProductProvider((U1, U1))``, documented to forward no
    ``DualBasis`` (#40). #312 forwarded it, so a product now expands a dual leg like any
    other provider -- asserted just below, so the widening is recorded rather than merely
    losing a test. ``helpers.NoBendProvider`` withholds the capability instead.
    """
    provider = NoBendProvider(UU)
    space = GradedSpace.new(
        provider,
        {
            ProductSector((U1Sector(0), U1Sector(0))): 2,
            ProductSector((U1Sector(1), U1Sector(0))): 1,
        },
    )
    leg = Leg(space, OUT, dual=True)
    t = SymmetricTensor.random((leg, Leg(space, IN)), seed=17)
    with pytest.raises(CapabilityError, match="DualBasis"):
        t.to_dense()
    with pytest.raises(CapabilityError, match="DualBasis"):
        SymmetricTensor.from_dense(np.zeros((space.dim, space.dim)), (leg, Leg(space, IN)))


def test_a_product_expands_a_dual_leg_since_the_forwarding():
    """The other side of the case above: the refusal it used to ride on is gone."""
    leg = Leg(P, OUT, dual=True)
    t = SymmetricTensor.random((leg, Leg(P, IN)), seed=17)
    dense = t.to_dense()
    assert dense.shape == (P.dim, P.dim)
    back = SymmetricTensor.from_dense(dense, (leg, Leg(P, IN)))
    assert tenet.allclose(back, t)


def test_both_directions_refuse_a_provider_without_clebsch_gordan():
    import dataclasses

    @dataclasses.dataclass(frozen=True, slots=True)
    class Bare:
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

    space = GradedSpace.new(Bare(), {TrivialSector(): 2})
    legs = (Leg(space, OUT), Leg(space, IN))
    with pytest.raises(CapabilityError):
        SymmetricTensor.random(legs, seed=18).to_dense()
    with pytest.raises(CapabilityError):
        SymmetricTensor.from_dense(np.zeros((2, 2)), legs)


# --- JAX ------------------------------------------------------------------------


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  # registration is the import's side effect

    return jax


@pytest.mark.parametrize("name", JAX_CASES)
def test_to_dense_on_jax_blocks_returns_a_jax_array_with_the_same_values(name):
    import autoray as ar

    use_jax()
    t = tensor(name, seed=19)
    dense = t.to_backend("jax").to_dense()
    assert ar.infer_backend(dense) == "jax"
    np.testing.assert_allclose(np.asarray(dense), t.to_dense(), atol=1e-13)


def test_to_numpy_still_returns_numpy_for_a_jax_backed_tensor():
    import autoray as ar

    use_jax()
    t = tensor("su2_3", seed=20)
    out = ar.do("to_numpy", t.to_backend("jax"))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, t.to_dense(), atol=1e-13)


def test_jit_traces_once_per_structure():
    jax = use_jax()
    count = 0

    @jax.jit
    def f(t):
        nonlocal count
        count += 1
        return (t.to_dense() ** 2).sum()

    a = tensor("su2_3", seed=21).to_backend("jax")
    b = tensor("su2_3", seed=22).to_backend("jax")
    f(a), f(b)
    assert count == 1
    f(tensor("u1_4", seed=23).to_backend("jax"))
    assert count == 2
    f(tensor("su2_3", seed=24).to_backend("jax"))
    assert count == 2


@pytest.mark.parametrize("name", ["u1_4", "su2_3", "su2_dual"])
def test_grad_through_to_dense_matches_finite_differences(name):
    jax = use_jax()
    t = tensor(name, seed=25)
    rng = np.random.default_rng(0)
    op = rng.standard_normal(t.shape)

    def loss_np(blocks):
        d = SymmetricTensor(t.structure, tuple(blocks)).to_dense()
        return float((d**2).sum() + (d * op).sum())

    def loss(x):
        d = x.to_dense()
        return (d**2).sum() + (d * op).sum()

    grads = jax.grad(loss)(t.to_backend("jax"))
    h = 1e-5
    for i, block in enumerate(t.blocks):
        got = np.asarray(grads.blocks[i])
        fd = np.zeros_like(block)
        for idx in np.ndindex(*block.shape):
            up, dn = [list(t.blocks) for _ in range(2)]
            up[i], dn[i] = block.copy(), block.copy()
            up[i][idx] += h
            dn[i][idx] -= h
            fd[idx] = (loss_np(up) - loss_np(dn)) / (2 * h)
        np.testing.assert_allclose(got, fd, atol=1e-6, rtol=1e-6)


def test_grad_of_a_real_objective_through_a_complex_to_dense():
    jax = use_jax()
    t = tensor("su2_3", seed=26)
    t = SymmetricTensor(t.structure, tuple(b.astype(np.complex128) + 0.5j for b in t.blocks))

    def loss_np(blocks):
        d = SymmetricTensor(t.structure, tuple(blocks)).to_dense()
        return float((abs(d) ** 2).sum())

    def loss(x):
        return (abs(x.to_dense()) ** 2).sum()

    grads = jax.grad(loss, holomorphic=False)(t.to_backend("jax"))
    h = 1e-5
    for i, block in enumerate(t.blocks):
        got = np.asarray(grads.blocks[i])
        for idx in np.ndindex(*block.shape):
            for part, step in ((0, h), (1, 1j * h)):
                up, dn = [list(t.blocks) for _ in range(2)]
                up[i], dn[i] = block.copy(), block.copy()
                up[i][idx] += step
                dn[i][idx] -= step
                fd = (loss_np(up) - loss_np(dn)) / (2 * h)
                # JAX's convention for a real loss: grad is the conjugate cotangent
                assert (got[idx].real if part == 0 else -got[idx].imag) == pytest.approx(
                    fd, abs=1e-6
                )


def test_from_dense_traces_only_with_atol_inf():
    """The boundary, both sides on one screen.

    ``atol=math.inf`` is pure slicing + reshape + matmul, hence traceable and
    differentiable. The default ``atol`` compares a residual to a tolerance, which
    is a concrete-value question, so JAX refuses it in its own voice.
    """
    jax = use_jax()
    jnp = pytest.importorskip("jax.numpy")
    legs = STRUCTURES["su2_3"]
    t = tensor("su2_3", seed=27)
    d = jnp.asarray(t.to_dense())

    @jax.jit
    def project(x):
        return from_dense(x, legs, atol=math.inf)

    back = project(d)
    for a, b in zip(t.blocks, back.blocks, strict=True):
        np.testing.assert_allclose(np.asarray(b), a, atol=1e-12)

    @jax.jit
    def checked(x):
        return from_dense(x, legs)

    with pytest.raises(jax.errors.ConcretizationTypeError):
        checked(d)


def test_grad_through_from_dense_with_atol_inf():
    jax = use_jax()
    jnp = pytest.importorskip("jax.numpy")
    legs = STRUCTURES["u1_4"]
    d0 = tensor("u1_4", seed=28).to_dense()

    def loss(x):
        return tenet.norm(from_dense(x, legs, atol=math.inf)) ** 2

    g = np.asarray(jax.grad(loss)(jnp.asarray(d0)))
    h = 1e-5
    for idx in [(0, 0, 0, 0), (1, 2, 1, 2), (3, 3, 0, 0)]:
        up, dn = d0.copy(), d0.copy()
        up[idx] += h
        dn[idx] -= h
        fd = (float(loss(jnp.asarray(up))) - float(loss(jnp.asarray(dn)))) / (2 * h)
        assert g[idx] == pytest.approx(fd, abs=1e-6)


def test_structural_refusals_are_identical_inside_a_trace():
    jax = use_jax()
    jnp = pytest.importorskip("jax.numpy")
    legs = STRUCTURES["su2_3"]

    @jax.jit
    def f(x):
        return from_dense(x, legs, atol=math.inf)

    with pytest.raises(ValueError, match="does not match"):
        f(jnp.zeros((2, 2, 2)))
