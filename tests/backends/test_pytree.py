"""Tests for `tenet.pytree` — opt-in JAX PyTree registration (issue #41).

x64 is enabled process-globally in ``tests/conftest.py``; the finite-difference
tolerances here depend on it.
"""

import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import tenet.pytree  # noqa: E402, F401  # registration is the import's side effect

HALF, ONE = SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 2, ONE: 1})
W = GradedSpace.new(SU2, {HALF: 1})
V_BIG = GradedSpace.new(SU2, {HALF: 3, ONE: 1})  # differs from V only in a degeneracy
Q = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 2})

# 3 legs, two qdim-distinct coupled sectors: the qdim weight in `norm` is visible
SU2_LEGS = (Leg(V, OUT), Leg(W, OUT), Leg(V, IN))
SU2_LEGS_BIG = (Leg(V_BIG, OUT), Leg(W, OUT), Leg(V_BIG, IN))
SU2_LEGS4 = (Leg(V, OUT), Leg(W, OUT), Leg(V, IN), Leg(W, IN))
U1_LEGS4 = (Leg(Q, OUT), Leg(Q, OUT), Leg(Q, IN), Leg(Q, IN))
# SU(2) is case A: only the OUT/IN interleaving may change, since a within-side
# reorder is a braid and needs PermutationCoefficients (M4). U(1) is case B and
# takes the within-side swap, so the coefficient path is exercised too.
SU2_PERM = (0, 2, 1, 3)
U1_PERM = (1, 0, 3, 2)


def jt(legs, seed):
    """A random float64 JAX-backed tensor."""
    return SymmetricTensor.random(legs, seed=seed).to_backend("jax")


def qdims(t):
    return [t.provider.qdim(k.coupled) for k in t.structure.block_order]


# --- core never imports jax -----------------------------------------------------


def test_only_pytree_module_imports_jax():
    """The file walk, not review, is what holds this invariant up.

    Set **equality**, never containment: ``tenet.ad`` (#76) is the second and last
    opt-in module allowed to import JAX, and the assertion has to fail the moment a
    third one appears.
    """
    src = pathlib.Path(tenet.__file__).parent
    offenders = {p.name for p in src.rglob("*.py") if "import jax" in p.read_text()}
    assert offenders == {"pytree.py", "ad.py"}
    # the modules #76 leaves at a zero-line diff, stated by name
    for rel in ("ops/linalg.py", "tensor.py", *(f"ops/{p.name}" for p in (src / "ops").iterdir())):
        if (src / rel).suffix == ".py":
            assert "import jax" not in (src / rel).read_text(), rel


def test_not_exported_from_tenet_namespace():
    assert "pytree" not in tenet.__all__
    assert tenet.pytree.__all__ == []


def test_import_error_without_jax_names_extra_and_alternative():
    """Simulate a JAX-less environment with a meta-path blocker, in a subprocess."""
    script = textwrap.dedent("""
        import importlib.abc, sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "jax" or name.startswith("jax."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, Blocker())
        import tenet                      # core must survive a JAX-less environment
        assert "jax" not in sys.modules
        try:
            import tenet.pytree
        except ImportError as exc:
            msg = str(exc)
            assert "JAX" in msg, msg
            assert "symtenet[jax]" in msg, msg
            assert "get_params/set_params" in msg, msg
        else:
            raise AssertionError("tenet.pytree imported without jax")
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "OK"


def test_double_import_is_a_no_op():
    import tenet.pytree
    import tenet.pytree as again

    assert again is tenet.pytree


# --- flatten / treedef ----------------------------------------------------------


def test_leaves_are_the_sector_matrix_objects_themselves():
    """The leaves are the tensor's storage: one matrix per coupled sector, no copies."""
    t = jt(SU2_LEGS, 0)
    leaves = jax.tree.leaves(t)
    assert len(leaves) == len(t.data) <= len(t.blocks)
    assert all(a is b for a, b in zip(leaves, t.data, strict=True))


def test_treedef_aux_is_the_structure_and_sentinel_path_does_not_raise():
    t = jt(SU2_LEGS, 1)
    # tree_structure walks the tree with object() sentinels as leaves: this is the
    # call that fails outright under a validating unflatten.
    treedef = jax.tree_util.tree_structure(t)
    assert treedef.node_data()[1] is t.structure


def test_treedef_hygiene_hash_equality_and_degeneracy_sensitivity():
    a, b = jt(SU2_LEGS, 2), jt(SU2_LEGS, 3)
    big = jt(SU2_LEGS_BIG, 4)
    hash(a.structure)
    da, db, dbig = (jax.tree_util.tree_structure(x) for x in (a, b, big))
    assert da == db and hash(da) == hash(db)
    assert da != dbig


def test_round_trip_preserves_structure_identity():
    t = jt(SU2_LEGS, 5)
    leaves, treedef = jax.tree_util.tree_flatten(t)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(back, SymmetricTensor)
    assert back.structure is t.structure
    assert back == t


# --- unflatten permissiveness ---------------------------------------------------


def test_unflatten_accepts_zero_d_leaves():
    t = jt(SU2_LEGS, 6)
    got = jax.tree.map(lambda b: b.sum(), t)
    assert got.structure is t.structure
    assert all(m.ndim == 0 for m in got.data)


def test_unflatten_accepts_python_float_leaves():
    t = jt(SU2_LEGS, 7)
    got = jax.tree.map(lambda b: 0.0, t)
    assert got.structure is t.structure
    assert got.data == (0.0,) * len(t.data)


def test_unflatten_accepts_stacked_leaves():
    ts = [jt(SU2_LEGS, s) for s in (8, 9, 10)]
    got = jax.tree.map(lambda *bs: jnp.stack(bs), *ts)
    assert got.structure is ts[0].structure
    for i, mat in enumerate(got.data):
        assert mat.shape == (3, *ts[0].data[i].shape)


@pytest.mark.parametrize(
    "mangle",
    [lambda t: tuple(b.sum() for b in t.blocks), lambda t: (0.0,) * len(t.blocks)],
)
def test_constructor_is_still_strict_about_the_same_blocks(mangle):
    """The strictness moved to the trust boundary; it was not deleted."""
    t = jt(SU2_LEGS, 11)
    with pytest.raises((ValueError, AttributeError)):
        SymmetricTensor(t.structure, mangle(t))


# --- norm traceability ----------------------------------------------------------


def test_numpy_norm_is_still_float_compatible():
    t = SymmetricTensor.random(SU2_LEGS, seed=12)
    v = tenet.norm(t)
    assert float(v) == v
    expected = np.sqrt(
        sum(q * np.sum(np.abs(b) ** 2) for q, b in zip(qdims(t), t.blocks, strict=True))
    )
    assert float(v) == float(expected)


def test_jit_norm_returns_a_zero_d_jax_array():
    t = jt(SU2_LEGS, 13)
    v = jax.jit(tenet.norm)(t)
    assert isinstance(v, jax.Array) and v.shape == ()
    assert float(v) == pytest.approx(float(tenet.norm(t.to_backend("numpy"))), abs=1e-12)


# --- jit ------------------------------------------------------------------------


def test_jit_traces_once_per_structure():
    count = 0

    @jax.jit
    def f(t):
        nonlocal count
        count += 1  # a Python side effect: it runs at trace time only
        return tenet.norm(t) ** 2

    a, b = jt(SU2_LEGS, 14), jt(SU2_LEGS, 15)
    ra, rb = f(a), f(b)
    assert count == 1
    for t, r in ((a, ra), (b, rb)):
        assert float(r) == pytest.approx(float(tenet.norm(t.to_backend("numpy"))) ** 2, rel=1e-12)

    f(jt(SU2_LEGS_BIG, 16))  # different bond dimension -> different treedef
    assert count == 2

    f(jt(SU2_LEGS, 17))  # back to the first structure: cache hit
    assert count == 2


@pytest.mark.parametrize(
    ("legs", "perm"), [(SU2_LEGS4, SU2_PERM), (U1_LEGS4, U1_PERM)], ids=["su2", "u1"]
)
def test_jit_transpose_and_fuse(legs, perm):
    t = jt(legs, 18)
    ref = t.to_backend("numpy")
    for op, want in (
        (lambda x: tenet.transpose(x, perm), tenet.transpose(ref, perm)),
        (lambda x: tenet.fuse(x, (0, 1)), tenet.fuse(ref, (0, 1))),
    ):
        got = jax.jit(op)(t)
        assert isinstance(got, SymmetricTensor)
        assert got.structure == want.structure
        assert tenet.allclose(got.to_backend("numpy"), want, rtol=0, atol=1e-12)


# --- grad -----------------------------------------------------------------------


def sq_norm(t):
    return tenet.norm(t) ** 2


def test_grad_matches_the_closed_form_for_su2():
    """d/dA_tau of sum_tau qdim(c_tau) ||A_tau||^2 is 2 qdim(c_tau) A_tau."""
    t = jt(SU2_LEGS, 19)
    g = jax.grad(sq_norm)(t)
    assert isinstance(g, SymmetricTensor)
    assert g.structure is t.structure
    assert len(g.blocks) == len(t.blocks)
    weights = set(qdims(t))
    assert len(weights) > 1, "pick legs where the qdim weight is actually visible"
    for q, got, block in zip(qdims(t), g.blocks, t.blocks, strict=True):
        np.testing.assert_allclose(np.asarray(got), 2 * q * np.asarray(block), atol=1e-12)


def central_differences(f, t, h=1e-5):
    """df/d(every entry of every block), by central differences."""
    out = []
    for i, block in enumerate(t.blocks):
        d = np.zeros(block.shape)
        for idx in np.ndindex(block.shape):

            def shifted(delta, i=i, idx=idx):
                blocks = list(t.blocks)
                blocks[i] = blocks[i].at[idx].add(delta)
                return float(f(SymmetricTensor(t.structure, tuple(blocks))))

            d[idx] = (shifted(h) - shifted(-h)) / (2 * h)
        out.append(d)
    return out


def test_grad_matches_finite_differences_su2():
    t = jt(SU2_LEGS, 20)
    g = jax.grad(sq_norm)(t)
    for got, want in zip(g.blocks, central_differences(sq_norm, t), strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-8)


def test_grad_matches_finite_differences_u1_through_transpose():
    """A composite objective, so a permutation's coefficient path is differentiated."""

    def f(t):
        return tenet.norm(tenet.transpose(t, U1_PERM)) ** 2

    t = jt(U1_LEGS4, 21)
    g = jax.grad(f)(t)
    for got, want in zip(g.blocks, central_differences(f, t), strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-8)


def test_grad_through_set_params_agrees_with_the_pytree_route():
    t = jt(SU2_LEGS, 22)
    # the parameters are the coupled-sector matrices, which are also the pytree leaves
    via_params = jax.grad(lambda params: sq_norm(t.set_params(params)))(t.get_params())
    via_pytree = jax.grad(sq_norm)(t)
    for a, b in zip(via_params, via_pytree.data, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)


# --- vmap -----------------------------------------------------------------------


@pytest.fixture
def batch():
    ts = [jt(SU2_LEGS, s) for s in (23, 24, 25)]
    return ts, jax.tree.map(lambda *bs: jnp.stack(bs), *ts)


def test_vmap_scalar_valued(batch):
    ts, batched = batch
    got = jax.vmap(sq_norm)(batched)
    assert got.shape == (3,)
    np.testing.assert_allclose(np.asarray(got), [float(sq_norm(t)) for t in ts], atol=1e-12)


def test_vmap_tensor_valued(batch):
    ts, batched = batch
    got = jax.vmap(lambda t: tenet.transpose(t, (0, 2, 1)))(batched)
    want = [tenet.transpose(t, (0, 2, 1)) for t in ts]
    for i, mat in enumerate(got.data):
        np.testing.assert_allclose(
            np.asarray(mat), np.stack([np.asarray(w.data[i]) for w in want]), atol=1e-12
        )


def test_the_stacked_object_is_not_a_valid_tensor(batch):
    """Honest framing: it is a transport container for vmap, nothing else."""
    _, batched = batch
    with pytest.raises(ValueError, match="shape"):
        SymmetricTensor(batched.structure, batched.data)


def test_traced_shapes_inside_vmap_are_unbatched(batch):
    ts, batched = batch
    unbatched = [b.shape for b in ts[0].blocks]

    def f(t):
        assert [b.shape for b in t.blocks] == unbatched
        return sq_norm(t)

    jax.vmap(f)(batched)


# --- composition ----------------------------------------------------------------


def test_jit_of_grad_and_vmap_of_grad(batch):
    ts, batched = batch
    plain = [jax.grad(sq_norm)(t) for t in ts]

    jitted = jax.jit(jax.grad(sq_norm))(ts[0])
    for a, b in zip(jitted.blocks, plain[0].blocks, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)

    vmapped = jax.vmap(jax.grad(sq_norm))(batched)
    for i, mat in enumerate(vmapped.data):
        np.testing.assert_allclose(
            np.asarray(mat), np.stack([np.asarray(p.data[i]) for p in plain]), atol=1e-12
        )


# --- what the storage costs a traced graph ------------------------------------------
# The same decision the leaves above record, seen from the other end. A block is a slice
# of a coupled-sector matrix; on NumPy that slice is a view and free, but a traced slice
# is a graph node whose backward pass is a ``pad``. So what an operation reads decides
# what it costs under ``jax.jit``, and these two facts are structural -- they are about
# the program, not about a machine or a duration.


def _primitives(fn, *args):
    """Every primitive a traced ``fn`` issues, counted, closed sub-jaxprs included."""
    counts: dict[str, int] = {}

    def walk(j):
        for eqn in j.eqns:
            name = str(eqn.primitive)
            counts[name] = counts.get(name, 0) + 1
            for param in eqn.params.values():
                for sub in param if isinstance(param, (list, tuple)) else (param,):
                    inner = getattr(sub, "jaxpr", None)
                    if inner is not None:
                        walk(getattr(inner, "jaxpr", inner))

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return counts


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(tenet.conj, id="conj"),
        pytest.param(tenet.negative, id="negative"),
        pytest.param(tenet.adjoint, id="adjoint"),
        pytest.param(lambda t: tenet.add(t, t), id="add"),
        pytest.param(lambda t: tenet.multiply(t, 2.0), id="multiply"),
    ],
)
def test_a_matrix_native_op_cuts_no_block_into_the_graph(op):
    """An operation defined on the coupled-sector matrices slices nothing at all.

    Not a bound with a constant in it: *zero*. Each of these is one backend call per
    coupled sector over ``data``, so no block is ever cut and the traced program carries
    no slice and no concatenate on their account -- however many blocks the tensor has.
    """
    t = jt(U1_LEGS4, 12)
    assert t.structure.num_blocks > len(t.data)  # the fixture has blocks to cut
    counts = _primitives(op, t)
    assert counts.get("slice", 0) == 0
    assert counts.get("concatenate", 0) == 0


def test_an_op_that_reads_blocks_does_not_also_gather_them_back():
    """A block-in, block-out operation gathers nothing: on JAX the matrices wait.

    ``transpose`` is handed blocks and produces blocks, so it pays the cut its operand
    needs -- that is what a slice is for -- and then nothing more, because on an
    immutable backend the result's matrices are not built until something asks for one.
    A ``concatenate`` here would be that gather, one per band, for a tensor whose blocks
    are what the caller went on to read.
    """
    t = jt(U1_LEGS4, 13)

    def fn(x):
        moved = tenet.transpose(x, (0, 3, 2, 1))
        return sum(jnp.sum(b) for b in moved.blocks)

    counts = _primitives(fn, t)
    assert counts.get("slice", 0) > 0  # the operand is matrix-backed, so it is cut
    assert counts.get("concatenate", 0) == 0
