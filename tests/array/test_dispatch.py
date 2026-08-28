"""autoray registration and the get_params/set_params protocol — issue #23."""

import importlib
import pathlib
import subprocess
import sys

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import map_layout
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {HALF: 2, ONE: 3})
W = GradedSpace.new(SU2, {SINGLET: 2, HALF: 3})
Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})

# interleaved sides: OUT at axes 0, 2 and IN at axes 1, 3
SU2_LEGS = (Leg(V, OUT), Leg(W, IN), Leg(W, OUT), Leg(V, IN))
U1_LEGS = (Leg(Q, OUT), Leg(Q, IN), Leg(Q, OUT))
# case-A permutations: interleave (0, 2) and (1, 3) without disturbing either
CASE_A = ((0, 2, 1, 3), (0, 1, 3, 2), (1, 0, 2, 3), (1, 3, 0, 2))


def su2(seed: int = 0) -> SymmetricTensor:
    return SymmetricTensor.random(SU2_LEGS, seed=seed)


def u1(seed: int = 0) -> SymmetricTensor:
    return SymmetricTensor.random(U1_LEGS, seed=seed)


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


# --- registration happens on import --------------------------------------------


def test_import_registers_without_setup_call():
    assert ar.infer_backend(su2()) == "tenet"


def test_reimport_does_not_double_register_or_raise():
    importlib.reload(importlib.import_module("tenet.array.dispatch"))
    assert ar.infer_backend(su2()) == "tenet"
    assert ar.do("linalg.norm", su2()) == pytest.approx(tenet.norm(su2()), abs=1e-15)


def test_infer_backend_is_the_tensor_type_not_the_block_type():
    """Backend is 'tenet' for NumPy blocks and for JAX blocks alike."""
    assert ar.infer_backend(su2()) == "tenet"
    use_jax()
    t = su2().to_backend("jax")
    assert ar.infer_backend(t.blocks[0]) == "jax"
    assert ar.infer_backend(t) == "tenet"


# --- the registered names -------------------------------------------------------


@pytest.mark.parametrize("p", CASE_A)
def test_do_transpose(p):
    t = su2()
    assert tenet.allclose(ar.do("transpose", t, p), t.transpose(*p))


def test_do_conj_and_conjugate():
    t = su2()
    assert ar.do("conj", t) == t.conj()
    assert ar.do("conjugate", t) == t.conj()


def test_do_linalg_norm():
    t = su2()
    assert ar.do("linalg.norm", t) == pytest.approx(tenet.norm(t), abs=1e-15)


def test_do_arithmetic():
    a, b = su2(0), su2(1)
    assert ar.do("add", a, b) == a + b
    assert ar.do("subtract", a, b) == a - b
    assert ar.do("multiply", a, 2.0) == a * 2.0
    assert ar.do("negative", a) == -a


def test_do_tensordot_and_trace():
    """#51 registered both; the closed list is a list, not a policy against growth."""
    a = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN)), seed=0)
    b = SymmetricTensor.random((Leg(W, OUT), Leg(V, IN)), seed=1)
    assert ar.do("tensordot", a, b, axes=([1], [0])) == tenet.tensordot(a, b, ([1], [0]))
    c = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN), Leg(W, IN)), seed=2)
    assert ar.do("trace", c, axes=(0, 1)) == tenet.trace(c, (0, 1))


def test_do_einsum_dispatches_despite_the_string_first_argument():
    """#52: autoray's own einsum dispatcher infers the backend from every argument,
    so the ``str`` equation in first position needs no extra registration."""
    a = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN)), seed=0)
    b = SymmetricTensor.random((Leg(W, OUT), Leg(V, IN)), seed=1)
    assert ar.do("einsum", "ab,bc->ac", a, b) == tenet.einsum("ab,bc->ac", a, b)


def test_do_fuse():
    t = su2()
    assert ar.do("fuse", t, (0, 2)) == t.fuse((0, 2))


def test_do_fuse_multi_group_is_refused_by_name():
    """#68: autoray's convention is ``fuse(x, *axes_groups)``; the multi-group form
    is a reshape in disguise and must say so instead of raising a bare TypeError."""
    with pytest.raises(ValueError, match=r"fuse.*reshape in disguise"):
        ar.do("fuse", su2(), (0, 2), (1, 3))


def test_do_reshape_is_refused_in_our_own_voice():
    """#68: registered only to refuse — the defined-operation list is unchanged."""
    with pytest.raises(ValueError, match=r"reshape by shape is not defined"):
        ar.do("reshape", su2(), (4, 9))


def test_do_shape_and_ndim():
    t = su2()
    assert ar.do("shape", t) == t.shape
    assert ar.do("ndim", t) == t.ndim


def test_do_to_numpy_is_to_dense_exactly():
    t = su2()
    assert np.array_equal(ar.do("to_numpy", t), t.to_dense())


# --- loud failure: the closed list ----------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda a, b: ar.do("exp", a),
        lambda a, b: ar.do("svd", a),
        # #93 ships tenet.block_sqrt / tenet.block_power as *blockwise* maps on the coefficients,
        # and refuses both here. Not an oversight: autoray's "sqrt" is the dense
        # elementwise operation, and no non-linear function commutes with
        # T = sum_tau A^(tau) (x) C^(tau) -- measured 1.673 off on a dense scale of
        # 3.82 for a rank-3 SU(2) tensor. A refusal about meaning, not effort.
        lambda a, b: ar.do("sqrt", a),
        lambda a, b: ar.do("power", a, 0.5),
        lambda a, b: ar.do("flip", a, 0),
    ],
    ids=["exp", "svd", "sqrt", "power", "flip"],
)
def test_unregistered_operations_raise(call):
    """Invariant 11 at the dispatch layer: nothing unimplemented leaks through."""
    a, b = su2(0), su2(1)
    with pytest.raises(Exception):  # noqa: B017  the exception *type* is autoray's business
        call(a, b)


@pytest.mark.parametrize(
    ("call", "wanted"),
    [
        (lambda a: ar.do("flip", a, 0), "tenet.flip_dual(t, axes)"),
        (lambda a: ar.do("sqrt", a), "tenet.block_sqrt(t)"),
        (lambda a: ar.do("power", a, 2), "tenet.block_power(t)"),
    ],
    ids=["flip", "sqrt", "power"],
)
def test_the_three_numpy_name_collisions_refuse_in_our_own_voice(call, wanted):
    """#185: the three names whose numpy meaning is not this package's.

    autoray resolves an unregistered name by importing the backend module and
    looking it up, so before M31 ``ar.do("flip", t, 0)`` reached ``tenet.flip``
    and returned a dual-toggled tensor with no error at all. The renames close
    the lookup; these registrations make the answer a sentence rather than an
    ``ImportError``, and each names the operation that *was* meant.
    """
    with pytest.raises(ValueError, match="not defined for a symmetric tensor") as exc:
        call(su2(0))
    assert wanted in str(exc.value)


def test_no_array_protocol_and_asarray_does_not_densify():
    t = su2()
    assert not hasattr(SymmetricTensor, "__array__")
    arr = np.asarray(t)
    assert arr.dtype == object
    assert arr.item() is t


def test_src_never_imports_jax_or_torch():
    src = pathlib.Path(tenet.__file__).parent
    for path in src.rglob("*.py"):
        if path.name in ("pytree.py", "ad.py"):
            continue  # the opt-in modules (#41, #76); core never imports either
        text = path.read_text()
        assert "import jax" not in text, path
        assert "import torch" not in text, path


def test_import_tenet_does_not_import_jax_or_torch():
    code = (
        "import sys, tenet; "
        "assert 'jax' not in sys.modules, 'tenet imported jax'; "
        "assert 'torch' not in sys.modules, 'tenet imported torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- get_params / set_params / copy ---------------------------------------------


def test_get_params_is_the_sector_matrices():
    t = su2()
    assert t.get_params() is t.data
    assert isinstance(t.get_params(), tuple)
    assert len(t.get_params()) == len(map_layout(t.structure).sectors)


def test_set_params_round_trip_returns_new_equal_tensor():
    t = su2()
    t2 = t.set_params(t.get_params())
    assert t2 == t
    assert t2 is not t
    assert tenet.allclose(t, su2())  # T itself unmodified


def test_set_params_scales_and_leaves_original_untouched():
    t = su2()
    before = tuple(np.array(b, copy=True) for b in t.blocks)
    t2 = t.set_params([2 * b for b in t.get_params()])
    assert tenet.allclose(t2, 2 * t)
    assert all(np.array_equal(a, b) for a, b in zip(t.blocks, before, strict=True))


def test_set_params_wrong_count_raises():
    t = su2()
    with pytest.raises(ValueError, match="argument"):
        t.set_params(t.get_params()[:-1])


def test_set_params_wrong_shape_raises():
    """The parameters are matrices, so the refusal is ``from_matrices``'s shape check."""
    t = su2()
    bad = (np.zeros((1, 1)), *t.get_params()[1:])
    with pytest.raises(ValueError, match=r"has shape .* expected"):
        t.set_params(bad)


def test_tree_map_round_trip_like_quimb_convert_raw_arrays():
    """quimb's ``x.set_params(tree_map(f, x.get_params()))``."""
    t = su2()

    def f(x):
        return x.astype(np.float32)

    t2 = t.set_params(tuple(map(f, t.get_params())))
    assert t2.dtype == np.float32
    assert t2.structure == t.structure


def test_copy():
    t = su2()
    c = t.copy()
    assert c is not t
    assert c.structure == t.structure
    assert c == t


def test_params_protocol_is_backend_agnostic():
    use_jax()
    t = u1().to_backend("jax")
    assert ar.infer_backend(t.get_params()[0]) == "jax"
    assert ar.infer_backend(t.set_params(t.get_params()).get_params()[0]) == "jax"
    assert ar.infer_backend(t.copy().get_params()[0]) == "jax"
