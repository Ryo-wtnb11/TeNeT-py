"""End-to-end proof that a ``SymmetricTensor`` works as the ``data`` of a
``quimb.tensor.Tensor`` — issue #68.

Nothing under ``src/tenet`` imports quimb or cotengra, and nothing ever will: the
integration is the ``autoray`` registration of ``tenet/array/dispatch.py`` plus the
``get_params``/``set_params`` protocol of #23. quimb is a **dev-group** dependency
whose only job is to make that claim testable; every test here is guarded by
``pytest.importorskip`` so the suite passes with quimb absent.

The file also pins the *refusals*. Two of them are correct mathematics that quimb
can walk into (a full scalar contraction, and dropping an index via
``output_inds``); two are ours to phrase well (``fuse``'s autoray multi-group form,
and ``reshape``).
"""

import pathlib

import autoray as ar
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

pytest.importorskip("quimb")
import quimb.tensor as qtn  # noqa: E402

SINGLET, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)

# --- U(1): a charge-conserving MPS with trivial boundaries ----------------------
U1_TRIV = GradedSpace.new(U1, {U1Sector(0): 1})
U1_PHYS = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
U1_BOND = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(0): 2, U1Sector(1): 1})

# --- SU(2): four spin-1/2 sites, so both boundaries can be the singlet ----------
SU2_TRIV = GradedSpace.new(SU2, {SINGLET: 1})
SU2_PHYS = GradedSpace.new(SU2, {HALF: 1})
SU2_BONDS = (
    SU2_TRIV,
    GradedSpace.new(SU2, {HALF: 1}),
    GradedSpace.new(SU2, {SINGLET: 1, ONE: 1}),
    GradedSpace.new(SU2, {HALF: 1}),
    SU2_TRIV,
)


def u1_chain(n: int = 3, seed: int = 0) -> list[SymmetricTensor]:
    """``n`` U(1) MPS site tensors, ``(bond OUT, physical OUT, bond IN)``."""
    bonds = [U1_TRIV, *[U1_BOND] * (n - 1), U1_TRIV]
    return [
        SymmetricTensor.random(
            (Leg(bonds[i], OUT), Leg(U1_PHYS, OUT), Leg(bonds[i + 1], IN)), seed=seed + i
        )
        for i in range(n)
    ]


def su2_chain(seed: int = 0) -> list[SymmetricTensor]:
    """A four-site SU(2) MPS. Three spin-1/2 sites cannot close on the singlet."""
    return [
        SymmetricTensor.random(
            (Leg(SU2_BONDS[i], OUT), Leg(SU2_PHYS, OUT), Leg(SU2_BONDS[i + 1], IN)),
            seed=seed + i,
        )
        for i in range(4)
    ]


def network(tensors, prefix: str = "b") -> qtn.TensorNetwork:
    return qtn.TensorNetwork(
        [
            qtn.Tensor(data=t, inds=(f"{prefix}{i}", f"p{i}", f"{prefix}{i + 1}"))
            for i, t in enumerate(tensors)
        ]
    )


def chain_inds(n: int, prefix: str = "b") -> tuple[str, ...]:
    return (f"{prefix}0", *(f"p{i}" for i in range(n)), f"{prefix}{n}")


def dense_chain(tensors) -> np.ndarray:
    """``np.einsum`` oracle for the open chain, indices ordered as ``chain_inds``."""
    letters = "abcdefgh"
    terms = ",".join(f"{letters[i]}{'ijklmn'[i]}{letters[i + 1]}" for i in range(len(tensors)))
    out = letters[0] + "ijklmn"[: len(tensors)] + letters[len(tensors)]
    return np.einsum(f"{terms}->{out}", *(t.to_dense() for t in tensors))


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    pytest.importorskip("tenet.pytree")
    return jax


@pytest.fixture
def quimb_set_params_forwards_the_return(monkeypatch):
    """quimb's ``Tensor.set_params`` mutates the data object in place and throws the
    return value away; ``SymmetricTensor`` is frozen (#23) and returns a new tensor,
    so without this shim quimb's parameter injection is a silent no-op and
    ``TNOptimizer`` reports a flat loss. Pinned by
    :func:`test_quimb_set_params_expects_in_place_mutation`; the shim is what makes
    the optimizer tests below exercise real optimization.
    """
    monkeypatch.setattr(
        qtn.Tensor,
        "set_params",
        lambda self, params: self._set_data(self.data.set_params(params)),
        raising=True,
    )


# --- Tensor round-trip -----------------------------------------------------------


def test_tensor_constructs_with_physical_shape_and_tenet_backend():
    t = u1_chain(1)[0]
    qt = qtn.Tensor(data=t, inds=("b0", "p0", "b1"))
    assert qt.shape == t.shape
    assert qt.size == int(np.prod(t.shape))
    assert qt.backend == "tenet"
    assert qt.data is t


@pytest.mark.parametrize("op", ["copy", "conj", "H", "transpose"])
def test_tensor_operations_preserve_the_symmetric_tensor(op):
    t = su2_chain()[1]
    qt = qtn.Tensor(data=t, inds=("b1", "p1", "b2"))
    out = {
        "copy": lambda: qt.copy(),
        "conj": lambda: qt.conj(),
        "H": lambda: qt.H,
        "transpose": lambda: qt.transpose("p1", "b2", "b1"),
    }[op]()
    assert isinstance(out.data, SymmetricTensor)
    assert out.shape == tuple(t.shape[qt.inds.index(i)] for i in out.inds)
    expected = t if op != "transpose" else t.transpose(1, 2, 0)
    assert out.data.legs == expected.legs


# --- contraction correctness against a dense oracle -------------------------------


def test_two_tensor_u1_contraction_matches_dense():
    a, b = u1_chain(2)
    ta = qtn.Tensor(data=a, inds=("b0", "p0", "b1"))
    tb = qtn.Tensor(data=b, inds=("b1", "p1", "b2"))
    out = ta @ tb
    assert isinstance(out.data, SymmetricTensor)
    assert np.allclose(out.transpose(*chain_inds(2)).data.to_dense(), dense_chain([a, b]))


def test_three_tensor_u1_network_matches_dense():
    ts = u1_chain(3)
    out = network(ts).contract(output_inds=chain_inds(3))
    assert np.allclose(out.data.to_dense(), dense_chain(ts))


def test_su2_chain_matches_dense():
    """Non-Abelian, four sites, contracted by quimb's own path."""
    ts = su2_chain()
    out = network(ts).contract(output_inds=chain_inds(4))
    assert np.allclose(out.data.to_dense(), dense_chain(ts))


def test_tensor_contract_helper_matches_dense():
    ts = u1_chain(3)
    out = qtn.tensor_contract(*network(ts).tensors, output_inds=chain_inds(3))
    assert np.allclose(out.data.to_dense(), dense_chain(ts))


# --- the hand-off is pairwise, asserted rather than assumed -----------------------


def test_quimb_only_ever_calls_us_with_one_or_two_operands():
    """cotengra lowers the network itself and hands us pairwise work, which is why
    the multi-operand ``opt_einsum`` path is irrelevant to this integration."""
    calls: list[tuple[str, int]] = []

    def spy_tensordot(a, b, axes, **kw):
        calls.append(("tensordot", 2))
        return tenet.tensordot(a, b, axes, **kw)

    def spy_einsum(eq, *arrays, **kw):
        calls.append(("einsum", len(arrays)))
        return tenet.einsum(eq, *arrays, **kw)

    ar.register_function("tenet", "tensordot", spy_tensordot)
    ar.register_function("tenet", "einsum", spy_einsum)
    try:
        ts = u1_chain(5)
        out = network(ts).contract(output_inds=chain_inds(5))
    finally:
        ar.register_function("tenet", "tensordot", tenet.tensordot)
        ar.register_function("tenet", "einsum", tenet.einsum)

    assert np.allclose(out.data.to_dense(), dense_chain(ts))
    assert calls, "the spies never fired — the contraction did not reach tenet"
    assert max(n for _, n in calls) <= 2, calls


# --- get_params / set_params ------------------------------------------------------


def test_get_params_set_params_round_trip_is_bitwise():
    ts = u1_chain(3)
    tn = network(ts)
    params = tn.get_params()
    assert isinstance(params, dict)
    assert all(isinstance(v, tuple) for v in params.values())

    tn2 = tn.copy()
    tn2.set_params(params)
    for before, after in zip(tn.tensors, tn2.tensors, strict=True):
        assert after.data.structure == before.data.structure
        for x, y in zip(before.data.blocks, after.data.blocks, strict=True):
            assert np.array_equal(x, y)


def test_quimb_set_params_expects_in_place_mutation():
    """The one genuine protocol mismatch this integration has, pinned so it cannot
    change silently. quimb's ``Tensor.set_params`` copies the data, calls
    ``data.set_params(params)`` for the side effect, and discards the return;
    ``SymmetricTensor`` is frozen, so the new blocks never land. Changing #23 is out
    of scope for #68, so the optimizer tests apply a one-line shim instead.
    """
    t = u1_chain(1)[0]
    qt = qtn.Tensor(data=t, inds=("b0", "p0", "b1"))
    doubled = tuple(2 * b for b in t.get_params())
    qt.set_params(doubled)
    assert all(np.array_equal(x, y) for x, y in zip(qt.data.blocks, t.blocks, strict=True)), (
        "quimb's in-place set_params finally works; drop the shim and this test"
    )


# --- TNOptimizer, end to end ------------------------------------------------------


def norm_loss(tn, output_inds, target=2.0):
    return (tenet.norm(tn.contract(output_inds=output_inds).data) - target) ** 2


def test_tn_optimizer_u1_decreases_the_loss(quimb_set_params_forwards_the_return):
    use_jax()
    ts = u1_chain(3)
    tn = network(ts)
    inds = chain_inds(3)
    opt = qtn.TNOptimizer(
        tn,
        loss_fn=lambda tn: norm_loss(tn, inds),
        autodiff_backend="jax",
        progbar=False,
    )
    out = opt.optimize(20)
    assert min(opt.losses) < opt.losses[0]
    assert norm_loss(out, inds) < norm_loss(tn, inds)
    for before, after in zip(tn.tensors, out.tensors, strict=True):
        assert isinstance(after.data, SymmetricTensor)
        assert after.data.structure == before.data.structure
        # __post_init__ is the trust boundary: re-running it must still pass
        SymmetricTensor(after.data.structure, after.data.blocks)


def test_tn_optimizer_runs_on_su2(quimb_set_params_forwards_the_return):
    """The repository rule: every optimization claim is exercised non-Abelian too."""
    use_jax()
    ts = su2_chain()
    tn = network(ts)
    inds = chain_inds(4)
    opt = qtn.TNOptimizer(
        tn,
        loss_fn=lambda tn: norm_loss(tn, inds),
        autodiff_backend="jax",
        progbar=False,
    )
    out = opt.optimize(3)
    assert min(opt.losses) < opt.losses[0]
    for before, after in zip(tn.tensors, out.tensors, strict=True):
        assert after.data.structure == before.data.structure
        SymmetricTensor(after.data.structure, after.data.blocks)


# --- fuse: autoray's convention ---------------------------------------------------


def test_do_fuse_multi_group_is_refused_in_our_own_voice():
    t = su2_chain()[1]
    with pytest.raises(ValueError, match=r"fuse.*reshape in disguise"):
        ar.do("fuse", t, (0,), (1, 2))


def test_do_fuse_single_group_still_matches_tenet_fuse():
    t = u1_chain(1)[0]
    assert ar.do("fuse", t, (0, 1)) == tenet.fuse(t, (0, 1))


# --- reshape: refused, in our own voice -------------------------------------------


def test_do_reshape_names_fuse_and_unfuse():
    t = u1_chain(1)[0]
    with pytest.raises(ValueError, match=r"reshape by shape is not defined"):
        ar.do("reshape", t, (2, 4))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda qt: qt.norm(), r"reshape by shape is not defined"),
        # ``Tensor.split`` matricises first, so it meets the multi-group ``fuse``
        # refusal before it ever reaches ``reshape``; both are ours, both name fuse.
        (lambda qt: qt.split(("b0",)), r"fuse.*reshape in disguise"),
    ],
    ids=["norm", "split"],
)
def test_quimb_shape_based_operations_surface_our_refusal(call, message):
    qt = qtn.Tensor(data=u1_chain(1)[0], inds=("b0", "p0", "b1"))
    with pytest.raises(ValueError, match=message):
        call(qt)


# --- the two *correct* refusals quimb can walk into --------------------------------


def test_full_scalar_contraction_is_refused():
    a = SymmetricTensor.random((Leg(U1_BOND, OUT), Leg(U1_PHYS, IN)), seed=0)
    b = SymmetricTensor.random((Leg(U1_BOND, IN), Leg(U1_PHYS, OUT)), seed=1)
    tn = qtn.TensorNetwork(
        [qtn.Tensor(data=a, inds=("i", "j")), qtn.Tensor(data=b, inds=("i", "j"))]
    )
    with pytest.raises(ValueError, match="leaves no free leg"):
        tn.contract()


def test_output_inds_dropping_an_index_is_refused():
    qt = qtn.Tensor(data=u1_chain(1)[0], inds=("b0", "p0", "b1"))
    with pytest.raises(ValueError, match="summing that axis away"):
        qtn.tensor_contract(qt, output_inds=("b0", "b1"))


def test_scalars_are_spelled_with_boundary_legs():
    """The documented pattern in place of a rank-0 tensor: contract down to the
    boundary legs, which for an MPS are trivial-sector legs of dimension 1, so the
    "scalar" is a rank-4 tensor holding one 1x1x1x1 block. Nothing is lost.
    """
    ts = su2_chain()
    ket = network(ts)
    bra = network([tenet.adjoint(t) for t in ts], prefix="c")
    out = (ket | bra).contract(output_inds=("b0", "b4", "c0", "c4"))
    assert out.shape == (1, 1, 1, 1)
    dense = dense_chain(ts)
    assert np.allclose(out.data.to_dense().reshape(()), np.vdot(dense, dense))


# --- quimb stays outside src/ ------------------------------------------------------


def test_src_never_imports_quimb_or_cotengra():
    src = pathlib.Path(tenet.__file__).parent
    for path in src.rglob("*.py"):
        # docstrings name quimb freely — the ban is on the import, not the word
        text = path.read_text()
        for banned in ("quimb", "cotengra"):
            assert f"import {banned}" not in text, path
            assert f"from {banned}" not in text, path
