"""Keyed block construction and replacement — one test per acceptance criterion of #208.

Reading a tensor's blocks is keyed (``items``) and so is writing them
(``from_blocks``, ``with_blocks``). The non-Abelian case is the motivating one, so the
round trip runs on SU(2) and SU(3)
as well as an Abelian provider — for a non-Abelian symmetry the reduced block per
fusion tree is the natural datum and the dense array is the derived object.
"""

import dataclasses

import numpy as np
import pytest

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure, map_layout
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector
from tenet.symmetry.sun import SUNProvider, SUNSector

SU3 = SUNProvider(3)

Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
V = GradedSpace.new(SU2, {SU2Sector(1): 2, SU2Sector(2): 1})
T3 = GradedSpace.new(SU3, {SUNSector((1, 0)): 2, SUNSector((0, 1)): 1})

U1_LEGS = (Leg(Q, OUT), Leg(Q, IN), Leg(Q, OUT))
SU2_LEGS = (Leg(V, OUT), Leg(V, IN), Leg(V, OUT))
SU3_LEGS = (Leg(T3, OUT), Leg(T3, OUT), Leg(T3, IN))

PROVIDERS = pytest.mark.parametrize("legs", [U1_LEGS, SU2_LEGS, SU3_LEGS], ids=["u1", "su2", "su3"])


def ones_for(structure, key):
    return np.ones(structure.block_shape(key))


# --- keyed construction ---------------------------------------------------------


@PROVIDERS
def test_from_blocks_names_one_key_without_building_a_zero_tensor_first(legs):
    """The layout is read off the structure, not off a throwaway tensor's blocks."""
    structure = TensorStructure(legs)
    assert structure.num_blocks > 1  # otherwise "the rest are zero" says nothing
    key = structure.block_order[0]

    t = SymmetricTensor.from_blocks(legs, {key: ones_for(structure, key)})

    assert t.structure == structure
    assert t.legs == legs
    np.testing.assert_array_equal(t.block(key), ones_for(structure, key))
    for other, block in t.items():
        if other != key:
            assert not np.asarray(block).any(), "an absent key is filled with zeros"


@PROVIDERS
def test_from_blocks_round_trips_through_items(legs):
    structure = TensorStructure(legs)
    rng = np.random.default_rng(0)
    want = {k: rng.standard_normal(structure.block_shape(k)) for k in structure.block_order}

    t = SymmetricTensor.from_blocks(legs, want)

    got = dict(t.items())
    assert got.keys() == want.keys()
    for key in want:
        np.testing.assert_array_equal(got[key], want[key])


@PROVIDERS
def test_from_blocks_with_every_key_holds_the_supplied_blocks_in_block_order(legs):
    structure = TensorStructure(legs)
    blocks = [ones_for(structure, k) for k in structure.block_order]
    keyed = SymmetricTensor.from_blocks(legs, dict(zip(structure.block_order, blocks, strict=True)))
    # value for value and in block order: the supplied blocks are gathered into the
    # tensor's coupled-sector matrices, and ``blocks`` is the view back out of them
    assert all(np.array_equal(a, b) for a, b in zip(keyed.blocks, blocks, strict=True))


def test_from_blocks_takes_dtype_and_backend_from_the_supplied_blocks():
    structure = TensorStructure(SU2_LEGS)
    key = structure.block_order[0]
    t = SymmetricTensor.from_blocks(SU2_LEGS, {key: ones_for(structure, key).astype(np.complex64)})
    assert t.dtype == np.complex64
    assert all(b.dtype == np.complex64 for b in t.blocks)
    assert t.backend == "numpy"


def test_from_blocks_zero_fill_lands_on_the_supplied_blocks_backend():
    pytest.importorskip("jax")
    structure = TensorStructure(SU2_LEGS)
    key = structure.block_order[0]
    block = (
        SymmetricTensor.from_blocks(SU2_LEGS, {key: ones_for(structure, key)})
        .to_backend("jax")
        .block(key)
    )
    t = SymmetricTensor.from_blocks(SU2_LEGS, {key: block})
    assert t.backend == "jax"
    assert all(b.dtype == np.float64 for b in t.blocks)


def test_from_blocks_refuses_an_empty_mapping_and_names_zeros():
    with pytest.raises(ValueError, match=r"SymmetricTensor\.zeros"):
        SymmetricTensor.from_blocks(SU2_LEGS, {})


# --- keyed replacement ----------------------------------------------------------


@PROVIDERS
def test_with_blocks_replaces_the_named_and_carries_the_rest(legs):
    t = SymmetricTensor.random(legs, seed=3)
    key = t.structure.block_order[0]
    new = ones_for(t.structure, key)

    u = t.with_blocks({key: new})

    assert u.structure == t.structure
    np.testing.assert_array_equal(u.block(key), new)
    for other, block in u.items():
        if other != key:
            np.testing.assert_array_equal(block, t.block(other))


def test_with_blocks_leaves_the_original_alone_and_an_empty_mapping_is_a_no_op():
    t = SymmetricTensor.random(SU2_LEGS, seed=3)
    key = t.structure.block_order[0]
    before = np.array(t.block(key))

    t.with_blocks({key: ones_for(t.structure, key)})

    np.testing.assert_array_equal(t.block(key), before)
    assert t.with_blocks({}) == t


@PROVIDERS
def test_with_blocks_round_trips_through_items(legs):
    t = SymmetricTensor.zeros(legs)
    rng = np.random.default_rng(1)
    want = {k: rng.standard_normal(t.structure.block_shape(k)) for k in t.structure.block_order}

    got = dict(t.with_blocks(want).items())

    assert got.keys() == want.keys()
    for key in want:
        np.testing.assert_array_equal(got[key], want[key])


# --- refusals -------------------------------------------------------------------


def foreign_key():
    """A well-formed key of a *different* structure, which no U(1) tensor here owns."""
    return TensorStructure(SU2_LEGS).block_order[0]


@pytest.mark.parametrize("build", ["from_blocks", "with_blocks"])
def test_a_foreign_key_raises_naming_the_legal_keys(build):
    structure = TensorStructure(U1_LEGS)
    blocks = {foreign_key(): np.ones((1, 1, 1))}
    call = (
        (lambda: SymmetricTensor.from_blocks(U1_LEGS, blocks))
        if build == "from_blocks"
        else (lambda: SymmetricTensor.zeros(U1_LEGS).with_blocks(blocks))
    )
    with pytest.raises(KeyError) as excinfo:
        call()
    message = str(excinfo.value)
    assert "block_order" in message, "the message must say where the legal keys live"
    assert str(structure.block_order[0]) in message, "and name some of them"


@pytest.mark.parametrize("build", ["from_blocks", "with_blocks"])
def test_a_wrong_shaped_block_raises_naming_the_expected_shape(build):
    structure = TensorStructure(SU2_LEGS)
    key = structure.block_order[0]
    expected = structure.block_shape(key)
    blocks = {key: np.ones(tuple(d + 1 for d in expected))}
    call = (
        (lambda: SymmetricTensor.from_blocks(SU2_LEGS, blocks))
        if build == "from_blocks"
        else (lambda: SymmetricTensor.zeros(SU2_LEGS).with_blocks(blocks))
    )
    with pytest.raises(ValueError, match=rf"expected {re_escape(expected)}"):
        call()


def re_escape(shape):
    return r"\(" + r", ".join(str(d) for d in shape) + r"\)"


# --- the contracts the new spellings must not touch -----------------------------


@PROVIDERS
def test_the_result_is_a_plain_frozen_symmetric_tensor(legs):
    structure = TensorStructure(legs)
    key = structure.block_order[0]
    for t in (
        SymmetricTensor.from_blocks(legs, {key: ones_for(structure, key)}),
        SymmetricTensor.zeros(legs).with_blocks({key: ones_for(structure, key)}),
    ):
        assert type(t) is SymmetricTensor
        assert dataclasses.is_dataclass(t)
        assert isinstance(t.blocks, tuple)
        with pytest.raises(dataclasses.FrozenInstanceError):
            t._data = ()


def test_the_result_is_still_a_jax_pytree():
    jax = pytest.importorskip("jax")
    pytest.importorskip("tenet.pytree")
    import tenet.pytree  # noqa: F401  # registration is the import's side effect

    structure = TensorStructure(SU2_LEGS)
    key = structure.block_order[0]
    t = SymmetricTensor.from_blocks(SU2_LEGS, {key: ones_for(structure, key)})

    leaves, treedef = jax.tree_util.tree_flatten(t)
    assert len(leaves) == len(map_layout(structure).sectors)
    assert jax.tree_util.tree_unflatten(treedef, leaves) == t
