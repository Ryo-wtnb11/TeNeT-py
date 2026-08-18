"""``tenet.save`` / ``tenet.load`` over a single ``.npz`` — issue #94.

Two things carry the file's correctness and both are asserted here: the provider
``name`` field (which is part of provider equality, so a bare registry string
would silently change a loaded tensor's structure) and the SU(2)/fZ2 gauge
fingerprints (which pin the coefficient convention the blocks were computed in).
Everything else is `np.savez` doing its job.
"""

import json
import pathlib
import zipfile

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.serialize import _LEGACY_GAUGES, FORMAT_VERSION
from tenet.symmetry import (
    SU2,
    U1,
    Z2,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Provider,
    U1Sector,
    Z2Sector,
    fZ2,
)
from tenet.symmetry.fz2 import _FZ2_GAUGE
from tenet.symmetry.su2 import _SU2_GAUGE

UU = ProductProvider((U1, U1))
NESTED = ProductProvider((UU, SU2))


def uu(a: int, b: int) -> ProductSector:
    return ProductSector((U1Sector(a), U1Sector(b)))


def nested(a: int, b: int, j: int) -> ProductSector:
    return ProductSector((uu(a, b), SU2Sector(j)))


SPACES = {
    "trivial": GradedSpace.new(Trivial, {TrivialSector(): 3}),
    "u1": GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1}),
    "z2": GradedSpace.new(Z2, {Z2Sector(0): 2, Z2Sector(1): 3}),
    "su2": GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 3}),
    "fz2": GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 3}),
    "product": GradedSpace.new(UU, {uu(0, 0): 2, uu(1, 0): 2, uu(0, 1): 1}),
    "nested": GradedSpace.new(NESTED, {nested(0, 0, 0): 2, nested(1, 0, 1): 1}),
}
PROVIDERS = list(SPACES)


def tensor(name: str, seed: int = 0) -> SymmetricTensor:
    """Rank 4, mixed sides, a dual leg, and all three supported name kinds."""
    v = SPACES[name]
    legs = (
        Leg(v, OUT, name="bond"),
        Leg(v, OUT, True),
        Leg(v, IN, name=3),
        Leg(v, IN, True, name=None),
    )
    return SymmetricTensor.random(legs, seed=seed)


def header_of(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(z["header"].item())


def rewrite_header(path, header: dict) -> None:
    """Rewrite just the ``header`` member of an existing ``.npz``.

    ``np.savez`` writes each array as a ``.npy`` member, so the file is rebuilt
    member-for-member with the header replaced and every block copied verbatim.
    """
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "header"}
    np.savez(path, header=np.array(json.dumps(header)), **arrays)


# --- round trip -----------------------------------------------------------------


def test_format_version_is_one():
    """Asserted directly so bumping it is a deliberate edit, never a drift."""
    assert FORMAT_VERSION == 1


@pytest.mark.parametrize("name", PROVIDERS)
def test_round_trip_is_exact(name, tmp_path):
    t = tensor(name)
    tenet.save(t, tmp_path / "t.npz")
    loaded = tenet.load(tmp_path / "t.npz")
    assert loaded == t  # __eq__: structure equality plus exact block equality
    assert loaded.structure is not t.structure
    assert loaded.structure == t.structure
    assert hash(loaded.structure) == hash(t.structure)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_round_trip_over_ranks_and_sides(rank, tmp_path):
    v = SPACES["su2"]
    sides = (OUT, IN, OUT, IN)[:rank]
    legs = tuple(Leg(v, s, i % 2 == 1) for i, s in enumerate(sides))
    t = SymmetricTensor.random(legs, seed=rank)
    tenet.save(t, tmp_path / "t.npz")
    assert tenet.load(tmp_path / "t.npz") == t


def test_round_trip_of_a_single_block_structure(tmp_path):
    v = GradedSpace.new(U1, {U1Sector(0): 2})
    t = SymmetricTensor.random((Leg(v, OUT), Leg(v, IN)), seed=1)
    assert t.structure.num_blocks == 1
    tenet.save(t, tmp_path / "t.npz")
    assert tenet.load(tmp_path / "t.npz") == t


def test_provider_name_field_round_trips(tmp_path):
    """``name`` is a dataclass field, so it is part of provider identity.

    Measured: ``U1Provider(name="q") == U1`` is ``False``. Writing only the
    registry string ``"U1"`` and rebuilding ``U1Provider()`` would therefore have
    silently *changed the tensor's structure* across a save/load — the loaded
    tensor would compare unequal to the original and refuse to ``add`` to it.
    """
    assert U1Provider(name="q") != U1
    charge = U1Provider(name="charge")
    v = GradedSpace.new(charge, {U1Sector(0): 2, U1Sector(1): 1})
    t = SymmetricTensor.random((Leg(v, OUT), Leg(v, IN)), seed=2)
    tenet.save(t, tmp_path / "t.npz")
    loaded = tenet.load(tmp_path / "t.npz")
    assert loaded.provider == charge
    assert loaded.provider != U1
    assert loaded == t


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.int64])
def test_dtypes_round_trip_exactly(dtype, tmp_path):
    t = tensor("su2")
    t = t.set_params([(b * 10).astype(dtype) for b in t.blocks])
    tenet.save(t, tmp_path / "t.npz")
    loaded = tenet.load(tmp_path / "t.npz")
    for a, b in zip(loaded.blocks, t.blocks, strict=True):
        assert a.dtype == b.dtype == dtype
        assert np.array_equal(a, b)


def test_empty_block_structure_round_trips(tmp_path):
    """No blocks: no ``b{i}`` members, and ``_first_block`` is never touched."""
    v = GradedSpace.new(U1, {U1Sector(1): 2})
    w = GradedSpace.new(U1, {U1Sector(0): 2})
    t = SymmetricTensor((tenet.TensorStructure((Leg(v, OUT), Leg(w, IN)))), ())
    assert t.blocks == ()
    tenet.save(t, tmp_path / "t.npz")
    with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
        assert z.files == ["header"]
    loaded = tenet.load(tmp_path / "t.npz")
    assert loaded.blocks == ()
    assert loaded == t


# --- backends -------------------------------------------------------------------


def test_jax_backed_tensor_saves_and_restores(tmp_path):
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    t = tensor("su2").to_backend("jax")
    tenet.save(t, tmp_path / "t.npz")
    loaded = tenet.load(tmp_path / "t.npz")
    assert loaded == t.to_backend("numpy")
    restored = loaded.to_backend("jax")
    assert restored == t


def test_serialize_module_imports_no_backend():
    src = pathlib.Path(tenet.serialize.__file__).read_text()
    assert "import jax" not in src
    assert "import torch" not in src


# --- the header schema ----------------------------------------------------------


def test_header_holds_nothing_derived(tmp_path):
    t = tensor("su2")
    tenet.save(t, tmp_path / "t.npz")
    header = header_of(tmp_path / "t.npz")
    assert set(header) == {"format", "tenet", "legs", "gauges", "num_blocks"}
    assert header["num_blocks"] == t.structure.num_blocks
    text = json.dumps(header)
    for derived in ("block_order", "block_shape", "coupled", "inner", "tree"):
        assert derived not in text


def test_header_records_a_gauge_per_kind_that_has_one(tmp_path):
    tenet.save(tensor("nested"), tmp_path / "t.npz")
    # (U(1) x U(1)) x SU(2): only SU(2) has a gauge fingerprint, and it is recorded
    # even though it sits inside a nested product.
    assert header_of(tmp_path / "t.npz")["gauges"] == {"SU2": _SU2_GAUGE}
    tenet.save(tensor("u1"), tmp_path / "u.npz")
    assert header_of(tmp_path / "u.npz")["gauges"] == {}
    tenet.save(tensor("fz2"), tmp_path / "f.npz")
    assert header_of(tmp_path / "f.npz")["gauges"] == {"fZ2": _FZ2_GAUGE}


# --- refusals -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kind", "gauge"),
    [("su2", "SU2", _SU2_GAUGE), ("fz2", "fZ2", _FZ2_GAUGE)],
)
def test_mutated_gauge_is_refused(name, kind, gauge, tmp_path):
    """A file written under a different coefficient convention is not readable.

    Every shape would match and every norm would look plausible; the physics would
    be silently gauge-rotated. So the string is compared, and both are named.
    """
    tenet.save(tensor(name), tmp_path / "t.npz")
    header = header_of(tmp_path / "t.npz")
    header["gauges"][kind] = gauge.replace("condon-shortley", "other") + ";mutated"
    rewrite_header(tmp_path / "t.npz", header)
    with pytest.raises(ValueError, match="different coefficient convention") as exc:
        tenet.load(tmp_path / "t.npz")
    assert gauge in str(exc.value)
    assert "mutated" in str(exc.value)


def test_the_one_legacy_su2_gauge_loads_and_a_fabricated_third_does_not(tmp_path):
    """#180 moved SU(2)'s coefficients to racah and the fingerprint moved with them.

    The two sets agree to 4.95e-14 over all 109,900 fixture rows, so blocks written
    under the old string are numerically comparable and refusing them would be a false
    alarm. That is one grandfathered string, not a relaxation: anything else, including
    a near-miss of the legacy string itself, is still refused.
    """
    original = tensor("su2")
    tenet.save(original, tmp_path / "t.npz")
    header = header_of(tmp_path / "t.npz")
    (legacy,) = _LEGACY_GAUGES["SU2"]
    header["gauges"]["SU2"] = legacy
    rewrite_header(tmp_path / "t.npz", header)
    assert tenet.load(tmp_path / "t.npz").structure == original.structure

    header["gauges"]["SU2"] = legacy.replace("tks-su2irrep", "fabricated-su2irrep")
    rewrite_header(tmp_path / "t.npz", header)
    with pytest.raises(ValueError, match="different coefficient convention"):
        tenet.load(tmp_path / "t.npz")


def test_future_format_version_is_refused(tmp_path):
    tenet.save(tensor("u1"), tmp_path / "t.npz")
    header = header_of(tmp_path / "t.npz")
    header["format"] = FORMAT_VERSION + 1
    rewrite_header(tmp_path / "t.npz", header)
    with pytest.raises(ValueError, match=f"format {FORMAT_VERSION + 1} is newer") as exc:
        tenet.load(tmp_path / "t.npz")
    assert str(FORMAT_VERSION) in str(exc.value)


def test_wrong_num_blocks_is_refused_before_any_block_is_read(tmp_path):
    tenet.save(tensor("u1"), tmp_path / "t.npz")
    header = header_of(tmp_path / "t.npz")
    header["num_blocks"] += 1
    rewrite_header(tmp_path / "t.npz", header)
    with pytest.raises(ValueError, match="different block enumeration"):
        tenet.load(tmp_path / "t.npz")


def test_missing_block_member_is_refused(tmp_path):
    t = tensor("u1")
    tenet.save(t, tmp_path / "t.npz")
    with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "header"}
        header = json.loads(z["header"].item())
    del arrays[f"b{len(arrays) - 1}"]
    np.savez(tmp_path / "t.npz", header=np.array(json.dumps(header)), **arrays)
    with pytest.raises(ValueError, match="missing block member"):
        tenet.load(tmp_path / "t.npz")


def test_extra_member_is_refused(tmp_path):
    t = tensor("u1")
    tenet.save(t, tmp_path / "t.npz")
    with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "header"}
        header = json.loads(z["header"].item())
    arrays["stowaway"] = np.zeros(3)
    np.savez(tmp_path / "t.npz", header=np.array(json.dumps(header)), **arrays)
    with pytest.raises(ValueError, match=r"unexpected members \['stowaway'\]"):
        tenet.load(tmp_path / "t.npz")


def test_wrong_block_shape_is_refused_by_post_init(tmp_path):
    """Not re-implemented in serialize.py: ``__post_init__`` already names the block."""
    t = tensor("u1")
    tenet.save(t, tmp_path / "t.npz")
    with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "header"}
        header = json.loads(z["header"].item())
    arrays["b0"] = np.zeros((1, 1, 1, 1))
    np.savez(tmp_path / "t.npz", header=np.array(json.dumps(header)), **arrays)
    with pytest.raises(ValueError, match=r"block 0 has shape .* expected"):
        tenet.load(tmp_path / "t.npz")


def test_unknown_provider_kind_is_refused(tmp_path):
    tenet.save(tensor("u1"), tmp_path / "t.npz")
    header = header_of(tmp_path / "t.npz")
    header["legs"][0]["space"]["provider"]["kind"] = "SO3"
    rewrite_header(tmp_path / "t.npz", header)
    with pytest.raises(KeyError, match="unknown provider kind"):
        tenet.load(tmp_path / "t.npz")


def test_object_dtype_member_hits_numpys_own_pickle_refusal(tmp_path):
    """``allow_pickle=False`` is passed explicitly; NumPy's refusal propagates."""
    t = tensor("u1")
    tenet.save(t, tmp_path / "t.npz")
    with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "header"}
        header = json.loads(z["header"].item())
    arrays["b0"] = np.array([object()], dtype=object)
    np.savez(tmp_path / "t.npz", header=np.array(json.dumps(header)), **arrays)
    with pytest.raises(ValueError, match="allow_pickle=False"):
        tenet.load(tmp_path / "t.npz")


def test_unsupported_leg_name_is_refused_at_save_time_and_no_file_is_written(tmp_path):
    v = SPACES["u1"]
    t = SymmetricTensor.random((Leg(v, OUT), Leg(v, IN, name=("a", "b"))), seed=1)
    path = tmp_path / "t.npz"
    with pytest.raises(TypeError, match=r"public axis 1: leg name \('a', 'b'\)"):
        tenet.save(t, path)
    assert not path.exists()


def test_unserializable_provider_is_refused_by_name(tmp_path):
    class SO3Provider(U1Provider):
        pass

    v = GradedSpace(SO3Provider(name="SO3"), ((U1Sector(0), 2),))
    t = SymmetricTensor.random((Leg(v, OUT), Leg(v, IN)), seed=1)
    with pytest.raises(ValueError, match="cannot serialize provider"):
        tenet.save(t, tmp_path / "t.npz")


# --- compression ----------------------------------------------------------------


def compress_types(path) -> set[int]:
    with zipfile.ZipFile(path) as z:
        return {info.compress_type for info in z.infolist()}


def test_default_is_uncompressed_and_compress_true_is_smaller(tmp_path):
    """Container overhead, measured, so nobody later reads it as a bug: a 4-block
    SU(2) tensor with 504 bytes of block data writes a 2282-byte file."""
    t = tensor("su2")
    tenet.save(t, tmp_path / "plain.npz")
    tenet.save(t, tmp_path / "zipped.npz", compress=True)
    assert compress_types(tmp_path / "plain.npz") == {zipfile.ZIP_STORED}
    assert compress_types(tmp_path / "zipped.npz") == {zipfile.ZIP_DEFLATED}
    assert (tmp_path / "zipped.npz").stat().st_size <= (tmp_path / "plain.npz").stat().st_size
    assert tenet.load(tmp_path / "zipped.npz") == t


# --- delegations ----------------------------------------------------------------


def test_method_delegations(tmp_path):
    t = tensor("su2")
    t.save(tmp_path / "t.npz")
    assert SymmetricTensor.load(tmp_path / "t.npz") == t
    t.save(tmp_path / "c.npz", compress=True)
    assert SymmetricTensor.load(tmp_path / "c.npz") == t
