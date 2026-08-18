"""``save`` / ``load`` over a single ``.npz`` — issue #94.

The file is a JSON header describing the legs plus one ``.npy`` member per block.
Nothing derived is written: ``block_order``, ``block_shape`` and every fusion tree
are pure functions of ``legs``, so putting them on disk would create a second
source of truth. ``num_blocks`` is in the header only as a *read count*, and it is
checked against the freshly derived value.

NumPy is correct here and only here for the same reason it is in
``tenet.ops.dense``: the *file format* is NumPy's. Blocks arrive through
``ar.to_numpy`` and leave through the ordinary constructor, so no backend is ever
imported and a JAX- or torch-backed tensor saves fine. ``load`` always returns
NumPy blocks; ``load(path).to_backend("jax")`` is the documented restore, because
a device placement is not a property of a tensor.

The SU(2), SU(N) and fZ2 gauge fingerprints are written and **verified** on load. Block
coefficients are only meaningful against the CG / F / R conventions that produced
them, and a file outlives the process in which provider identity pins those
conventions; loading gauge-mismatched coefficients would be silently wrong in a
way no shape check catches.
"""

import dataclasses
import json
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import autoray as ar
import numpy as np

from tenet.leg import Leg, Side
from tenet.space import GradedSpace
from tenet.structure import TensorStructure
from tenet.symmetry.base import (
    FusionRules,
    Sector,
    TrivialProvider,
    TrivialSector,
    _DualFusionRules,
)
from tenet.symmetry.fz2 import _FZ2_GAUGE, FZ2Provider, FZ2Sector
from tenet.symmetry.product import ProductProvider, ProductSector
from tenet.symmetry.su2 import _SU2_GAUGE, SU2Provider, SU2Sector
from tenet.symmetry.sun import _SUN_GAUGE, SUNProvider, SUNSector
from tenet.symmetry.u1 import U1Provider, U1Sector
from tenet.symmetry.z2 import Z2Provider, Z2Sector

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["FORMAT_VERSION", "load", "save"]

FORMAT_VERSION = 1
"""On-disk format version. Files are not guaranteed readable across ``tenet``
versions before 1.0; a *future* version is refused loudly rather than misread."""

# The provider registry. ``name`` is a dataclass *field* on every non-product
# provider and participates in __eq__/__hash__ -- measured, ``U1Provider(name="q")
# == U1`` is False -- so writing a bare registry string would silently change a
# loaded tensor's structure. ProductProvider is the one recursive case; its
# ``name`` is a property derived from ``factors``, so only the factors are written.
_PROVIDERS: dict[str, Any] = {
    "Trivial": TrivialProvider,
    "U1": U1Provider,
    "SU2": SU2Provider,
    "fZ2": FZ2Provider,
    "Z2": Z2Provider,
}
_SECTORS: dict[str, Any] = {
    "Trivial": TrivialSector,
    "U1": U1Sector,
    "SU2": SU2Sector,
    "fZ2": FZ2Sector,
    "Z2": Z2Sector,
}
_KINDS: dict[type, str] = {
    TrivialProvider: "Trivial",
    U1Provider: "U1",
    SU2Provider: "SU2",
    FZ2Provider: "fZ2",
    Z2Provider: "Z2",
    ProductProvider: "Product",
}
_GAUGES: dict[str, str] = {"SU2": _SU2_GAUGE, "fZ2": _FZ2_GAUGE}

# racah is a core dependency since #180, so SU(N) registers unconditionally.
_PROVIDERS["SUN"] = SUNProvider
_SECTORS["SUN"] = SUNSector
_KINDS[SUNProvider] = "SUN"
_GAUGES["SUN"] = _SUN_GAUGE


# --- header encoding ---------------------------------------------------------


def _kind(provider: FusionRules) -> str:
    try:
        return _KINDS[type(provider)]
    except KeyError:
        raise ValueError(
            f"cannot serialize provider {provider!r}: {type(provider).__name__} is not one of "
            f"{sorted(_KINDS.values())}"
        ) from None


def _encode_provider(provider: FusionRules) -> dict[str, Any]:
    kind = _kind(provider)
    if kind == "Product":
        # kind == "Product" pins the concrete ProductProvider, which has factors
        return {
            "kind": kind,
            "factors": [_encode_provider(f) for f in provider.factors],  # ty: ignore[unresolved-attribute]
        }
    # Every dataclass field, not just ``name``: SUNProvider also carries ``n``, and a
    # provider reconstructed without it would be a different symmetry.
    # every concrete provider is a frozen dataclass; the protocol cannot say so
    fields = {f.name: getattr(provider, f.name) for f in dataclasses.fields(provider)}  # ty: ignore[invalid-argument-type]
    return {"kind": kind, **fields}


def _decode_provider(d: Any) -> _DualFusionRules:
    kind = d["kind"]
    if kind == "Product":
        return ProductProvider(tuple(_decode_provider(f) for f in d["factors"]))
    if kind not in _PROVIDERS:
        raise KeyError(f"unknown provider kind {kind!r}; known kinds are {sorted(_KINDS.values())}")
    return _PROVIDERS[kind](**{k: v for k, v in d.items() if k != "kind"})


def _encode_sector(a: Sector) -> list[Any]:
    """A sector's scalar fields, in field order; nested lists for a product."""
    if isinstance(a, ProductSector):
        return [_encode_sector(c) for c in a.components]
    return [getattr(a, f.name) for f in dataclasses.fields(a)]


def _decode_sector(provider: FusionRules, args: list[Any]) -> Sector:
    kind = _kind(provider)
    if kind == "Product":
        return ProductSector(
            # kind == "Product" pins ProductProvider, which has factors
            tuple(
                _decode_sector(f, x)
                for f, x in zip(provider.factors, args, strict=True)  # ty: ignore[unresolved-attribute]
            )
        )
    return _SECTORS[kind](*args)


def _encode_leg(leg: Leg, axis: int) -> dict[str, Any]:
    if not (leg.name is None or isinstance(leg.name, str | int)):
        raise TypeError(
            f"public axis {axis}: leg name {leg.name!r} of type {type(leg.name).__name__} "
            "cannot be written to JSON; only None, str and int names are supported. "
            "Rename it first: leg.renamed('bond')"
        )
    return {
        "side": leg.side.value,
        "dual": leg.dual,
        "name": leg.name,
        "space": {
            "provider": _encode_provider(leg.space.provider),
            "sectors": [[_encode_sector(a), m] for a, m in leg.space.sectors],
        },
    }


def _decode_leg(d: Any) -> Leg:
    provider = _decode_provider(d["space"]["provider"])
    # GradedSpace.new re-canonicalizes and re-checks the sector type against the
    # provider, so the loader gets #5's validator for free.
    space = GradedSpace.new(
        provider, [(_decode_sector(provider, a), m) for a, m in d["space"]["sectors"]]
    )
    return Leg(space, Side(d["side"]), bool(d["dual"]), d["name"])


def _gauges(providers: "Iterable[FusionRules]") -> dict[str, str]:
    """The gauge fingerprint of every kind present that has one, product factors included."""
    out: dict[str, str] = {}
    stack = list(providers)
    while stack:
        p = stack.pop()
        kind = _kind(p)
        if kind == "Product":
            # kind == "Product" pins the concrete ProductProvider, which has factors
            stack.extend(p.factors)  # ty: ignore[unresolved-attribute]
        elif kind in _GAUGES:
            out[kind] = _GAUGES[kind]
    return out


# --- public API --------------------------------------------------------------


def save(t: "SymmetricTensor", path: str | os.PathLike, *, compress: bool = False) -> None:
    """Write ``t`` to ``path`` as a single ``.npz``: a JSON header plus one array per block.

    Parameters
    ----------
    t : SymmetricTensor
        The tensor written; any backend. Blocks are converted with
        ``ar.to_numpy``, so a JAX- or torch-backed tensor saves fine.
    path : str or os.PathLike
        The destination file.
    compress : bool, optional
        ``False`` (the default) writes uncompressed; see Notes for why.

    Raises
    ------
    TypeError
        If a ``Leg``'s ``name`` is not ``None``, ``str`` or ``int`` — refused
        here, before anything is written, with the public axis named.
    ValueError
        If a leg's provider is not one of the serializable kinds.

    Examples
    --------
    >>> import os, tempfile
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> with tempfile.TemporaryDirectory() as d:
    ...     tenet.save(t, os.path.join(d, "t.npz"))
    ...     t2 = tenet.load(os.path.join(d, "t.npz"))
    >>> t2.structure == t.structure
    True
    >>> bool(tenet.allclose(t2, t))
    True

    Notes
    -----
    A saved tensor loads back as **NumPy** — ``load(...).to_backend("jax")`` is the
    documented restore, because a device placement is not a property of the tensor.

    ``compress=False`` by default: reduced blocks are dense float arrays that do
    not compress well, so paying ``zlib`` on every checkpoint buys nothing. The
    zip container costs a constant overhead — a 4-block SU(2) tensor with 504
    bytes of block data writes a 2282-byte file — which is not a bug.
    """
    import tenet

    header = {
        "format": FORMAT_VERSION,
        "tenet": tenet.__version__,
        "legs": [_encode_leg(leg, i) for i, leg in enumerate(t.legs)],
        "gauges": _gauges(leg.space.provider for leg in t.legs),
        "num_blocks": len(t.blocks),
    }
    writer = np.savez_compressed if compress else np.savez
    writer(
        path,
        header=np.array(json.dumps(header)),
        **{f"b{i}": ar.to_numpy(b) for i, b in enumerate(t.blocks)},
    )


def load(path: str | os.PathLike) -> "SymmetricTensor":
    """Read a tensor written by [save][tenet.save]. NumPy blocks; structure exactly equal.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.npz`` file to read.

    Returns
    -------
    SymmetricTensor
        The saved tensor, blocks NumPy-backed; ``load(path).to_backend("jax")``
        is the documented restore.

    Raises
    ------
    ValueError
        If — for SU(2), SU(N) or fZ2 — the file's gauge fingerprint is not the
        running build's: block coefficients are only meaningful against the
        CG / F / R conventions that produced them, so a gauge-mismatched file
        is refused rather than silently misread. Also for a future ``format``
        version, a header block count that contradicts the structure, or a
        member set that is not exactly the header plus ``b0..b{n-1}``.
    KeyError
        For an unknown provider kind.

    Examples
    --------
    See [save][tenet.save] for the round trip.

    Notes
    -----
    Block count and per-block shape are validated by
    ``SymmetricTensor.__post_init__``, unmodified.
    """
    from tenet.tensor import SymmetricTensor

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"].item()))
        version = header["format"]
        if version > FORMAT_VERSION:
            raise ValueError(
                f"{path}: file format {version} is newer than this tenet's format "
                f"{FORMAT_VERSION}; upgrade tenet to read it"
            )
        for kind, gauge in header["gauges"].items():
            running = _GAUGES.get(kind)
            if running is not None and gauge != running:
                raise ValueError(
                    f"{path}: {kind} coefficient gauge on disk is {gauge!r}, but this build "
                    f"runs {running!r}; the file was written under a different coefficient "
                    "convention and its block coefficients are not comparable"
                )
        structure = TensorStructure(tuple(_decode_leg(d) for d in header["legs"]))
        num_blocks = header["num_blocks"]
        if num_blocks != structure.num_blocks:
            raise ValueError(
                f"{path}: header says {num_blocks} blocks but the structure it describes has "
                f"{structure.num_blocks}; the file was written against a different block "
                "enumeration"
            )
        expected = {"header", *(f"b{i}" for i in range(num_blocks))}
        extra = sorted(set(z.files) - expected)
        if extra:
            raise ValueError(f"{path}: unexpected members {extra} in the archive")
        try:
            blocks = tuple(z[f"b{i}"] for i in range(num_blocks))
        except KeyError as exc:
            raise ValueError(f"{path}: missing block member {exc.args[0]!r}") from None
    return SymmetricTensor(structure, blocks)
