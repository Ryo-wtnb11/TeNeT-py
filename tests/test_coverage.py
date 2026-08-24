"""The two mechanical guards of the coverage policy (issue #146, `tests/COVERAGE.md`).

Everything else the policy asks for is review discipline; these two are greps.
E1 is the failure that has happened three times — `flip_dual` (#142), `full_trace` and
`inner` (#126) each landed in `tenet.__all__` after #95's torch pass and never
joined it. E2 is #145's gap turned into an invariant: the AD suite must keep a
provider whose F/R/B symbols are matrix-valued, or `grad x SU(N)` silently stops
running again.
"""

import inspect
import pathlib
import re
import sys

import tenet
from tenet.symmetry.base import BMatrixData, FMatrixData, RMatrixData

TESTS = pathlib.Path(__file__).parent

# Public callables that legitimately never touch a torch block, with the reason.
NOT_ON_TORCH = {
    "as_map": "a zero-copy structural view over the tensor's own blocks; no backend kernel",
    "coupled_sectors": "pure sector combinatorics over legs; consumes no array blocks",
    "enable_jax": (
        "the JAX opt-in seam (pytree registration, and tenet.ad's broadened VJPs); "
        "eager torch needs no registration of its own, so there is nothing to run "
        "on a torch block -- tests/test_enable_jax.py is its suite"
    ),
    "fusion_trees": "pure sector combinatorics; enumerates trees, consumes no array blocks",
    "map_layout": "static layout metadata computed from the structure alone; no array blocks",
}


def test_e1_every_public_operation_reaches_the_torch_suite():
    src = (TESTS / "backends" / "test_torch.py").read_text()
    public = {n: getattr(tenet, n) for n in tenet.__all__}
    ops = [n for n, o in public.items() if callable(o) and not isinstance(o, type)]
    for name in ops:
        if name in NOT_ON_TORCH:
            assert NOT_ON_TORCH[name].strip(), f"NOT_ON_TORCH[{name!r}] needs a reason"
        else:
            assert re.search(rf"\b{name}\b", src), (
                f"tenet.{name} is public but never appears in tests/backends/test_torch.py; "
                f"add it there or to NOT_ON_TORCH with a reason (tests/COVERAGE.md)"
            )
    stale = set(NOT_ON_TORCH) - set(ops)
    assert not stale, f"NOT_ON_TORCH names nothing public: {sorted(stale)}"


def test_e2_the_ad_suite_carries_a_multiplicity_bearing_provider():
    src = (TESTS / "backends" / "test_ad.py").read_text()
    square = src.split("SQUARE = {", 1)[1].split("}", 1)[0]
    assert '"su3"' in square, "test_ad.py's SQUARE lost its multiplicity-bearing column"
    sys.path.insert(0, str(TESTS / "symmetry"))
    from _su3_fixture import SU3  # noqa: PLC0415  # the provider that column instantiates

    for capability in (FMatrixData, RMatrixData, BMatrixData):
        assert isinstance(SU3, capability)


# E3 (#167): the docstring standard's mechanical half. Scoped to stages 1+2+3+4 —
# every class in tenet.__all__, the tenet.__all__ functions defined in the
# converted modules (ops, fusion_tree, map_view), tenet.linalg.__all__, all of
# tenet.network.__all__, and (stage 4) tenet.symmetry.__all__,
# tenet.serialize.__all__ and tenet.ad's install/uninstall.
# Stage-4 decisions, recorded:
# - A capability *protocol* has no constructor, so the class itself is not
#   checked for Parameters; its public method stubs ARE — they are the provider
#   contract, documented once here and inherited by every provider whose method
#   carries no docstring of its own.
# - Concrete provider classes whose only field is the identity label ``name``
#   enter NO_PARAMS: users take the module singleton, never the constructor.
# - tenet.pytree exports nothing public (empty __all__; registration is the
#   import's side effect), so it adds no names — and importing it here would
#   register the pytree node as a test side effect, so it is not imported.
# A NamedTuple documents its fields in an Attributes section, not a Parameters
# one — griffe warns (fatal under mkdocs --strict) on a Parameters entry absent
# from the signature, and it does not synthesize a NamedTuple __init__ — so
# every NamedTuple in the guard enters NO_PARAMS below, with the section that
# does carry its field docs named in the reason.
# Public names that legitimately carry no Parameters section, with the reason.
NO_PARAMS: dict[str, str] = {
    "tenet.Side": "an Enum; the members OUT/IN are the API, there is no constructor",
    "tenet.StructureChangingError": (
        "an exception raised by the library; it has no introspectable signature "
        "and users never construct it"
    ),
    "tenet.network.CheckerboardLattice": (
        "the 2x2 bipartite geometry takes no arguments at all -- its dims, sites and "
        "bonds are what the class name fixes"
    ),
    "tenet.network.Site": (
        "a NamedTuple of two coordinates; its fields are documented in the class "
        "docstring's Attributes section, which griffe accepts where a Parameters "
        "section on a synthesized __init__ would warn"
    ),
    "tenet.network.Bond": (
        "a NamedTuple of two sites; its fields are documented in the class "
        "docstring's Attributes section"
    ),
    "tenet.network.DoubleLayer": (
        "a NamedTuple holding a bra and a ket; its fields are documented in the "
        "class docstring's Attributes section"
    ),
    "tenet.network.CTM_out": (
        "a result record built by EnvCTM.iterate_, never by users; its fields are "
        "documented in the class docstring's Attributes section"
    ),
    "tenet.network.EnvLocal": (
        "a mutable record of eight optional tensors a move fills in; its fields are "
        "documented in the class docstring's Attributes section"
    ),
    "tenet.network.EnvLocalC4v": (
        "a mutable record of one corner and one edge a move fills in; its fields are "
        "documented in the class docstring's Attributes section"
    ),
    "tenet.network.EnvProjectors": (
        "a mutable record of eight optional projectors a move fills in; its fields are "
        "documented in the class docstring's Attributes section"
    ),
    "tenet.network.DMRG_out": (
        "a result record built by dmrg_, never by users; its fields are "
        "documented in the class docstring's Attributes section"
    ),
    "tenet.network.Sweep": (
        "a NamedTuple; its fields are documented in the class docstring's "
        "Attributes section, which griffe accepts where a Parameters section "
        "on a synthesized signature is a --strict failure"
    ),
    "tenet.symmetry.CapabilityError": (
        "an exception raised by the library; it has no introspectable signature "
        "and users never construct it"
    ),
    "tenet.symmetry.StructureChangingError": (
        "an exception raised by the library; it has no introspectable signature "
        "and users never construct it"
    ),
    "tenet.symmetry.Sector": "a field-less marker base; subclasses add the label fields",
    "tenet.symmetry.TrivialSector": (
        "the single field-less sector of the trivial symmetry; nothing to parameterize"
    ),
    "tenet.symmetry.TrivialProvider": (
        "constructed once as the module singleton Trivial; the `name` field is an "
        "identity label that participates in equality, not a configuration knob"
    ),
    "tenet.symmetry.U1Provider": (
        "constructed once as the module singleton U1; the `name` field is an "
        "identity label that participates in equality, not a configuration knob"
    ),
    "tenet.symmetry.Z2Provider": (
        "constructed once as the module singleton Z2; the `name` field is an "
        "identity label that participates in equality, not a configuration knob"
    ),
    "tenet.symmetry.FZ2Provider": (
        "constructed once as the module singleton fZ2; the `name` field is an "
        "identity label that participates in equality, not a configuration knob"
    ),
    "tenet.symmetry.SU2Provider": (
        "constructed once as the module singleton SU2; the `name` field is an "
        "identity label that participates in equality, not a configuration knob"
    ),
    "tenet.ad.uninstall": "takes no arguments; restores autoray's stock JAX bindings",
}


def _documented_names():
    # scoped with the guard that uses them
    from tenet import ad, linalg, network, serialize, symmetry  # noqa: PLC0415

    for name in tenet.__all__:
        obj = getattr(tenet, name)
        converted = ("tenet.ops", "tenet.fusion_tree", "tenet.map_view", "tenet.serialize")
        if inspect.isclass(obj) or (
            inspect.isfunction(obj) and obj.__module__.startswith(converted)
        ):
            yield f"tenet.{name}", obj
    for name in linalg.__all__:
        yield f"tenet.linalg.{name}", getattr(linalg, name)
    for name in network.__all__:
        obj = getattr(network, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            yield f"tenet.network.{name}", obj
    for name in symmetry.__all__:
        obj = getattr(symmetry, name)
        if inspect.isclass(obj) and getattr(obj, "_is_protocol", False):
            # A Protocol has no constructor; its public method stubs ARE the
            # provider contract, so each one is held to the Parameters standard.
            for attr, member in vars(obj).items():
                if not attr.startswith("_") and inspect.isfunction(member):
                    yield f"tenet.symmetry.{name}.{attr}", member
        elif inspect.isclass(obj) or inspect.isfunction(obj):
            yield f"tenet.symmetry.{name}", obj
    for name in serialize.__all__:
        obj = getattr(serialize, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            yield f"tenet.serialize.{name}", obj
    yield "tenet.ad.install", ad.install
    yield "tenet.ad.uninstall", ad.uninstall


def test_e3_every_documented_name_parameterizes_its_docstring():
    seen = set()
    for qualname, fn in _documented_names():
        seen.add(qualname)
        if qualname in NO_PARAMS:
            assert NO_PARAMS[qualname].strip(), f"NO_PARAMS[{qualname!r}] needs a reason"
            continue
        params = [
            p.name
            for p in inspect.signature(fn).parameters.values()
            if p.name not in ("self", "cls")
        ]
        doc = fn.__doc__ or ""
        section = re.search(
            r"^\s*Parameters\n\s*-+\n(.*?)(?=^\s*\w[\w ]*\n\s*-+\n|\Z)", doc, re.M | re.S
        )
        assert section, (
            f"{qualname} has no numpy-style Parameters section; add one, or add the name "
            f"to NO_PARAMS with a reason (#167)"
        )
        for p in params:
            # numpy style allows grouped entries ("a, b : Sector"), so the name
            # may sit after other comma-separated names on the entry line
            assert re.search(rf"^\s*(?:\*{{0,2}}\w+, *)*\*{{0,2}}{p}\b", section.group(1), re.M), (
                f"{qualname}'s Parameters section does not name its parameter {p!r}"
            )
    stale = set(NO_PARAMS) - seen
    assert not stale, f"NO_PARAMS names nothing under the guard: {sorted(stale)}"
