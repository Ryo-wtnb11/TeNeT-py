import tenet
import tenet.symmetry


def test_import():
    assert tenet.__version__


def test_the_provider_api_surface():
    """Every provider is reachable from ``tenet.symmetry`` and listed in its ``__all__``.

    ``Z2``/``Z2Provider``/``Z2Sector`` are #104's three new names; ``Z2_GAUGE`` is
    deliberately absent, because bosonic Z2 pins no coefficient convention (every F, R, B
    and FS is ``+1``), exactly as ``U1`` and ``Trivial`` pin none.
    """
    exported = set(tenet.symmetry.__all__)
    assert {
        "Trivial",
        "TrivialProvider",
        "TrivialSector",
        "U1",
        "U1Provider",
        "U1Sector",
        "SU2",
        "SU2Provider",
        "SU2Sector",
        "fZ2",
        "FZ2Provider",
        "FZ2Sector",
        "Z2",
        "Z2Provider",
        "Z2Sector",
    } <= exported
    assert "Z2_GAUGE" not in exported
    for name in exported:
        assert hasattr(tenet.symmetry, name), name


def test_no_public_namespace_exports_one_object_under_two_names():
    """One name per thing — #120's rule, made mechanical by #185.

    The four M24a aliases (``ClebschGordan``, ``QuantumDimension``,
    ``RecouplingData``, ``MultiplicityRecoupling``) were each a second name
    bound by ``is`` to an existing protocol, and they survived a milestone past
    the one that promised to remove them. This asserts by identity rather than
    by spelling, so a re-introduced alias fails here on the day it lands, not
    on the day someone reads the module again.
    """
    import tenet.network
    import tenet.serialize

    for module in (tenet, tenet.network, tenet.symmetry, tenet.linalg, tenet.serialize):
        seen: dict[int, str] = {}
        for name in module.__all__:
            obj = getattr(module, name)
            first = seen.setdefault(id(obj), name)
            assert first == name, (
                f"{module.__name__}.__all__ exports one object under two names: "
                f"{first!r} and {name!r} are the same object"
            )
