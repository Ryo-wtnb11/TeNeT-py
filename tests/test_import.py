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
