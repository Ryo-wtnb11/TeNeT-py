"""SU(2) F/R/B/FS symbols, cross-checked against the vendored TensorKitSectors fixtures.

Pentagon and unitarity are gauge-*covariant* and hold in every valid gauge
simultaneously, so only a full-table comparison can tell ours apart from a
plausible-looking convention slip.

Since #180 the coefficients come from racah at Dynkin label ``(two_j,)``, so the
fixtures are the cross-check rather than the specification: the two agree, but they
are two independent computations and the tolerance below says so.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from tenet.symmetry import (
    SU2,
    U1,
    AssociatorData,
    BraidingData,
    DualityData,
    FSIndicatorData,
    SU2Provider,
    SU2Sector,
    Trivial,
)
from tenet.symmetry.coherence import (
    validate_hexagon,
    validate_pentagon,
    validate_snake,
    validate_spherical,
    validate_unitary,
)
from tenet.symmetry.su2 import triangle

FIXTURES = Path(__file__).parent.parent / "fixtures"


# The doubled-spin entry points this file was written against. They used to be
# a private pure-Python module; that module is gone (#180) and the provider is the
# only coefficient surface left, so they are thin adapters over it.
def f_symbol(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
    return SU2.f_symbol(*(SU2Sector(x) for x in (a, b, c, d, e, f)))


def r_symbol(a: int, b: int, c: int) -> int:
    return SU2.r_symbol(SU2Sector(a), SU2Sector(b), SU2Sector(c))


def b_symbol(a: int, b: int, c: int) -> float:
    return SU2.b_symbol(SU2Sector(a), SU2Sector(b), SU2Sector(c))


def frobenius_schur(dj: int) -> int:
    return SU2.frobenius_schur(SU2Sector(dj))


@pytest.fixture(scope="session")
def f_table() -> tuple[np.ndarray, np.ndarray]:
    """``(dj[N, 6], fval[N])`` parsed once per session (109,900 rows)."""
    data = np.loadtxt(FIXTURES / "su2_f.txt")
    return data[:, :6].astype(int), data[:, 6]


@pytest.fixture(scope="session")
def r_table() -> tuple[np.ndarray, np.ndarray]:
    """``(dj[N, 3], rval[N])`` parsed once per session (616 rows)."""
    data = np.loadtxt(FIXTURES / "su2_r.txt", dtype=int)
    return data[:, :3], data[:, 3]


# --- provenance -------------------------------------------------------------


@pytest.mark.parametrize("name", ["su2_f.txt", "su2_r.txt"])
def test_fixture_header_names_tensorkitsectors_as_oracle(name: str) -> None:
    """A silent regeneration from a different oracle must show up in the diff."""
    header = [line for line in (FIXTURES / name).read_text().splitlines() if line.startswith("#")]
    assert len(header) == 9, "the nine racah provenance lines must be preserved verbatim"
    assert header[1] == "# oracle: TensorKitSectors (via TensorKit)"
    assert "# TensorKitSectors version: 0.3.6" in header
    assert "# generator: tools/gen_fr_fixtures.jl" in header


def test_fixture_row_counts(f_table, r_table) -> None:
    assert f_table[0].shape == (109_900, 6)
    assert r_table[0].shape == (616, 3)


# --- the tables themselves --------------------------------------------------


def test_f_symbol_matches_full_fixture_table(f_table) -> None:
    """The bound is 1e-13, deliberately looser than the 1e-14 this asserted before #180.

    Two independent computations of the same gauge cannot be compared at bit level:
    over the first 3000 rows only 7.2% of values are exactly equal, and the fixture
    text carries its own rounding (``su2_f.txt``:11 stores 1.0000000000000002 for a
    value of 1). The measured worst row is
    ``dj = (12, 8, 12, 12, 6, 10)`` at 4.95e-14 — racah 0.40004034223679297 against
    fixture 0.40004034223674345 — so 1e-13 is the first round bound above what the
    numbers support, not a tolerance chosen to make a failure go away.

    The loop iterates the fixture **in file order** and must keep doing so: racah's
    internal coefficient caches are locality-sensitive, and a shuffled or set-ordered
    rewrite of this loop costs ~780 s against 4 s here.
    """
    dj, expected = f_table
    got = np.fromiter((f_symbol(*row) for row in map(tuple, dj.tolist())), float, len(dj))
    dev = np.abs(got - expected)
    worst = int(dev.argmax())
    assert dev[worst] <= 1e-13, (
        f"max deviation {dev[worst]:.3e} at dj={tuple(dj[worst])}: "
        f"got {got[worst]!r}, fixture {expected[worst]!r}; the worst row measured at "
        f"#180 was dj=(12, 8, 12, 12, 6, 10) at 4.95e-14"
    )


def test_r_symbol_matches_full_fixture_table_in_sign_and_magnitude(r_table) -> None:
    """Sign against the fixture, magnitude on its own.

    racah returns R as a float carrying noise (``-0.9999999999999998``), so the sign
    is what the fixture pins; the exact ``+-1`` is the provider's own snap
    (``round`` behind a unit-modulus assert) and is asserted separately rather than
    folded into one ``==`` that would silently cover both.
    """
    dj, expected = r_table
    got = [r_symbol(*row) for row in map(tuple, dj.tolist())]
    bad = [
        (tuple(d), g, int(e))
        for d, g, e in zip(dj, got, expected, strict=True)
        if (g > 0) != (int(e) > 0)
    ]
    assert not bad, f"{len(bad)} sign mismatches, first: {bad[0]}"
    assert all(isinstance(g, int) and abs(g) == 1 for g in got)


# --- structure --------------------------------------------------------------


def _admissible(a: int, b: int, c: int, d: int, e: int, f: int) -> bool:
    return triangle(a, b, e) and triangle(e, c, d) and triangle(b, c, f) and triangle(a, f, d)


def test_f_symbol_is_exactly_zero_on_violated_triangles() -> None:
    checked = 0
    for dj in product(range(5), repeat=6):
        if not _admissible(*dj):
            assert f_symbol(*dj) == 0.0, f"{dj} is not a true zero"
            checked += 1
    assert checked > 1000


def _fuse(a: int, b: int) -> range:
    return range(abs(a - b), a + b + 1, 2)


def test_f_symbol_matrices_are_unitary() -> None:
    """Promoted to :func:`tenet.symmetry.coherence.validate_unitary` (M24a); the
    validator raises on the first non-unitary F-matrix or non-unimodular R, and
    returns the number of F-matrices checked over the same ``range(7)`` budget
    the inline loop used."""
    checked = validate_unitary(SU2, tuple(SU2Sector(x) for x in range(7)), atol=1e-13)
    assert checked > 100


def test_pentagon_identity() -> None:
    """``[F^{fcd}_e]_{g,l} [F^{abl}_e]_{f,k} = sum_h [F^{abc}_g]_{f,h} [F^{ahd}_e]_{g,k}
    [F^{bcd}_k]_{h,l}`` — an arithmetic cross-check on the phase/dimension folding,
    not a gauge check (pentagon holds in every valid gauge). Promoted to
    :func:`tenet.symmetry.coherence.validate_pentagon` (M24a) over the same
    ``range(5)`` budget the inline loop used.
    """
    checked = validate_pentagon(SU2, tuple(SU2Sector(x) for x in range(5)), atol=1e-12)
    assert checked > 1000

    # the multi-term guard the inline loop carried: at least one pentagon sum is a
    # genuine expansion over h, so the validator was not only fed one-term sums
    def multiterm_exists() -> bool:
        for a, b, c, d in product(range(3), repeat=4):
            for f in _fuse(a, b):
                for g in _fuse(f, c):
                    for ell in _fuse(c, d):
                        for e in (x for x in _fuse(g, d) if triangle(f, ell, x)):
                            for k in (x for x in _fuse(b, ell) if triangle(a, x, e)):
                                terms = [
                                    f_symbol(a, b, c, g, f, h)
                                    * f_symbol(a, h, d, e, g, k)
                                    * f_symbol(b, c, d, k, h, ell)
                                    for h in _fuse(b, c)
                                ]
                                if sum(t != 0.0 for t in terms) > 1:
                                    return True
        return False

    assert multiterm_exists(), "pentagon was only exercised on one-term sums"


def test_hexagon_identity() -> None:
    """The R-move hexagon, promoted to ``validate_hexagon`` (M24a): SU(2)'s
    symmetric R is consistent with its associator."""
    assert validate_hexagon(SU2, tuple(SU2Sector(x) for x in range(5))) > 100


def test_snake_and_spherical() -> None:
    """``B`` from ``F`` and ``qdim(a) == qdim(dual(a))``, via the M24a validators."""
    sectors = tuple(SU2Sector(x) for x in range(7))
    assert validate_snake(SU2, sectors) > 50
    assert validate_spherical(SU2, sectors) == 7


# --- the vertex-normalization oracle ----------------------------------------


def _tree_left(a: int, b: int, c: int, d: int, e: int) -> np.ndarray:
    """Dense CG tensor of ``((a b) c) -> d`` with inner line ``e``, from ``SU2.cgc`` only."""
    ab = SU2.cgc(SU2Sector(a), SU2Sector(b), SU2Sector(e))[..., 0]
    ec = SU2.cgc(SU2Sector(e), SU2Sector(c), SU2Sector(d))[..., 0]
    return np.tensordot(ab, ec, axes=([2], [0]))  # ma mb mc md


def _tree_right(a: int, b: int, c: int, d: int, f: int) -> np.ndarray:
    """Dense CG tensor of ``(a (b c)) -> d`` with inner line ``f``."""
    bc = SU2.cgc(SU2Sector(b), SU2Sector(c), SU2Sector(f))[..., 0]
    af = SU2.cgc(SU2Sector(a), SU2Sector(f), SU2Sector(d))[..., 0]
    return np.tensordot(af, bc, axes=([1], [2])).transpose(0, 2, 3, 1)  # ma mb mc md


def _f_from_cg(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
    """``[F^{abc}_d]_{e,f}`` by contracting the two trees over all four magnetic indices.

    The trees are orthogonal but not unit-norm: each carries ``d_d`` from the sum over
    the coupled index, hence the division.
    """
    overlap = np.tensordot(_tree_left(a, b, c, d, e), _tree_right(a, b, c, d, f), axes=4)
    return float(overlap) / (d + 1)


def test_f_symbol_from_cg_contraction() -> None:
    """The one check that fails on a vertex-normalization mismatch (M1 CG normalization)."""
    checked = 0
    for a, b, c, d, e, f in product(range(5), repeat=6):
        if not _admissible(a, b, c, d, e, f):
            continue
        assert abs(_f_from_cg(a, b, c, d, e, f) - f_symbol(a, b, c, d, e, f)) < 1e-12
        checked += 1
    assert checked > 500


# --- R, FS, B ---------------------------------------------------------------


def test_r_symbol_is_symmetric_and_its_own_inverse() -> None:
    for a, b, c in product(range(9), repeat=3):
        if not triangle(a, b, c):
            continue
        assert r_symbol(a, b, c) == r_symbol(b, a, c)
        assert r_symbol(a, b, c) ** 2 == 1


def test_r_symbol_spot_values_for_two_spin_half_lines() -> None:
    assert r_symbol(1, 1, 0) == -1  # singlet is antisymmetric
    assert r_symbol(1, 1, 2) == 1  # triplet is symmetric


def test_frobenius_schur_signs() -> None:
    for dj in range(13):
        assert frobenius_schur(dj) == (-1) ** dj
        assert frobenius_schur(dj) == (1 if dj % 2 == 0 else -1)


def test_b_symbol_has_unit_modulus() -> None:
    for a, b, c in product(range(9), repeat=3):
        if not triangle(a, b, c):
            continue
        assert abs(abs(b_symbol(a, b, c)) - 1.0) < 1e-13


def test_b_symbol_unit_codomain_collapse() -> None:
    """``B^{0 b}_b == 1``, to float tolerance: racah derives B through the F-symbol and
    a dimension ratio, so it lands on 1.0000000000000002 where the closed form hit 1.0
    exactly. The identity is what matters, not the last bit."""
    for b in range(13):
        assert abs(b_symbol(0, b, b) - 1.0) < 1e-13


def test_b_symbol_from_cg_contraction() -> None:
    """``B`` re-derived through the CG oracle, so it cannot inherit a ``w6j`` error."""
    for a, b, c in product(range(5), repeat=3):
        if not triangle(a, b, c):
            continue
        expected = np.sqrt((a + 1) * (b + 1) / (c + 1)) * _f_from_cg(a, b, b, a, c, 0)
        assert abs(b_symbol(a, b, c) - expected) < 1e-12


# --- reality ----------------------------------------------------------------


def test_all_coefficients_are_real_so_domain_side_conjugation_is_a_noop() -> None:
    """#36 applies the *conjugate* coefficient on the fusion (domain) tree; in this
    gauge every coefficient is real, so that conjugation is a no-op.
    """
    for dj in product(range(5), repeat=6):
        v = f_symbol(*dj)
        assert isinstance(v, float)
        assert v == v.conjugate()
    for a, b, c in product(range(7), repeat=3):
        if not triangle(a, b, c):
            continue
        assert isinstance(r_symbol(a, b, c), int)
        assert r_symbol(a, b, c) in (-1, 1)
        assert isinstance(b_symbol(a, b, c), float)
    for dj in range(13):
        assert isinstance(frobenius_schur(dj), int)
        assert frobenius_schur(dj) in (-1, 1)


# --- the protocol and the provider adapters ---------------------------------


def test_recoupling_capabilities() -> None:
    for capability in (AssociatorData, BraidingData, DualityData, FSIndicatorData):
        assert isinstance(SU2, capability)
    for capability in (AssociatorData, BraidingData, DualityData):
        assert not isinstance(U1, capability)
        assert not isinstance(Trivial, capability)
    # the FS indicator alone is carried by every provider, Abelian ones included
    assert isinstance(U1, FSIndicatorData)
    assert isinstance(Trivial, FSIndicatorData)


def test_provider_f_symbol_raises_on_multiplicity() -> None:
    """Unreachable for SU(2), so the contract is asserted through a stub provider."""

    class _Degenerate(SU2Provider):
        def n_symbol(self, a: SU2Sector, b: SU2Sector, c: SU2Sector) -> int:
            return 2 * super().n_symbol(a, b, c)

    s = SU2Sector
    with pytest.raises(ValueError, match="scalar-valued"):
        _Degenerate().f_symbol(s(1), s(1), s(1), s(1), s(0), s(0))


def test_provider_stays_array_free_and_hashable() -> None:
    hash(SU2)
    assert not SU2Provider.__dataclass_fields__.keys() - {"name"}
