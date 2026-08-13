"""SU(2) F/R/B/FS symbols, pinned to the vendored racah (TensorKitSectors) fixtures.

The fixtures are the specification of the gauge: pentagon and unitarity are
gauge-*covariant* and hold in every valid gauge simultaneously, so only a full-table
comparison can tell ours apart from a plausible-looking convention slip.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from tenet.symmetry import SU2, U1, RecouplingData, SU2Provider, SU2Sector, Trivial
from tenet.symmetry._su2_coeff import b_symbol, f_symbol, frobenius_schur, r_symbol, triangle

FIXTURES = Path(__file__).parent.parent / "fixtures"


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
    dj, expected = f_table
    got = np.fromiter((f_symbol(*row) for row in map(tuple, dj.tolist())), float, len(dj))
    dev = np.abs(got - expected)
    worst = int(dev.argmax())
    assert dev[worst] <= 1e-14, (
        f"max deviation {dev[worst]:.3e} at dj={tuple(dj[worst])}: "
        f"got {got[worst]!r}, fixture {expected[worst]!r}"
    )


def test_r_symbol_matches_full_fixture_table_exactly(r_table) -> None:
    dj, expected = r_table
    got = [r_symbol(*row) for row in map(tuple, dj.tolist())]
    bad = [(tuple(d), g, int(e)) for d, g, e in zip(dj, got, expected, strict=True) if g != e]
    assert not bad, f"{len(bad)} exact mismatches, first: {bad[0]}"


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


def _inner_lines(a: int, b: int, c: int, d: int) -> tuple[list[int], list[int]]:
    e = [x for x in _fuse(a, b) if triangle(x, c, d)]
    f = [x for x in _fuse(b, c) if triangle(a, x, d)]
    return e, f


def test_f_symbol_matrices_are_unitary() -> None:
    checked = 0
    for a, b, c, d in product(range(7), repeat=4):
        es, fs = _inner_lines(a, b, c, d)
        if not es:
            assert not fs
            continue
        assert len(es) == len(fs)
        m = np.array([[f_symbol(a, b, c, d, e, f) for f in fs] for e in es])
        eye = np.eye(len(es))
        assert np.allclose(m @ m.conj().T, eye, atol=1e-13)
        assert np.allclose(m.conj().T @ m, eye, atol=1e-13)
        checked += 1
    assert checked > 100


def test_pentagon_identity() -> None:
    """``[F^{fcd}_e]_{g,l} [F^{abl}_e]_{f,k} = sum_h [F^{abc}_g]_{f,h} [F^{ahd}_e]_{g,k}
    [F^{bcd}_k]_{h,l}`` — an arithmetic cross-check on the phase/dimension folding,
    not a gauge check (pentagon holds in every valid gauge).
    """
    checked = multiterm = 0
    for a, b, c, d in product(range(5), repeat=4):
        for f in _fuse(a, b):
            for g in _fuse(f, c):
                for ell in _fuse(c, d):
                    for e in _fuse(g, d):
                        if not triangle(f, ell, e):
                            continue
                        for k in _fuse(b, ell):
                            if not triangle(a, k, e):
                                continue
                            lhs = f_symbol(f, c, d, e, g, ell) * f_symbol(a, b, ell, e, f, k)
                            terms = [
                                f_symbol(a, b, c, g, f, h)
                                * f_symbol(a, h, d, e, g, k)
                                * f_symbol(b, c, d, k, h, ell)
                                for h in _fuse(b, c)
                            ]
                            assert abs(lhs - sum(terms)) < 1e-12, (a, b, c, d, e, f, g, k, ell)
                            checked += 1
                            multiterm += sum(t != 0.0 for t in terms) > 1
    assert checked > 1000
    assert multiterm > 0, "pentagon was only exercised on one-term sums"


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
    for b in range(13):
        assert b_symbol(0, b, b) == 1.0


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


def test_recoupling_data_capability() -> None:
    assert isinstance(SU2, RecouplingData)
    assert not isinstance(U1, RecouplingData)
    assert not isinstance(Trivial, RecouplingData)


def test_provider_adapters_agree_with_doubled_spin_functions() -> None:
    rng = np.random.default_rng(35)
    s = SU2Sector
    for _ in range(200):
        dj = [int(x) for x in rng.integers(0, 9, 6)]
        if not _admissible(*dj):
            continue
        assert SU2.f_symbol(*(s(x) for x in dj)) == f_symbol(*dj)
    for a, b, c in product(range(7), repeat=3):
        if not triangle(a, b, c):
            continue
        assert SU2.r_symbol(s(a), s(b), s(c)) == r_symbol(a, b, c)
        assert SU2.b_symbol(s(a), s(b), s(c)) == b_symbol(a, b, c)
    for dj in range(13):
        assert SU2.frobenius_schur(s(dj)) == frobenius_schur(dj)


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
