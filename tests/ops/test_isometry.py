"""Tests for ``tenet.isometry`` and ``tenet.random_isometry`` — issue #89.

Two things here are measured rather than assumed, and both are the reason the
functions are shaped the way they are:

* ``isometry`` is asserted to *be* ``embed(identity(...))`` block for block, so
  the implementation cannot drift from the composition it claims to be — which is
  also what keeps its refusals ``_check_containment``'s, unduplicated.
* ``random_isometry``'s Mezzadri sign fix is pinned by a **statistic**, not a
  shape check: the same run with the fix removed is performed inline and asserted
  to fail it. Orthogonality alone cannot see the difference, and neither can
  ``E|W[0,0]|²`` — see the comment on that test.
"""

from dataclasses import replace

import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.map_view import to_matrices
from tenet.symmetry import (
    SU2,
    U1,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

TS = TrivialSector()
UF = ProductProvider((U1, fZ2))


def uf(charge: int, parity: int) -> ProductSector:
    return ProductSector((U1Sector(charge), FZ2Sector(parity)))


# (small, large) with `small` contained in `large` sector-wise, per provider.
CASES = {
    "trivial": ({TS: 2}, {TS: 4}),
    "u1": (
        {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1},
        {U1Sector(-1): 3, U1Sector(0): 4, U1Sector(1): 2, U1Sector(2): 1},
    ),
    "su2": (
        {SU2Sector(0): 2, SU2Sector(1): 1},
        {SU2Sector(0): 3, SU2Sector(1): 2, SU2Sector(2): 1},
    ),
    "fz2": ({FZ2Sector(0): 2, FZ2Sector(1): 3}, {FZ2Sector(0): 3, FZ2Sector(1): 4}),
    "product": (
        {uf(0, 0): 2, uf(1, 1): 3},
        {uf(0, 0): 3, uf(1, 1): 4, uf(2, 0): 2},
    ),
}
PROVIDERS = {
    "trivial": Trivial,
    "u1": U1,
    "su2": SU2,
    "fz2": fZ2,
    "product": UF,
}
NAMES = list(CASES)
WIDTHS = [1, 2]  # a single-leg domain and a two-leg one: band interleaving


def spaces(name):
    small, large = CASES[name]
    return GradedSpace.new(PROVIDERS[name], small), GradedSpace.new(PROVIDERS[name], large)


def pair(name, width=1, *, equal=False):
    """``(codomain legs, domain legs)`` — ``width`` legs on each side.

    ``equal=True`` is the containment-is-an-equality case, and it takes the
    *domain's* names on both sides so the result is comparable to
    ``identity(legs)`` leg for leg (``name`` is user bookkeeping and ``embed``
    takes it from the target).
    """
    small, large = spaces(name)
    cod = small if equal else large
    return (
        tuple(Leg(cod, OUT, name=f"d{i}" if equal else f"c{i}") for i in range(width)),
        tuple(Leg(small, IN, name=f"d{i}") for i in range(width)),
    )


def mirrored(domain):
    """``identity``'s legs for ``domain``: the same legs, side OUT."""
    return tuple(replace(leg, side=OUT) for leg in domain)


def target(codomain, domain):
    return (
        *(replace(leg, side=OUT) for leg in codomain),
        *(replace(leg, side=IN) for leg in domain),
    )


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  registration is the import's side effect

    return jax


def entries(w, position=(0, 0)):
    """``{c: W_c[position]}`` — one coupled-sector matrix element per sector."""
    return {c: np.asarray(m)[position] for c, m in to_matrices(w).items()}


# --- exports and legs ---------------------------------------------------------------


def test_exported_from_tenet_and_ops_and_not_as_a_method():
    for fn in ("isometry", "random_isometry"):
        assert fn in tenet.__all__ and fn in tenet.ops.__all__
        assert getattr(tenet, fn) is getattr(tenet.ops, fn)
        assert not hasattr(SymmetricTensor, fn)  # constructors, not operations


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", WIDTHS)
def test_legs_keep_space_dual_and_name_and_only_side_is_set(name, width):
    """``identity``'s documented stance: ``side`` is *set*, never compared — so a
    codomain leg handed in as IN still comes back OUT."""
    codomain, domain = pair(name, width)
    flipped = tuple(replace(leg, side=IN) for leg in codomain)
    for w in (tenet.isometry(codomain, domain), tenet.isometry(flipped, domain)):
        assert w.legs == target(codomain, domain)
        for got, want in zip(w.legs, (*codomain, *domain), strict=True):
            assert (got.space, got.dual, got.name) == (want.space, want.dual, want.name)
    assert tenet.random_isometry(flipped, domain, seed=0).legs == target(codomain, domain)


# --- isometry IS embed(identity(...)) -----------------------------------------------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", WIDTHS)
def test_isometry_is_embed_of_identity_block_for_block(name, width):
    codomain, domain = pair(name, width)
    want = tenet.embed(tenet.identity(domain), target(codomain, domain))
    got = tenet.isometry(codomain, domain)
    assert got.structure == want.structure
    for a, b in zip(got.blocks, want.blocks, strict=True):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() == 0.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", WIDTHS)
def test_w_dagger_w_is_the_identity_on_the_domain_exactly(name, width):
    codomain, domain = pair(name, width)
    w = tenet.isometry(codomain, domain)
    want = tenet.identity(mirrored(domain))
    got = tenet.adjoint(w) @ w
    assert got.legs == want.legs
    for a, b in zip(got.blocks, want.blocks, strict=True):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() == 0.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", WIDTHS)
def test_w_w_dagger_is_a_projector(name, width):
    codomain, domain = pair(name, width)
    p = tenet.isometry(codomain, domain)
    p = p @ tenet.adjoint(p)
    assert tenet.allclose(p @ p, p, rtol=0, atol=1e-13)


@pytest.mark.parametrize("name", NAMES)
def test_an_equality_containment_gives_the_identity_and_needs_no_unitary_function(name):
    """``unitary(V, W)`` would be this plus one assertion, i.e. a criterion."""
    codomain, domain = pair(name, equal=True)
    p = tenet.isometry(codomain, domain)
    p = p @ tenet.adjoint(p)
    want = tenet.identity(mirrored(codomain))
    assert p.legs == want.legs
    for a, b in zip(p.blocks, want.blocks, strict=True):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() == 0.0


@pytest.mark.parametrize("name", NAMES)
def test_the_identity_case_copies_nothing(name, monkeypatch):
    """``embed``'s allocation-light path: no ``pad`` runs, and the blocks are the
    identity's own."""
    import autoray as ar

    codomain, domain = pair(name, equal=True)
    want = tenet.identity(mirrored(domain))

    real_do = ar.do

    def no_pad(fn, *args, **kwargs):
        # "concatenate" since #95: ``embed`` pads by concatenating a zero slab,
        # autoray's torch ``pad`` translation being wrong. Same assertion.
        if fn in ("pad", "concatenate"):  # pragma: no cover - it is unused
            raise AssertionError("isometry(legs, legs) must not pad")
        return real_do(fn, *args, **kwargs)

    monkeypatch.setattr(ar, "do", no_pad)
    got = tenet.isometry(codomain, domain)
    assert got.structure == want.structure
    for a, b in zip(got.blocks, want.blocks, strict=True):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() == 0.0


# --- refusals, which are embed's ----------------------------------------------------


def test_refuses_a_domain_sector_with_a_larger_degeneracy():
    small, large = spaces("u1")
    with pytest.raises(ValueError) as excinfo:
        tenet.isometry((Leg(small, OUT),), (Leg(large, IN),))
    message = str(excinfo.value)
    assert message.startswith("embed:")  # _check_containment's, not a duplicate
    assert "axis 0" in message and repr(U1Sector(-1)) in message
    assert "degeneracy 3 in the tensor but 2 in the target space" in message


def test_refuses_a_sector_absent_from_the_codomain():
    small, _ = spaces("u1")
    partial = GradedSpace.new(U1, {U1Sector(0): 5})
    with pytest.raises(ValueError) as excinfo:
        tenet.isometry((Leg(partial, OUT),), (Leg(small, IN),))
    assert "absent from the target space entirely" in str(excinfo.value)


def test_refuses_a_dual_mismatch():
    small, large = spaces("u1")
    with pytest.raises(ValueError, match=r"dual is categorical"):
        tenet.isometry((replace(Leg(large, OUT), dual=True),), (Leg(small, IN),))


def test_refuses_a_provider_mismatch():
    _, large = spaces("u1")
    other, _ = spaces("su2")
    with pytest.raises(ValueError, match=r"never casts between symmetries"):
        tenet.isometry((Leg(large, OUT),), (Leg(other, IN),))


def test_refuses_a_leg_count_mismatch():
    small, large = spaces("u1")
    with pytest.raises(ValueError, match=r"never adds or drops an axis"):
        tenet.isometry((Leg(large, OUT), Leg(large, OUT)), (Leg(small, IN),))


def test_side_cannot_mismatch_because_isometry_sets_it():
    """``_check_containment``'s ``side`` refusal is unreachable from here — the
    legs' sides are *set* from the argument position, ``identity``'s stance. The
    other five refusals above are the ones this function can actually produce."""
    small, large = spaces("u1")
    for side in (IN, OUT):
        w = tenet.isometry((Leg(large, side),), (Leg(small, side),))
        assert [leg.side for leg in w.legs] == [OUT, IN]


def test_the_fused_containment_case_is_refused_per_leg():
    """A target whose *fused* space contains the domain's while no single leg does.

    TensorKit accepts this (its condition is ``domain(t) ≾ codomain(t)`` on the
    fused spaces); we refuse it, because the only construction available is a
    coupled-sector ``eye``, which names an arbitrary basis correspondence — see
    the ``ponytail:`` note in ``isometry``'s docstring.
    """
    domain = (
        Leg(GradedSpace.new(U1, {U1Sector(0): 2}), IN),
        Leg(GradedSpace.new(U1, {U1Sector(1): 3}), IN),
    )
    codomain = (
        Leg(GradedSpace.new(U1, {U1Sector(1): 5}), OUT),
        Leg(GradedSpace.new(U1, {U1Sector(0): 5}), OUT),
    )
    # the fused spaces do contain: charge 1 with degeneracy 6 inside 25
    with pytest.raises(ValueError, match=r"absent from the target space entirely"):
        tenet.isometry(codomain, domain)


# --- random_isometry ----------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128], ids=["real", "complex"])
@pytest.mark.parametrize("equal", [False, True], ids=["tall", "square"])
def test_random_isometry_is_an_isometry(name, width, dtype, equal):
    codomain, domain = pair(name, width, equal=equal)
    w = tenet.random_isometry(codomain, domain, seed=7, dtype=dtype)
    assert w.dtype == dtype
    want = tenet.identity(mirrored(domain), dtype=dtype)
    got = tenet.adjoint(w) @ w
    assert got.legs == want.legs
    assert tenet.allclose(got, want, rtol=0, atol=1e-13)


def test_the_same_seed_is_byte_identical_and_different_seeds_differ():
    codomain, domain = pair("su2", 2)
    a = tenet.random_isometry(codomain, domain, seed=11)
    b = tenet.random_isometry(codomain, domain, seed=11)
    c = tenet.random_isometry(codomain, domain, seed=12)
    for x, y in zip(a.blocks, b.blocks, strict=True):
        assert np.array_equal(x, y)
    assert any(not np.array_equal(x, y) for x, y in zip(a.blocks, c.blocks, strict=True))


def test_structural_refusal_names_the_sector_and_both_dimensions():
    small, large = spaces("u1")
    with pytest.raises(ValueError) as excinfo:
        tenet.random_isometry((Leg(small, OUT),), (Leg(large, IN),))
    message = str(excinfo.value)
    assert "random_isometry" in message
    assert "rows=2 < cols=3" in message and repr(U1Sector(-1)) in message
    # and it says which condition this is, against isometry's per-leg one
    assert "fused" in message and "per-leg" in message


def test_the_fused_condition_is_weaker_than_isometrys_per_leg_one():
    """A codomain leg *smaller* than its domain partner, which ``isometry``
    refuses, still has every coupled sector tall — so ``random_isometry`` accepts
    it. That gap is the point of the two messages."""
    domain = (
        Leg(GradedSpace.new(U1, {U1Sector(0): 4}), IN),
        Leg(GradedSpace.new(U1, {U1Sector(1): 1}), IN),
    )
    codomain = (
        Leg(GradedSpace.new(U1, {U1Sector(0): 3}), OUT),
        Leg(GradedSpace.new(U1, {U1Sector(1): 3}), OUT),
    )
    with pytest.raises(ValueError, match=r"^embed:"):
        tenet.isometry(codomain, domain)
    w = tenet.random_isometry(codomain, domain, seed=1)
    assert tenet.allclose(
        tenet.adjoint(w) @ w, tenet.identity(mirrored(domain)), rtol=0, atol=1e-13
    )


# --- Haar-ness, measured ------------------------------------------------------------


def unfixed(shape, rng, complex_draw=False):
    """``random_isometry``'s draw with ``Q * sign(diag(R))`` **removed**."""
    a = rng.standard_normal(shape)
    if complex_draw:
        a = a + 1j * rng.standard_normal(shape)
    return np.linalg.qr(a)[0]


def test_haar_statistic_and_the_bias_the_sign_fix_removes():
    """``E[W_c[0,0]] == 0`` is the discriminating statistic, and it is stated on the
    *signed* mean on purpose.

    ``E[|W_c[0,0]|²] == 1/rows_c`` is the natural "is it Haar" check, and it is
    asserted below — but it is **blind to the sign fix**: multiplying a column by
    a phase does not change ``|W[0,0]|``. Measured over 2000 draws of a ``(5, 3)``
    block: ``0.2029`` with the fix and ``0.2029`` without, against ``1/5``. The
    signed mean is what catches it — ``0.0015`` with the fix, ``-0.3776`` without,
    i.e. a bias some 37 sampling errors wide, and exactly the LAPACK ``R``-diagonal
    sign bias Mezzadri's correction exists to remove. Do not delete that line.
    """
    codomain, domain = pair("su2", 1)
    draws = 400
    samples: dict = {}
    for seed in range(draws):
        for c, value in entries(tenet.random_isometry(codomain, domain, seed=seed)).items():
            samples.setdefault(c, []).append(value)

    for raw in samples.values():
        values = np.asarray(raw)
        sem = values.std(ddof=1) / np.sqrt(draws)
        assert abs(values.mean()) <= 4 * sem  # unbiased: the sign fix is in

    # the same statistic on the unfixed draw, at one of the sector shapes above
    rows, cols = next(
        iter(to_matrices(tenet.random_isometry(codomain, domain, seed=0)).values())
    ).shape
    rng = np.random.default_rng(0)
    biased = np.asarray([unfixed((rows, cols), rng)[0, 0] for _ in range(draws)])
    sem = biased.std(ddof=1) / np.sqrt(draws)
    assert abs(biased.mean()) > 4 * sem  # and without it, the draw is not Haar


def test_the_modulus_statistic_matches_one_over_rows():
    codomain, domain = pair("su2", 1)
    draws = 400
    samples: dict = {}
    for seed in range(draws):
        for c, value in entries(tenet.random_isometry(codomain, domain, seed=seed)).items():
            samples.setdefault(c, []).append(abs(value) ** 2)

    shapes = {c: m.shape for c, m in to_matrices(tenet.random_isometry(codomain, domain)).items()}
    for c, values in samples.items():
        values = np.asarray(values)
        sem = values.std(ddof=1) / np.sqrt(draws)
        assert abs(values.mean() - 1 / shapes[c][0]) <= 4 * sem


def test_complex_draws_are_genuinely_complex_and_not_a_real_cast():
    """A real orthogonal matrix cast to ``complex128`` is an isometry and passes
    every other criterion in this file; its entries just have no imaginary part
    and its column phases are ``O(n)``'s, not ``U(n)``'s. Measured over 400 draws:
    ``E|Im W|² / E|W|²`` is ``0.49`` for the Ginibre draw and exactly ``0`` for a
    real draw cast — half, because a circularly-symmetric entry splits its modulus
    evenly between the two components."""
    codomain, domain = pair("su2", 2)
    w = tenet.random_isometry(codomain, domain, seed=3, dtype=np.complex128)
    for block in w.blocks:
        assert np.abs(np.imag(np.asarray(block))).max() > 0.0

    draws = 400
    values = np.asarray(
        [
            v
            for seed in range(draws)
            for v in entries(
                tenet.random_isometry(codomain, domain, seed=seed, dtype=np.complex128)
            ).values()
        ]
    )
    fraction = np.mean(np.abs(values.imag) ** 2) / np.mean(np.abs(values) ** 2)
    assert fraction == pytest.approx(0.5, abs=0.05)


# --- backends and the cross-check with left_null ------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_both_return_numpy_blocks_and_survive_to_backend_jax(name):
    use_jax()
    import autoray as ar

    codomain, domain = pair(name, 2)
    for w in (
        tenet.isometry(codomain, domain),
        tenet.random_isometry(codomain, domain, seed=5),
    ):
        assert {ar.infer_backend(b) for b in w.blocks} == {"numpy"}
        moved = w.to_backend("jax")
        assert {ar.infer_backend(b) for b in moved.blocks} == {"jax"}
        assert tenet.allclose(
            tenet.adjoint(moved) @ moved,
            tenet.identity(mirrored(domain)).to_backend("jax"),
            rtol=0,
            atol=1e-13,
        )


def test_map_module_imports_no_jax():
    import pathlib
    import sys

    source = pathlib.Path(sys.modules["tenet.ops.map"].__file__).read_text()
    assert "import jax" not in source


def test_left_nulls_isometry_is_the_same_statement_isometry_guarantees():
    """The two definitions of "isometry" cannot drift apart: ``N† N`` from
    ``left_null`` satisfies exactly what ``isometry`` guarantees, ``W† W = id``."""
    from test_linalg_null import TALL, tall

    t = tall("su2", seed=2)
    n = tenet.linalg.left_null(t, TALL)
    bond = (Leg(n.legs[-1].space, IN),)
    want = tenet.identity(mirrored(bond))
    assert tenet.allclose(tenet.adjoint(n) @ n, want, rtol=0, atol=1e-13)
    # and the same bond, drawn at random, is an isometry into the same codomain
    w = tenet.random_isometry(tuple(replace(leg, side=OUT) for leg in bond), bond, seed=4)
    assert tenet.allclose(tenet.adjoint(w) @ w, want, rtol=0, atol=1e-13)
