"""M12c: ``SUNProvider``, SU(3) first, behind ``tenet-py[sun]`` — issue #127.

Two oracles run side by side here. The vendored ``tests/fixtures/su3_*.txt`` are
the *specification* (M12a pinned them, wheel-free); the installed ``racah`` wheel
is the *implementation*, and the two are checked against each other in **both**
directions — every fixture row must come back from the provider, and every symbol
the provider produces over the fixture sector set must be in the fixture.
``to_dense`` is the third, independent oracle for the tensor-level tests.

SU(3) is also the first provider with ``dual(a) != a`` *and* ``d_a > 1``, so the
cup/cap section is the SU(3) counterpart of ``test_su2_dual.py``.

Everything except the packaging test skips cleanly without ``racah-py``.
"""

import dataclasses
import itertools
import json
import subprocess
import sys
import textwrap
from math import sqrt

import numpy as np
import pytest

try:
    import racah
except ImportError:  # the packaging test below is the one that must still run
    racah = None

needs_racah = pytest.mark.skipif(
    racah is None, reason="SU(N) needs racah-py: pip install 'tenet-py[sun]'"
)

if racah is not None:
    import _su3_fixture as fx

    import tenet
    from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    from tenet.symmetry.base import (
        BendingCoefficients,
        ClebschGordan,
        DualBasis,
        FusionProvider,
        MultiplicityRecoupling,
        PermutationCoefficients,
        QuantumDimension,
        RecouplingData,
    )
    from tenet.symmetry.sun import _SUN_GAUGE, SUNProvider, SUNSector

    SU3 = SUNProvider(3)
    ONE = SUNSector((0, 0))
    THREE = SUNSector((1, 0))
    THREEBAR = SUNSector((0, 1))
    SIX = SUNSector((2, 0))
    EIGHT = SUNSector((1, 1))
    TEN = SUNSector((3, 0))
    TENBAR = SUNSector((0, 3))
    TWENTYSEVEN = SUNSector((2, 2))

    # The five sectors the backward fixture sweeps run over; the fixtures cover eleven,
    # and 11**6 sextuples is a sweep nobody needs to pay for on every run. Inner lines are
    # filtered against the vendored set rather than against this one, because the set is
    # not fusion-closed (``6 x 6`` reaches ``15' = (4, 0)``, which the fixtures omit).
    SUB = (ONE, THREE, THREEBAR, SIX, EIGHT)
    VENDORED = {a for labels in fx._N for a in labels}

    E = GradedSpace.new(SU3, {EIGHT: 1})
    V = GradedSpace.new(SU3, {THREE: 1})
    W = GradedSpace.new(SU3, {THREEBAR: 1})

    # Three adjoints on the OUT side, so the block set contains the N = 2 vertex.
    LEGS = (Leg(E, OUT), Leg(E, OUT), Leg(E, OUT), Leg(E, IN))
    # Non-self-dual legs, interleaved sides, one dual flag set.
    MIXED = (Leg(E, OUT), Leg(V, IN), Leg(E, OUT), Leg(W, IN))
    DUALED = (Leg(E, OUT), Leg(V, IN, dual=True), Leg(E, OUT), Leg(W, IN))


# Parametrization runs at collection time, so anything a decorator touches has to exist
# without racah. Sectors are therefore passed by *name* and resolved through globals().
SPLITS = (((0, 1, 2), (3,)), ((0, 1), (2, 3)), ((0,), (1, 2, 3)), ((0, 1, 2, 3), ()))
SMALL = ("ONE", "THREE", "THREEBAR", "EIGHT")


def su3(legs=None, seed=11):
    return SymmetricTensor.random(LEGS if legs is None else legs, seed=seed)


# --- packaging ----------------------------------------------------------------


def test_import_error_without_racah_names_the_extra():
    """Simulate a racah-less environment with a meta-path blocker, in a subprocess."""
    script = textwrap.dedent("""
        import importlib.abc, sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "racah" or name.startswith("racah."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, Blocker())
        import tenet                      # core must survive a racah-less environment
        assert "racah" not in sys.modules
        try:
            import tenet.symmetry.sun
        except ImportError as exc:
            msg = str(exc)
            assert "tenet-py[sun]" in msg, msg
            assert "no pure-Python fallback" in msg, msg
            assert "second gauge" in msg, msg
        else:
            raise AssertionError("tenet.symmetry.sun imported without racah")
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "OK"


# --- sector and provider hygiene ----------------------------------------------


@needs_racah
def test_provider_supplies_every_capability_m12c_claims():
    for capability in (
        QuantumDimension,
        ClebschGordan,
        DualBasis,
        MultiplicityRecoupling,
        RecouplingData,
        PermutationCoefficients,
        BendingCoefficients,
    ):
        assert isinstance(SU3, capability), capability.__name__
    _: FusionProvider = SU3  # static conformance


@needs_racah
def test_provider_is_frozen_hashable_and_array_free():
    assert hash(SU3) == hash(SUNProvider(3))
    assert SUNProvider(3) != SUNProvider(4)
    assert {f.name for f in dataclasses.fields(SU3)} == {"n", "name"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        SU3.n = 4


@needs_racah
def test_sector_is_ordered_validated_and_tuple_normalized():
    assert sorted([EIGHT, ONE, THREE]) == [ONE, THREE, EIGHT]
    assert SUNSector([1, 1]) == EIGHT and SUNSector([1, 1]).dynkin == (1, 1)
    with pytest.raises(ValueError):
        SUNSector((-1, 0))
    with pytest.raises(ValueError):
        SUNSector(())
    with pytest.raises(TypeError):
        SUNSector((1.0, 0))


@needs_racah
@pytest.mark.parametrize("method", ["dual", "qdim", "irrep_dim"])
def test_wrong_rank_sector_fails_at_the_first_query(method):
    """``ProductProvider._split``'s type check cannot tell SU(3) from SU(4); this can."""
    with pytest.raises(ValueError, match="SU\\(4\\) label"):
        getattr(SU3, method)(SUNSector((1, 0, 0)))
    with pytest.raises(ValueError, match="not an SU\\(3\\) sector"):
        SU3.qdim(ONE.dynkin)


# --- fusion -------------------------------------------------------------------


@needs_racah
def test_known_su3_decompositions():
    assert SU3.fusion(THREE, THREEBAR) == (ONE, EIGHT)
    assert SU3.fusion(THREE, THREE) == (THREEBAR, SIX)
    assert set(SU3.fusion(EIGHT, EIGHT)) == {ONE, EIGHT, TEN, TENBAR, TWENTYSEVEN}
    multiplicities = {c: SU3.n_symbol(EIGHT, EIGHT, c) for c in SU3.fusion(EIGHT, EIGHT)}
    assert multiplicities == {ONE: 1, EIGHT: 2, TEN: 1, TENBAR: 1, TWENTYSEVEN: 1}
    # 8 x 8 = 64 = 1 + 2·8 + 10 + 10bar + 27
    assert sum(n * SU3.irrep_dim(c) for c, n in multiplicities.items()) == 64


@needs_racah
def test_duality_and_dimensions():
    assert SU3.unit == ONE
    assert SU3.dual(THREE) == THREEBAR and SU3.dual(THREEBAR) == THREE
    assert SU3.dual(EIGHT) == EIGHT
    assert [SU3.irrep_dim(a) for a in (ONE, THREE, THREEBAR, SIX, EIGHT)] == [1, 3, 3, 6, 8]
    assert SU3.qdim(EIGHT) == 8.0


# --- Clebsch-Gordan -----------------------------------------------------------


@needs_racah
@pytest.mark.parametrize("a_name", ["THREE", "THREEBAR", "SIX", "EIGHT"])
@pytest.mark.parametrize("b_name", ["THREE", "THREEBAR", "EIGHT"])
def test_cgc_orthogonality(a_name, b_name):
    """``sum_{m1 m2} C[m1,m2,m3,mu] C[m1,m2,m3',mu'] == delta`` across channels too."""
    a, b = globals()[a_name], globals()[b_name]
    channels = [(c, mu) for c in SU3.fusion(a, b) for mu in range(SU3.n_symbol(a, b, c))]
    rows = []
    for c, mu in channels:
        block = SU3.cgc(a, b, c)
        d = (SU3.irrep_dim(a), SU3.irrep_dim(b), SU3.irrep_dim(c), SU3.n_symbol(a, b, c))
        assert block.shape == d
        for m3 in range(SU3.irrep_dim(c)):
            rows.append(block[:, :, m3, mu].ravel())
    m = np.array(rows)
    np.testing.assert_allclose(m @ m.T, np.eye(len(rows)), rtol=0.0, atol=1e-10)


@needs_racah
def test_cgc_is_read_only_and_refuses_a_forbidden_triple():
    assert not SU3.cgc(THREE, THREEBAR, EIGHT).flags.writeable
    with pytest.raises(ValueError, match="does not appear in the fusion"):
        SU3.cgc(THREE, THREE, EIGHT)


# --- fixture agreement, both directions ---------------------------------------


@needs_racah
def test_every_fixture_f_row_comes_back_from_the_provider():
    """The fixture is the spec: 14767 F blocks, pinned wheel-free in M12a."""
    assert len(fx._F) > 10_000
    for labels in fx._F:
        got = SU3.f_matrix(*(SUNSector(x) for x in labels))
        want = fx._f_block(labels)
        assert got.shape == want.shape, labels
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12, err_msg=str(labels))


@needs_racah
def test_every_fixture_r_row_comes_back_from_the_provider():
    assert len(fx._R) > 200
    for labels in fx._R:
        got = SU3.r_matrix(*(SUNSector(x) for x in labels))
        np.testing.assert_allclose(got, fx._r_block(labels), rtol=0.0, atol=1e-12)


@needs_racah
def test_every_provider_symbol_over_the_sector_subset_is_in_the_fixture():
    """The other direction: the wheel may not invent a coefficient the spec lacks."""
    seen = 0
    for a, b, c, d in itertools.product(SUB, repeat=4):
        for e in SU3.fusion(a, b):
            if e.dynkin not in VENDORED or not SU3.n_symbol(e, c, d):
                continue
            for f in SU3.fusion(b, c):
                if f.dynkin not in VENDORED or not SU3.n_symbol(a, f, d):
                    continue
                labels = tuple(x.dynkin for x in (a, b, c, d, e, f))
                assert labels in fx._F, labels
                np.testing.assert_allclose(
                    SU3.f_matrix(a, b, c, d, e, f), fx._f_block(labels), rtol=0.0, atol=1e-12
                )
                seen += 1
    assert seen > 400
    for a, b in itertools.product(SUB, repeat=2):
        for c in SU3.fusion(a, b):
            if c.dynkin not in VENDORED:
                continue
            labels = (a.dynkin, b.dynkin, c.dynkin)
            assert labels in fx._R, labels
            np.testing.assert_allclose(
                SU3.r_matrix(a, b, c), fx._r_block(labels), rtol=0.0, atol=1e-12
            )


@needs_racah
def test_derived_symbols_agree_with_the_fixture_provider():
    """``b_matrix``, ``frobenius_schur`` and ``z_matrix`` are derivations, not table reads."""
    for a, b in itertools.product(SUB, repeat=2):
        for c in SU3.fusion(a, b):
            if c.dynkin not in VENDORED:
                continue
            np.testing.assert_allclose(
                SU3.b_matrix(a, b, c),
                fx.SU3.b_matrix(*(fx.SU3Sector(x.dynkin) for x in (a, b, c))),
                rtol=0.0,
                atol=1e-12,
            )
    for a in SUB:
        fixture_a = fx.SU3Sector(a.dynkin)
        assert SU3.frobenius_schur(a) == fx.SU3.frobenius_schur(fixture_a)
        np.testing.assert_allclose(
            SU3.z_matrix(a), fx.SU3.z_matrix(fixture_a), rtol=0.0, atol=1e-12
        )


# --- the categorical identities, from racah's own checks -----------------------


@needs_racah
@pytest.mark.parametrize("quad", list(itertools.product(SMALL, repeat=4)))
def test_pentagon_and_f_unitarity(quad):
    irreps = [racah.Irrep(globals()[name].dynkin) for name in quad]
    racah.check_pentagon(*irreps)
    racah.check_f_unitarity(*irreps)


@needs_racah
@pytest.mark.parametrize("triple", list(itertools.product(SMALL, repeat=3)))
def test_hexagon(triple):
    racah.check_hexagon(*(racah.Irrep(globals()[name].dynkin) for name in triple))


# --- tensor level: transpose / repartition / to_dense --------------------------


@needs_racah
def test_block_set_contains_a_multiplicity_two_vertex():
    t = su3()
    assert any(
        SU3.n_symbol(a, b, c) == 2
        for key in t.structure.block_order
        for tree in (key.output_tree, key.input_tree)
        for a, b, c, _ in tree.vertices()
    )
    assert any(k.output_tree.multiplicities[-1] == 1 for k in t.structure.block_order)


@needs_racah
@pytest.mark.parametrize("legs_name", ["LEGS", "MIXED", "DUALED"])
@pytest.mark.parametrize("perm", [(0, 2, 1, 3), (1, 0, 2, 3), (2, 1, 0, 3), (0, 1, 3, 2)])
def test_transpose_matches_the_dense_oracle(legs_name, perm):
    t = su3(globals()[legs_name])
    got = np.asarray(t.transpose(perm).to_dense())
    np.testing.assert_allclose(got, np.transpose(np.asarray(t.to_dense()), perm), atol=1e-10)


@needs_racah
@pytest.mark.parametrize("perm", [(0, 2, 1, 3), (2, 1, 0, 3), (1, 2, 0, 3)])
def test_transpose_round_trips(perm):
    t = su3()
    inverse = tuple(sorted(range(4), key=perm.__getitem__))
    back = t.transpose(perm).transpose(inverse)
    assert back.structure == t.structure
    for got, want in zip(back.blocks, t.blocks, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-10)


@needs_racah
@pytest.mark.parametrize("legs_name", ["LEGS", "MIXED"])
@pytest.mark.parametrize("split", SPLITS)
def test_repartition_matches_the_dense_oracle(legs_name, split):
    outputs, inputs = split
    t = su3(globals()[legs_name])
    got = np.asarray(t.repartition(outputs, inputs).to_dense())
    want = np.transpose(np.asarray(t.to_dense()), (*outputs, *inputs))
    np.testing.assert_allclose(got, want, atol=1e-10)


@needs_racah
@pytest.mark.parametrize("split", SPLITS)
def test_repartition_round_trips(split):
    outputs, inputs = split
    t = su3()
    back = t.repartition(outputs, inputs).repartition((0, 1, 2), (3,))
    assert back.structure == t.structure
    for got, want in zip(back.blocks, t.blocks, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-10)


@needs_racah
def test_bend_is_multi_term_at_a_multiplicity_vertex():
    from tenet.fusion_tree import fusion_trees
    from tenet.structure import FusionBlockKey

    src = next(
        t
        for t in fusion_trees(SU3, (EIGHT, EIGHT, EIGHT), EIGHT)
        if t.inner == (EIGHT,) and t.multiplicities == (0, 0)
    )
    dst = fusion_trees(SU3, (EIGHT,), EIGHT)[0]
    terms = SU3.bend_right(FusionBlockKey(src, dst), dual=False)
    assert {k.input_tree.multiplicities[-1] for k, _ in terms} == {0, 1}


# --- DualBasis: the first provider with dual(a) != a AND d_a > 1 ---------------


@needs_racah
def test_three_is_the_case_su2_and_u1_could_not_reach():
    assert SU3.dual(THREE) != THREE and SU3.irrep_dim(THREE) > 1


@needs_racah
@pytest.mark.parametrize("a_name", ["THREE", "THREEBAR", "SIX", "EIGHT"])
def test_z_matrix_shape_unitarity_and_frobenius_schur(a_name):
    a = globals()[a_name]
    z, zbar = SU3.z_matrix(a), SU3.z_matrix(SU3.dual(a))
    assert z.shape == (SU3.irrep_dim(a), SU3.irrep_dim(SU3.dual(a)))
    assert not z.flags.writeable
    np.testing.assert_allclose(z @ z.conj().T, np.eye(z.shape[0]), rtol=0.0, atol=1e-12)
    # Z_a Z_dual(a) is the Frobenius-Schur sign, keyed on a *different* label than a
    np.testing.assert_allclose(
        z @ zbar, SU3.frobenius_schur(a) * np.eye(z.shape[0]), rtol=0.0, atol=1e-12
    )


def cup(out_space, partner, *, dual):
    """``1 -> V (x) W`` with the single block set to 1, as ``test_su2_dual`` builds it."""
    legs = (Leg(out_space, OUT), Leg(partner, OUT, dual=dual))
    t = SymmetricTensor.zeros(legs)
    blocks = tuple(np.ones_like(b) for b in t.blocks)
    assert len(blocks) == 1 and blocks[0].shape == (1, 1)
    return np.asarray(SymmetricTensor.from_legs(legs, blocks).to_dense())


@needs_racah
@pytest.mark.parametrize("a_name", ["THREE", "THREEBAR", "SIX"])
def test_cup_on_two_direct_legs_is_z(a_name):
    """``V_a (x) V_dual(a)``: the two legs carry *different* labels, unlike SU(2)."""
    a = globals()[a_name]
    space = GradedSpace.new(SU3, {a: 1})
    partner = GradedSpace.new(SU3, {SU3.dual(a): 1})
    c = cup(space, partner, dual=False)
    np.testing.assert_allclose(c * sqrt(SU3.irrep_dim(a)), SU3.z_matrix(a), atol=1e-12)


@needs_racah
@pytest.mark.parametrize("a_name", ["THREE", "THREEBAR", "SIX", "EIGHT"])
def test_cup_with_a_dual_leg_is_the_evaluation_map(a_name):
    """``V_a (x) V_a^* -> 1``: the Z insertion turns the cup into the identity."""
    a = globals()[a_name]
    d = SU3.irrep_dim(a)
    space = GradedSpace.new(SU3, {a: 1})
    np.testing.assert_allclose(cup(space, space, dual=True) * sqrt(d), np.eye(d), atol=1e-12)


@needs_racah
@pytest.mark.parametrize("a_name", ["THREE", "SIX"])
def test_cap_against_cup_gives_the_identity_with_the_fs_sign(a_name):
    a = globals()[a_name]
    d = SU3.irrep_dim(a)
    c = cup(GradedSpace.new(SU3, {a: 1}), GradedSpace.new(SU3, {SU3.dual(a): 1}), dual=False)
    np.testing.assert_allclose((c @ c.conj().T) * d, np.eye(d), rtol=0.0, atol=1e-12)
    partner = cup(GradedSpace.new(SU3, {SU3.dual(a): 1}), GradedSpace.new(SU3, {a: 1}), dual=False)
    np.testing.assert_allclose((c @ partner) * d, SU3.frobenius_schur(a) * np.eye(d), atol=1e-12)


# --- save / load ---------------------------------------------------------------


@needs_racah
def test_gauge_embeds_the_racah_fingerprint():
    assert racah.sun_authority_fingerprint() in _SUN_GAUGE
    from tenet.serialize import _GAUGES

    assert _GAUGES["SUN"] == _SUN_GAUGE


@needs_racah
def test_save_load_round_trips_an_su3_tensor(tmp_path):
    t = su3(MIXED)
    tenet.save(t, tmp_path / "t.npz")
    back = tenet.load(tmp_path / "t.npz")
    assert back.structure == t.structure
    assert back.legs[0].space.provider == SU3  # ``n`` survived the header
    for got, want in zip(back.blocks, t.blocks, strict=True):
        np.testing.assert_array_equal(got, np.asarray(want))


def header_of(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(z["header"].item())


def rewrite_header(path, header: dict) -> None:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "header"}
    np.savez(path, header=np.array(json.dumps(header)), **arrays)


@needs_racah
def test_doctored_gauge_is_refused_with_the_fingerprint_in_the_message(tmp_path):
    path = tmp_path / "t.npz"
    tenet.save(su3(MIXED), path)
    header = header_of(path)
    assert header["gauges"] == {"SUN": _SUN_GAUGE}
    header["gauges"]["SUN"] = _SUN_GAUGE.replace("epoch=1", "epoch=2")
    rewrite_header(path, header)
    with pytest.raises(ValueError, match="different coefficient convention") as exc:
        tenet.load(path)
    assert racah.sun_authority_fingerprint() in str(exc.value)
    assert "epoch=2" in str(exc.value)


@needs_racah
def test_json_header_holds_the_dynkin_label_as_a_list(tmp_path):
    """``SUNSector`` normalizes back to a tuple, which is what makes the round trip exact."""
    path = tmp_path / "t.npz"
    tenet.save(su3(MIXED), path)
    header = header_of(path)
    assert header["legs"][0]["space"]["provider"] == {"kind": "SUN", "n": 3, "name": "SUN"}
    assert header["legs"][0]["space"]["sectors"] == [[[[1, 1]], 1]]
