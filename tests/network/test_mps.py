"""``tenet.network.MPS``: the write barrier, the canonical form, and the two exits.

The write barrier is the point of the class (#112): every factorization in
``tenet.linalg`` lowers its input to a *map* first, so a rank-3 factor comes back on the
map's partition, and ``examples/toy_codes/dmrg.py`` used to repair that at each of two call sites
with a private ``_as_site``. Storing the factor is now the repair, and these tests are
what say so.
"""

import json

import heisenberg_walkthrough as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import MPS, Env, expectation_1site, expectation_2site
from tenet.network.mps import MPS_FORMAT_VERSION
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

# A four-site SU(2) chain of spin-1/2s: the container is provider-generic even though the
# Hamiltonian is not (REPOSITORY_RULES:61). ``SU2Sector(2 j)``, so 1 is the doublet.
SU2_PHYS = GradedSpace.new(SU2, {SU2Sector(1): 1})
SU2_BONDS = [
    GradedSpace.new(SU2, {SU2Sector(0): 1}),
    GradedSpace.new(SU2, {SU2Sector(1): 1}),
    GradedSpace.new(SU2, {SU2Sector(0): 1, SU2Sector(2): 1}),
    GradedSpace.new(SU2, {SU2Sector(1): 1}),
    GradedSpace.new(SU2, {SU2Sector(0): 1}),
]


def u1_mps(n_sites: int = 4, seed: int = 0) -> MPS:
    return MPS.random(example.PHYS, example.bond_spaces(n_sites), seed=seed)


def right_isometry_error(t: SymmetricTensor) -> float:
    """``|| A A^dag - 1 ||`` on the left bond of a right-canonical site tensor."""
    closed = tenet.einsum("apr,Apr->aA", t, tenet.adjoint(t))
    eye = tenet.identity((t.legs[0],))
    return float(tenet.norm(tenet.subtract(closed, eye)))


# --- the write barrier --------------------------------------------------------------


def test_an_lq_factor_stores_without_a_repartition():
    """The whole reason ``_as_site`` no longer exists anywhere.

    ``lq``'s ``q`` comes back on the map partition ``((0,), (1, 2))`` -- its physical leg
    IN-dual in the domain -- and storing it must land it on ``(bond OUT, phys OUT, bond
    IN)`` with the same numbers a caller's own ``tenet.repartition`` would have produced.
    """
    psi = u1_mps()
    _, q = tenet.linalg.lq(psi[2], ((0,), (1, 2)))
    assert tuple(leg.side for leg in q.legs) != (OUT, OUT, IN)  # bent, as delivered

    psi[2] = q
    assert tuple(leg.side for leg in psi[2].legs) == (OUT, OUT, IN)
    assert psi[2].structure == tenet.repartition(q, (0, 1), (2,)).structure
    assert float(tenet.norm(tenet.subtract(psi[2], tenet.repartition(q, (0, 1), (2,))))) == 0.0


def test_the_write_barrier_refuses_the_wrong_rank():
    """Rank 2 and rank 4 are not MPS sites, and silence would be a structure mismatch."""
    psi = u1_mps()
    rank2 = tenet.einsum("apr,Apr->aA", psi[1], tenet.adjoint(psi[1]))
    rank4 = tenet.einsum("apx,xqr->apqr", psi[1], psi[2])
    for bad in (rank2, rank4):
        with pytest.raises(ValueError, match="rank 3"):
            psi[1] = bad


def test_the_write_barrier_refuses_a_non_int_index():
    """YASTN ``_mps_parent.py``:88-105: a slice is not a site."""
    psi = u1_mps()
    with pytest.raises(TypeError, match="int"):
        psi[0:2] = psi[0]


# --- the canonical form -------------------------------------------------------------


def test_canonize_sets_the_centre_it_claims():
    psi = u1_mps(6).canonize_()
    assert psi.center == 0
    assert psi.canonize_(0) is psi  # in place, returns self -- YASTN _mps_obc.py:390


def test_canonize_leaves_every_site_but_the_centre_right_isometric():
    """``A_n A_n^dag = 1`` for ``n > 0``, to 1e-12, against ``tenet.identity``."""
    psi = u1_mps(6, seed=5).canonize_()
    for n in range(1, len(psi)):
        assert right_isometry_error(psi[n]) < 1e-12, n
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)


def test_canonize_refuses_a_centre_it_does_not_implement():
    with pytest.raises(NotImplementedError):
        u1_mps(4).canonize_(3)


# --- the constructors and the exits --------------------------------------------------


def test_random_uses_the_bond_spaces_it_is_given():
    spaces = example.bond_spaces(5)
    psi = MPS.random(example.PHYS, spaces, seed=2)
    assert len(psi) == 5
    for n in range(5):
        assert psi[n].legs[0].space == spaces[n]
        assert psi[n].legs[1].space == example.PHYS
        assert psi[n].legs[2].space == spaces[n + 1]
        assert tuple(leg.side for leg in psi[n].legs) == (OUT, OUT, IN)
    assert psi.center is None  # no claim made until canonize_


def test_from_tensors_round_trips_through_the_barrier():
    psi = u1_mps(4)
    again = MPS.from_tensors(psi.sites)
    assert [t.structure for t in again.sites] == [t.structure for t in psi.sites]


def test_to_dense_matches_a_hand_rolled_einsum_chain_at_n4():
    """The oracle exit against the chain ``tests/integration/test_dmrg.py``:305 writes."""
    psi = u1_mps(4, seed=7).canonize_()
    amplitudes = np.asarray(psi[0].to_dense())
    for site in psi.sites[1:]:
        amplitudes = np.einsum("a...b,bcd->a...cd", amplitudes, np.asarray(site.to_dense()))
    expected = amplitudes[0, ..., 0]
    assert np.abs(np.asarray(psi.to_dense()) - expected).max() < 1e-13
    assert np.linalg.norm(expected) == pytest.approx(psi.norm(), abs=1e-12)


def test_norm_matches_the_dense_norm_before_any_canonization():
    """The transfer pass against the exponential expansion, on a state with no structure.

    After ``canonize_`` the norm is 1 and the check is nearly free; this one runs on a raw
    random MPS, where the two routes share nothing but the answer.
    """
    psi = u1_mps(4, seed=3)
    dense = float(np.linalg.norm(np.asarray(psi.to_dense())))
    assert psi.norm() == pytest.approx(dense, rel=1e-12)


# --- M14: MPS.product -----------------------------------------------------------------


def test_product_puts_the_total_charge_on_bond_zero():
    """The target sector is readable off the result, and the amplitudes are exact.

    Dense physical index 0 is charge -1 (down) and 1 is charge +1 (up), so the Neel
    string's single amplitude sits at ``(1, 0, 1, 0, ...)`` -- exactly 1.0, everything
    else exactly zero, no tolerance.
    """
    neel = MPS.product(example.PHYS, [U1Sector(1), U1Sector(-1)] * 4)
    assert neel[0].legs[0].space == example.BOUNDARY
    expected = np.zeros((2,) * 8)
    expected[(1, 0) * 4] = 1.0
    assert np.array_equal(np.asarray(neel.to_dense()), expected)

    up = MPS.product(example.PHYS, [U1Sector(1)] * 4)
    assert up[0].legs[0].space == GradedSpace.new(U1, {U1Sector(4): 1})
    assert up[len(up) - 1].legs[2].space == example.BOUNDARY  # the other end is the unit


def test_product_is_normalized_and_measures_its_own_sectors():
    """``norm()`` is 1 and ``expectation_1site`` reads +-0.5 -- the #122 oracle."""
    states = [U1Sector(1), U1Sector(-1)] * 4
    psi = MPS.product(example.PHYS, states)
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)
    sz = SymmetricTensor.from_dense(
        np.diag([-0.5, 0.5]), (Leg(example.PHYS, OUT), Leg(example.PHYS, IN))
    )
    for n, a in enumerate(states):
        want = 0.5 if a == U1Sector(1) else -0.5
        assert expectation_1site(psi, sz, n) == pytest.approx(want, abs=1e-12)


def test_product_refuses_a_sector_absent_from_phys():
    with pytest.raises(ValueError, match="not in the physical space; available"):
        MPS.product(example.PHYS, [U1Sector(1), U1Sector(3)])


def test_product_refuses_a_degenerate_sector():
    """A product state names a *basis vector*; this constructor has no degeneracy slot."""
    wide = GradedSpace.new(U1, {U1Sector(0): 2})
    with pytest.raises(ValueError, match=r"degeneracy 2.*Sequence\[tuple\[Sector, int\]\]"):
        MPS.product(wide, [U1Sector(0), U1Sector(0)])


def test_product_refuses_a_non_abelian_fusion():
    """Abelian-only, permanently: a single spin-up is not an SU(2) multiplet."""
    with pytest.raises(ValueError, match=r"SU\(2\) multiplet.*MPS\.random"):
        MPS.product(SU2_PHYS, [SU2Sector(1)] * 4)


# --- provider-generic: the same container on SU(2) legs ------------------------------


def test_the_container_is_structural_on_su2_legs():
    """The write barrier and the canonical form on SU(2), where the bend is not free.

    U(1) bending coefficients are all 1, so a U(1)-only test cannot tell a correct
    repartition from a missing one. SU(2) carries genuine recoupling and Frobenius-Schur
    data, and ``A_n A_n^dag = 1`` here is a statement about the qdim weights as much as
    about the isometry. The *Hamiltonian* stays U(1) (#112 out of scope); the container
    does not have to.
    """
    psi = MPS.random(SU2_PHYS, SU2_BONDS, seed=1)
    assert len(psi) == 4
    for n in range(4):
        assert tuple(leg.side for leg in psi[n].legs) == (OUT, OUT, IN)

    psi.canonize_()
    assert psi.center == 0
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)
    for n in range(1, len(psi)):
        assert right_isometry_error(psi[n]) < 1e-12, n

    _, q = tenet.linalg.lq(psi[2], ((0,), (1, 2)))
    psi[2] = q
    assert tuple(leg.side for leg in psi[2].legs) == (OUT, OUT, IN)
    assert psi[2].legs[1].space == SU2_PHYS


# --- M11c: save / load ---------------------------------------------------------------


def su2_mps() -> MPS:
    return MPS.random(SU2_PHYS, SU2_BONDS, seed=1)


def assert_same(a: MPS, b: MPS) -> None:
    """Bit-for-bit, not ``allclose``: the same structure and the same block bytes."""
    assert len(a) == len(b) and a.center == b.center
    for x, y in zip(a.sites, b.sites, strict=True):
        assert x.structure == y.structure
        for p, q in zip(x.blocks, y.blocks, strict=True):
            assert np.array_equal(np.asarray(p), np.asarray(q))


def rewrite_header(path, edit) -> None:
    """Reopen a per-tensor ``.npz``, run ``edit`` on its JSON header, write it back."""
    with np.load(path, allow_pickle=False) as z:
        members = {k: z[k] for k in z.files}
    header = json.loads(str(members.pop("header").item()))
    edit(header)
    np.savez(path, header=np.array(json.dumps(header)), **members)


@pytest.mark.parametrize("build", [lambda: u1_mps(6, seed=4), su2_mps])
def test_save_load_round_trips_bit_for_bit(build, tmp_path):
    """U(1) at N=6 and SU(2) at N=4, through the per-tensor ``tenet.save``."""
    psi = build().canonize_()
    psi.save(tmp_path / "psi")
    assert_same(psi, MPS.load(tmp_path / "psi"))
    assert sorted(p.name for p in (tmp_path / "psi").iterdir()) == [
        f"{n:03d}.npz" for n in range(len(psi))
    ] + ["mps.json"]


def test_center_none_is_written_as_json_null_and_read_back_as_none(tmp_path):
    psi = u1_mps(4)
    assert psi.center is None
    psi.save(tmp_path / "psi")
    meta = json.loads((tmp_path / "psi" / "mps.json").read_text())
    assert meta == {"format": MPS_FORMAT_VERSION, "n_sites": 4, "center": None}
    assert MPS.load(tmp_path / "psi").center is None  # not 0


def test_the_gauge_check_propagates_through_the_per_tensor_route(tmp_path):
    """Why the format is per-tensor: #94's SU(2) gauge verification is inherited, not
    re-implemented, and a mutated header is refused with the offending file named."""
    su2_mps().save(tmp_path / "psi")
    rewrite_header(tmp_path / "psi" / "002.npz", lambda h: h["gauges"].update(SU2="not-the-gauge"))
    with pytest.raises(ValueError, match="002.npz.*gauge"):
        MPS.load(tmp_path / "psi")


def test_load_refuses_a_missing_manifest(tmp_path):
    (tmp_path / "psi").mkdir()
    with pytest.raises(ValueError, match="mps.json"):
        MPS.load(tmp_path / "psi")


def test_load_refuses_a_future_directory_format(tmp_path):
    u1_mps(4).save(tmp_path / "psi")
    meta = json.loads((tmp_path / "psi" / "mps.json").read_text())
    meta["format"] = MPS_FORMAT_VERSION + 1
    (tmp_path / "psi" / "mps.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="newer than this tenet"):
        MPS.load(tmp_path / "psi")


def test_load_refuses_a_missing_site_file(tmp_path):
    u1_mps(4).save(tmp_path / "psi")
    (tmp_path / "psi" / "002.npz").unlink()
    with pytest.raises(ValueError, match=r"missing \['002.npz'\]"):
        MPS.load(tmp_path / "psi")


def test_load_refuses_an_extra_site_file(tmp_path):
    u1_mps(4).save(tmp_path / "psi")
    (tmp_path / "psi" / "004.npz").write_bytes((tmp_path / "psi" / "003.npz").read_bytes())
    with pytest.raises(ValueError, match=r"unexpected \['004.npz'\]"):
        MPS.load(tmp_path / "psi")


def test_load_refuses_an_unexpected_member(tmp_path):
    u1_mps(4).save(tmp_path / "psi")
    (tmp_path / "psi" / "notes.txt").write_text("hello")
    with pytest.raises(ValueError, match=r"unexpected \['notes.txt'\]"):
        MPS.load(tmp_path / "psi")


@pytest.mark.parametrize("center", [4, -1])
def test_load_refuses_an_out_of_range_centre(center, tmp_path):
    u1_mps(4).save(tmp_path / "psi")
    meta = json.loads((tmp_path / "psi" / "mps.json").read_text())
    meta["center"] = center
    (tmp_path / "psi" / "mps.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="neither null nor in range"):
        MPS.load(tmp_path / "psi")


def test_load_refuses_a_neighbour_bond_mismatch(tmp_path):
    """The trust-boundary check. It is *not* in ``__setitem__``: ``sweep_`` writes
    ``psi[n + 1]`` before ``psi[n]``, so a sweeping MPS is transiently inconsistent by
    construction and the same check in the write barrier would fire on correct code."""
    u1_mps(4).save(tmp_path / "psi")
    (tmp_path / "psi" / "001.npz").write_bytes((tmp_path / "psi" / "002.npz").read_bytes())
    with pytest.raises(ValueError, match="right bond space does not match"):
        MPS.load(tmp_path / "psi")


def test_save_refuses_a_non_empty_destination_before_writing_anything(tmp_path):
    psi = u1_mps(4)
    psi.save(tmp_path / "psi")
    before = {p.name: p.read_bytes() for p in (tmp_path / "psi").iterdir()}
    with pytest.raises(FileExistsError, match="delete it or choose another"):
        u1_mps(6).save(tmp_path / "psi")
    after = {p.name: p.read_bytes() for p in (tmp_path / "psi").iterdir()}
    assert after == before  # the previous good checkpoint is untouched
    assert_same(psi, MPS.load(tmp_path / "psi"))


def test_a_converged_state_round_trips_into_the_same_energy(tmp_path):
    """The criterion that ties the format to a caller: a checkpoint is worth what it
    measures, and this is the only exit in the repository that says so."""
    out = example.dmrg(6, chi=16)
    out.psi.save(tmp_path / "psi")
    loaded = MPS.load(tmp_path / "psi")
    assert Env(loaded, example.mpo(6)).measure() == pytest.approx(out.energy, abs=1e-12)


def test_a_jax_backed_mps_saves_and_loads_back_as_numpy(tmp_path):
    pytest.importorskip("jax")
    psi = MPS.from_tensors(t.to_backend("jax") for t in u1_mps(4).canonize_())
    psi.save(tmp_path / "psi")
    loaded = MPS.load(tmp_path / "psi")
    assert all(isinstance(b, np.ndarray) for t in loaded.sites for b in t.blocks)
    assert loaded.norm() == pytest.approx(1.0, abs=1e-12)


# --- M11c: compress_ -----------------------------------------------------------------


def test_compress_at_a_large_chi_is_a_gauge_transformation():
    """No-op: nothing is discarded and the amplitudes are the same state."""
    psi = u1_mps(8, seed=11).canonize_()
    before = np.asarray(psi.to_dense())
    # sqrt of an O(eps) Pythagoras residual; linux BLAS measured 1.5e-8.
    assert psi.compress_(chi=64) < 1e-7
    assert np.abs(np.asarray(psi.to_dense()) - before).max() < 1e-12
    assert psi.norm() == pytest.approx(1.0, abs=1e-12)
    assert psi.center == len(psi) - 1


def test_compress_discards_monotonically_less_as_chi_grows():
    weights = [u1_mps(8, seed=11).compress_(chi=chi) for chi in (2, 4, 8, 16)]
    assert all(b <= a + 1e-14 for a, b in zip(weights, weights[1:], strict=False)), weights


def test_compress_is_idempotent_at_the_same_chi():
    psi = u1_mps(8, seed=11)
    psi.compress_(chi=4)
    before = np.asarray(psi.to_dense())
    assert psi.compress_(chi=4) < 1e-7  # same sqrt(eps) floor
    assert np.abs(np.asarray(psi.to_dense()) - before).max() < 1e-12


def test_compress_is_variational_on_a_converged_state():
    """The energy of a truncated ground state can only rise, and at the DMRG's own chi it
    does not move at all -- which is what makes the sweep a compression and not a bug."""
    out = example.dmrg(8, chi=8)
    h = example.mpo(8)
    energies = []
    for chi in (16, 8, 4, 2):
        psi = out.psi.copy()
        psi.compress_(chi=chi)
        energies.append(Env(psi, h).measure())
    assert energies[0] == pytest.approx(out.energy, abs=1e-10)
    assert all(b >= a - 1e-12 for a, b in zip(energies, energies[1:], strict=False)), energies


# --- M11c: expectation_1site / expectation_2site --------------------------------------


def h2_dense() -> np.ndarray:
    """The Heisenberg two-site term, ``[P, Q, p, q]`` in ``example.PHYS``'s basis."""
    _, sz, sp, sm = example._spin_half()
    pair = np.einsum("Pp,Qq->PQpq", sz, sz)
    return pair + 0.5 * np.einsum("Pp,Qq->PQpq", sp, sm) + 0.5 * np.einsum("Pp,Qq->PQpq", sm, sp)


def test_expectation_1site_matches_a_dense_oracle():
    for phys, psi in ((example.PHYS, u1_mps(4, seed=7)), (SU2_PHYS, su2_mps())):
        o = SymmetricTensor.random((Leg(phys, OUT), Leg(phys, IN)), seed=13)
        v, od = np.asarray(psi.to_dense()), np.asarray(o.to_dense())
        for n in range(len(psi)):
            applied = np.moveaxis(np.tensordot(od, v, axes=([1], [n])), 0, n)
            want = np.vdot(v, applied) / np.vdot(v, v)
            assert expectation_1site(psi, o, n) == pytest.approx(float(want), abs=1e-12)


def test_expectation_2site_matches_a_dense_oracle():
    for phys, psi in ((example.PHYS, u1_mps(4, seed=7)), (SU2_PHYS, su2_mps())):
        legs = (Leg(phys, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(phys, IN))
        o = SymmetricTensor.random(legs, seed=17)
        v, od = np.asarray(psi.to_dense()), np.asarray(o.to_dense())
        for n in range(len(psi) - 1):
            applied = np.tensordot(od, v, axes=([2, 3], [n, n + 1]))
            applied = np.moveaxis(applied, (0, 1), (n, n + 1))
            want = np.vdot(v, applied) / np.vdot(v, v)
            assert expectation_2site(psi, o, n) == pytest.approx(float(want), abs=1e-12)


def test_summing_expectation_2site_reproduces_the_mpo_energy():
    """The first cross-check in the repository between the MPO route and a direct
    observable: two independent contractions of the same Hamiltonian."""
    out = example.dmrg(6, chi=16)
    phys = example.PHYS
    legs = (Leg(phys, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(phys, IN))
    h2 = SymmetricTensor.from_dense(h2_dense(), legs)
    total = sum(expectation_2site(out.psi, h2, n) for n in range(len(out.psi) - 1))
    assert total == pytest.approx(Env(out.psi, example.mpo(6)).measure(), abs=1e-10)


@pytest.mark.parametrize("canonize", [True, False])
def test_the_identity_expectation_is_one_canonical_or_not(canonize):
    """The non-canonical half is what proves the ``<psi|psi>`` division is doing work: a
    raw random MPS has a norm far from 1, and an undivided ``measure_*`` would return it."""
    psi = u1_mps(4, seed=3)
    if canonize:
        psi.canonize_()
    else:
        assert abs(psi.norm() - 1.0) > 1e-3
    eye = tenet.identity((Leg(example.PHYS, OUT),))
    for n in range(len(psi)):
        assert expectation_1site(psi, eye, n) == pytest.approx(1.0, abs=1e-12)


def test_the_measurements_refuse_a_bad_site_or_a_bad_rank():
    psi = u1_mps(4)
    eye = tenet.identity((Leg(example.PHYS, OUT),))
    with pytest.raises(ValueError, match="out of range; expected 0 <= n <= 3"):
        expectation_1site(psi, eye, 4)
    with pytest.raises(ValueError, match="out of range; expected 0 <= n <= 2"):
        expectation_2site(psi, eye, 3)
    with pytest.raises(ValueError, match="must be rank 4"):
        expectation_2site(psi, eye, 0)
    with pytest.raises(ValueError, match="must be rank 2"):
        expectation_1site(psi, tenet.einsum("Pp,Qq->PQpq", eye, eye), 0)
