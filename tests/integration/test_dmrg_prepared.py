"""M16 (#141): the prepared and dense two-site paths are the same operator.

The prepared path's one-sided terms use the sweep's mixed-canonical gauge, so the
comparison harness *walks* that gauge -- an exact-SVD two-site sweep with the
eigensolver removed -- and compares ``heff2``, ``update_`` (both directions) and
``measure`` at every bond, in both directions, for four Hamiltonians chosen to
populate every field pattern: nearest-neighbour Heisenberg, an R=4 power-law chain, a
width-6 cylinder, and a mixed set with a rank-3 charged operator next to rank-2k
invariant ones on non-adjacent sites. Bit-identity is deliberately not asserted: the
prepared path sums its terms in a different order.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples"))

import dmrg as example  # noqa: E402

import tenet  # noqa: E402
from tenet import GradedSpace  # noqa: E402
from tenet.network import MPO, MPS, Env, dmrg_, local_op  # noqa: E402
from tenet.symmetry import SU2, SU2Sector, U1Sector  # noqa: E402

E_OBC_12 = -5.142090632840532  # the open-boundary N=12 reference test_dmrg.py derives


def _ops():
    _, sz, sp, sm = example._spin_half()
    return (
        local_op(sz, phys=example.PHYS, charge=U1Sector(0)),
        local_op(sp, phys=example.PHYS, charge=U1Sector(-2)),
        local_op(sm, phys=example.PHYS, charge=U1Sector(2)),
    )


def _pair_terms(pairs, coeff=None):
    op_sz, op_sp, op_sm = _ops()
    terms = []
    for i, j in pairs:
        c = 1.0 if coeff is None else coeff(i, j)
        terms += [
            (c, [(op_sz, i), (op_sz, j)]),
            (0.5 * c, [(op_sp, i), (op_sm, j)]),
            (0.5 * c, [(op_sm, i), (op_sp, j)]),
        ]
    return terms


def _cylinder_pairs(n, ly):
    pairs = []
    for x in range(-(-n // ly)):
        for y in range(ly):
            i = x * ly + y
            pairs.append((i, x * ly + (y + 1) % ly))
            pairs.append((i, i + ly))
    return sorted({(min(i, j), max(i, j)) for i, j in pairs if i != j and max(i, j) < n})


def _sweep_worst(n, h, phys, bonds, seed=5):
    """Walk the sweep's gauge with exact SVDs; return the worst prepared-dense gap."""
    dense = MPO(h.sites)  # same tensors, no table: the dense path on the same input
    psi = MPS.random(phys, bonds, seed=seed).canonize_()
    envs = [Env(psi, h).setup_(), Env(psi, dense).setup_()]
    worst = 0.0
    for direction in ("right", "left"):
        bonds_iter = range(n - 1) if direction == "right" else range(n - 2, -1, -1)
        for b in bonds_iter:
            aa = tenet.einsum("apx,xqr->apqr", psi[b], psi[b + 1])
            yp, yd = (env.heff2(b, aa) for env in envs)
            worst = max(worst, float(tenet.norm(tenet.subtract(yp, yd))))
            u, s, vh = tenet.linalg.svd(aa, ((0, 1), (2, 3)))
            psi[b + 1] = vh
            if direction == "right":
                psi[b] = u
                psi[b + 1] = tenet.einsum("xy,yqr->xqr", s, psi[b + 1])
            else:
                psi[b] = tenet.einsum("apx,xy->apy", u, s)
            for env in envs:
                env.clear_(b, b + 1)
                if direction == "right":
                    env.update_(b, to="last")
                else:
                    env.update_(b + 1, to="first")
            key = (b, b + 1) if direction == "right" else (b + 1, b)
            worst = max(worst, float(tenet.norm(tenet.subtract(envs[0].F[key], envs[1].F[key]))))
    worst = max(worst, abs(Env(psi, h).measure() - Env(psi, dense).measure()))
    return worst


def test_prepared_agrees_with_dense_on_the_nn_heisenberg_chain():
    h = MPO.from_terms(8, _pair_terms([(i, i + 1) for i in range(7)]), cutoff=None)
    assert _sweep_worst(8, h, example.PHYS, example.bond_spaces(8)) < 1e-12


def test_prepared_agrees_with_dense_on_the_r4_power_law_chain():
    pairs = [(i, j) for i in range(12) for j in range(i + 1, min(i + 5, 12))]
    h = MPO.from_terms(12, _pair_terms(pairs, coeff=lambda i, j: (j - i) ** -2.0), cutoff=None)
    assert h[6].legs[0].space.dim == 14  # measurement 2's R=4 row: D_w = 14 uncompressed
    assert _sweep_worst(12, h, example.PHYS, example.bond_spaces(12)) < 1e-12


def test_prepared_agrees_with_dense_on_the_ly6_cylinder():
    h = MPO.from_terms(12, _pair_terms(_cylinder_pairs(12, 6)), cutoff=None)
    assert _sweep_worst(12, h, example.PHYS, example.bond_spaces(12)) < 1e-12


def test_prepared_agrees_with_dense_on_a_mixed_rank3_and_rank2k_term_set():
    op_sz, op_sp, op_sm = _ops()
    _, sz, _, _ = example._spin_half()
    op_zz = local_op(np.kron(sz, sz), phys=example.PHYS)
    terms = [
        (0.5, [(op_sp, 1), (op_sm, 4)]),
        (0.7, [(op_zz, (0, 3))]),  # invariant rank-4 on non-adjacent sites
        (2.5, [(op_sz, 0), (op_sz, 2), (op_sz, 5)]),  # a genuine continuing (A) edge
        (-1.3, [(op_sz, 3)]),  # an onsite (D) edge, so ID and DE are exercised
    ]
    h = MPO.from_terms(6, terms, cutoff=None)
    assert _sweep_worst(6, h, example.PHYS, example.bond_spaces(6)) < 1e-12


def test_prepared_agrees_with_dense_on_the_su2_heisenberg_chain():
    """The dual-convention MPO: k-site splits, capped boundary legs, graded aux bonds."""
    su2_phys = GradedSpace.new(SU2, {SU2Sector(1): 1})
    _, sz, sp, sm = example._spin_half()
    ss = local_op(np.kron(sz, sz) + (np.kron(sp, sm) + np.kron(sm, sp)) / 2, phys=su2_phys)
    h = MPO.from_terms(8, [(1.0, [(ss, (i, i + 1))]) for i in range(7)], cutoff=None)
    tri = GradedSpace.new(SU2, {SU2Sector(0): 1})
    mid = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2, SU2Sector(2): 1})
    assert _sweep_worst(8, h, su2_phys, [tri] + [mid] * 7 + [tri]) < 1e-12


def test_dmrg_reaches_the_n12_reference_on_both_paths():
    """The whole pipeline, both routes, to the tolerance ``test_dmrg.py`` already uses."""
    h = MPO.from_terms(12, _pair_terms([(i, i + 1) for i in range(11)]), cutoff=None)
    for ham in (h, MPO(h.sites)):
        psi = MPS.random(example.PHYS, example.bond_spaces(12), seed=0)
        out = dmrg_(psi, ham, chi=64)
        assert out.energy == pytest.approx(E_OBC_12, abs=1e-10)


def test_the_prepared_path_issues_fewer_dispatches_on_the_n24_ly10_cylinder():
    """#141's inequality on the measurement-2 object itself: N=24, width 10, D_w=32."""
    import autoray  # noqa: PLC0415

    h = MPO.from_terms(24, _pair_terms(_cylinder_pairs(24, 10)), cutoff=None)
    psi = MPS.random(example.PHYS, example.bond_spaces(24), seed=1).canonize_()
    envs = [Env(psi, h).setup_(), Env(psi, MPO(h.sites)).setup_()]
    for env in envs:
        for m in range(12):
            env.update_(m, to="last")
    aa = tenet.einsum("apx,xqr->apqr", psi[12], psi[13])

    def count(fn):
        state = {"n": 0, "depth": 0}
        orig = autoray.do

        def do(*args, **kwargs):
            state["n"] += state["depth"] == 0
            state["depth"] += 1
            try:
                return orig(*args, **kwargs)
            finally:
                state["depth"] -= 1

        autoray.do = do
        try:
            fn()
        finally:
            autoray.do = orig
        return state["n"]

    for env in envs:
        env.heff2(12, aa)  # warm the prepared, compiled and plan caches
    assert count(lambda: envs[0].heff2(12, aa)) < count(lambda: envs[1].heff2(12, aa))
