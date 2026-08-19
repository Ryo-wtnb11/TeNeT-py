"""Gate 1 for #204: the corner-pinned compression's bond width against the free one.

The design comment on #204 pins the two corner channels (``_IDL``/``_IDR``) through both
truncating sweeps so that a *compressed* MPO still carries the ``IdL (+) open (+) IdR``
partition ``Env.heff2``'s prepared path eats. Pinning restricts the gauge freedom the two
SVDs have, so it can only cost bond width, never save it. **This script is the
measurement of that cost**, per cut, on the same input set ``bench_qc_mpo.py`` uses:

    uv run python benchmarks/bench_pinned_mpo.py                # the shipped input set
    uv run python benchmarks/bench_pinned_mpo.py --synthetic-only
    uv run python benchmarks/bench_pinned_mpo.py --only H4.STO6G.R1.8

Gate: ``max(pinned / free)`` over the cuts of every fixture must stay at or under 1.10.
The two boundary cuts are reported separately as well, because a cut whose free width is
2 or 3 moves the ratio by a third for one kept direction and says nothing about the
ab initio bond the issue is about.

The prototype is also **checked for correctness**, not only for width: on every fixture
small enough to expand, the pinned operator's ``to_dense`` is compared against the freely
compressed one and against the uncompressed ``from_terms``. That check is what a width
number is worth nothing without.

Not a test, not part of the package, on no CI path. It reuses ``bench_qc_mpo.py``'s
FCIDUMP fetch, its synthetic generator and its term folding unchanged, so the licence
decision recorded in that module's docstring covers this one too.
"""

import argparse
import json
import sys
import time

import numpy as np

import tenet
from tenet import IN, GradedSpace, Leg, SymmetricTensor
from tenet.network import MPO, local_op
from tenet.network.mps import _as_w, _compress_forward, _instantiate
from tenet.symmetry import U1, FZ2Sector, U1Sector, fZ2

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import bench_qc_mpo as qc  # noqa: E402


# --- the prototype ------------------------------------------------------------------
#
# Two sweeps, mirroring ``_instantiate`` and ``_compress_forward`` exactly except for
# what the SVD is allowed to rotate. The layout fact both halves ride on: a cut's bond is
# ``_merge``'s direct sum over ``ordered[i] = [_IDL, *open, _IDR]``, and ``_IDL``/``_IDR``
# both carry the trivial ``D=1`` unit space (``_Walk.__init__``), so the two corner
# channels are the *first* and *last* degeneracy slot of the bond's unit sector and
# nothing else. Pinning is then: take those two rows out, rotate what is left, put the
# three slabs back in that order.


def corner_rows(bond, sym):
    """``{'idl': row, 'idr': row}``: the dense row a cut's corner channels occupy."""
    off = bond.sector_offset(sym.unit)
    m = dict(bond.sectors)[sym.unit]
    return {"idl": off, "idr": off + m - 1}


def corner_map(bond, row, legs, *, column):
    """The 1-row / 1-column selector of one corner channel, on the given legs.

    A dense ``(D, 1)`` (or ``(1, D)``) indicator through ``from_dense`` with the legs
    declared, the way ``_channel_map`` and ``_group_embedding`` build theirs: the slab is
    *read off* a tensor rather than hand-derived, which is what keeps the dual flags and
    any braiding sign honest.
    """
    dense = np.zeros((bond.dim, 1))
    dense[row, 0] = 1.0
    return SymmetricTensor.from_dense(dense if column else dense.T, legs)


def _match(t, dual, axis):
    return t if t.legs[axis].dual == dual else tenet.flip_dual(t, axis)


def _chain(parts, axes):
    out = parts[0]
    for p in parts[1:]:
        out = tenet.direct_sum(out, p, axes=axes)
    return out


def pinned_instantiate(tab, cutoff):
    """``_instantiate`` with the corner channels pinned; returns ``(sites, corners)``."""
    sym = tab.phys.provider
    unit = GradedSpace.new(sym, {sym.unit: 1})
    out, carry, corners = [], None, [None] * (len(tab.edges) + 1)
    for n in reversed(range(len(tab.edges))):
        w = tab.site(n, carry)
        if n:
            bond, db = tab.bonds[n], tab.dual
            rows = corner_rows(bond, sym)
            live = [g for g in ("idl", "idr") if tab.groups[n][g] is not None]
            slabs, rest = {}, w
            for g in live:
                right = corner_map(
                    bond, rows[g], (Leg(unit, IN, db), Leg(bond, tenet.OUT, db)), column=False
                )
                left = corner_map(
                    bond, rows[g], (Leg(bond, IN, db), Leg(unit, tenet.OUT, db)), column=True
                )
                slabs[g] = tenet.einsum("xpqb,vx->vpqb", w, right)
                rest = tenet.subtract(rest, tenet.einsum("vpqb,xv->xpqb", slabs[g], left))
            u, s, vh = tenet.linalg.svd_truncated(rest, ((0,), (1, 2, 3)), cutoff=cutoff)
            open_site, open_carry = _as_w(vh), tenet.einsum("xy,yz->xz", u, s)
            d0, ref = open_site.legs[0].dual, open_carry.legs[1]
            site_parts, carry_parts = [], []
            for g in ("idl", "open", "idr"):
                if g == "open":
                    site_parts.append(open_site)
                    carry_parts.append(open_carry)
                elif g in live:
                    site_parts.append(_match(slabs[g], d0, 0))
                    carry_parts.append(
                        corner_map(
                            bond,
                            rows[g],
                            (open_carry.legs[0], Leg(unit, ref.side, ref.dual)),
                            column=True,
                        )
                    )
            w = _chain(site_parts, 0)
            carry = tenet.repartition(_chain(carry_parts, 1), (), (0, 1))
            corners[n] = (
                unit if "idl" in live else None,
                open_site.legs[0].space,
                unit if "idr" in live else None,
            )
        out.append(w)
    out.reverse()
    return out, corners


def pinned_forward(sites, corners, cutoff):
    """``_compress_forward`` mirrored onto the open column slab."""
    sym = sites[0].legs[1].space.provider
    unit = GradedSpace.new(sym, {sym.unit: 1})
    out = list(sites)
    for n in range(len(out) - 1):
        idl, _open_space, idr = corners[n + 1]
        bond, db = out[n].legs[3].space, out[n].legs[3].dual
        rows = corner_rows(bond, sym)
        live = [g for g, s in (("idl", idl), ("idr", idr)) if s is not None]
        slabs, rest = {}, out[n]
        for g in live:
            left = corner_map(
                bond, rows[g], (Leg(bond, IN, db), Leg(unit, tenet.OUT, db)), column=True
            )
            right = corner_map(
                bond, rows[g], (Leg(unit, IN, db), Leg(bond, tenet.OUT, db)), column=False
            )
            slabs[g] = tenet.einsum("xv,apqx->apqv", left, out[n])
            rest = tenet.subtract(rest, tenet.einsum("vx,apqv->apqx", right, slabs[g]))
        u, s, vh = tenet.linalg.svd_truncated(rest, ((0, 1, 2), (3,)), cutoff=cutoff)
        open_site, open_carry = _as_w(u), tenet.einsum("xy,yz->xz", s, vh)
        d3, ref = open_site.legs[3].dual, open_carry.legs[0]
        site_parts, carry_parts = [], []
        for g in ("idl", "open", "idr"):
            if g == "open":
                site_parts.append(open_site)
                carry_parts.append(open_carry)
            elif g in live:
                site_parts.append(_match(slabs[g], d3, 3))
                carry_parts.append(
                    corner_map(
                        bond,
                        rows[g],
                        (Leg(unit, ref.side, ref.dual), open_carry.legs[1]),
                        column=False,
                    )
                )
        out[n] = _chain(site_parts, 3)
        carry = tenet.repartition(_chain(carry_parts, 0), (0, 1), ())
        out[n + 1] = _as_w(tenet.einsum("ypqr,xy->xpqr", out[n + 1], carry))
        corners[n + 1] = (idl, open_site.legs[3].space, idr)
    return out


def pinned_sites(tab, cutoff):
    sites, corners = pinned_instantiate(tab, cutoff)
    return pinned_forward(sites, corners, cutoff), corners


# --- the corner-exactness check -----------------------------------------------------


def corners_are_unit(sites, corners):
    """Every compressed cut's two corner channels are exact unit slabs.

    Read off the dense site tensor: with the bond ordered ``IdL, open, IdR``, the ``IdL``
    channel is row 0 / column 0 of the unit sector and the ``IdR`` channel the last of
    each, so ``W[IdL, :, :, IdL]`` and ``W[IdR, :, :, IdR]`` must be the identity on the
    physical space and the ``IdL`` *column* / ``IdR`` *row* must be zero everywhere else
    -- the partition ``EdgeBlocks`` needs, stated as an assertion on the numbers.
    """
    worst = 0.0
    for n, w in enumerate(sites):
        dense = np.asarray(w.to_dense())
        d = w.legs[1].space.dim
        eye = np.eye(d)
        for side, cut in (("l", n), ("r", n + 1)):
            pass
        for name, index in (("idl", 0), ("idr", -1)):
            lo = corners[n]
            hi = corners[n + 1]
            if lo is None or hi is None:
                continue
            g = 0 if name == "idl" else 2
            if lo[g] is None or hi[g] is None:
                continue
            r = _corner_row(w.legs[0].space, w.legs[0].space.provider, name)
            c = _corner_row(w.legs[3].space, w.legs[3].space.provider, name)
            worst = max(worst, float(np.abs(dense[r, :, :, c] - eye).max()))
            if name == "idl":  # nothing enters IdL from the open block or from IdR
                col = dense[:, :, :, c].copy()
                col[r] = 0.0
                worst = max(worst, float(np.abs(col).max()))
            else:  # nothing leaves IdR into the open block or into IdL
                row = dense[r].copy()
                row[:, :, c] = 0.0
                worst = max(worst, float(np.abs(row).max()))
    return worst


def _corner_row(space, sym, name):
    off = space.sector_offset(sym.unit)
    return off if name == "idl" else off + dict(space.sectors)[sym.unit] - 1


# --- the measurement ----------------------------------------------------------------


def widths(sites):
    return [t.legs[0].space.dim for t in sites] + [sites[-1].legs[3].space.dim]


def measure(name, n_sites, terms, cutoff=1e-13, dense_check=False):
    t0 = time.perf_counter()
    free = MPO.from_terms(n_sites, terms, cutoff=cutoff)
    t_free = time.perf_counter() - t0
    t0 = time.perf_counter()
    tab = MPO.from_terms(n_sites, terms, cutoff=None).edges
    sites, corners = pinned_sites(tab, cutoff)
    t_pin = time.perf_counter() - t0
    wf, wp = widths(free.sites), widths(sites)
    row = {
        "name": name,
        "n_sites": n_sites,
        "free": wf,
        "pinned": wp,
        "wall_free": round(t_free, 2),
        "wall_pinned": round(t_pin, 2),
        "corner_err": corners_are_unit(sites, corners) if n_sites <= 12 else None,
    }
    if dense_check:
        ref = np.asarray(MPO.from_terms(n_sites, terms, cutoff=None).to_dense())
        row["err_free"] = float(np.abs(np.asarray(free.to_dense()) - ref).max())
        row["err_pinned"] = float(np.abs(np.asarray(MPO(sites).to_dense()) - ref).max())
    return row


def ratios(row):
    pairs = [
        (p / f, i) for i, (f, p) in enumerate(zip(row["free"], row["pinned"], strict=True)) if f
    ]
    interior = [r for r, i in pairs if 2 < i < len(row["free"]) - 3]
    return max(r for r, _ in pairs), (max(interior) if interior else 1.0)


# --- small graded fixtures, for the correctness half --------------------------------

_A = np.array([[0.0, 1.0], [0.0, 0.0]])
U1_PHYS = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
FZ2_PHYS = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
PHYS4 = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})
_C_UP = np.zeros((4, 4))
_C_UP[0, 2] = _C_UP[3, 1] = 1.0
_C_DN = np.zeros((4, 4))
_C_DN[0, 3] = 1.0
_C_DN[2, 1] = -1.0


def spin_chain(n):
    """U(1) Heisenberg plus a power-law ``SzSz`` tail -- something for the sweep to cut."""
    sp = local_op(_A.T, phys=U1_PHYS, charge=U1Sector(-2))
    sm = local_op(_A, phys=U1_PHYS, charge=U1Sector(2))
    sz = local_op(np.diag([-0.5, 0.5]), phys=U1_PHYS, charge=U1Sector(0))
    t = []
    for i in range(n - 1):
        t += [(0.5, [(sp, i), (sm, i + 1)]), (0.5, [(sm, i), (sp, i + 1)])]
    for i in range(n):
        for j in range(i + 1, n):
            t.append((1.0 / (j - i) ** 3, [(sz, i), (sz, j)]))
    return t


def spinless_chain(n):
    """fZ2 hopping plus a power-law density tail."""
    cd = local_op(_A.T, phys=FZ2_PHYS, charge=FZ2Sector(1))
    c = local_op(_A, phys=FZ2_PHYS, charge=FZ2Sector(1))
    nop = local_op(_A.T @ _A, phys=FZ2_PHYS, charge=FZ2Sector(0))
    t = []
    for i in range(n - 1):
        t += [(-1.0, [(cd, i), (c, i + 1)]), (-1.0, [(cd, i + 1), (c, i)])]
    for i in range(n):
        for j in range(i + 1, n):
            t.append((0.8 / (j - i) ** 3, [(nop, i), (nop, j)]))
    return t


def hubbard_chain(n):
    """The spinful ``d=4`` Hubbard chain of ``tests/network/test_hubbard.py``."""
    ops = {}
    for label, m in (("cu", _C_UP), ("cd", _C_DN)):
        ops[label] = local_op(m, phys=PHYS4, charge=FZ2Sector(1))
        ops[label + "+"] = local_op(m.T, phys=PHYS4, charge=FZ2Sector(1))
    nn = local_op((_C_UP.T @ _C_UP) @ (_C_DN.T @ _C_DN), phys=PHYS4, charge=FZ2Sector(0))
    t = [(4.0, [(nn, i)]) for i in range(n)]
    for i in range(n - 1):
        for f in ("cu", "cd"):
            t += [
                (-1.0, [(ops[f + "+"], i), (ops[f], i + 1)]),
                (-1.0, [(ops[f + "+"], i + 1), (ops[f], i)]),
            ]
    return t


SMALL = {
    "spin-U1-6": (6, spin_chain(6)),
    "spinless-fZ2-6": (6, spinless_chain(6)),
    "hubbard-fZ2-4": (4, hubbard_chain(4)),
}


# --- driver -------------------------------------------------------------------------


def qc_terms(name):
    norb, _nelec, recs = (
        qc.synthetic(int(name[4:])) if name.startswith("syn-") else qc.fetch(name)
    )
    screen = 1e-6 if name.startswith("syn-") else qc.SCREEN
    folded, _refused = qc.fold_terms(qc.spin_orbital_terms(recs, screen=screen))
    return 2 * norb, qc.to_tenet_terms(folded)


FIXTURES = [
    "H4.STO6G.R1.8",
    "H8.STO6G.R1.8",
    "N2.STO3G",
    "H10.STO6G.R1.8",
    "N2.CAS.6-31G",
    "C2.CAS.PVDZ",
    "syn-42",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--cutoff", type=float, default=1e-13)
    a = ap.parse_args()

    print("# correctness: pinned vs free vs uncompressed to_dense\n")
    for name, (n, terms) in SMALL.items():
        row = measure(name, n, terms, cutoff=a.cutoff, dense_check=True)
        hi, _ = ratios(row)
        print(
            f"{name:<18} err(free) {row['err_free']:.2e}  err(pinned) {row['err_pinned']:.2e}"
            f"  corner {row['corner_err']:.2e}  max ratio {hi:.3f}"
        )
        print(f"{'':<18} free   {row['free']}")
        print(f"{'':<18} pinned {row['pinned']}")

    names = FIXTURES
    if a.synthetic_only:
        names = [x for x in names if x.startswith("syn-")]
    if a.only:
        names = [x for x in names if x in a.only]
    print("\n# gate 1: pinned vs free bond width per cut\n")
    print(f"{'fixture':<16} {'N':>4} {'max free':>9} {'max pin':>8} {'max ratio':>10} "
          f"{'interior':>9} {'verdict':>8}")
    verdicts = []
    for name in names:
        n, terms = qc_terms(name)
        row = measure(name, n, terms, cutoff=a.cutoff)
        hi, interior = ratios(row)
        ok = hi <= 1.10
        verdicts.append(ok)
        print(
            f"{name:<16} {n:>4} {max(row['free']):>9} {max(row['pinned']):>8} "
            f"{hi:>10.3f} {interior:>9.3f} {'PASS' if ok else 'FAIL':>8}"
        )
        qc.note(kind="pinned-width", **row)
    print("\n# gate 1 verdict:", "PASS" if all(verdicts) else "FAIL")


if __name__ == "__main__":
    main()
