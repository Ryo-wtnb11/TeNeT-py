"""Gate 1 for #204: the corner-pinned compression's bond width against the free one.

The design comment on #204 pins the two corner channels (``_IDL``/``_IDR``) through both
truncating sweeps so that a *compressed* MPO still carries the ``IdL (+) open (+) IdR``
partition ``Env.heff2``'s prepared path eats. Pinning restricts the gauge freedom the two
SVDs have, so per cut it can cost bond width. **This script is the measurement of that
cost**, per cut, on the same input set ``bench_qc_mpo.py`` uses:

    uv run python benchmarks/bench_pinned_mpo.py                # the shipped input set
    uv run python benchmarks/bench_pinned_mpo.py --synthetic-only
    uv run python benchmarks/bench_pinned_mpo.py --only H4.STO6G.R1.8

The pinned side is the shipped ``MPO.from_terms``; the free side is the sweep as it stood
before #204, kept here in ``free_sites`` because that is the only place a comparison
needs it. Gate: ``max(pinned / free)`` over the cuts of every fixture at or under 1.10,
reported per cut, with the two cuts adjacent to the boundary called out separately --
a cut whose free width is 4 moves the ratio by a quarter for one kept direction and says
nothing about the ab initio bond the issue is about.

The script also checks that the pinned operator is the *same operator*: on every fixture
small enough to expand, ``to_dense`` against the free sweep and against the uncompressed
``from_terms``. A width number is worth nothing without it.

``bench_qc_mpo.py``'s own ``MVC`` row is the third opinion and the sharper one: it computes
a minimum vertex cover of each cut combinatorially, and the pinned bond equals it at every
cut. The free sweep's boundary-adjacent 4 is *below* the cover, which is what mixing the
``IdL``/``IdR`` channels away buys and what pinning declines to buy.

Not a test, not part of the package, on no CI path. It reuses ``bench_qc_mpo.py``'s
FCIDUMP fetch, its synthetic generator and its term folding unchanged, so the licence
decision recorded in that module's docstring covers this one too.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

import tenet
from tenet import GradedSpace
from tenet.network import MPO, local_op
from tenet.network.mps import _as_w
from tenet.symmetry import U1, FZ2Sector, U1Sector, fZ2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bench_qc_mpo as qc  # noqa: E402

# --- the free sweeps, as they stood before #204 -------------------------------------


def free_sites(tab, cutoff):
    """``_instantiate`` + ``_compress_forward`` with the SVD free to rotate the whole bond."""
    out, carry = [], None
    for n in reversed(range(len(tab.edges))):
        w = tab.site(n, carry)
        if n:
            u, s, vh = tenet.linalg.svd_truncated(w, ((0,), (1, 2, 3)), cutoff=cutoff)
            w = _as_w(vh)
            carry = tenet.repartition(tenet.einsum("xy,yz->xz", u, s), (), (0, 1))
        out.append(w)
    out.reverse()
    for n in range(len(out) - 1):
        u, s, vh = tenet.linalg.svd_truncated(out[n], ((0, 1, 2), (3,)), cutoff=cutoff)
        out[n] = _as_w(u)
        carry = tenet.repartition(tenet.einsum("xy,yz->xz", s, vh), (0, 1), ())
        out[n + 1] = _as_w(tenet.einsum("ypqr,xy->xpqr", out[n + 1], carry))
    return out


# --- the measurement ----------------------------------------------------------------


def widths(sites):
    return [t.legs[0].space.dim for t in sites] + [sites[-1].legs[3].space.dim]


def measure(name, n_sites, terms, cutoff=1e-13, dense_check=False):
    t0 = time.perf_counter()
    free = free_sites(MPO.from_terms(n_sites, terms, cutoff=None).edges, cutoff)
    t_free = time.perf_counter() - t0
    t0 = time.perf_counter()
    pinned = MPO.from_terms(n_sites, terms, cutoff=cutoff)
    t_pin = time.perf_counter() - t0
    row = {
        "name": name,
        "n_sites": n_sites,
        "free": widths(free),
        "pinned": widths(pinned.sites),
        "wall_free": round(t_free, 2),
        "wall_pinned": round(t_pin, 2),
    }
    if dense_check:
        ref = np.asarray(MPO.from_terms(n_sites, terms, cutoff=None).to_dense())
        row["err_free"] = float(np.abs(np.asarray(MPO(free).to_dense()) - ref).max())
        row["err_pinned"] = float(np.abs(np.asarray(pinned.to_dense()) - ref).max())
    return row


def ratios(row):
    """``(max over every cut, max away from the two boundary-adjacent cuts)``."""
    n = len(row["free"])
    pairs = [
        (p / f, i) for i, (f, p) in enumerate(zip(row["free"], row["pinned"], strict=True)) if f
    ]
    inner = [r for r, i in pairs if 1 < i < n - 2]
    return max(r for r, _ in pairs), (max(inner) if inner else 1.0)


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
    norb, _nelec, recs = qc.synthetic(int(name[4:])) if name.startswith("syn-") else qc.fetch(name)
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


# --- gate 2: a full DMRG run through each of the three routes -----------------------


def dmrg_run(name, variant, chi, sweeps):
    """One full DMRG at fixed sweep count; peak RSS and wall, per operator route.

    Three routes, the same arithmetic:

    * ``pinned`` -- ``from_terms(cutoff=1e-13)``, the compressed **prepared** path #204
      builds;
    * ``fsm`` -- ``from_terms(cutoff=None)``, the prepared path on the uncompressed
      finite-state-machine bond, which is #203's measurement and the number to beat;
    * ``free-dense`` -- the freely compressed sites in a bare ``MPO``, so ``heff2`` takes
      its dense path. This is gate 2's zeroth-order baseline: the alternative that needs
      no new code at all, reported beside the others whichever way it comes out.

    ``sweep_`` is driven directly rather than through ``dmrg_`` so the sweep count is
    fixed and the three routes do the same work, which is ``bench_qc_mpo.dmrg_run``'s
    reason too.
    """
    import resource  # noqa: PLC0415

    from tenet.network import MPS, Env, sweep_  # noqa: PLC0415

    def rss():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**30 if sys.platform == "darwin" else peak / 2**20

    t0 = time.perf_counter()
    n_sites, terms = qc_terms(name)
    if variant == "pinned":
        h = MPO.from_terms(n_sites, terms, cutoff=1e-13)
    elif variant == "fsm":
        h = MPO.from_terms(n_sites, terms, cutoff=None)
    else:
        h = MPO(free_sites(MPO.from_terms(n_sites, terms, cutoff=None).edges, 1e-13))
    t_build = time.perf_counter() - t0
    qc.note(event="build", name=name, variant=variant, t=round(t_build, 1), rss=round(rss(), 2))

    phys = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
    triv = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
    mid = GradedSpace.new(fZ2, {FZ2Sector(0): chi - chi // 2, FZ2Sector(1): chi // 2})
    psi = MPS.random(phys, [triv] + [mid] * (n_sites - 1) + [triv], seed=0)
    t1 = time.perf_counter()
    psi.canonize_(0)
    env = Env(psi, h).setup_(0)
    schmidt, energies = {}, []
    for it in range(sweeps):
        t2 = time.perf_counter()
        energy, _ = sweep_(psi, h, env, schmidt, chi=chi, cutoff=1e-10)
        energies.append(energy)
        qc.note(event="sweep", name=name, variant=variant, sweep=it, energy=energy,
                t=round(time.perf_counter() - t2, 1), rss=round(rss(), 2))  # fmt: skip
    qc.note(event="dmrg", name=name, variant=variant, chi=chi, sweeps=sweeps,
            energy=energies[-1], energies=energies, t_build=round(t_build, 1),
            t_sweeps=round(time.perf_counter() - t1, 1), rss=round(rss(), 2))  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--cutoff", type=float, default=1e-13)
    ap.add_argument("--dmrg", choices=("pinned", "fsm", "free-dense"))
    ap.add_argument("--chi", type=int, default=16)
    ap.add_argument("--sweeps", type=int, default=3)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if a.dmrg:
        for name in a.only or ["N2.CAS.6-31G"]:
            dmrg_run(name, a.dmrg, a.chi, a.sweeps)
        return

    print("# correctness: pinned vs free vs uncompressed to_dense\n")
    for name, (n, terms) in SMALL.items():
        row = measure(name, n, terms, cutoff=a.cutoff, dense_check=True)
        print(f"{name:<18} err(free) {row['err_free']:.2e}  err(pinned) {row['err_pinned']:.2e}")
        print(f"{'':<18} free   {row['free']}")
        print(f"{'':<18} pinned {row['pinned']}")

    names = FIXTURES
    if a.synthetic_only:
        names = [x for x in names if x.startswith("syn-")]
    if a.only:
        names = [x for x in names if x in a.only]
    print("\n# gate 1: pinned vs free bond width per cut\n")
    print(
        f"{'fixture':<16} {'N':>4} {'max free':>9} {'max pin':>8} {'all cuts':>9} "
        f"{'inner cuts':>11} {'verdict':>8}"
    )
    verdicts = []
    for name in names:
        n, terms = qc_terms(name)
        row = measure(name, n, terms, cutoff=a.cutoff)
        hi, inner = ratios(row)
        verdicts.append(hi <= 1.10)
        print(
            f"{name:<16} {n:>4} {max(row['free']):>9} {max(row['pinned']):>8} "
            f"{hi:>9.3f} {inner:>11.3f} {'PASS' if hi <= 1.10 else 'FAIL':>8}"
        )
        qc.note(kind="pinned-width", **row)
    print("\n# gate 1 verdict, every cut:", "PASS" if all(verdicts) else "FAIL")


if __name__ == "__main__":
    main()
