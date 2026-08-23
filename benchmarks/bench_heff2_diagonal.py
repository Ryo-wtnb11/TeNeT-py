"""Can ``diag(H_eff)`` be built from the same blocks the matvec uses? (#219)

M53 asked for block2's other half: the diagonal of the two-site effective Hamiltonian
(``effective_hamiltonian.hpp``:141-146 allocates it, :188-201 fills it through a dedicated
``tensor_product_diagonal``) as the preconditioner of a Davidson solve
(``iterative_matrix_functions.hpp``:66-72). The premise to establish first is that the
diagonal can be produced from the same environments and blocks ``_cores2``/``_build2``
already assemble, without forming ``H_eff``.

This instrument answers that, and it is a **refusal** instrument: it measures the
candidate against the truth and reports the gap.

* ``exact_diagonal`` -- the truth, and it needs no new machinery: probe
  [Env.heff2][tenet.network.Env.heff2] with the reduced basis' unit vectors and read the
  probed entry back. ``dim(aa)`` matvecs, so it is a fixture-sized oracle and nothing else.
* ``factorized_diagonal`` -- the candidate every "take only the diagonal in the two-site
  index" reading of the issue means:
  ``sum_{x,m,y} GL[a,x,a] W1[x,p,p,m] W2[m,q,q,y] GR[r,y,r]``, evaluated on reduced
  blocks over the site-tensor (compatibility) path, where the two ``W``'s are in hand.
* ``cost`` -- what the candidate *would* cost, dense, against one matvec of the same
  shape, so that the refusal is not mistaken for a cost refusal.

Run: ``uv run python benchmarks/bench_heff2_diagonal.py``. On no CI path.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples" / "toy_codes"))

import dmrg as example  # noqa: E402

import tenet  # noqa: E402
from tenet import GradedSpace, SymmetricTensor  # noqa: E402
from tenet.network import MPO, MPS, Env, local_op  # noqa: E402
from tenet.symmetry import SU2, FZ2Sector, SU2Sector, U1Sector, fZ2  # noqa: E402


def exact_diagonal(env, n, aa):
    """``{block key: array}``, the true diagonal in the reduced storage basis.

    One ``heff2`` per reduced entry: seed a unit vector at that entry, apply, read the
    same entry back. This is the definition of the diagonal of the matrix the solver
    actually iterates on, and it assumes nothing about how it might be contracted.
    """
    out = {}
    for i, (key, blk) in enumerate(aa.items()):
        arr = np.zeros(blk.shape)
        for j in range(blk.size):
            z = [np.zeros_like(b) for b in aa.blocks]
            z[i].flat[j] = 1.0
            probe = SymmetricTensor(aa.structure, tuple(z))
            arr.flat[j] = env.heff2(n, probe).blocks[i].flat[j]
        out[key] = arr
    return out


def _sectors(t):
    """``{axis-sector tuple: reduced block}``, summed over keys that share the tuple."""
    grouped = {}
    for key, blk in t.items():
        grouped.setdefault(tuple(t.structure.axis_sectors(key)), []).append(blk)
    return {k: sum(v) for k, v in grouped.items()}


def factorized_diagonal(env, h, n):
    """``{axis-sector tuple: array}``, the leg-factorized candidate on reduced blocks.

    ``GL[a, x, a]``, ``W[x, p, p, m]`` and ``GR[r, y, r]`` are numpy diagonals of the
    reduced blocks -- deliberately below the public API, which is half of what the
    measurement is about (``tests/network/test_hygiene.py`` forbids exactly this inside
    ``src/tenet/network/``).
    """
    gl, gr = _sectors(env.F[n - 1, n]), _sectors(env.F[n + 2, n + 1])
    w1, w2 = _sectors(h[n]), _sectors(h[n + 1])
    gld = {(k[0], k[1]): np.diagonal(v, axis1=0, axis2=2).T for k, v in gl.items() if k[0] == k[2]}
    grd = {(k[0], k[1]): np.diagonal(v, axis1=0, axis2=2).T for k, v in gr.items() if k[0] == k[2]}
    w1d = {k[:2] + k[3:]: np.diagonal(v, axis1=1, axis2=2) for k, v in w1.items() if k[1] == k[2]}
    w2d = {k[:2] + k[3:]: np.diagonal(v, axis1=1, axis2=2) for k, v in w2.items() if k[1] == k[2]}
    out = {}
    for (sa, sx), a in gld.items():
        for (sx1, sp, sm), b in w1d.items():
            for (sm1, sq, sy), c in w2d.items():
                for (sr, sy1), d in grd.items():
                    if sx1 != sx or sm1 != sm or sy1 != sy:
                        continue
                    part = np.einsum("ax,xmp,myq,ry->apqr", a, b, c, d)
                    key = (sa, sp, sq, sr)
                    out[key] = part if key not in out else out[key] + part
    return out


def compare(name, h, phys, bonds, n, seed=3):
    """Print the worst gap between the two, and the diagonal's own scale."""
    psi = MPS.random(phys, bonds, seed=seed).canonize_()
    env = Env(psi, MPO(h.sites)).setup_()
    for site in range(n):
        env.update_(site, to="last")
    aa = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
    truth = exact_diagonal(env, n, aa)
    guess = factorized_diagonal(env, h, n)
    worst = scale = 0.0
    trees = {}
    for key, arr in truth.items():
        secs = tuple(aa.structure.axis_sectors(key))
        scale = max(scale, float(np.abs(arr).max()))
        trees.setdefault(secs, []).append((key.output_tree.inner, arr))
        got = guess.get(secs)
        if got is not None:
            worst = max(worst, float(np.abs(arr - got).max()))
    split = max((len(v) for v in trees.values()), default=0)
    print(f"{name}: entries={sum(a.size for a in truth.values())} scale={scale:.4f}")
    print(
        f"    worst |exact - factorized| = {worst:.3e}"
        f"   most fusion trees sharing one sector tuple: {split}"
    )
    for rows in trees.values():
        if len(rows) > 1:
            labels = [
                (tuple(getattr(s, "two_j", s) for s in inner), round(float(a.flat[0]), 6))
                for inner, a in rows
            ]
            print(f"    one sector tuple, {len(rows)} inner lines -> {labels}")
    return worst


def u1_chain(n=6):
    sz, sp, sm = (
        local_op(example._spin_half()[1], phys=example.PHYS, charge=U1Sector(0)),
        local_op(example._spin_half()[2], phys=example.PHYS, charge=U1Sector(-2)),
        local_op(example._spin_half()[3], phys=example.PHYS, charge=U1Sector(2)),
    )
    terms = [(0.37, [(sz, 2)])]
    for i in range(n - 1):
        terms += [
            (1.0, [(sz, i), (sz, i + 1)]),
            (0.5, [(sp, i), (sm, i + 1)]),
            (0.5, [(sm, i), (sp, i + 1)]),
        ]
    return (
        MPO.from_terms(n, terms, cutoff=None, symbolic=True),
        example.PHYS,
        example.bond_spaces(n),
    )


def fz2_chain(n=5):
    phys = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = local_op(a.T, phys=phys, charge=FZ2Sector(1))
    c = local_op(a, phys=phys, charge=FZ2Sector(1))
    terms = [(0.8, [(local_op(np.diag([0.0, 1.0]), phys=phys, charge=FZ2Sector(0)), 2)])]
    for i, j in [(m, m + 1) for m in range(n - 1)] + [(1, 3)]:
        terms += [(1.0, [(cd, i), (c, j)]), (1.0, [(cd, j), (c, i)])]
    unit = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
    both = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})
    return (
        MPO.from_terms(n, terms, cutoff=None, symbolic=True),
        phys,
        [unit] + [both] * (n - 1) + [unit],
    )


def su2_chain(n=6):
    phys = GradedSpace.new(SU2, {SU2Sector(1): 1})
    _, sz, sp, sm = example._spin_half()
    ss = local_op(np.kron(sz, sz) + (np.kron(sp, sm) + np.kron(sm, sp)) / 2, phys=phys)
    terms = [(1.0, [(ss, (i, i + 1))]) for i in range(n - 1)]
    terms += [(0.41, [(ss, (i, i + 2))]) for i in range(n - 2)]  # J2, so J1's degeneracy lifts
    tri = GradedSpace.new(SU2, {SU2Sector(0): 1})
    mid = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2, SU2Sector(2): 1})
    return (
        MPO.from_terms(n, terms, cutoff=None, symbolic=True),
        phys,
        [tri] + [mid] * (n - 1) + [tri],
    )


def cost():
    """Dense wall of the candidate contraction against one matvec of the same shape."""
    rng = np.random.default_rng(0)
    print("\ndense cost, no symmetry (the FLOP ratio the term families inherit)")
    for chi, d, dw in [(64, 2, 5), (128, 2, 5), (64, 4, 100), (128, 4, 100), (128, 4, 300)]:
        gl, gr = rng.normal(size=(chi, dw, chi)), rng.normal(size=(chi, dw, chi))
        w1, w2 = rng.normal(size=(dw, d, d, dw)), rng.normal(size=(dw, d, d, dw))
        aa = rng.normal(size=(chi, d, d, chi))

        def matvec(gl=gl, gr=gr, w1=w1, w2=w2, aa=aa):
            t = np.einsum("apqr,rys->apqys", aa, gr, optimize=True)
            t = np.einsum("apqys,mQqy->apQms", t, w2, optimize=True)
            t = np.einsum("apQms,xPpm->aPQxs", t, w1, optimize=True)
            return np.einsum("aPQxs,axB->BPQs", t, gl, optimize=True)

        parts = (
            np.einsum("axa->ax", gl),
            np.einsum("xppm->xpm", w1),
            np.einsum("mqqy->mqy", w2),
            np.einsum("ryr->ry", gr),
        )

        def diagonal(parts=parts):
            return np.einsum("ax,xpm,mqy,ry->apqr", *parts, optimize=True)

        def wall(fn, reps=3):
            fn()
            start = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - start) / reps

        tm, td = wall(matvec), wall(diagonal)
        print(
            f"    chi={chi:4d} d={d} Dw={dw:4d}   matvec {tm * 1e3:9.3f} ms"
            f"   diagonal {td * 1e3:8.3f} ms   ratio {td / tm:7.4f}"
        )


if __name__ == "__main__":
    compare("U(1) Heisenberg (ungraded Abelian)", *u1_chain(), 2)
    compare("fZ2 spinless fermions (graded Abelian)", *fz2_chain(), 2)
    compare("SU(2) J1-J2 chain (non-Abelian)", *su2_chain(), 2)
    cost()
