"""What the *correct* two-site diagonal costs, against one matvec (#232 Stage A).

`benchmarks/bench_heff2_diagonal.py` (M53, #230) measured the leg-factorized candidate --
`diag[a,p,q,r] = sum_{x,m,y} GL[a,x,a] W1[x,p,p,m] W2[m,q,q,y] GR[r,y,r]` on reduced
blocks -- and refused it: exact on U(1), sign-wrong on fZ2, and wrong in *shape* on SU(2).
Its cost table put that candidate at **0.2 % of a matvec** at quantum-chemistry scale.
That figure is a cost for an object that does not exist, and this instrument replaces it
with the cost of the object that does, `tenet.map_diagonal`.

Three sections, each answering one question:

* `agreement` -- is it right on the real DMRG object? The truth is #230's own oracle,
  `exact_diagonal`, which probes `Env.heff2` with reduced-basis unit vectors. The
  effective Hamiltonian is formed explicitly from the same environment tensors the matvec
  uses, into the square partition `(a p q r | a' p' q' r')`, and its diagonal is compared
  entry for entry. The two bases differ by one bend of the right bond -- a scalar per
  block, hence a diagonal similarity, hence the same diagonal -- and the numbers say so.
* `cost` -- the ratio on the real implementation, symmetric tensors and all: one
  `Env.heff2` (the `_apply2` path) against forming the map and against extracting its
  diagonal.
* `dense` -- #230's five `(chi, d, D_w)` points with #230's own matvec, so the 0.2 %
  figure is answered on its own ground.

Run: `uv run python benchmarks/bench_map_diagonal.py`. On no CI path.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples" / "toy_codes"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import bench_heff2_diagonal as m53  # noqa: E402

import tenet  # noqa: E402
from tenet import FusionTree  # noqa: E402
from tenet.network import MPO, MPS, Env  # noqa: E402

# GL is `(bond IN, w OUT, bond OUT)`, GR is `(bond OUT, w IN, bond IN)` and W is
# `(w IN, p OUT, p IN, w OUT)`. The environments' *contracted* bond indices are GL's
# first and GR's first, so the ket indices `(a p q r)` are GL's last, the two W's `p`
# / `q` and GR's last -- which is why `r` arrives IN, exactly as it sits on `aa`.
HEFF = "Axa,xpPm,mqQy,Ryr->apqrAPQR"


def walk(chain, n=2, seed=3):
    """One `(env, aa, h, n)` at a bond, in the two-site sweep's own mixed-canonical gauge."""
    h, phys, bonds = chain
    psi = MPS.random(phys, bonds, seed=seed).canonize_()
    env = Env(psi, MPO(h.sites)).setup_()
    for site in range(n):
        env.update_(site, to="last")
    aa = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
    return env, aa, h, n


def form(env, h, n):
    """The two-site effective Hamiltonian, explicitly, on the square partition.

    The contraction lands it on the partition `aa` itself has -- the right bond IN on
    the ket side, OUT on the bra side -- and one `repartition` bends the pair into
    `(a p q r | a' p' q' r')`. That bend is the whole distance between the operator as
    the environments hand it over and the operator as a square map, and it is one
    scalar per block on each side, so it leaves the diagonal alone.
    """
    nat = tenet.einsum(HEFF, env.F[n - 1, n], h[n], h[n + 1], env.F[n + 2, n + 1])
    return nat.repartition(outputs=(0, 1, 2, 3), inputs=(4, 5, 6, 7))


def bent_key(structure, key):
    """`aa`'s block key, read in the all-OUT basis `map_diagonal` returns.

    Bending the right bond turns the pair `(tree over (a, p, q) -> c, tree over (r) -> c)`
    into one tree over `(a, p, q, r*)` coupling to the unit whose last inner line is `c`.
    The correspondence is a relabelling; the coefficient it carries is one scalar per
    block and cancels out of a diagonal.
    """
    ot, it = key.output_tree, key.input_tree
    dual = structure.provider.dual
    return FusionTree(
        (*ot.uncoupled, *(dual(u) for u in it.uncoupled)),
        (*ot.inner, ot.coupled),
        (*ot.multiplicities, 0),
        structure.provider.unit,
    )


def agreement(name, chain):
    """`map_diagonal` of the formed map against #230's probing oracle, entry for entry."""
    env, aa, h, n = walk(chain)
    truth = m53.exact_diagonal(env, n, aa)
    got = tenet.map_diagonal(form(env, h, n))
    index = {k.output_tree: i for i, k in enumerate(got.structure.block_order)}
    worst = scale = 0.0
    for key, arr in truth.items():
        mine = np.asarray(got.blocks[index[bent_key(aa.structure, key)]])
        worst = max(worst, float(np.abs(arr - mine).max(initial=0.0)))
        scale = max(scale, float(np.abs(arr).max(initial=0.0)))
    print(f"{name}: scale={scale:.4f}   worst |exact - map_diagonal| = {worst:.3e}")


def wall(fn, reps=3):
    fn()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps


def nbytes(t):
    return sum(np.asarray(b).nbytes for b in t.blocks)


def cost(name, chain):
    """One real `heff2` against forming the map and against extracting its diagonal."""
    env, aa, h, n = walk(chain)
    heff = form(env, h, n)
    tm = wall(lambda: env.heff2(n, aa))
    tf = wall(lambda: form(env, h, n))
    td = wall(lambda: tenet.map_diagonal(heff))
    print(
        f"{name}: heff2 {tm * 1e3:8.3f} ms   form {tf * 1e3:9.3f} ms ({tf / tm:7.2f}x)"
        f"   map_diagonal {td * 1e3:7.3f} ms ({td / tm:7.4f}x)"
        f"   bytes(H)/bytes(aa) = {nbytes(heff) / nbytes(aa):.1f}"
    )


def dense(ceiling_gib=2.0):
    """#230's own five points, its own matvec, and the correct diagonal beside it.

    The extraction is a strided read of `chi^2 d^2` entries out of the formed map; the
    formation is the `chi^4 d^4` array it reads them from, and past the ceiling that array
    is what makes the whole route unavailable, not the reading.
    """
    rng = np.random.default_rng(0)
    print("\ndense, no symmetry -- #230's five points")
    for chi, d, dw in [(64, 2, 5), (128, 2, 5), (64, 4, 100), (128, 4, 100), (128, 4, 300)]:
        gl, gr = rng.normal(size=(chi, dw, chi)), rng.normal(size=(chi, dw, chi))
        w1, w2 = rng.normal(size=(dw, d, d, dw)), rng.normal(size=(dw, d, d, dw))
        aa = rng.normal(size=(chi, d, d, chi))

        def matvec(gl=gl, gr=gr, w1=w1, w2=w2, aa=aa):
            t = np.einsum("apqr,rys->apqys", aa, gr, optimize=True)
            t = np.einsum("apqys,mQqy->apQms", t, w2, optimize=True)
            t = np.einsum("apQms,xPpm->aPQxs", t, w1, optimize=True)
            return np.einsum("aPQxs,axB->BPQs", t, gl, optimize=True)

        tm = wall(matvec)
        gib = (chi * d * d * chi) ** 2 * 8 / 2**30
        line = f"  chi={chi:4d} d={d} Dw={dw:4d}  matvec {tm * 1e3:9.3f} ms  H {gib:8.2f} GiB"
        if gib > ceiling_gib:
            print(f"{line}   form/extract: unavailable")
            continue

        def build(gl=gl, gr=gr, w1=w1, w2=w2):
            return np.einsum("axA,xpPm,mqQy,ryR->apqrAPQR", gl, w1, w2, gr, optimize=True)

        heff = build()
        tf = wall(build, reps=1)
        te = wall(lambda heff=heff: np.einsum("apqrapqr->apqr", heff))
        print(f"{line}   form {tf * 1e3:9.1f} ms ({tf / tm:8.1f}x)   extract {te / tm:8.4f}x")


if __name__ == "__main__":
    chains = {
        "U(1) Heisenberg (ungraded Abelian)": m53.u1_chain(),
        "fZ2 spinless fermions (graded Abelian)": m53.fz2_chain(),
        "SU(2) J1-J2 chain (non-Abelian)": m53.su2_chain(),
    }
    for label, chain in chains.items():
        agreement(label, chain)
    print()
    for label, chain in chains.items():
        cost(label, chain)
    dense()
