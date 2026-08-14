"""TeNeT-py vs symmray: U(1) pairwise tensordot, CPU.

Rank-4 tensors, every leg carrying U(1) charges {-1, 0, 1}, two axes contracted
(numpy's `axes=2` form). Both libraries get identical sector content,
degeneracies and block structure (19 stored blocks per operand).

Two shapes, because uniform degeneracy m explodes as m**4:
  "uniform"  all four legs have degeneracy m         (m=128 would be ~40 GB/tensor)
  "wide"     free legs pinned at m=4, contracted legs degeneracy m
             -- reaches m=128 with sane memory, and is the interesting regime
             anyway: fixed block *count*, growing block *size*.

The JAX columns are float32 and float64 (x64 is enabled below): symmray runs
float64, so f64 is the fair column and f32 is the it-if-you-can-afford-it one.

Not a test, not part of the package. Run from the repo root:
  uv run --with symmray --with cotengra python benchmarks/bench_u1_tensordot.py
"""

import statistics
import sys
import time

import jax
import numpy as np
import symmray as sr

import tenet
import tenet.pytree  # noqa: F401  -- registers SymmetricTensor as a jax pytree
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import U1, U1Sector

jax.config.update("jax_enable_x64", True)

CHARGES = (-1, 0, 1)
AXES = ((2, 3), (0, 1))  # == numpy axes=2
REPS = 10
MEM_BUDGET = 1e9  # bytes per operand


def space(m):
    return GradedSpace.new(U1, {U1Sector(q): m for q in CHARGES})


def tenet_pair(mf, mc, seed=0, dtype=np.float64):
    # dual=False throughout: outward duality comes from side (OUT vs IN),
    # which is exactly what symmray's per-index `dual` flag encodes.
    legs = (
        Leg(space(mf), OUT),
        Leg(space(mf), OUT),
        Leg(space(mc), IN),
        Leg(space(mc), IN),
    )
    other = (
        Leg(space(mc), OUT),
        Leg(space(mc), OUT),
        Leg(space(mf), IN),
        Leg(space(mf), IN),
    )
    return (
        SymmetricTensor.random(legs, seed=seed, dtype=dtype),
        SymmetricTensor.random(other, seed=seed + 1, dtype=dtype),
    )


def symmray_pair(mf, mc, seed=0):
    def ix(m, dual):
        return sr.BlockIndex({q: m for q in CHARGES}, dual=dual)

    a = (ix(mf, False), ix(mf, False), ix(mc, True), ix(mc, True))
    b = (ix(mc, False), ix(mc, False), ix(mf, True), ix(mf, True))
    return sr.U1Array.random(a, seed=seed), sr.U1Array.random(b, seed=seed + 1)


def clear_caches():
    for name, mod in list(sys.modules.items()):
        if not name.startswith("tenet"):
            continue
        for obj in vars(mod).values():
            if hasattr(obj, "cache_clear"):
                obj.cache_clear()


def timed(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def sanity(m=4):
    """Identical numbers in => identical numbers out, both against np.tensordot."""
    a, b = tenet_pair(m, m)
    da, db = a.to_dense(), b.to_dense()
    imap = [[q for q in CHARGES for _ in range(m)]] * 4
    sa = sr.U1Array.from_dense(da, imap, (False, False, True, True), invalid_sectors="raise")
    sb = sr.U1Array.from_dense(db, imap, (False, False, True, True), invalid_sectors="raise")
    assert np.allclose(sa.to_dense(), da) and np.allclose(sb.to_dense(), db)

    ref = np.tensordot(da, db, axes=2)
    got_t = tenet.tensordot(a, b, axes=AXES).to_dense()
    got_s = sr.tensordot(sa, sb, axes=2).to_dense()
    assert np.allclose(got_t, ref), np.abs(got_t - ref).max()
    assert np.allclose(got_s, ref), np.abs(got_s - ref).max()
    print(
        f"sanity ok (m={m}): tenet == symmray == np.tensordot on the dense expansion; "
        f"{len(a.blocks)} blocks / operand, {len(sa.blocks)} on the symmray side",
        flush=True,
    )


def bench_jax(mf, mc, dtype):
    a, b = tenet_pair(mf, mc, dtype=dtype)
    a, b = a.to_backend("jax"), b.to_backend("jax")
    f = jax.jit(lambda x, y: tenet.tensordot(x, y, axes=AXES))

    def run():
        for blk in f(a, b).blocks:
            blk.block_until_ready()

    run()  # trace + compile
    return timed(run)


def row(mf, mc):
    nbytes = 19 * mf * mf * mc * mc * 8
    if nbytes > MEM_BUDGET:
        return None

    a, b = tenet_pair(mf, mc)
    clear_caches()
    t0 = time.perf_counter()
    tenet.tensordot(a, b, axes=AXES)
    first = (time.perf_counter() - t0) * 1e3
    steady = timed(lambda: tenet.tensordot(a, b, axes=AXES))

    sa, sb = symmray_pair(mf, mc)
    sr.tensordot(sa, sb, axes=2)  # warm up
    sym = timed(lambda: sr.tensordot(sa, sb, axes=2))

    return (
        first,
        steady,
        bench_jax(mf, mc, np.float32),
        bench_jax(mf, mc, np.float64),
        sym,
        nbytes,
    )


def main():
    sanity()
    for label, shapes in (
        ("uniform (all legs m)", [(m, m) for m in (4, 16, 64, 128)]),
        ("wide (free legs m=4, contracted legs m)", [(4, m) for m in (4, 16, 64, 128)]),
    ):
        print(f"\n{label}   ms, median of {REPS}")
        print(
            f"{'m':>5} {'tenet-first':>12} {'tenet-steady':>13} {'jax-jit-f32':>12} "
            f"{'jax-jit-f64':>12} {'symmray':>10} {'MB/operand':>11}"
        )
        for mf, mc in shapes:
            r = row(mf, mc)
            m = max(mf, mc)
            if r is None:
                print(f"{m:>5}   -- skipped, ~{19 * mf * mf * mc * mc * 8 / 1e9:.0f} GB/operand --")
                continue
            f, s, j32, j64, y, nb = r
            print(
                f"{m:>5} {f:>12.3f} {s:>13.3f} {j32:>12.3f} {j64:>12.3f} {y:>10.3f} "
                f"{nb / 1e6:>11.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
