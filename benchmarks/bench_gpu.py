"""GPU benchmark: TeNeT-py (jax-jit, A100) vs symmray (jax-GPU eager, numpy-CPU).

U(1) rank-4 tensordot, "wide" shape (free legs m=4, contracted legs degeneracy m),
plus an SU(2) general-path (bending) contraction, GPU vs same-machine CPU.

symmray 0.2.1 registers no jax pytree, so `jax.jit(sr.tensordot)` fails
("Expected SymmrayCommon, got DynamicJaxprTracer") -- its GPU column is eager,
one XLA dispatch per block.

Run on the GPU box:
  CUDA_VISIBLE_DEVICES=2 venv/bin/python benchmarks/bench_gpu.py
"""

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import symmray as sr

import tenet
import tenet.pytree  # noqa: F401  -- registers SymmetricTensor as a jax pytree
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, SU2Sector, U1, U1Sector

jax.config.update("jax_enable_x64", True)

CHARGES = (-1, 0, 1)
AXES = ((2, 3), (0, 1))  # == numpy axes=2
REPS = 10
GPU = jax.devices()[0]
CPU = jax.devices("cpu")[0]


def space(m):
    return GradedSpace.new(U1, {U1Sector(q): m for q in CHARGES})


def tenet_pair(mf, mc, seed=0, dtype=np.float64):
    legs = (Leg(space(mf), OUT), Leg(space(mf), OUT), Leg(space(mc), IN), Leg(space(mc), IN))
    other = (Leg(space(mc), OUT), Leg(space(mc), OUT), Leg(space(mf), IN), Leg(space(mf), IN))
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


def timed(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def wait(blocks):
    for blk in blocks:
        blk.block_until_ready()


def bench_tenet(mf, mc, dtype, device):
    """(compile_ms, steady_ms) for jitted tenet.tensordot on `device`."""
    with jax.default_device(device):
        a, b = tenet_pair(mf, mc, dtype=dtype)
        a, b = a.to_backend("jax"), b.to_backend("jax")
        f = jax.jit(lambda x, y: tenet.tensordot(x, y, axes=AXES))

        def run():
            wait(f(a, b).blocks)

        t0 = time.perf_counter()
        run()  # trace + compile + first execute
        compile_ms = (time.perf_counter() - t0) * 1e3
        return compile_ms, timed(run)


def bench_symmray(mf, mc, on_gpu):
    sa, sb = symmray_pair(mf, mc)
    if on_gpu:
        dev = GPU
        sa.apply_to_arrays(lambda x: jax.device_put(jnp.asarray(x), dev))
        sb.apply_to_arrays(lambda x: jax.device_put(jnp.asarray(x), dev))

        def run():
            wait(sr.tensordot(sa, sb, axes=2).blocks.values())
    else:

        def run():
            sr.tensordot(sa, sb, axes=2)

    run()  # warm up
    return timed(run)


def sanity(m=4):
    a, b = tenet_pair(m, m)
    da, db = a.to_dense(), b.to_dense()
    ref = np.tensordot(da, db, axes=2)

    with jax.default_device(GPU):
        ja, jb = a.to_backend("jax"), b.to_backend("jax")
        got = jax.jit(lambda x, y: tenet.tensordot(x, y, axes=AXES))(ja, jb)
    assert np.allclose(np.asarray(got.to_dense()), ref), "tenet gpu != numpy ref"

    imap = [[q for q in CHARGES for _ in range(m)]] * 4
    duals = (False, False, True, True)
    sa = sr.U1Array.from_dense(da, imap, duals, invalid_sectors="raise")
    sb = sr.U1Array.from_dense(db, imap, duals, invalid_sectors="raise")
    sa.apply_to_arrays(jnp.asarray)
    sb.apply_to_arrays(jnp.asarray)
    assert np.allclose(np.asarray(sr.tensordot(sa, sb, axes=2).to_dense()), ref), "symmray gpu != ref"
    print(f"sanity ok (m={m}): tenet-gpu == symmray-gpu == np.tensordot\n", flush=True)


def u1():
    print(f"U(1) wide (free legs m=4, contracted m), ms median of {REPS}")
    hdr = (
        f"{'m':>5} {'gpu-f32':>9} {'gpu-f64':>9} {'sym-gpu':>9} {'sym-numpy':>10} "
        f"{'cpu-f64':>9} {'MB/op':>8} {'cc-f64':>9}"
    )
    print(hdr)
    for m in (16, 64, 128, 256, 512):
        mf, mc = 4, m
        nbytes = 19 * mf * mf * mc * mc * 8
        try:
            _, g32 = bench_tenet(mf, mc, np.float32, GPU)
            cc, g64 = bench_tenet(mf, mc, np.float64, GPU)
            sg = bench_symmray(mf, mc, on_gpu=True)
            sn = bench_symmray(mf, mc, on_gpu=False)
            _, c64 = bench_tenet(mf, mc, np.float64, CPU)
        except Exception as e:  # OOM or anything else -> record and continue
            print(f"{m:>5}   -- skipped: {type(e).__name__}: {str(e)[:90]} --", flush=True)
            continue
        print(
            f"{m:>5} {g32:>9.3f} {g64:>9.3f} {sg:>9.3f} {sn:>10.3f} {c64:>9.3f} "
            f"{nbytes / 1e6:>8.1f} {cc:>9.1f}",
            flush=True,
        )


SECTORS = (SU2Sector(0), SU2Sector(1), SU2Sector(2))
GENERAL = ((1, 2), (0, 3))


def su2_operands(m, seed=0):
    def sp():
        return GradedSpace.new(SU2, {s: m for s in SECTORS})

    a = (Leg(sp(), OUT, dual=True), Leg(sp(), OUT), Leg(sp(), IN), Leg(sp(), IN))
    b = (Leg(sp(), OUT, dual=True), Leg(sp(), OUT), Leg(sp(), IN), Leg(sp(), IN, dual=True))
    return tuple(
        SymmetricTensor.random(legs, seed=seed + i, dtype=np.float64) for i, legs in enumerate((a, b))
    )


def su2():
    print(f"\nSU(2) rank-4 general (bending) path, f64, ms median of {REPS}")
    print(f"{'m':>5} {'gpu':>9} {'cpu':>9} {'blocks':>8} {'cc-gpu':>9}")
    for m in (16, 32, 64):
        try:
            out = []
            for dev in (GPU, CPU):
                with jax.default_device(dev):
                    a, b = su2_operands(m)
                    a, b = a.to_backend("jax"), b.to_backend("jax")
                    f = jax.jit(lambda x, y: tenet.tensordot(x, y, axes=GENERAL))

                    def run(f=f, a=a, b=b):
                        wait(f(a, b).blocks)

                    t0 = time.perf_counter()
                    run()
                    out.append(((time.perf_counter() - t0) * 1e3, timed(run), len(a.blocks)))
        except Exception as e:
            print(f"{m:>5}   -- skipped: {type(e).__name__}: {str(e)[:90]} --", flush=True)
            continue
        (ccg, g, nb), (_, c, _) = out
        print(f"{m:>5} {g:>9.3f} {c:>9.3f} {nb:>8} {ccg:>9.1f}", flush=True)


if __name__ == "__main__":
    print(f"devices: gpu={GPU}  cpu={CPU}  symmray={sr.__version__}  jax={jax.__version__}\n")
    sanity()
    u1()
    su2()
