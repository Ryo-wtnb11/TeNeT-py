"""Where the SU(2) general-path jit time goes: trace, XLA compile, or execution (#74).

The A100 sweep in #74 reported "compile" (really trace + compile + first exec) of
2.1 s / 2.3 s / 54 s at m = 16 / 32 / 64 for the SU(2) rank-4 general (bending)
path, against a flat ~0.9 s for the U(1) aligned path, and read the 54 s as graph
size: block count x term count unrolled into a big jaxpr. This script measures the
three phases separately, and the reading does not survive it.

**The emitted graph does not depend on m.** Shapes are the only thing m changes,
and shapes are not part of the jaxpr's structure, so at m = 4, 8, 16, 32 and 64 the
general path traces to the *same* 707 equations -- 244 transpose, 181 mul, 138
reshape, 60 add, 45 slice, 34 concatenate and 5 dot_general -- and the whole
Python-side trace plus XLA compile stays flat on CPU (0.09-0.17 s across the whole
sweep, most of it XLA). Whatever costs 52 s on the A100 at m = 64 and ~1 s at m = 32
cannot be the graph: it is byte-for-byte identical in the two runs.

Stage decomposition of those 707 equations (SU(2), m = 16, CPU):

    stage              eqns   trace_s   lower_s   compile_s
    repartition a       147     0.003     0.003       0.067
    repartition b       147     0.003     0.003       0.066
    compose             222     0.005     0.006       0.055
    repartition c       145     0.003     0.003       0.066
    final transpose      46     0.001     0.002       0.018
    whole tensordot     707     0.010     0.008       0.127

Three repartitions at ~147 equations each are the bulk, as #61 predicted -- and
they cost 70 ms of XLA compile, not 52 s. The U(1) aligned control traces to 99
equations and compiles in 0.06 s. A 7x graph is a 70 ms difference; it is not the
50-second one.

**It is gemm shape, not graph size.** Both paths emit exactly 5 `dot_general`s --
blocks are already packed into one dense matrix per fusion channel before the
matmul, so "same-shape block bucketing" is a thing `to_matrices` already did. What
differs is the shape of those five matmuls:

    U(1) aligned, m = 512   (16, 262144) x (262144, 16)   ... and 4 more, all skinny
    SU(2) general, m = 64   (16384, 16384) x (16384, 16384), 12288^2, 8192^2, 4096^2

The U(1) benchmark shape holds the free legs at m = 4, so its matmuls stay skinny
at every m: one XLA kernel choice fits all of them and compile stays flat. The
SU(2) general path at m = 64 hands XLA:GPU four large square f64 gemms, and
XLA:GPU picks a gemm algorithm by *running* candidates. On an A100 a 16384^3 f64
gemm is ~0.5 s per trial, so a few dozen trials over four distinct shapes is tens
of seconds. That is the 52 s, and it is exactly what `--xla_gpu_autotune_level=0`
turns off (untested here -- this box has no GPU; measure it on qg1 before quoting
a number for it).

CPU execution confirms the shapes are genuinely that big: at m = 64 the first call
and the steady call both take ~57 s (16384^3 f64 is ~8.8 TFLOP for the largest
gemm alone; the A100 does the same work in 1.25 s). Nothing is hiding in there.

    m    exec1_s   steady_s
    16     0.170      0.013
    32     0.772      0.628
    64    56.691     56.752

**Conclusion: #74 option 4, amortize.** Options 1-3 (block bucketing, `lax` loops,
cross-block CSE) all shrink the graph, and the graph is already m-independent,
already fused to 5 matmuls, and already worth 0.13 s. There is no 5x to win there.
The one-time A100 cost at m = 64 is 54 s against 1.251 s of steady execution:

    overhead after N calls = 54 / (54 + 1.251 N)

    N =   389 calls -> 10 % overhead
    N = 4 277 calls ->  1 % overhead

A VMC loop runs one structure for 10^4-10^6 contractions, where the compile is
0.1 % of wall clock and below. Paying it is correct. The knob for someone who
cannot (interactive work, a structure used a handful of times) is
`XLA_FLAGS=--xla_gpu_autotune_level=0`, which trades steady gemm throughput for
compile time and is the honest lever, since compile time here *is* autotuning.

Not a test, not part of the package. Run from the repo root:
  uv run python benchmarks/bench_compile.py
"""

import statistics
import time
from collections import Counter

import jax
import numpy as np

import tenet
import tenet.pytree  # noqa: F401  -- registers SymmetricTensor as a jax pytree
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops.contraction import contraction_plan
from tenet.ops.map import compose
from tenet.ops.permutation import transpose
from tenet.ops.repartition import repartition
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

jax.config.update("jax_enable_x64", True)

SECTORS = (SU2Sector(0), SU2Sector(1), SU2Sector(2))
CHARGES = (-1, 0, 1)
GENERAL = ((1, 2), (0, 3))
ALIGNED = ((2, 3), (0, 1))
# m = 64 executes for ~57 s per call on CPU; the compile-time question is answered
# by m <= 32, so the big point is opt-in.
SWEEP = (4, 8, 16, 32)


def su2_operands(m, seed=0):
    def sp():
        return GradedSpace.new(SU2, {s: m for s in SECTORS})

    a = (Leg(sp(), OUT, dual=True), Leg(sp(), OUT), Leg(sp(), IN), Leg(sp(), IN))
    b = (Leg(sp(), OUT, dual=True), Leg(sp(), OUT), Leg(sp(), IN), Leg(sp(), IN, dual=True))
    return tuple(
        SymmetricTensor.random(legs, seed=seed + i, dtype=np.float64).to_backend("jax")
        for i, legs in enumerate((a, b))
    )


def u1_operands(mf, mc, seed=0):
    def sp(m):
        return GradedSpace.new(U1, {U1Sector(q): m for q in CHARGES})

    a = (Leg(sp(mf), OUT), Leg(sp(mf), OUT), Leg(sp(mc), IN), Leg(sp(mc), IN))
    b = (Leg(sp(mc), OUT), Leg(sp(mc), OUT), Leg(sp(mf), IN), Leg(sp(mf), IN))
    return tuple(
        SymmetricTensor.random(legs, seed=seed + i, dtype=np.float64).to_backend("jax")
        for i, legs in enumerate((a, b))
    )


def phases(fn, *args):
    """``(equations, trace_s, lower_s, compile_s)`` -- the three phases, separately."""
    t0 = time.perf_counter()
    jaxpr = jax.make_jaxpr(fn)(*args)
    t_trace = time.perf_counter() - t0
    t0 = time.perf_counter()
    lowered = jax.jit(fn).lower(*args)
    t_lower = time.perf_counter() - t0
    t0 = time.perf_counter()
    lowered.compile()
    t_compile = time.perf_counter() - t0
    return jaxpr, t_trace, t_lower, t_compile


def wait(t):
    for blk in t.blocks:
        blk.block_until_ready()
    return t


def timed(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def m_scaling():
    """The load-bearing measurement: equation count is constant, so compile is too."""
    print("SU(2) general path vs m -- graph size and compile time")
    print(f"{'m':>4} {'blocks':>7} {'eqns':>6} {'trace_s':>8} {'lower_s':>8} {'compile_s':>10}")
    for m in SWEEP:
        a, b = su2_operands(m)
        jaxpr, tt, tl, tc = phases(lambda x, y: tenet.tensordot(x, y, axes=GENERAL), a, b)
        n = len(jaxpr.jaxpr.eqns)
        print(f"{m:>4} {len(a.blocks):>7} {n:>6} {tt:>8.3f} {tl:>8.3f} {tc:>10.3f}", flush=True)

    print("\nU(1) aligned control (free legs m=4, contracted m)")
    print(f"{'m':>4} {'eqns':>6} {'trace_s':>8} {'lower_s':>8} {'compile_s':>10}")
    for m in (16, 512):
        a, b = u1_operands(4, m)
        jaxpr, tt, tl, tc = phases(lambda x, y: tenet.tensordot(x, y, axes=ALIGNED), a, b)
        n = len(jaxpr.jaxpr.eqns)
        print(f"{m:>4} {n:>6} {tt:>8.3f} {tl:>8.3f} {tc:>10.3f}", flush=True)


def stages(m=16):
    """Which part of the lowering owns the 707 equations."""
    a, b = su2_operands(m)
    plan = contraction_plan(a.structure, b.structure, GENERAL)
    ra = repartition(a, plan.a_outputs, plan.a_inputs)
    rb = repartition(b, plan.b_outputs, plan.b_inputs)
    c = compose(ra, rb)
    rc = repartition(c, plan.restore_outputs, plan.restore_inputs)
    todo = (
        ("repartition a", lambda x: repartition(x, plan.a_outputs, plan.a_inputs), (a,)),
        ("repartition b", lambda x: repartition(x, plan.b_outputs, plan.b_inputs), (b,)),
        ("compose", compose, (ra, rb)),
        (
            "repartition c",
            lambda x: repartition(x, plan.restore_outputs, plan.restore_inputs),
            (c,),
        ),
        ("final transpose", lambda x: transpose(x, plan.final_transpose), (rc,)),
        ("whole tensordot", lambda x, y: tenet.tensordot(x, y, axes=GENERAL), (a, b)),
    )
    print(f"\nstage decomposition, SU(2) m={m}")
    print(f"{'stage':>16} {'eqns':>6} {'trace_s':>8} {'lower_s':>8} {'compile_s':>10}")
    for name, fn, args in todo:
        jaxpr, tt, tl, tc = phases(fn, *args)
        n = len(jaxpr.jaxpr.eqns)
        print(f"{name:>16} {n:>6} {tt:>8.3f} {tl:>8.3f} {tc:>10.3f}", flush=True)


def shapes():
    """Graph size is equal; matmul shape is not. This is the whole diagnosis."""
    print("\nemitted dot_general shapes")
    # m=32, not the 64 quoted in the docstring: same story, a quarter of the memory
    for label, ops, axes in (
        ("SU(2) general m=16", su2_operands(16), GENERAL),
        ("SU(2) general m=32", su2_operands(32), GENERAL),
        ("U(1) aligned  m=512", u1_operands(4, 512), ALIGNED),
    ):
        jaxpr = jax.make_jaxpr(lambda x, y, ax=axes: tenet.tensordot(x, y, axes=ax))(*ops)
        eqns = jaxpr.jaxpr.eqns
        dots = [e for e in eqns if str(e.primitive) == "dot_general"]
        counted = Counter(tuple(v.aval.shape for v in e.invars) for e in dots)
        print(f"  {label}: {len(eqns)} eqns, {len(dots)} dots")
        for shape, k in counted.most_common():
            print(f"      x{k}  {shape[0]} x {shape[1]}")


def execution():
    """First call against steady call: nothing is hiding in the first execution."""
    print("\nexecution (CPU)")
    print(f"{'m':>4} {'exec1_s':>9} {'steady_s':>9}")
    for m in SWEEP:
        a, b = su2_operands(m)
        f = jax.jit(lambda x, y: tenet.tensordot(x, y, axes=GENERAL))
        t0 = time.perf_counter()
        wait(f(a, b))
        e1 = time.perf_counter() - t0
        steady = timed(lambda f=f, a=a, b=b: wait(f(a, b)))
        print(f"{m:>4} {e1:>9.3f} {steady:>9.3f}", flush=True)


def amortization(compile_s=54.0, steady_s=1.251):
    """Break-even from the #74 A100 numbers at m=64 (override to re-run elsewhere)."""
    print(f"\namortization: one-time {compile_s:.1f} s, steady {steady_s * 1e3:.0f} ms/call")
    for frac in (0.5, 0.1, 0.01):
        n = compile_s * (1 - frac) / (frac * steady_s)
        print(f"  overhead <= {frac:>5.0%} after {n:>10.0f} calls")


if __name__ == "__main__":
    print(f"jax={jax.__version__} devices={jax.devices()}\n")
    m_scaling()
    stages()
    shapes()
    execution()
    amortization()
