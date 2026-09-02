"""tenet's SU(2) core against TensorKit.jl and frostspin, with a dense BLAS floor.

The question this answers is not "who wins a wall-clock race". It is *why*: for one
SU(2) contraction, how much of the wall is arithmetic and how much is the bookkeeping
around it. So every row carries the FLOPs the contraction actually performs -- counted
from the block structure, not estimated -- the number of ``gemm`` calls those FLOPs are
spread over, and ``ratio_to_peak``: the measured wall over what the same FLOPs would cost
as **one** dense ``gemm`` in the same runtime on the same machine. A ratio of 1 means the
library is pure arithmetic; 100 means it spends 99% of the wall on dispatch. Separately,
``speedup_vs_dense`` is the same contraction with the symmetry thrown away -- what the
symmetry actually buys, which is the number this project cares about.

Neither opponent is a dependency of this project. Install them for the benchmark only::

    uv pip install "frostspin @ git+https://github.com/ogauthe/frostspin.git"
    julia --project=benchmarks/su2_libraries_jl -e 'using Pkg; Pkg.instantiate()'

and then run with ``uv run --no-sync`` so the sync does not remove frostspin again.

The contraction
---------------
One composition, in every arm: ``a`` maps into a middle space, ``b`` maps out of it, and
the result is ``a ∘ b``. In tenet that is ``a @ b`` (``tenet.compose``), in frostspin
``fa @ fb`` (``SymmetricTensor.__matmul__``), in TensorKit ``a * b``, and in the dense
arm one ``numpy`` / ``LinearAlgebra`` ``gemm`` on the reshaped dense arrays. Composition
and not a general ``tensordot`` on purpose: it is the one contraction all three libraries
express *without* a leg permutation, so what is measured is the SU(2) block machinery
itself and not three different transpose implementations. Two shapes:

* ``rank3`` -- ``(V⊗V ← V) ∘ (V ← V⊗V)``, one leg contracted, rank-4 result.
* ``rank5`` -- ``(V⊗V⊗V ← V⊗V) ∘ (V⊗V ← V⊗V⊗V)``, two legs contracted, rank-6 result.

Both are PEPS shapes: a rank-5 site tensor cut into a 3+2 partition is exactly what a
boundary absorption composes.

The spaces
----------
``--space grid`` sweeps the two axes the question is about, independently:

* ``--irreps n`` keeps the ``n`` lowest SU(2) irreps, ``2j = 0 .. n-1`` (so ``n = 1`` is
  the no-symmetry corner: one sector, one big block), and
* ``--deg m`` gives every kept irrep the same degeneracy ``m``.

``--space production`` is the real thing instead: the virtual space read off
``P4_BPcap64_thenIdentityTruncate_reflectionOrbit_D28_Dxe32/P=4/final_tenet``, which is
``{2j=0: 3, 2j=2: 5, 2j=4: 2}`` -- three irreps, degeneracies 2-5, dense dimension 28.
That is deep in the many-small corner, and it is the point this project actually runs at.

What is held equal, and what could not be
-----------------------------------------
* **The same block structure, asserted.** Every arm's operands carry the same coupled
  sectors with the same ``(M, K, N)`` per sector -- checked, not assumed, against
  ``tenet.to_matrices`` -- so the FLOP column is one number for all of them and the wall
  is comparable. The *values* are shared where the cross-check runs: on every point whose
  dense form fits, tenet expands ``a`` and ``b`` with ``to_dense()`` and frostspin
  projects those very arrays back with ``from_array``, which refuses anything outside its
  own invariant subspace, and the dense results are then compared elementwise
  (``--verify``, ``verify_vs_dense`` and ``verify_vs_tenet``). Where the dense form does
  not fit, the blocks are drawn independently; the wall does not depend on the values.
* **f64 everywhere**, one thread of BLAS everywhere (env vars below for NumPy,
  ``-t 1`` plus ``BLAS.set_num_threads(1)`` for Julia), one process per arm.
* **Not equal: the BLAS itself.** NumPy here is linked against Apple Accelerate and
  Julia against OpenBLAS, and there is no way to make one use the other without changing
  what is being measured. That is why the headline diagnostic is ``ratio_to_peak``: each
  arm is divided by *its own* runtime's ``gemm`` rate (a 512^3 f64 ``gemm``, measured in
  that process), so the BLAS difference cancels out of the overhead column. Absolute
  ``steady_ms`` and ``gemm_rate_gflops`` are reported too, so the BLAS gap stays visible
  rather than hidden -- and it is a caveat, not a control: the ceiling is calibrated on a
  512^3 ``gemm`` while the arms run 30x30 ones, and the two BLAS libraries do not have
  the same small-``gemm`` constant.
* **Not equal: warm-up.** Julia JITs, JAX compiles, and every library builds plans on
  first sight of a shape. The first call is reported as ``warmup_ms`` and never enters
  ``steady_ms``.
* **Measured, not assumed: run-to-run noise.** On this machine Accelerate's small-``gemm``
  wall moves by up to 3x between processes for byte-identical work -- ``blocks_ms`` on one
  fixed point came back as 17 us and as 43 us on repeat runs. So read
  ``ratio_to_blocks``, which is measured against the floor *in the same process*, before
  reading absolute milliseconds across arms, and repeat a point before believing a
  factor under 2.

Running it
----------
One arm, one case, one point, appending a JSON line per point::

    uv run --no-sync python benchmarks/bench_su2_libraries.py --arm tenet \\
        --case rank3 --space production --out /tmp/su2.jsonl

The whole grid for every Python arm, plus the TensorKit arm through Julia::

    uv run --no-sync python benchmarks/bench_su2_libraries.py --sweep \\
        --out /tmp/su2.jsonl
    uv run --no-sync python benchmarks/bench_su2_libraries.py --sweep \\
        --arm tensorkit --out /tmp/su2.jsonl

and then the table::

    uv run --no-sync python benchmarks/bench_su2_libraries.py --report /tmp/su2.jsonl

A point already present in ``--out`` is skipped, so the sweep is resumable.
"""

import os

# Before NumPy is imported by anything, including tenet and frostspin: a multi-threaded
# BLAS would put a different number of cores behind each arm's ``dgemm`` and the wall
# column would measure the thread pool rather than the library.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

# JAX's CPU backend ignores every one of those: XLA runs its own thread pool. And
# without x64 it would silently answer a *different question* in f32 -- half the
# arithmetic, at f32 gemm rates, against everyone else's f64.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import tenet  # noqa: E402
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor  # noqa: E402
from tenet.symmetry import SU2, SU2Sector  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
JULIA_PROJECT = HERE / "su2_libraries_jl"
JULIA_SCRIPT = HERE / "bench_su2_libraries.jl"

# The virtual space of the production state, read off its saved leg metadata:
# projects/circuit_synthesis/experiments/l4_heisenberg_reflection_fit_20260826/data/
#   results/P4_BPcap64_thenIdentityTruncate_reflectionOrbit_D28_Dxe32/P=4/final_tenet
# {2j: degeneracy}; dense dimension 3*1 + 5*3 + 2*5 = 28, which is the state's D=28.
PRODUCTION = {0: 3, 2: 5, 4: 2}

# ``a``'s legs then ``b``'s, as (n_out, n_in) per operand. The middle space is a's IN
# legs, which is b's OUT legs; composition contracts exactly those.
CASES = {"rank3": (2, 1), "rank5": (3, 2)}

ARMS = ("tenet", "tenet-jax", "frostspin", "tensorkit")

GRID_IRREPS = (1, 2, 3, 4, 5, 6)
GRID_DEG = (1, 2, 4, 8, 16, 32)

# Three separate ceilings, because three separate things blow up at different rates.
# A rank-6 dense result is d^6 f64 and a rank-5 dense operand is d^5, so at the
# production space (d = 28) the operands are 137 MB apiece and the dense *result* would
# be 3.9 GB. That is why ``ratio_to_peak`` is measured against a calibrated ``gemm``
# rate rather than against the dense arm: the diagnostic survives where the dense arm
# cannot run, and the dense arm still answers "what does the symmetry buy" wherever it
# fits.
MAX_SYM_FLOPS = 5.0e9  # keeps the sweep finite
MAX_OPERAND_ELEMENTS = 4.0e7  # dense operand, needed by the dense/frostspin/verify paths
MAX_DENSE_ELEMENTS = 2.0e7  # dense result, ~160 MB
MAX_DENSE_FLOPS = 4.0e10

# The calibration gemm: big enough to be pure arithmetic, so its rate is the machine's
# ceiling for f64 on one thread. Everything's ``ratio_to_peak`` is against this.
CALIBRATION_N = 512


# --------------------------------------------------------------------------------------
# Spaces and operands.


def degeneracies(irreps: int | None, deg: int | None) -> dict[int, int]:
    """``{2j: degeneracy}`` for the requested space."""
    if irreps is None:
        return dict(PRODUCTION)
    return {two_j: deg for two_j in range(irreps)}


def space(degs: dict[int, int]) -> GradedSpace:
    return GradedSpace.new(SU2, {SU2Sector(k): d for k, d in degs.items()})


def dense_dim(degs: dict[int, int]) -> int:
    return sum(d * (k + 1) for k, d in degs.items())


def operands(case: str, degs: dict[int, int], seed: int = 0) -> tuple[SymmetricTensor, ...]:
    """``a`` and ``b`` such that ``a @ b`` is the case's composition.

    Every leg carries the same space; ``a``'s IN legs are the middle, and ``b``'s OUT
    legs repeat them so ``compose`` accepts the pair with no bending.
    """
    n_out, n_mid = CASES[case]
    v = space(degs)
    a = (Leg(v, OUT),) * n_out + (Leg(v, IN),) * n_mid
    b = (Leg(v, OUT),) * n_mid + (Leg(v, IN),) * n_out
    return tuple(
        SymmetricTensor.random(legs, seed=seed + i, dtype=np.float64)
        for i, legs in ((0, a), (1, b))
    )


# --------------------------------------------------------------------------------------
# The cost model: FLOPs from the block structure, not from a guess.


def block_costs(a: SymmetricTensor, b: SymmetricTensor) -> tuple[float, int, list]:
    """``(flops, n_gemm, shapes)`` for the composition, counted off the coupled blocks.

    ``a ∘ b`` is one ``(M_c, K_c) @ (K_c, N_c)`` per coupled sector ``c`` the two
    operands share; a sector only one of them carries contributes a zero block and no
    arithmetic. Real ``gemm``, so ``2 M K N``.
    """
    ma, mb = tenet.to_matrices(a), tenet.to_matrices(b)
    flops = 0.0
    shapes = []
    for c in sorted(set(ma) & set(mb), key=repr):
        m, k = ma[c].shape
        k2, n = mb[c].shape
        assert k == k2, (c, ma[c].shape, mb[c].shape)
        flops += 2.0 * m * k * n
        shapes.append((repr(c), m, k, n))
    return flops, len(shapes), shapes


def dense_shape(case: str, degs: dict[int, int]) -> tuple[int, int, int]:
    """``(M, K, N)`` of the same composition done with no symmetry at all."""
    n_out, n_mid = CASES[case]
    d = dense_dim(degs)
    return d**n_out, d**n_mid, d**n_out


# --------------------------------------------------------------------------------------
# Timing.


def measure(fn, *, budget: float = 0.5, min_reps: int = 5, max_reps: int = 200):
    """``(warmup_ms, steady_ms, reps)``: the first call alone, then the median of the rest.

    The first call carries plan construction, JIT and any cache the library fills on
    first sight of a shape, and smearing it into the steady number is the single easiest
    way to make this benchmark a lie.
    """
    t0 = time.perf_counter()
    fn()
    warmup = (time.perf_counter() - t0) * 1e3

    walls = []
    start = time.perf_counter()
    while len(walls) < min_reps or (time.perf_counter() - start < budget and len(walls) < max_reps):
        t0 = time.perf_counter()
        fn()
        walls.append((time.perf_counter() - t0) * 1e3)
    return warmup, statistics.median(walls), len(walls)


# --------------------------------------------------------------------------------------
# The arms.


def gemm_rate() -> float:
    """FLOP/s of one square f64 ``gemm``, this process's BLAS, one thread. Cached."""
    if not hasattr(gemm_rate, "_v"):
        x = np.random.default_rng(0).standard_normal((CALIBRATION_N, CALIBRATION_N))
        _, ms, _ = measure(lambda: x @ x, budget=0.3)
        gemm_rate._v = 2.0 * CALIBRATION_N**3 / (ms * 1e-3)
    return gemm_rate._v


def blocks_only(shapes, backend: str):
    """The arithmetic with no library at all: the same per-sector ``gemm`` list, raw.

    This is the honest floor, and the reason it exists is that the peak-``gemm`` ratio is
    not comparable across runtimes -- NumPy's Accelerate reaches 7x OpenBLAS's f64 rate on
    this machine, and neither number says anything about how a *30x30* ``gemm`` behaves.
    Dividing each arm by this instead asks the one question that transfers: on top of the
    arithmetic it dispatches, how much does the SU(2) bookkeeping cost?
    """
    rng = np.random.default_rng(0)
    pairs = [(rng.standard_normal((m, k)), rng.standard_normal((k, n))) for _, m, k, n in shapes]
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        pairs = [(jnp.asarray(x), jnp.asarray(y)) for x, y in pairs]
        return lambda: jax.block_until_ready([x @ y for x, y in pairs])
    return lambda: [x @ y for x, y in pairs]


def arm_dense(a: SymmetricTensor, b: SymmetricTensor, case: str, degs: dict[int, int]):
    """The floor: the same contraction with the symmetry thrown away, one ``gemm``."""
    m, k, n = dense_shape(case, degs)
    da = np.ascontiguousarray(a.to_dense().reshape(m, k))
    db = np.ascontiguousarray(b.to_dense().reshape(k, n))
    return lambda: da @ db


def arm_tenet(a: SymmetricTensor, b: SymmetricTensor, backend: str):
    if backend != "numpy":
        a, b = a.to_backend(backend), b.to_backend(backend)
    if backend == "jax":
        import jax

        def run():
            return jax.block_until_ready((a @ b).data)

        return run
    return lambda: a @ b


def arm_frostspin(
    a: SymmetricTensor, b: SymmetricTensor, case: str, degs: dict[int, int], *, from_dense: bool
):
    """frostspin's operands for the same composition.

    ``from_dense`` builds them by projecting tenet's own dense arrays, which is what
    makes the cross-library check a check; it is also the only constructor frostspin
    offers that takes values, and its cost is quadratic in the dense operand, so where
    the dense arrays are too big to be useful the blocks are drawn independently instead.
    Nothing in the wall depends on the values -- only on the block shapes, which are
    asserted equal to tenet's either way.
    """
    from frostspin import SU2SymmetricTensor

    n_out, n_mid = CASES[case]
    # frostspin labels an SU(2) irrep by its dimension 2j+1, tenet by 2j.
    rep = np.array([[d for d in degs.values()], [k + 1 for k in degs]])
    rows, cols = (rep,) * n_out, (rep,) * n_mid
    sig_a = [False] * n_out + [True] * n_mid
    sig_b = [False] * n_mid + [True] * n_out
    if from_dense:
        fa = SU2SymmetricTensor.from_array(a.to_dense(), rows, cols, signature=sig_a)
        fb = SU2SymmetricTensor.from_array(b.to_dense(), cols, rows, signature=sig_b)
    else:
        fa = SU2SymmetricTensor.random(rows, cols, signature=sig_a)
        fb = SU2SymmetricTensor.random(cols, rows, signature=sig_b)
    return (lambda: fa @ fb), fa, fb


# --------------------------------------------------------------------------------------
# One point.


def run_point(arm: str, case: str, degs: dict[int, int], *, budget: float, verify: bool) -> dict:
    a, b = operands(case, degs)
    flops, n_gemm, shapes = block_costs(a, b)
    m, k, n = dense_shape(case, degs)
    dflops = 2.0 * m * k * n

    operand_elements = float(np.prod(a.shape))
    do_dense = (
        operand_elements <= MAX_OPERAND_ELEMENTS
        and m * n <= MAX_DENSE_ELEMENTS
        and dflops <= MAX_DENSE_FLOPS
    )

    row = {
        "arm": arm,
        "case": case,
        "degeneracies": {str(kk): vv for kk, vv in degs.items()},
        "n_irreps": len(degs),
        "dense_dim": dense_dim(degs),
        "flops": flops,
        "n_gemm": n_gemm,
        "dense_flops": dflops,
        "dense_mkn": [m, k, n],
        "block_shapes": shapes,
        "gemm_rate_gflops": gemm_rate() / 1e9,
    }

    if do_dense:
        row["dense_warmup_ms"], row["dense_ms"], _ = measure(
            arm_dense(a, b, case, degs), budget=budget
        )
    else:
        row["dense_warmup_ms"] = row["dense_ms"] = None

    if arm == "tenet":
        fn = arm_tenet(a, b, "numpy")
    elif arm == "tenet-jax":
        fn = arm_tenet(a, b, "jax")
    elif arm == "frostspin":
        fn, fa, fb = arm_frostspin(a, b, case, degs, from_dense=do_dense)
        # frostspin fuses the row legs into one degeneracy index and tenet keeps one
        # block per fusion tree, but the coupled-sector matrices they multiply are the
        # same shapes -- so the FLOP column really is the same arithmetic on both sides.
        ma = tenet.to_matrices(a)
        theirs = sorted(bl.shape for bl in fa.blocks)
        ours = sorted(mm.shape for mm in ma.values())
        assert theirs == ours, (theirs, ours)
        assert sorted(bl.shape for bl in fb.blocks) == sorted(
            mm.shape for mm in tenet.to_matrices(b).values()
        )
    else:
        raise ValueError(arm)

    row["warmup_ms"], row["steady_ms"], row["reps"] = measure(fn, budget=budget)
    # The diagnostic: measured wall over the wall the *same* FLOPs would take at the
    # machine's one-thread gemm ceiling. 1 is pure arithmetic, 100 is 99% dispatch.
    row["ratio_to_peak"] = row["steady_ms"] * 1e-3 * gemm_rate() / flops if flops else None
    _, row["blocks_ms"], _ = measure(
        blocks_only(shapes, "jax" if arm == "tenet-jax" else "numpy"), budget=budget
    )
    row["ratio_to_blocks"] = row["steady_ms"] / row["blocks_ms"]
    # And what the symmetry actually buys, where the dense arm could run at all.
    row["speedup_vs_dense"] = row["dense_ms"] / row["steady_ms"] if do_dense else None

    if verify and do_dense:
        got = (a @ b).to_dense()
        ref = np.tensordot(
            a.to_dense(), b.to_dense(), axes=(range(CASES[case][0], a.ndim), range(CASES[case][1]))
        )
        row["verify_vs_dense"] = float(np.abs(got - ref).max())
        if arm == "frostspin":
            row["verify_vs_tenet"] = float(np.abs(fn().toarray() - got).max())
    return row


def points(case: str, space_kind: str):
    if space_kind == "production":
        yield dict(PRODUCTION)
        return
    for n_irr in GRID_IRREPS:
        for m in GRID_DEG:
            degs = degeneracies(n_irr, m)
            # Cheap stand-in for the symmetric FLOPs, which need the blocks to count
            # exactly: the reduced (degeneracy-only) dimension raised to the number of
            # reduced indices the composition touches.
            n_out, n_mid = CASES[case]
            if sum(degs.values()) ** (2 * n_out + n_mid) > MAX_SYM_FLOPS:
                continue
            yield degs


# --------------------------------------------------------------------------------------
# Driver.


def key_of(row: dict) -> tuple:
    return (row["arm"], row["case"], tuple(sorted(row["degeneracies"].items())))


def existing(out: pathlib.Path | None) -> set:
    if out is None or not out.exists():
        return set()
    return {key_of(json.loads(line)) for line in out.read_text().splitlines() if line.strip()}


def julia_sweep(cases, space_kind, out, budget):
    """Hand the TensorKit arm its grid and let Julia append the same JSON lines."""
    spec = [
        {"case": c, "degeneracies": {str(k): v for k, v in degs.items()}}
        for c in cases
        for degs in points(c, space_kind)
    ]
    cmd = [
        "julia",
        "-t",
        "1",
        f"--project={JULIA_PROJECT}",
        str(JULIA_SCRIPT),
        json.dumps(spec),
        str(out),
        str(budget),
    ]
    print(" ".join(cmd[:5]) + " ...", file=sys.stderr)
    subprocess.run(cmd, check=True)


def report(path: pathlib.Path) -> None:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows.sort(key=lambda r: (r["case"], r["n_irreps"], r["dense_dim"], r["arm"]))
    head = (
        f"{'case':>6} {'irr':>4} {'deg':>4} {'Ddense':>7} {'arm':>10} "
        f"{'warmup ms':>10} {'steady ms':>10} {'MFLOP':>9} {'gemms':>6} "
        f"{'ns/FLOP':>9} {'blocks ms':>10} {'xblocks':>8} {'dense ms':>9} {'xdense':>7}"
    )
    print(head)
    print("-" * len(head))

    def cell(v, fmt):
        return " " * len(format(0.0, fmt)) if v is None else format(v, fmt)

    for r in rows:
        deg = max(r["degeneracies"].values())
        print(
            f"{r['case']:>6} {r['n_irreps']:>4} {deg:>4} {r['dense_dim']:>7} {r['arm']:>10} "
            f"{r['warmup_ms']:>10.3f} {r['steady_ms']:>10.4f} {r['flops'] / 1e6:>9.3f} "
            f"{r['n_gemm']:>6} {r['steady_ms'] * 1e6 / r['flops']:>9.3f} "
            f"{cell(r.get('blocks_ms'), '10.4f')} {cell(r.get('ratio_to_blocks'), '8.1f')} "
            f"{cell(r.get('dense_ms'), '9.4f')} "
            f"{cell(r.get('speedup_vs_dense'), '7.1f')}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, default="tenet")
    p.add_argument("--case", choices=tuple(CASES), default="rank3")
    p.add_argument("--space", choices=("grid", "production"), default="production")
    p.add_argument("--irreps", type=int, help="grid point: keep 2j = 0..irreps-1")
    p.add_argument("--deg", type=int, help="grid point: degeneracy of every kept irrep")
    p.add_argument("--sweep", action="store_true", help="every arm, case and grid point")
    p.add_argument("--budget", type=float, default=0.5, help="seconds of steady-state per point")
    p.add_argument("--verify", action="store_true", help="check the dense results agree")
    p.add_argument("--out", type=pathlib.Path, help="JSONL to append to; resumable")
    p.add_argument("--report", type=pathlib.Path, help="print the table from a JSONL and exit")
    a = p.parse_args()

    if a.report is not None:
        report(a.report)
        return

    done = existing(a.out)

    def emit(row: dict | None) -> None:
        if row is None:  # an arm that cannot be handed this point
            return
        if a.out is not None:
            with a.out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        else:
            print(json.dumps(row))

    if a.sweep:
        cases = tuple(CASES)
        if a.arm == "tensorkit":
            julia_sweep(cases, a.space, a.out, a.budget)
            return
        arms = ARMS[:-1] if a.arm == "tenet" else (a.arm,)
        for case in cases:
            for degs in points(case, a.space):
                for arm in arms:
                    row = {
                        "arm": arm,
                        "case": case,
                        "degeneracies": {str(k): v for k, v in degs.items()},
                    }
                    if key_of(row) in done:
                        continue
                    print(f"{arm} {case} {degs}", file=sys.stderr, flush=True)
                    emit(run_point(arm, case, degs, budget=a.budget, verify=a.verify))
        return

    if a.arm == "tensorkit":
        julia_sweep((a.case,), a.space, a.out, a.budget)
        return
    degs = degeneracies(a.irreps, a.deg) if a.space == "grid" else dict(PRODUCTION)
    emit(run_point(a.arm, a.case, degs, budget=a.budget, verify=a.verify))


if __name__ == "__main__":
    main()
