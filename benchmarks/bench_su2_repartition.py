"""SU(2) tensordot: the general (bending) path against the aligned (no-bend) path.

Rank-4 SU(2) tensors, three sectors per leg, NumPy f64. Two contractions of the
same operands:

  general   axes=((1, 2), (0, 3))  -- legs cross sides, so `repartition` bends
  aligned   axes=((2, 3), (0, 1))  -- a's domain against b's codomain, zero bends

The general column is what #61 fuses; the aligned column is the control that must
not regress (it early-returns before any repartition plan is built).

Not a test, not part of the package. Run from the repo root:
  uv run python benchmarks/bench_su2_repartition.py
"""

import statistics
import time
from functools import partial

import numpy as np

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, SU2Sector

SECTORS = (SU2Sector(0), SU2Sector(1), SU2Sector(2))
GENERAL = ((1, 2), (0, 3))
ALIGNED = ((2, 3), (0, 1))
REPS = 15


def space(m):
    return GradedSpace.new(SU2, {s: m for s in SECTORS})


def operands(m, seed=0):
    """``a``, and the two partners that make the general/aligned contractions legal.

    Contractibility is opposite ``dual xor (side is IN)``, so the partner's duals
    differ between the two cases; the block content and sizes do not.
    """
    a = (Leg(space(m), OUT, dual=True), Leg(space(m), OUT), Leg(space(m), IN), Leg(space(m), IN))
    # a's axes 1 (OUT) and 2 (IN) are contracted: both operands' legs must cross
    general = (
        Leg(space(m), OUT, dual=True),
        Leg(space(m), OUT),
        Leg(space(m), IN),
        Leg(space(m), IN, dual=True),
    )
    # a's whole domain against the partner's whole codomain: zero bends
    aligned = (Leg(space(m), OUT), Leg(space(m), OUT), Leg(space(m), IN), Leg(space(m), IN))
    return tuple(
        SymmetricTensor.random(legs, seed=seed + i, dtype=np.float64)
        for i, legs in enumerate((a, general, aligned))
    )


def timed(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    print(f"SU(2) rank-4, {len(SECTORS)} sectors, f64   ms, median of {REPS}")
    print(f"{'m':>5} {'general':>10} {'aligned':>10} {'blocks':>8}")
    for m in (2, 4, 8, 16):
        a, bg, ba = operands(m)
        tenet.tensordot(a, bg, axes=GENERAL)  # warm the plan caches
        tenet.tensordot(a, ba, axes=ALIGNED)
        g = timed(partial(tenet.tensordot, a, bg, axes=GENERAL))
        al = timed(partial(tenet.tensordot, a, ba, axes=ALIGNED))
        print(f"{m:>5} {g:>10.3f} {al:>10.3f} {len(a.blocks):>8}", flush=True)


if __name__ == "__main__":
    main()
