# Torch contraction execution (#353)

Measured 2026-09-06 against `560c19078492167f23fbfa6ecda1b6882c7433a7`.
macOS 15.5 arm64, Torch 2.13.0, CPU, one Torch thread, complex64 with nonzero
imaginary components. Each version ran in a fresh process. Repartition times
are medians of seven warm calls, including result assembly; backward includes
forward, a squared-magnitude loss and differentiation with respect to all input
matrices. The timings below are not GPU or 20-core-node measurements.

| Repartition case | Old forward (ms) | New forward (ms) | Old forward + backward (ms) | New forward + backward (ms) |
|---|---:|---:|---:|---:|
| su2-small (3 blocks) | 0.032 | 0.025 | 0.113 | 0.105 |
| u1-ragged (141 blocks) | 0.801 | 0.051 | 3.011 | 0.315 |
| su2-ragged (977 blocks) | 21.283 | 0.366 | 61.958 | 1.349 |
| su2-wide (23,872 blocks) | 1628.783 | 6.561 | 4599.110 | 14.185 |

All four output SHA-256 digests match. On the wide SU(2) case, profiled call
counts fall from 1,185,135 to 624.
Its cold call, including categorical planning, takes
5.54 -> 4.23 s, so the warm speedup
must not be used as a first-call claim. Peak process RSS across the complete
four-case benchmark is 2306 -> 659 MiB.
The flat executor saves index/coefficient buffers for backward (4.34 MiB of
unique saved storage in the wide case); the old path saves no buffers visible
to that hook but builds many more autograd nodes. Saved-tensor bytes alone
therefore do not measure total graph memory.

## Representative BP contractions

`bench_torch_bp.py` uses PEPS_VMC
`5515b260917cb1a30a239b643a740a25aff14f6e` and a seeded random 6x6 OBC
SU(2) PEPS: physical spin 1/2, virtual degeneracies `(5, 6, 5)` in sectors
`j=(0, 1/2, 1)`, hence physical bond dimension 32. It runs one untied BP
iteration from the same initial messages, with five warm repetitions.

| One BP iteration | Old (ms) | New (ms) |
|---|---:|---:|
| Forward, no graph | 442.4 | 296.8 |
| Forward, with graph | 737.4 | 365.9 |
| Forward + backward | 2231.5 | 737.9 |

Maximum absolute differences: messages `8.94e-8`, input gradients `9.02e-8`.
Relative L2 gradient difference: `1.75e-7`. Backward follows the application's
existing eager graph, including its scalar normalization controls; this is not
an implicit derivative of a converged BP fixed point.

This is **not** the issue's full P=3 circuit realization. The referenced
`run.py` lacks its reported `gauged`/`tied` arguments, and attempts to run the
available circuit API encountered dtype/provider mismatches. No end-to-end
realization speedup or GPU claim is made.

## Implementation and bounds

- Reuse the reduced-storage position helper already used by dense expansion;
  the new path does not expand a tensor into its dense physical basis.
- Group flat source indices by summand count, preserve addition order, and
  place the complete result in one gather. There is no array call per block.
- Skip unit coefficient multiplication and one-input concatenation.
- Expanded indices and coefficients share a cost-bounded cache. Tables larger
  than its byte budget are declined before expansion; the existing shape-bucket
  path handles them, with out-of-place arithmetic under autograd.
- Equal-shape sector matmuls are batched; singleton groups keep ordinary matmul.
  Stacking can still cost more for tiny CPU matrices. There is no unmeasured
  size threshold or runtime autotuner, and no claim of universal optimality.

## Reproduce

With Torch installed in the repository environment:

```sh
baseline_dir=$(mktemp -d)
git archive 560c19078492167f23fbfa6ecda1b6882c7433a7 src/tenet | tar -x -C "$baseline_dir"
PYTHONPATH="$baseline_dir/src" .venv/bin/python benchmarks/bench_torch_lowering.py
.venv/bin/python benchmarks/bench_torch_lowering.py
```

For BP, install the application dependencies and use the same application
checkout for both library versions:

```sh
PYTHONPATH="$baseline_dir/src" .venv/bin/python benchmarks/bench_torch_bp.py --peps-vmc /path/to/PEPS_VMC --output /tmp/bp-old.json
.venv/bin/python benchmarks/bench_torch_bp.py --peps-vmc /path/to/PEPS_VMC --output /tmp/bp-new.json
```

The BP script also saves messages and gradients in adjacent NPZ files. Both
scripts create fresh tensor wrappers with differentiable leaves before AD
measurements; toggling `requires_grad` after caching no-grad block views gives
an invalid gradient baseline on the old implementation.
