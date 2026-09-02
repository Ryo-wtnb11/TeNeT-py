"""tenet's ``svd_truncated`` against TensorKit.jl's ``svd_trunc(...; trunc=truncrank(D))``.

Not "are the singular values the same" -- they are, both libraries diagonalize the same
dense matrix in the same basis -- but *what the truncation convention is*, which is
where symmetric-tensor libraries quietly disagree. Four questions, one per column of the
report:

1. **What does ``D`` count?** A kept singular value in SU(2) sector ``j`` stands for
   ``2j+1`` dense basis states. ``D`` can mean ``D`` dense dimensions or ``D`` reduced
   (degeneracy-only) ones.
2. **Are multiplets split?** Can a cut keep part of an irrep's ``2j+1``?
3. **Tie-breaking.** With the same singular value in two sectors, which survives, and is
   the answer deterministic?
4. **The reported truncation error** -- norm or squared norm, relative or absolute,
   multiplicity-weighted or not.

How the two libraries are made comparable
-----------------------------------------
The same way ``bench_su2_libraries.py`` does it on the ``bench/su2-library-comparison``
branch, whose pinned Julia project (``su2_libraries_jl/``) this reuses: tenet builds the
tensor, expands it with
``to_dense()``, and TensorKit reads that very array back with ``TensorMap(A, V ← W)``,
which *refuses* an array outside its own invariant subspace. The Julia arm asserts the
round-trip (``convert(Array, t) == A``) before it truncates, so a basis difference would
be a loud failure rather than a silent one. Both arms then report the kept space, the
kept singular values, the truncation error, and the dense ``U S V†``, and this driver
compares them.

The cases
---------
* ``generic`` -- distinct singular values; the sort and the cut are unambiguous. If this
  disagrees, nothing else matters.
* ``isometry`` -- every singular value is exactly 1, so the cut is *entirely*
  tie-breaking and sector accounting.
* ``ties`` -- a spectrum built by hand from three values shared across every sector, so
  a cut has to choose between sectors at equal magnitude.

each in SU(2) and in U(1). U(1) is the control: its irreps have multiplicity 1, so a
disagreement there is about sorting or tie-breaking alone, while SU(2) adds ``2j+1``.

Running it
----------
::

    julia --project=benchmarks/su2_libraries_jl -e 'using Pkg; Pkg.instantiate()'
    uv run --no-sync python benchmarks/bench_svd_truncation.py

``--keep DIR`` writes the dense arrays somewhere durable instead of a temporary
directory; ``--verbose`` prints every case's per-``D`` rows rather than only the
disagreements.
"""

import argparse
import json
import pathlib
import subprocess
import tempfile

import numpy as np

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

HERE = pathlib.Path(__file__).resolve().parent
JULIA_PROJECT = HERE / "su2_libraries_jl"
JULIA_SCRIPT = HERE / "bench_svd_truncation.jl"

# {label: degeneracy}; the label is 2j for SU(2) and the charge for U(1), which is what
# the Julia arm parses back. Dense dimension 2*1 + 2*2 + 1*3 = 9 for SU(2), 5 for U(1),
# so the rank-4 map V⊗V ← V⊗V is 81x81 resp. 25x25 -- small enough to compare densely,
# wide enough that the bond crosses several sector boundaries.
SPACES = {"su2": {0: 2, 1: 2, 2: 1}, "u1": {0: 2, 1: 2, 2: 1}}
PROVIDERS = {"su2": (SU2, SU2Sector), "u1": (U1, U1Sector)}

# Below, at and above the natural sector boundaries, plus D past the full rank.
DS = {
    "su2": (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 50, 81, 200),
    "u1": (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 60),
}

# The tie pool: three values, cycled inside every sector, so equal magnitudes recur
# across sectors and the cut has to break the tie.
TIE_POOL = (1.0, 0.5, 0.25)

AXES = ((0, 1), (2, 3))


def build(sym: str, kind: str, seed: int = 0) -> SymmetricTensor:
    """The case's tensor: a rank-4 map ``V⊗V ← V⊗V`` over ``SPACES[sym]``."""
    provider, sector = PROVIDERS[sym]
    v = GradedSpace.new(provider, {sector(k): d for k, d in SPACES[sym].items()})
    cod, dom = (Leg(v, OUT),) * 2, (Leg(v, IN),) * 2
    if kind == "isometry":
        # square, so this is unitary per coupled sector: every singular value is 1
        return tenet.random_isometry(cod, dom, seed=seed)
    t = SymmetricTensor.random(cod + dom, seed=seed)
    if kind == "generic":
        return t
    # Exact ties, and exact on purpose: every coupled block is *diagonal* with entries
    # drawn from TIE_POOL, and LAPACK returns a diagonal matrix's singular values
    # bit-exactly. Rotating a prescribed spectrum into a dense block instead (U diag V†)
    # would come back as 1 +/- 1e-16, and then the cut is decided by last-bit noise
    # rather than by either library's tie-break -- which is the ``isometry`` case below,
    # kept precisely because that is what happens in practice.
    mats = {}
    for c, b in tenet.to_matrices(t).items():
        m = np.zeros_like(b)
        n = min(m.shape)
        np.fill_diagonal(m, np.take(TIE_POOL, np.arange(n) % len(TIE_POOL)))
        mats[c] = m
    return tenet.from_matrices(t.structure, mats)


def label(c) -> str:
    return str(c.two_j if isinstance(c, SU2Sector) else c.charge)


def tenet_row(t: SymmetricTensor, sym: str, D: int) -> dict:
    provider = PROVIDERS[sym][0]
    try:
        sel = tenet.linalg.select_bond(t, AXES, max_bond=D)
    except ValueError:
        # the budget could not pay for even the largest multiplet: tenet refuses to
        # return a bond with no sectors at all
        return {"refused": True}
    u, s, vh = tenet.linalg.svd_truncated(t, AXES, max_bond=D)
    recon = (u @ s @ vh).to_dense()
    return {
        "kept": {label(c): m for c, m in sel.bond.sectors},
        "dense_dim": sel.dense_dim,
        "reduced_dim": sel.reduced_dim,
        "values": sorted((sigma for sigma, _, _ in sel.kept), reverse=True),
        # tenet reports the *squared* norm of the discarded part; TensorKit reports the
        # norm. This is the one place the two numbers are put on the same footing.
        "error_reported": sel.discarded_weight,
        "error_recomputed": float(np.linalg.norm(t.to_dense() - recon)),
        "recon": recon,
        "qdim": {label(c): provider.qdim(c) for c, _ in sel.bond.sectors},
    }


def cases() -> list[tuple[str, str, SymmetricTensor]]:
    return [
        (f"{sym}_{kind}", sym, build(sym, kind))
        for sym in SPACES
        for kind in ("generic", "isometry", "ties")
    ]


def run_julia(dir_: pathlib.Path, built) -> dict:
    spec = []
    for name, sym, t in built:
        dense = np.ascontiguousarray(t.to_dense(), dtype=np.float64)
        dense.tofile(dir_ / f"{name}.bin")
        spec.append(
            {
                "name": name,
                "symmetry": sym,
                "space": {str(k): v for k, v in SPACES[sym].items()},
                "n_out": 2,
                "n_in": 2,
                "array": f"{name}.bin",
                "shape": list(dense.shape),
                "Ds": list(DS[sym]),
            }
        )
    (dir_ / "spec.json").write_text(json.dumps(spec))
    subprocess.run(
        ["julia", f"--project={JULIA_PROJECT}", str(JULIA_SCRIPT), str(dir_)], check=True
    )
    return json.loads((dir_ / "tk_results.json").read_text())


def compare(dir_: pathlib.Path, verbose: bool) -> int:
    built = cases()
    tk = run_julia(dir_, built)
    print(f"TensorKit {tk['tensorkit_version']} on Julia {tk['julia_version']}")
    worst_roundtrip = max(tk["roundtrip"].values())
    print(f"dense array round-trip tenet -> TensorKit -> dense: max |diff| = {worst_roundtrip:.2e}")
    rows = {(r["case"], r["D"]): r for r in tk["rows"]}

    head = (
        f"{'case':>12} {'D':>4} | {'kept (sector:mult)':>26} {'Ddense':>7} {'Dred':>5} "
        f"{'err':>10} | {'TensorKit kept':>26} {'Ddense':>7} {'Dred':>5} {'err':>10} | agree"
    )
    print()
    print(head)
    print("-" * len(head))
    disagreements = []
    for name, sym, t in built:
        for D in DS[sym]:
            ours = tenet_row(t, sym, D)
            theirs = rows[(name, D)]
            if ours.get("refused"):
                print(
                    f"{name:>12} {D:>4} | {'REFUSED (empty bond)':>26} {'':>7} {'':>5} {'':>10} | "
                    f"{fmt_kept(theirs['kept']):>26} {theirs['dense_dim']:>7} "
                    f"{theirs['reduced_dim']:>5} {theirs['error_recomputed']:>10.3e} | n/a"
                )
                continue
            theirs_recon = (
                np.fromfile(dir_ / theirs["recon"], dtype=np.float64)
                .reshape(ours["recon"].shape[::-1])
                .T
            )
            same_space = ours["kept"] == theirs["kept"]
            same_vals = len(ours["values"]) == len(theirs["values"]) and np.allclose(
                ours["values"], theirs["values"], atol=1e-10
            )
            # tenet's discarded_weight is the *square* of TensorKit's truncation error
            same_err = abs(ours["error_reported"] ** 0.5 - theirs["error_recomputed"]) < 1e-9
            checks = {"space": same_space, "values": same_vals, "error": same_err}
            # U and V are only unique up to a rotation inside each degenerate group, so
            # comparing the dense U S V† elementwise is a statement about the truncation
            # only where the kept spectrum has no repeats. Where it does, the error
            # column above is the gauge-invariant statement and this one is skipped.
            if not degenerate(ours["values"]):
                checks["recon"] = np.allclose(ours["recon"], theirs_recon, atol=1e-10)
            ok = all(checks.values())
            if not ok:
                disagreements.append((name, D, checks))
            if verbose or not ok:
                print(
                    f"{name:>12} {D:>4} | {fmt_kept(ours['kept']):>26} "
                    f"{ours['dense_dim']:>7.0f} {ours['reduced_dim']:>5} "
                    f"{ours['error_reported'] ** 0.5:>10.3e} | "
                    f"{fmt_kept(theirs['kept']):>26} {theirs['dense_dim']:>7} "
                    f"{theirs['reduced_dim']:>5} {theirs['error_recomputed']:>10.3e} | "
                    f"{'yes' if ok else 'NO'}"
                )
    print()
    if disagreements:
        print(f"{len(disagreements)} disagreement(s):")
        for name, D, checks in disagreements:
            print(f"  {name} D={D}: {', '.join(k for k, v in checks.items() if not v)}")
    else:
        print("kept space, kept singular values, reconstruction and error agree everywhere.")
    return len(disagreements)


def degenerate(values, tol: float = 1e-10) -> bool:
    """Does this kept spectrum repeat a value? Then ``U``/``V`` are not unique."""
    v = np.asarray(values)
    return bool(v.size > 1 and np.any(np.abs(np.diff(v)) < tol))


def fmt_kept(kept: dict) -> str:
    return " ".join(f"{k}:{v}" for k, v in sorted(kept.items(), key=lambda kv: int(kv[0])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep", type=pathlib.Path, help="write the dense arrays here, and keep them")
    p.add_argument("--verbose", action="store_true", help="print every row, not only mismatches")
    a = p.parse_args()
    if a.keep is not None:
        a.keep.mkdir(parents=True, exist_ok=True)
        raise SystemExit(1 if compare(a.keep, a.verbose) else 0)
    with tempfile.TemporaryDirectory() as d:
        raise SystemExit(1 if compare(pathlib.Path(d), a.verbose) else 0)


if __name__ == "__main__":
    main()
