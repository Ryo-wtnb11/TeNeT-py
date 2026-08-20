"""Does the compiled matvec hold at quantum-chemistry scale? (#220)

``Env(compile=)`` -- reachable from ``dmrg_`` since #220 -- wraps the prepared two-site
matvec once per structure key. #141 measured that wrapper as ~20x on one bond of a lattice
model; nothing had measured it over a *sweep*, and nothing at all at ``K = 16``/``K = 26``.
The named risk is structural: block2's ``SparseMatrixInfo::ConnectionInfo`` is a flat
index-and-coefficient table whose build cost is linear in the number of blocks, while an
XLA graph grows with the traced operations, so trace and compile time need not scale that
way.

One process, one point of the grid ``(model, chi, arm)``; peak resident memory is this
process's own, which is why each point is run in its own subprocess::

    for m in lattice N2.CAS.6-31G C2.CAS.PVDZ; do
      for a in numpy jax jax-jit; do
        uv run python benchmarks/bench_dmrg_compile.py --model $m --chi 64 --arm $a
      done
    done

Reported per point as one JSON line, and the split is the finding rather than the total:
**trace + compile time separately from run time**, first-call latency, steady-state
per-matvec time, sweep wall, peak RSS, and how many ``compile()`` invocations one sweep
provokes -- a trace that is cheap once but happens per bond per sweep is a third outcome,
distinct from "holds" and "does not hold".

Not a test, on no CI path, nothing here is asserted. The quantum-chemistry inputs are
``bench_qc_mpo.py``'s fetched FCIDUMPs and carry that module's licence decision.
"""

import argparse
import json
import resource
import sys
import time

import bench_qc_mpo as qc
import numpy as np

from tenet import GradedSpace
from tenet.network import MPO, MPS, Env, dmrg_, local_op
from tenet.symmetry import U1, FZ2Sector, U1Sector, fZ2


def rss_gib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def lattice(n):
    """U(1) Heisenberg with a next-nearest-neighbour term, so the MPO bond is not trivial.

    Nearest neighbour alone gives ``D_w = 5``; ``J2 = 0.5`` at range 2 takes it to 8, which
    is the point -- wide enough that the prepared operator has several live fields.
    """
    phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    sz = np.diag([-0.5, 0.5])
    sp = np.array([[0.0, 0.0], [1.0, 0.0]])
    op = {
        0: local_op(sz, phys=phys, charge=U1Sector(0)),
        -2: local_op(sp, phys=phys, charge=U1Sector(-2)),
        2: local_op(sp.T, phys=phys, charge=U1Sector(2)),
    }
    terms = []
    for j, d in ((1.0, 1), (0.5, 2)):
        for i in range(n - d):
            terms.append((j, [(op[0], i), (op[0], i + d)]))
            terms.append((0.5 * j, [(op[-2], i), (op[2], i + d)]))
            terms.append((0.5 * j, [(op[2], i), (op[-2], i + d)]))
    return MPO.from_terms(n, terms, cutoff=None), phys, U1


def qc_model(name):
    """One of ``bench_qc_mpo.py``'s FCIDUMPs at the **default** cutoff -- M39's pinned
    operator, which is what #218's chi grid measured and the only route that finishes at
    K=26 (``cutoff=None`` did not complete ``Env.setup_`` there, at 19--24 GiB). It still
    carries its edge description, so the prepared matvec is still what runs.
    """
    norb, _, recs = qc.fetch(name)
    terms = qc.to_tenet_terms(qc.fold_terms(qc.spin_orbital_terms(recs))[0])
    phys = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
    return MPO.from_terms(2 * norb, terms), phys, fZ2


def bond_spaces(sym, n, chi):
    """``n + 1`` virtual spaces of dimension ``<= chi``, full-rank from sweep one.

    ``examples/toy_codes/dmrg.py``'s ``bond_spaces`` seeds degeneracy 1 per sector, so a
    chi=128 bond would be several sweeps away and the measured sweep would not be the one
    named. Here the degeneracy is spread over the reachable sectors instead, which is what
    makes the chi column mean what it says.
    """
    if sym is fZ2:
        mid = GradedSpace.new(fZ2, {FZ2Sector(0): chi - chi // 2, FZ2Sector(1): chi // 2})
        triv = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
        return [triv] + [mid] * (n - 1) + [triv]
    spaces = []
    for i in range(n + 1):
        w = min(i, n - i)  # the reachable S^z sectors on bond i, as in the toy code
        qs = list(range(-w, w + 1, 2))
        # The two boundary bonds stay one-dimensional -- that is the ``S^z_tot = 0``
        # statement, and a degenerate boundary would make the run a batch of states.
        deg = 1 if i in (0, n) else max(1, chi // len(qs))
        spaces.append(GradedSpace.new(U1, {U1Sector(q): deg for q in qs}))
    return spaces


class Timing:
    """``compile=`` instrumented: one counter per invocation, first call split from the rest.

    A ``jax.jit`` wrapper traces and compiles inside its *first* call, so the honest split
    is per compiled callable: ``first`` is trace + compile + one run, ``steady`` is one run
    alone, and the difference charged over the ``first`` list is what tracing cost.
    """

    def __init__(self, inner, sync):
        self.inner, self.sync = inner, sync
        self.n_compile = 0
        self.first = []
        self.steady = []

    def __call__(self, fn):
        self.n_compile += 1
        wrapped = fn if self.inner is None else self.inner(fn)
        seen = []

        def call(*args):
            t0 = time.perf_counter()
            out = wrapped(*args)
            if self.sync is not None:
                self.sync(out)
            dt = time.perf_counter() - t0
            (self.steady if seen else self.first).append(dt)
            seen.append(1)
            return out

        return call


def record_keys(seen):
    """Wrap ``Env.heff2`` to record ``(bond, structure key)`` -- what ``_compiled`` keys on."""
    orig = Env.heff2

    def heff2(self, n, aa):
        seen.add((n, tuple(aa.legs)))
        return orig(self, n, aa)

    Env.heff2 = heff2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="lattice", help="lattice, or a bench_qc_mpo input name")
    ap.add_argument("--chi", type=int, default=64)
    ap.add_argument("--arm", default="numpy", choices=("numpy", "jax", "jax-jit"))
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--sites", type=int, default=20, help="lattice only")
    a = ap.parse_args(argv)

    inner = sync = None
    if a.arm != "numpy":
        import jax  # application level: ``tenet.network`` never names an accelerator

        import tenet.pytree  # noqa: F401  (registers SymmetricTensor as a JAX pytree)

        jax.config.update("jax_enable_x64", True)  # the sweep's SVD is float64 either way
        sync = jax.block_until_ready
        if a.arm == "jax-jit":
            inner = jax.jit

    t0 = time.perf_counter()
    h, phys, sym = lattice(a.sites) if a.model == "lattice" else qc_model(a.model)
    t_build = time.perf_counter() - t0
    n = len(h)
    psi = MPS.random(phys, bond_spaces(sym, n, a.chi), seed=0)
    if a.arm != "numpy":
        psi = MPS.from_tensors(t.to_backend("jax") for t in psi)

    timing, keys = Timing(inner, sync), set()
    record_keys(keys)
    t1 = time.perf_counter()
    out = dmrg_(psi, h, chi=a.chi, cutoff=1e-10, max_sweeps=a.sweeps, compile=timing)
    t_sweep = time.perf_counter() - t1

    first, steady = timing.first, timing.steady
    mean_steady = float(np.mean(steady)) if steady else None
    print(
        json.dumps(
            {
                "model": a.model,
                "sites": n,
                "chi": a.chi,
                "arm": a.arm,
                "sweeps": a.sweeps,
                "energy": out.energy,
                "t_build_s": round(t_build, 3),
                "t_sweep_s": round(t_sweep, 3),
                "n_compile": timing.n_compile,
                "n_keys": len(keys),
                "n_bonds": len({b for b, _ in keys}),
                "n_matvec": len(first) + len(steady),
                "first_ms": round(1e3 * float(np.mean(first)), 3) if first else None,
                "first_total_s": round(sum(first), 3),
                "steady_ms": round(1e3 * mean_steady, 3) if steady else None,
                "steady_total_s": round(sum(steady), 3),
                # What tracing cost: the first-call latencies minus the run they each
                # contain, charged once per ``compile()``.
                "trace_total_s": round(sum(first) - len(first) * mean_steady, 3)
                if steady
                else None,
                "peak_rss_gib": round(rss_gib(), 3),
            }
        )
    )


if __name__ == "__main__":
    sys.exit(main())
