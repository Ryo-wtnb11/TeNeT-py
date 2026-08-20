"""Where does one DMRG sweep actually spend its time? (#225)

#224/M54 measured the compiled two-site matvec and found it is **not** where the time is
at scale: arithmetic on its numbers puts the matvec at roughly 40 % of an N2 K=16,
chi=128 sweep, and the other ~60 % had never been decomposed. This instrument decomposes
it, on the **NumPy backend**, per sweep, into the phases the sweep is made of:

* ``assemble`` -- the two-site einsum ``"apx,xqr->apqr"``;
* ``lanczos_own`` -- the Krylov recurrence, the tridiagonal ``eigh`` and the
  recombination, with the ``heff2`` calls it makes subtracted out;
* ``heff2_prepare`` -- ``Env._prepare2``, the fold of the two environments into the
  bond's static cores, paid once per bond visit;
* ``heff2_apply`` -- ``_apply2``, paid ``ncv`` times per solve;
* ``heff2_other`` -- what is left inside ``Env.heff2``: the structure key and the
  compiled-callable cache;
* ``svd`` -- ``tenet.linalg.svd_truncated``;
* ``env_update`` -- ``Env.update_``, i.e. ``_fold_last`` / ``_fold_first``;
* ``writeback`` -- the ``MPS.__setitem__`` write barrier plus the two gauge einsums
  ``"xy,yqr->xqr"`` and ``"apx,xy->apy"``;
* ``spectrum`` -- the Schmidt read;
* ``residual`` -- **wall minus the rest**, reported and never distributed. It is the
  instrument's own honesty check: a large residual means a phase is missing, and the
  right response is to name it, not to normalise it away.

How it is instrumented, and what that costs
-------------------------------------------
By **wrapping call sites**, not by sampling. Nine wrappers go on:
``tenet.linalg.svd_truncated``, ``MPS.__setitem__``, ``Env.heff2``, ``Env._prepare2``,
``Env.update_``, ``dmrg.lanczos``, ``dmrg.spectrum``, ``Env(compile=)`` around
``_apply2``, and a proxy over ``dmrg.py``'s own ``tenet`` module global so that the three
einsums *the sweep itself* spells are timed while the hundreds inside the matvec are not.
Nested timers are subtracted, so no second of wall time is counted twice.

The wrappers are Python frames around calls that are otherwise BLAS-bound, so the
overhead is a fixed cost per call rather than a fraction of the work, and it is measured
rather than assumed: every point is run in **both** arms, ``plain`` (no wrapper at all,
one number: the sweep wall) and ``wrapped``, and both walls are reported. ``compile=`` is
a Python identity stub in the wrapped arm, so the matvec is the same plain ``_apply2``
the ``compile=None`` default runs.

Not a test, on no CI path, nothing here is asserted. The model fixtures and the
quantum-chemistry inputs are ``bench_dmrg_compile.py``'s and ``bench_qc_mpo.py``'s, and
carry that module's licence decision. Run from the repo root::

    uv run python benchmarks/bench_sweep_phases.py --model N2.CAS.6-31G --chis 16,64,128
    uv run python benchmarks/bench_sweep_phases.py --model C2.CAS.PVDZ --chis 16,64,128
    uv run python benchmarks/bench_sweep_phases.py --model lattice --chis 64

One JSON line per ``(model, chi)``, carrying both walls and every phase in seconds and as
a share of the wrapped wall.
"""

import argparse
import collections
import json
import sys
import time

import bench_dmrg_compile as compile_bench

import tenet
from tenet.network import MPS, Env
from tenet.network import dmrg as dmrg_module

#: The three einsums ``sweep_`` spells itself: the two-site assembly and the two gauge
#: multiplications that put ``s`` back into the MPS. Every other einsum reached through
#: ``dmrg.py``'s ``tenet`` global belongs to a phase that is already timed by its own
#: wrapper, so it rides through the proxy untimed.
SWEEP_EINSUMS = {
    "apx,xqr->apqr": "assemble",
    "xy,yqr->xqr": "writeback",
    "apx,xy->apy": "writeback",
}


class Clock:
    """Accumulated seconds and call counts per phase, with nesting subtracted.

    ``heff2`` runs inside ``lanczos`` and ``_prepare2``/``_apply2`` run inside ``heff2``,
    so an outer timer contains its inner ones. Rather than thread a stack through every
    wrapper, each phase is accumulated gross and the containment is undone once at the
    end, in ``report`` -- the containment is static and there are exactly two levels of
    it.
    """

    def __init__(self):
        self.gross = collections.defaultdict(float)
        self.calls = collections.defaultdict(int)

    def wrap(self, phase, fn):
        """Return ``fn`` with its wall time accumulated under ``phase``."""

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.gross[phase] += time.perf_counter() - t0
                self.calls[phase] += 1

        return timed

    def report(self, wall):
        """Net seconds per phase plus the residual, as ``{phase: seconds}``."""
        g = self.gross
        heff2_own = g["heff2"] - g["heff2_prepare"] - g["heff2_apply"]
        net = {
            "assemble": g["assemble"],
            "lanczos_own": g["lanczos"] - g["heff2"],
            "heff2_prepare": g["heff2_prepare"],
            "heff2_apply": g["heff2_apply"],
            "heff2_other": heff2_own,
            "svd": g["svd"],
            "env_update": g["env_update"],
            "writeback": g["writeback"],
            "spectrum": g["spectrum"],
        }
        net["residual"] = wall - sum(net.values())
        return net


class _TenetProxy:
    """``dmrg.py``'s ``tenet`` module with ``einsum`` timed for the three sweep equations.

    A proxy rather than a patch on ``tenet.einsum`` itself: the sweep's own three einsums
    are the phase, and patching the package attribute would put a Python frame in front
    of every contraction the matvec and the environment folds make -- hundreds per bond,
    all of them already timed by an outer wrapper.
    """

    def __init__(self, clock):
        self._clock = clock

    def __getattr__(self, name):
        return getattr(tenet, name)

    def einsum(self, equation, *operands):
        phase = SWEEP_EINSUMS.get(equation)
        if phase is None:
            return tenet.einsum(equation, *operands)
        t0 = time.perf_counter()
        try:
            return tenet.einsum(equation, *operands)
        finally:
            self._clock.gross[phase] += time.perf_counter() - t0
            self._clock.calls[phase] += 1


def install(clock):
    """Put every wrapper on, and return the undo callable."""
    import tenet.ops.linalg as linalg_module

    saved = [
        (linalg_module, "svd_truncated", linalg_module.svd_truncated),
        (MPS, "__setitem__", MPS.__setitem__),
        (Env, "heff2", Env.heff2),
        (Env, "_prepare2", Env._prepare2),
        (Env, "update_", Env.update_),
        (dmrg_module, "lanczos", dmrg_module.lanczos),
        (dmrg_module, "spectrum", dmrg_module.spectrum),
        (dmrg_module, "tenet", dmrg_module.tenet),
    ]
    linalg_module.svd_truncated = clock.wrap("svd", linalg_module.svd_truncated)
    MPS.__setitem__ = clock.wrap("writeback", MPS.__setitem__)
    Env.heff2 = clock.wrap("heff2", Env.heff2)
    Env._prepare2 = clock.wrap("heff2_prepare", Env._prepare2)
    Env.update_ = clock.wrap("env_update", Env.update_)
    dmrg_module.lanczos = clock.wrap("lanczos", dmrg_module.lanczos)
    dmrg_module.spectrum = clock.wrap("spectrum", dmrg_module.spectrum)
    dmrg_module.tenet = _TenetProxy(clock)

    def undo():
        for owner, name, original in saved:
            setattr(owner, name, original)

    return undo


def one_sweep(h, phys, sym, chi, clock, compiles):
    """Canonize a fresh state, build its environments, sweep once, return the wall."""
    n = len(h)
    psi = MPS.random(phys, compile_bench.bond_spaces(sym, n, chi), seed=0)
    psi.canonize_(0)

    def stub(fn):  # identity: the wrapped arm still runs the plain ``_apply2``
        compiles.append(fn)
        return clock.wrap("heff2_apply", fn) if clock is not None else fn

    env = Env(psi, h, compile=stub).setup_(0)
    undo = install(clock) if clock is not None else (lambda: None)
    try:
        t0 = time.perf_counter()
        dmrg_module.sweep_(psi, h, env, {}, chi=chi, cutoff=1e-10)
        return time.perf_counter() - t0
    finally:
        undo()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="lattice", help="lattice, or a bench_qc_mpo name")
    ap.add_argument("--chis", default="16,64,128")
    ap.add_argument("--sites", type=int, default=20, help="lattice only")
    ap.add_argument("--warmup", type=int, default=1, help="discarded sweeps before each point")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    if a.model == "lattice":
        h, phys, sym = compile_bench.lattice(a.sites)
    else:
        h, phys, sym = compile_bench.qc_model(a.model)
    t_build = time.perf_counter() - t0

    for chi in (int(c) for c in a.chis.split(",")):
        # A discarded sweep first: the edge tables, the group embeddings and the merged
        # cores are all built on a bond's *first* visit, so whichever arm ran first would
        # otherwise carry the whole operator's construction. Measured on the lattice at
        # chi=16 that bias is 7x, which is larger than anything the table is about.
        for _ in range(a.warmup):
            one_sweep(h, phys, sym, chi, None, [])
        plain_compiles = []
        plain = one_sweep(h, phys, sym, chi, None, plain_compiles)
        clock, compiles = Clock(), []
        wrapped = one_sweep(h, phys, sym, chi, clock, compiles)
        net = clock.report(wrapped)
        print(
            json.dumps(
                {
                    "model": a.model,
                    "sites": len(h),
                    "chi": chi,
                    "warmup": a.warmup,
                    "t_build_s": round(t_build, 3),
                    "wall_plain_s": round(plain, 3),
                    "wall_wrapped_s": round(wrapped, 3),
                    "overhead_pct": round(100.0 * (wrapped - plain) / plain, 2),
                    "n_compile": len(compiles),
                    "n_compile_plain": len(plain_compiles),
                    "phase_s": {k: round(v, 3) for k, v in net.items()},
                    "phase_pct": {k: round(100.0 * v / wrapped, 2) for k, v in net.items()},
                    "calls": dict(clock.calls),
                    "peak_rss_gib": round(compile_bench.rss_gib(), 3),
                },
                sort_keys=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
