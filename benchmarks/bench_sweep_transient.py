"""Where do the *first* sweeps spend their time, and what does a moved bond cost? (#248)

``bench_sweep_phases.py`` decomposes one **steady** sweep. #250 showed the gap that
matters is not there: on the site-tensor path at N=64/chi=256 U(1) the first sweep is
38.66 s against a 2.70 s steady sweep, and none of the prepared machinery runs. So this
instrument reports the same decomposition **per sweep over the first twelve**, on
``bench_vs_yastn.py``'s exact fixture, plus two columns that sweep instrument does not
have:

* **the structural cross-cut** -- the wall spent inside the ``functools.cache``d plan
  layer on a *miss*: ``TensorStructure``'s block enumeration (``_block_order`` and the
  tables derived from it), ``contraction_plan``, the repartition / fusion / map /
  permutation layouts. Every one of those caches is keyed on a ``TensorStructure``, and a
  ``TensorStructure`` hashes its legs, and a ``Leg`` hashes its ``GradedSpace``
  *including the degeneracies*. So a bond whose degeneracies moved is a new key
  everywhere, and the whole plan layer is rebuilt for it. This is the mechanism under
  test.
* **the bond churn** -- how many of the ``n - 1`` internal bonds changed their
  ``GradedSpace`` since the previous sweep, split into "the sector set moved" and "only
  the degeneracies moved". The transient should track this column and nothing else.

The phase columns partition the sweep wall (``residual`` is the honesty check, reported
and never distributed). The structural columns are a **cross-cut**, not a further split:
the plan-layer wall sits *inside* ``heff2``/``svd``/``env_update``, so it is also
attributed per phase (``struct_by_phase``) rather than added to the partition. Nested
structural calls are timed at the outermost frame only, so no second is counted twice.

Not a test, on no CI path, nothing here is asserted. One JSON line per sweep. Run from
the repo root::

    uv run python benchmarks/bench_sweep_transient.py --model heisenberg --n 64 --chi 256
    uv run python benchmarks/bench_sweep_transient.py --model hubbard --n 32 --chi 256
"""

import os

# Before NumPy is imported by anything: one thread per arm, as ``bench_vs_yastn.py``.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import bench_vs_yastn as fixture  # noqa: E402
import numpy as np  # noqa: E402

import tenet  # noqa: E402
from tenet.network import MPO, MPS, Env  # noqa: E402
from tenet.network import dmrg as dmrg_module  # noqa: E402

#: The three einsums ``sweep_`` spells itself (``bench_sweep_phases.py``'s table).
SWEEP_EINSUMS = {
    "apx,xqr->apqr": "assemble",
    "xy,yqr->xqr": "writeback",
    "apx,xy->apy": "writeback",
}

#: The ``functools.cache``d plan layer, ``(module path, attribute)``. Every one is keyed
#: on a ``TensorStructure`` (or on legs), so every one misses when a bond space moves.
PLAN_CACHES = (
    ("tenet.structure", "_block_order"),
    ("tenet.structure", "_index_map"),
    ("tenet.structure", "_axis_sectors_table"),
    ("tenet.structure", "_block_shape_table"),
    ("tenet.ops.contraction", "contraction_plan"),
    ("tenet.ops.repartition", "repartition_plan"),
    ("tenet.ops.repartition", "bend_plan"),
    ("tenet.ops.fusion", "fusion_plan"),
    ("tenet.ops.fusion", "fuse_spaces"),
    ("tenet.ops.map", "adjoint_plan"),
    ("tenet.ops.map", "_diagonal_subscripts"),
    ("tenet.ops.permutation", "permutation_plan"),
    ("tenet.ops.dense", "dense_plan"),
    ("tenet.map_view", "map_layout"),
    # The same three plans again where a caller imported them by name: patching the
    # defining module alone would leave those call sites untimed.
    ("tenet.ops.map", "map_layout"),
    ("tenet.ops.linalg", "map_layout"),
    ("tenet.ops.repartition", "permutation_plan"),
    # Keyed on sectors alone, with no degeneracy in the key: the control column.
    ("tenet.fusion_tree", "_fusion_trees"),
    ("tenet.fusion_tree", "_coupled_sectors"),
    ("tenet.fusion_tree", "_all_trees"),
)


class Clock:
    """Per-phase and per-plan-cache wall, with nesting subtracted.

    ``heff2`` runs inside ``lanczos``, so phases are accumulated gross and the containment
    undone in ``report``. Plan-cache calls nest inside one another (``_index_map`` calls
    ``_block_order``) and inside a phase; only the outermost plan-cache frame is timed,
    and it is attributed to the phase on top of the stack at that moment.
    """

    def __init__(self):
        self.gross = collections.defaultdict(float)
        self.calls = collections.defaultdict(int)
        self.struct_s = collections.defaultdict(float)
        self.struct_calls = collections.defaultdict(int)
        self.struct_by_phase = collections.defaultdict(float)
        self.structures_built = 0
        self.phase_stack = []
        self.struct_depth = 0

    def wrap(self, phase, fn):
        """Return ``fn`` with its wall accumulated under ``phase``."""

        def timed(*args, **kwargs):
            self.phase_stack.append(phase)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.gross[phase] += time.perf_counter() - t0
                self.calls[phase] += 1
                self.phase_stack.pop()

        return timed

    def wrap_plan(self, name, fn):
        """Return a cached ``fn`` timed only when it is the outermost plan-cache frame."""

        def timed(*args, **kwargs):
            # Counting only on the hot path: these are called ~10^5 times a sweep, so a
            # ``cache_info()`` per call would be the measurement rather than the work.
            # Misses are read once per sweep, from the caches themselves.
            self.struct_calls[name] += 1
            if self.struct_depth:
                return fn(*args, **kwargs)
            self.struct_depth = 1
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                self.struct_depth = 0
                self.struct_s[name] += dt
                self.struct_by_phase[self.phase_stack[-1] if self.phase_stack else "outside"] += dt

        return timed

    def report(self, wall):
        """Net seconds per phase plus the residual, as ``{phase: seconds}``."""
        g = self.gross
        net = {
            "assemble": g["assemble"],
            "lanczos_own": g["lanczos"] - g["heff2"],
            "heff2": g["heff2"],
            "svd": g["svd"],
            "env_update": g["env_update"],
            "env_clear": g["env_clear"],
            "writeback": g["writeback"],
            "spectrum": g["spectrum"],
        }
        net["residual"] = wall - sum(net.values())
        return net


class _TenetProxy:
    """``dmrg.py``'s ``tenet`` module with ``einsum`` timed for the three sweep equations."""

    def __init__(self, clock):
        self._clock = clock

    def __getattr__(self, name):
        return getattr(tenet, name)

    def einsum(self, equation, *operands):
        phase = SWEEP_EINSUMS.get(equation)
        if phase is None:
            return tenet.einsum(equation, *operands)
        self._clock.phase_stack.append(phase)
        t0 = time.perf_counter()
        try:
            return tenet.einsum(equation, *operands)
        finally:
            self._clock.gross[phase] += time.perf_counter() - t0
            self._clock.calls[phase] += 1
            self._clock.phase_stack.pop()


def _plan_targets():
    """Resolve ``PLAN_CACHES`` to ``(module, name, function)``, skipping absent ones."""
    import importlib

    out = []
    for path, name in PLAN_CACHES:
        module = importlib.import_module(path)
        fn = getattr(module, name, None)
        if fn is not None and hasattr(fn, "cache_info"):
            out.append((module, name, fn))
    return out


def install(clock):
    """Put every wrapper on, and return the undo callable."""
    import tenet.ops.linalg as linalg_module
    import tenet.structure as structure_module

    saved = [
        (linalg_module, "svd_truncated", linalg_module.svd_truncated),
        (MPS, "__setitem__", MPS.__setitem__),
        (Env, "heff2", Env.heff2),
        (Env, "update_", Env.update_),
        (Env, "clear_", Env.clear_),
        (dmrg_module, "lanczos", dmrg_module.lanczos),
        (dmrg_module, "spectrum", dmrg_module.spectrum),
        (dmrg_module, "tenet", dmrg_module.tenet),
    ]
    linalg_module.svd_truncated = clock.wrap("svd", linalg_module.svd_truncated)
    MPS.__setitem__ = clock.wrap("writeback", MPS.__setitem__)
    Env.heff2 = clock.wrap("heff2", Env.heff2)
    Env.update_ = clock.wrap("env_update", Env.update_)
    Env.clear_ = clock.wrap("env_clear", Env.clear_)
    dmrg_module.lanczos = clock.wrap("lanczos", dmrg_module.lanczos)
    dmrg_module.spectrum = clock.wrap("spectrum", dmrg_module.spectrum)
    dmrg_module.tenet = _TenetProxy(clock)

    for module, name, fn in _plan_targets():
        saved.append((module, name, fn))
        setattr(module, name, clock.wrap_plan(f"{module.__name__.split('.')[-1]}.{name}", fn))

    # Counting only: how many ``TensorStructure`` objects the sweep constructs.
    original_post_init = structure_module.TensorStructure.__post_init__

    def counted_post_init(self):
        clock.structures_built += 1
        original_post_init(self)

    saved.append((structure_module.TensorStructure, "__post_init__", original_post_init))
    structure_module.TensorStructure.__post_init__ = counted_post_init

    def undo():
        for owner, name, original in saved:
            setattr(owner, name, original)

    return undo


def plan_cache_info(field: str):
    """``{name: cache_info().<field>}`` for every plan cache."""
    return {
        f"{m.__name__.split('.')[-1]}.{n}": getattr(fn.cache_info(), field)
        for m, n, fn in _plan_targets()
    }


def build(model: str, n: int, chi: int, sites: bool):
    """``bench_vs_yastn.run_tenet``'s fixture, up to the point the sweeps start."""
    from tenet import GradedSpace
    from tenet.network import local_op
    from tenet.symmetry import U1, FZ2Sector, U1Sector, fZ2

    if model == "heisenberg":
        sym, sector = U1, U1Sector
        phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        sp = np.array([[0.0, 0.0], [1.0, 0.0]])
        ops = {
            "sz": local_op(np.diag([-0.5, 0.5]), phys=phys, charge=U1Sector(0)),
            "sp": local_op(sp, phys=phys, charge=U1Sector(-2)),
            "sm": local_op(sp.T, phys=phys, charge=U1Sector(2)),
        }
    else:
        sym, sector = fZ2, FZ2Sector
        phys = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})
        odd, even = FZ2Sector(1), FZ2Sector(0)
        ops = {
            "cpu": local_op(fixture.C_UP.T, phys=phys, charge=odd),
            "cu": local_op(fixture.C_UP, phys=phys, charge=odd),
            "cpd": local_op(fixture.C_DN.T, phys=phys, charge=odd),
            "cd": local_op(fixture.C_DN, phys=phys, charge=odd),
            "nud": local_op(fixture.N_UP @ fixture.N_DN, phys=phys, charge=even),
        }
    terms = [(a, [(ops[name], i) for name, i in prod]) for a, prod in fixture.model_terms(model, n)]
    h = MPO.from_terms(n, terms)
    if sites:
        # The edge description dropped: ``Env.heff2`` takes the site-tensor path (#250).
        h = MPO(h.sites) if not hasattr(h, "materialize") else h.materialize()
    bonds = [
        GradedSpace.new(sym, {sector(q): d for q, d in space.items()})
        for space in fixture.bond_charges(model, n, chi)
    ]
    psi = MPS.random(phys, bonds, seed=245)
    psi.canonize_(0)
    return h, psi


def bond_spaces(psi: MPS) -> list:
    """The ``n - 1`` internal bond spaces, as ``GradedSpace`` objects."""
    return [psi[i].legs[2].space for i in range(len(psi) - 1)]


def churn(old: list, new: list) -> tuple[int, int]:
    """``(bonds whose sector set moved, bonds where only degeneracies moved)``."""
    sectors_moved = degen_moved = 0
    for a, b in zip(old, new, strict=True):
        if a == b:
            continue
        if tuple(a) != tuple(b):
            sectors_moved += 1
        else:
            degen_moved += 1
    return sectors_moved, degen_moved


def rss_gib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=("heisenberg", "hubbard"), default="heisenberg")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--chi", type=int, default=256)
    p.add_argument("--sweeps", type=int, default=12)
    p.add_argument("--prepared", action="store_true", help="keep the edge description")
    p.add_argument("--plain", action="store_true", help="no wrappers: the sweep wall alone")
    p.add_argument("--tag", default="", help="free label carried into every row")
    p.add_argument("--out", type=pathlib.Path, help="JSONL to append the rows to")
    a = p.parse_args(argv)

    h, psi = build(a.model, a.n, a.chi, sites=not a.prepared)
    env = Env(psi, h).setup_(0)
    schmidt: dict[int, list[float]] = {}
    spaces = bond_spaces(psi)

    fh = a.out.open("a") if a.out else None
    for sweep in range(1, a.sweeps + 1):
        clock = Clock()
        misses_before = plan_cache_info("misses")
        undo = install(clock) if not a.plain else (lambda: None)
        try:
            t0 = time.perf_counter()
            energy, _ = dmrg_module.sweep_(psi, h, env, schmidt, chi=a.chi, cutoff=0.0, ncv=3)
            wall = time.perf_counter() - t0
        finally:
            undo()
        misses = {k: v - misses_before[k] for k, v in plan_cache_info("misses").items()}
        new_spaces = bond_spaces(psi)
        sectors_moved, degen_moved = churn(spaces, new_spaces)
        spaces = new_spaces
        net = clock.report(wall)
        row = {
            "model": a.model,
            "n": a.n,
            "chi": a.chi,
            "path": "prepared" if a.prepared else "sites",
            "tag": a.tag,
            "sweep": sweep,
            "wall_s": round(wall, 3),
            "energy": energy,
            "bonds_sector_moved": sectors_moved,
            "bonds_degen_moved": degen_moved,
            "bonds_total": a.n - 1,
            "phase_s": {k: round(v, 3) for k, v in net.items()},
            "phase_pct": {k: round(100.0 * v / wall, 2) for k, v in net.items()},
            "struct_total_s": round(sum(clock.struct_s.values()), 3),
            "struct_pct": round(100.0 * sum(clock.struct_s.values()) / wall, 2),
            "struct_s": {k: round(v, 3) for k, v in sorted(clock.struct_s.items()) if v >= 5e-4},
            "struct_misses": {k: v for k, v in sorted(misses.items()) if v},
            "struct_calls": dict(sorted(clock.struct_calls.items())),
            "struct_by_phase_s": {k: round(v, 3) for k, v in sorted(clock.struct_by_phase.items())},
            "structures_built": clock.structures_built,
            "plan_cache_sizes": plan_cache_info("currsize"),
            "peak_rss_gib": round(rss_gib(), 3),
        }
        line = json.dumps(row)
        print(line, flush=True)
        if fh is not None:
            fh.write(line + "\n")
            fh.flush()
    if fh is not None:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
