"""tenet against YASTN, head to head, on U(1) Heisenberg and fermionic Hubbard (#245).

YASTN is the fairest available opponent: pure Python + NumPy, symmetric blocked tensors,
two-site DMRG with the same ``ncv = 3`` Lanczos. Against block2 the gap would be a
C++/MKL constant factor; against YASTN what shows is design.

YASTN is **not** a dependency of this project -- not core, not test, not dev. Install it
for the benchmark only::

    uv pip install "yastn @ git+https://github.com/yastn/yastn.git"

and then run with ``uv run --no-sync`` so the sync does not remove it again.

The three arms
--------------
``--arm tenet`` builds its Hamiltonian with ``MPO.from_terms``, which keeps an
``EdgeTable``, so ``Env.heff2`` -- which routes on ``self.h.edges is not None`` -- takes
the **prepared** path. ``--arm tenet-sites`` is the same run with ``h.materialize()`` --
M67's recommended lattice spelling, which is what M64b spelled ``MPO(h.sites)``: the
same tensors, the description dropped, so the same call takes the **site-tensor** path,
which is YASTN's ``Heff2`` contraction order. Nothing else differs between the two --
same state, same seed, same ``chi``, same sweeps -- so ``tenet / tenet-sites`` prices the
prepared machinery and ``tenet-sites / yastn`` prices the rest. ``--arm yastn`` is YASTN
itself (#245). The split is the question #247 left open, measured rather than argued.

The conditions, each held equal
-------------------------------
Every one of these is a knob that, left unmatched, would make the numbers a lie.

* **The same term list builds both Hamiltonians.** ``model_terms`` spells the model once
  as ``(amplitude, [(op_name, site), ...])`` and the two arms translate it, tenet through
  ``MPO.from_terms`` and YASTN through ``Hterm`` + ``generate_mpo``. The on-site matrices
  are the same convention on both sides, checked by the ``--ed`` anchor below.
* **Truncation.** YASTN ``opts_svd={'D_total': chi}``; tenet ``chi=chi, cutoff=0.0``.
  YASTN's ``tol`` keeps ``sigma > tol * max(sigma)`` (``tensor/linalg.py``,
  ``truncation_mask``), which is exactly tenet's ``cutoff_mode="rel"`` -- the rules are
  identical, not merely similar. tenet's sweep does not expose ``cutoff_mode`` and uses
  its ``"rsum2"`` default, so rather than compare two different rules the cutoff is
  switched **off on both sides** (``cutoff=0.0`` admits the whole spectrum in ``rsum2``;
  YASTN's ``tol`` defaults to ``-inf``) and ``chi``/``D_total`` is the only rule acting.
  Those two are the same rule: both keep the largest values up to a total kept count, and
  on U(1)/Z2 every sector has quantum dimension 1, so tenet's ``qdim``-weighted dense
  budget is a plain count.
* **Lanczos:** ``ncv=3`` both (tenet's default was deliberately YASTN's).
* **Two-site both**, and one sweep means the same thing on both sides: left-to-right then
  right-to-left (tenet ``sweep_``, YASTN ``_dmrg_sweep_2site_``).
* **Fixed sweep count, convergence disabled.** YASTN ``energy_tol=None,
  Schmidt_tol=None``; tenet ``energy_tol=0.0, schmidt_tol=0.0`` (``denergy`` is an
  absolute value, so ``< 0.0`` never fires). The comparison is per sweep, not per
  "convergence".
* **The same initial state spec.** ``bond_charges`` gives one list of virtual spaces --
  charge to degeneracy, per bond -- and both arms seed a random MPS on exactly those
  spaces, full rank from sweep one so ``chi`` means what it says. The *entries* differ:
  the two libraries have different RNGs and there is no way to make one draw the other's.
  That is the one condition that could not be made identical, and it costs nothing in the
  wall column (the block shapes are what the arithmetic sees) while showing up in the
  energy column only as the residual convergence difference at a fixed sweep count.
* **Single-threaded BLAS**, set in this process before NumPy is imported, and one process
  per point.

Fermionic grading
-----------------
tenet grades the Hubbard site by fZ2 -- the Jordan-Wigner string *is* the braiding
(#147) -- so the ``d = 4`` site is even ``{|0>, |ud>}`` and odd ``{|u>, |d>}``, two blocks
of 2. YASTN is run through ``SpinfulFermions(sym='Z2')``, whose site is the same two
blocks of 2 in the same basis order. Z2 rather than U(1)xU(1) on purpose: U(1)xU(1)
grades by ``(n_up, n_dn)`` and would hand YASTN four blocks of 1 on the site and
correspondingly finer virtual blocks, i.e. strictly less arithmetic for the same chi.
That would be a comparison of gradings, not of implementations.

What is measured
----------------
One JSON line per ``(model, n, chi, arm)``, appended to ``--out`` the moment the point
finishes, so a kill loses at most one point; a point already in ``--out`` is skipped, so
the driver loop is resumable. Per point: the energy after the last sweep, the per-sweep
walls (the first is discarded as warm-up -- it carries canonization and environment
setup), the steady mean over the last ``--steady`` sweeps, this process's peak RSS, and
the realized bond dimensions so the "same bond spaces" claim is checkable rather than
asserted. A point that runs past ``--budget`` seconds stops at the next sweep boundary
and is recorded with ``status="budget"`` rather than dropped.

Exact diagonalization is out of reach at every grid point (the smallest is N=16 spinful,
i.e. 4**16). The ``--ed`` flag therefore serves a separate, small anchor run whose job is
to prove the two term lists mean the same Hamiltonian:

    uv run --no-sync python benchmarks/bench_vs_yastn.py --model heisenberg --n 10 \\
        --chi 32 --arm tenet --sweeps 6 --ed --out anchors.jsonl

Not a test, on no CI path, nothing here is asserted. Run from the repo root::

    for m in heisenberg hubbard; do
      for n in 32 64; do for chi in 64 128 256; do for a in tenet tenet-sites yastn; do
        uv run --no-sync python benchmarks/bench_vs_yastn.py \\
            --model $m --n $n --chi $chi --arm $a --out vs_yastn.jsonl
      done; done; done
    done
"""

import os

# Before NumPy is imported by anything, including tenet and yastn: a multi-threaded BLAS
# would put a different number of cores behind each arm's ``dgemm`` and the wall column
# would measure the thread pool rather than the library.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

# --------------------------------------------------------------------------------------
# The models, spelled once.
#
# ``model_terms`` returns ``(amplitude, [(op_name, site), ...])`` products, the shape both
# ``MPO.from_terms`` and ``Hterm`` take. Operator names are resolved per arm.
# --------------------------------------------------------------------------------------

#: ``U/t`` for the Hubbard runs, and ``t`` itself. Not a flag: the issue fixes them.
HUBBARD_T, HUBBARD_U = 1.0, 4.0


def model_terms(model: str, n: int) -> list:
    """The Hamiltonian as a list of ``(amplitude, [(op_name, site), ...])`` products."""
    terms = []
    if model == "heisenberg":
        # H = sum_i S^z_i S^z_{i+1} + (1/2) (S^+_i S^-_{i+1} + S^-_i S^+_{i+1})
        for i in range(n - 1):
            terms.append((1.0, [("sz", i), ("sz", i + 1)]))
            terms.append((0.5, [("sp", i), ("sm", i + 1)]))
            terms.append((0.5, [("sm", i), ("sp", i + 1)]))
        return terms
    if model == "hubbard":
        # H = -t sum_{i,s} (c+_{i,s} c_{i+1,s} + h.c.) + U sum_i n_up n_dn
        for i in range(n - 1):
            for s in "ud":
                terms.append((-HUBBARD_T, [(f"cp{s}", i), (f"c{s}", i + 1)]))
                terms.append((-HUBBARD_T, [(f"cp{s}", i + 1), (f"c{s}", i)]))
        for i in range(n):
            terms.append((HUBBARD_U, [("nud", i)]))
        return terms
    raise SystemExit(f"unknown model {model!r}")


def bond_charges(model: str, n: int, chi: int) -> list[dict[int, int]]:
    """``n + 1`` virtual spaces as ``{charge: degeneracy}``, one spec for both arms.

    Full rank from sweep one -- ``bench_dmrg_compile.py``'s rule -- so the measured sweep
    is the one the ``chi`` column names rather than one on the way up from a product
    state. The two boundary bonds stay one-dimensional in the unit sector: that is the
    target-sector statement (``S^z_tot = 0``; even total parity), and a degenerate
    boundary would make the run a batch of states.
    """
    if model == "hubbard":  # fZ2 / Z2: even and odd, split as evenly as chi allows
        mid = {0: chi - chi // 2, 1: chi // 2}
        return [{0: 1}] + [mid] * (n - 1) + [{0: 1}]
    spaces = []
    for i in range(n + 1):
        w = min(i, n - i)  # the S^z sectors bond i can reach, in units of 1/2
        qs = list(range(-w, w + 1, 2))
        deg = 1 if i in (0, n) else max(1, chi // len(qs))
        spaces.append({q: deg for q in qs})
    return spaces


# --------------------------------------------------------------------------------------
# Exact diagonalization, for the anchor run only.
# --------------------------------------------------------------------------------------

# The graded d=4 Hubbard site, (|0>, |ud>, |u>, |d>): even sector first, then odd, which
# is how a dense array over a GradedSpace is laid out. The intra-site convention is
# ``|ud> = c+_up c+_dn |0>`` with up the first mode, so ``c_up`` carries no intra-site
# sign and ``c_dn`` pays the Jordan-Wigner Z on the up mode. Same matrices as
# ``tests/network/test_hubbard.py``, which checks them against the two-mode kron.
C_UP = np.zeros((4, 4))
C_UP[0, 2] = 1.0
C_UP[3, 1] = 1.0
C_DN = np.zeros((4, 4))
C_DN[0, 3] = 1.0
C_DN[2, 1] = -1.0
N_UP, N_DN = C_UP.T @ C_UP, C_DN.T @ C_DN
P_SITE = np.diag([1.0, 1.0, -1.0, -1.0])  # the inter-site JW string


def ed_energy(model: str, n: int) -> float:
    """Dense ground-state energy. Only for the anchor run; every grid point is far past it."""
    if model == "heisenberg":
        sz, sp = np.diag([-0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 0.0]])
        eye = np.eye(2)

        def chain(op, i):
            full = np.array([[1.0]])
            for f in [eye] * i + [op] + [eye] * (n - i - 1):
                full = np.kron(full, f)
            return full

        h = np.zeros((2**n, 2**n))
        for i in range(n - 1):
            h += chain(sz, i) @ chain(sz, i + 1)
            h += 0.5 * chain(sp, i) @ chain(sp.T, i + 1)
            h += 0.5 * chain(sp.T, i) @ chain(sp, i + 1)
        return float(np.linalg.eigvalsh(h).min())

    def chain_c(i, local):  # P on every site to the left: the inter-site JW string
        full = np.array([[1.0]])
        for f in [P_SITE] * i + [local] + [np.eye(4)] * (n - i - 1):
            full = np.kron(full, f)
        return full

    h = np.zeros((4**n, 4**n))
    for i in range(n - 1):
        for local in (C_UP, C_DN):
            cd, c = chain_c(i, local.T), chain_c(i + 1, local)
            h += -HUBBARD_T * (cd @ c + c.T @ cd.T)
    for i in range(n):  # an even on-site operator: no string
        h += HUBBARD_U * np.kron(np.kron(np.eye(4**i), N_UP @ N_DN), np.eye(4 ** (n - i - 1)))
    # The MPS boundary legs pin the total parity to even, so the oracle is the even block.
    par = np.array([0, 0, 1, 1])
    idx, total = np.arange(4**n), np.zeros(4**n, dtype=int)
    for _ in range(n):
        total += par[idx % 4]
        idx //= 4
    even = np.flatnonzero(total % 2 == 0)
    return float(np.linalg.eigvalsh(h[np.ix_(even, even)]).min())


# --------------------------------------------------------------------------------------
# The two arms.
#
# Each returns ``(energy, per_sweep_walls, bond_dims)`` and each stops at the first sweep
# boundary past ``budget``, so a point that does not finish is recorded as such.
# --------------------------------------------------------------------------------------


def run_tenet(
    model: str, n: int, chi: int, sweeps: int, budget: float, sites: bool = False
) -> tuple:
    from tenet import GradedSpace
    from tenet.network import MPO, MPS, dmrg_, local_op
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
            "cpu": local_op(C_UP.T, phys=phys, charge=odd),
            "cu": local_op(C_UP, phys=phys, charge=odd),
            "cpd": local_op(C_DN.T, phys=phys, charge=odd),
            "cd": local_op(C_DN, phys=phys, charge=odd),
            "nud": local_op(N_UP @ N_DN, phys=phys, charge=even),
        }

    terms = [(a, [(ops[name], i) for name, i in prod]) for a, prod in model_terms(model, n)]
    h = MPO.from_terms(n, terms)
    if sites:
        # The same tensors, with the edge description dropped: ``Env.heff2`` routes on
        # ``self.h.edges is not None``, so this is the only difference between the two
        # tenet arms and it selects the site-tensor path (YASTN's ``Heff2`` order).
        h = h.materialize()
    bonds = [
        GradedSpace.new(sym, {sector(q): d for q, d in space.items()})
        for space in bond_charges(model, n, chi)
    ]
    psi = MPS.random(phys, bonds, seed=245)

    walls: list[float] = []
    mark = [time.perf_counter()]

    def tick(_out):
        now = time.perf_counter()
        walls.append(now - mark[0])
        mark[0] = now
        if sum(walls) > budget:
            raise TimeoutError

    try:
        out = dmrg_(
            psi,
            h,
            chi=chi,
            cutoff=0.0,
            ncv=3,
            energy_tol=0.0,
            schmidt_tol=0.0,
            max_sweeps=sweeps,
            callback=tick,
        )
        energy = out.energy
    except TimeoutError:
        energy = None
    dims = [psi[i].legs[0].space.dim for i in range(n)] + [psi[n - 1].legs[2].space.dim]
    return energy, walls, dims


def run_yastn(model: str, n: int, chi: int, sweeps: int, budget: float) -> tuple:
    import yastn
    import yastn.tn.mps as mps

    if model == "heisenberg":
        ops = yastn.operators.Spin12(sym="U1")
        local = {"sz": ops.sz(), "sp": ops.sp(), "sm": ops.sm()}
    else:
        ops = yastn.operators.SpinfulFermions(sym="Z2")
        local = {
            "cpu": ops.cp("u"),
            "cu": ops.c("u"),
            "cpd": ops.cp("d"),
            "cd": ops.c("d"),
            "nud": ops.n("u") @ ops.n("d"),
        }

    identity = mps.product_mpo(ops.I(), N=n)
    hterms = [
        mps.Hterm(a, [i for _, i in prod], [local[name] for name, _ in prod])
        for a, prod in model_terms(model, n)
    ]
    h = mps.generate_mpo(identity, hterms)

    # The initial state on ``bond_charges``'s spaces rather than on ``random_mps``'s
    # Gaussian charge profile: the point is that both arms start on the *same* bond
    # spaces. YASTN's MPS site legs are ``(s=-1 left, s=+1 physical, s=+1 right)`` and the
    # right leg of site i is the conjugate of the left leg of site i+1, which is the
    # convention ``_initialize.random_mps`` builds and this loop copies.
    spec = bond_charges(model, n, chi)
    phys_leg = identity[0].get_legs(axes=1)
    legs = [yastn.Leg(ops.config, s=-1, t=tuple(s), D=tuple(s.values())) for s in spec]
    psi = mps.Mps(n)
    for i in range(n):
        psi.A[i] = yastn.rand(
            ops.config, legs=[legs[i], phys_leg, legs[i + 1].conj()], dtype="float64"
        )

    walls: list[float] = []
    energy = None
    run = mps.dmrg_(
        psi,
        h,
        method="2site",
        energy_tol=None,
        Schmidt_tol=None,
        max_sweeps=sweeps,
        iterator_step=1,
        opts_eigs={"hermitian": True, "ncv": 3, "which": "SR"},
        opts_svd={"D_total": chi},
    )
    mark = time.perf_counter()
    for out in run:
        now = time.perf_counter()
        walls.append(now - mark)
        mark = now
        energy = float(out.energy)
        if sum(walls) > budget:
            energy = None
            break
    dims = list(psi.get_bond_dimensions())
    return energy, walls, dims


ARMS = {
    "tenet": run_tenet,
    "tenet-sites": functools.partial(run_tenet, sites=True),
    "yastn": run_yastn,
}


def rss_gib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("heisenberg", "hubbard"), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--chi", type=int, required=True)
    p.add_argument("--arm", choices=tuple(ARMS), required=True)
    p.add_argument("--sweeps", type=int, default=8)
    p.add_argument("--steady", type=int, default=3, help="sweeps averaged for the steady wall")
    p.add_argument("--budget", type=float, default=3600.0, help="seconds per point")
    p.add_argument("--ed", action="store_true", help="add the dense oracle; anchor runs only")
    p.add_argument("--out", type=pathlib.Path, help="JSONL to append one row to")
    a = p.parse_args()

    key = {"model": a.model, "n": a.n, "chi": a.chi, "arm": a.arm}
    if a.out and a.out.exists():
        for line in a.out.read_text().splitlines():
            if line.strip() and all(json.loads(line).get(k) == v for k, v in key.items()):
                print(f"skip (already in {a.out}): {key}", file=sys.stderr)
                return

    t0 = time.perf_counter()
    energy, walls, dims = ARMS[a.arm](a.model, a.n, a.chi, a.sweeps, a.budget)
    # The first sweep carries canonization and environment setup on both sides, so it is
    # the warm-up and never enters the steady mean.
    steady = walls[1:][-a.steady :]
    row = {
        **key,
        "status": "ok" if energy is not None else "budget",
        "sweeps_done": len(walls),
        "sweeps_asked": a.sweeps,
        "energy": energy,
        "energy_per_site": None if energy is None else energy / a.n,
        "walls": [round(w, 4) for w in walls],
        "steady_wall": round(sum(steady) / len(steady), 4) if steady else None,
        "total_wall": round(time.perf_counter() - t0, 2),
        "peak_rss_gib": round(rss_gib(), 3),
        "max_bond": max(dims),
        "bond_dims": dims,
        "ncv": 3,
        "cutoff": "off (chi only)",
    }
    if a.ed:
        row["ed_energy"] = ed_energy(a.model, a.n)
    line = json.dumps(row)
    if a.out:
        with a.out.open("a") as fh:  # appended the moment the point finishes
            fh.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
