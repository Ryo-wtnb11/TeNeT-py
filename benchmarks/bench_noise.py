"""Does the perturbative noise reach a lower energy than the wavefunction noise? (M61 Stage C)

The gate #232's Stage C carries, and the one #223 recorded before it was absorbed: a
measured case where the wavefunction noise plateaus and the perturbative noise does not,
**or the honest finding that it does not plateau**. Either way the table is what closes
the stage; the mechanism is adopted because it is block2's default decimation, not
because it is a performance patch.

The instrument, per model: one fixed schedule, five noise settings, the energy after
every sweep. Nothing else varies -- same seed, same starting MPS, same ``chi``, same
sweep count, same operator. The noise is on for the first ``--hot`` sweeps and off for
the rest, so every column ends on a cooled state and the last row is comparable.

Not a test, not part of the package, on no CI path. Run from the repo root::

    uv run python benchmarks/bench_noise.py                    # both models
    uv run python benchmarks/bench_noise.py --only heisenberg  # lattice only, no network
    uv run python benchmarks/bench_noise.py --only n2 --chi 32 --sweeps 12

The ab initio input is ``N2.CAS.6-31G`` (K=16, 32 spin-orbital ``fZ2`` sites), fetched
and sha256-verified by ``bench_qc_mpo.py`` -- the licence reasoning for fetching rather
than vendoring is in that file's docstring and is not repeated here.
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))

import bench_qc_mpo as qc  # noqa: E402
import heisenberg_walkthrough as lattice  # noqa: E402

from tenet import GradedSpace  # noqa: E402
from tenet.network import MPO, MPS, Env, sweep_  # noqa: E402
from tenet.symmetry import FZ2Sector, fZ2  # noqa: E402

#: ``(label, noise, noise_type)`` per column, in the order the table prints them.
SETTINGS = (
    ("none", 0.0, "wavefunction"),
    ("wfn 1e-4", 1e-4, "wavefunction"),
    ("wfn 1e-5", 1e-5, "wavefunction"),
    ("pert 1e-4", 1e-4, "perturbative"),
    ("pert 1e-5", 1e-5, "perturbative"),
)


def n2_model(chi):
    """``N2.CAS.6-31G`` at ``cutoff=None``: the operator, and a fresh random MPS."""
    norb, _, recs = qc.fetch("N2.CAS.6-31G")
    terms = qc.to_tenet_terms(qc.fold_terms(qc.spin_orbital_terms(recs, screen=qc.SCREEN))[0])
    n_sites = 2 * norb
    h = MPO.from_terms(n_sites, terms, cutoff=None)
    phys = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
    triv = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
    mid = GradedSpace.new(fZ2, {FZ2Sector(0): chi - chi // 2, FZ2Sector(1): chi // 2})
    bonds = [triv] + [mid] * (n_sites - 1) + [triv]
    return h, lambda: MPS.random(phys, bonds, seed=0)


def heisenberg_model(n_sites):
    """The U(1) Heisenberg chain, seeded from the Neel product state.

    The product seed is the point: its bonds are ``D=1``, so what a sweep can reach is
    decided by what the eigensolver and the mixer between them put on the bond. A random
    seed already carries every sector and measures the mixer much less.
    """
    h = lattice.mpo_from_terms(n_sites)
    pattern = [lattice.U1Sector(1), lattice.U1Sector(-1)] * (n_sites // 2)
    return h, lambda: MPS.product(lattice.PHYS, pattern)


def run(h, seed_state, *, chi, sweeps, hot, noise, noise_type):
    """One column: the per-sweep energies of a fixed schedule at one noise setting."""
    psi = seed_state()
    psi.canonize_(0)
    env = Env(psi, h).setup_(0)
    schmidt, energies = {}, []
    for it in range(sweeps):
        energy, _ = sweep_(
            psi,
            h,
            env,
            schmidt,
            chi=chi,
            cutoff=1e-10,
            noise=noise if it < hot else 0.0,
            noise_type=noise_type,
            seed=977 * it,
        )
        energies.append(energy)
    return energies


def table(name, h, seed_state, *, chi, sweeps, hot, settings=SETTINGS):
    print(f"\n== {name}: chi={chi}, {sweeps} sweeps, noise on for the first {hot}")
    columns, walls = {}, {}
    for label, noise, noise_type in settings:
        t0 = time.perf_counter()
        columns[label] = run(
            h, seed_state, chi=chi, sweeps=sweeps, hot=hot, noise=noise, noise_type=noise_type
        )
        walls[label] = time.perf_counter() - t0
        print(f"   ... {label} done in {walls[label]:.1f} s", flush=True)
    width = max(len(label) for label in columns) + 2
    print(f"\n{'sweep':>6}" + "".join(f"{label:>{width + 12}}" for label in columns))
    for it in range(sweeps):
        cells = "".join(f"{columns[label][it]:>{width + 12}.9f}" for label in columns)
        print(f"{it + 1:>6}" + cells)
    print(f"{'wall s':>6}" + "".join(f"{walls[label]:>{width + 12}.1f}" for label in columns))
    best = min(columns, key=lambda label: columns[label][-1])
    print(f"lowest final energy: {best} at {columns[best][-1]:.9f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=("n2", "heisenberg"), help="run just these")
    ap.add_argument("--chi", type=int, default=24)
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--hot", type=int, default=5, help="sweeps the noise is on for")
    ap.add_argument("--sites", type=int, default=20, help="Heisenberg chain length")
    ap.add_argument(
        "--settings", nargs="*", choices=[s[0] for s in SETTINGS],
        help="run just these columns; an hour-long table is worth resuming rather than redoing",
    )  # fmt: skip
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    wanted = a.only or ["n2", "heisenberg"]
    chosen = tuple(s for s in SETTINGS if a.settings is None or s[0] in a.settings)
    if "n2" in wanted:
        h, seed_state = n2_model(a.chi)
        table(
            "N2.CAS.6-31G (K=16)",
            h,
            seed_state,
            chi=a.chi,
            sweeps=a.sweeps,
            hot=a.hot,
            settings=chosen,
        )
    if "heisenberg" in wanted:
        h, seed_state = heisenberg_model(a.sites)
        table(
            f"U(1) Heisenberg N={a.sites}, Neel seed",
            h,
            seed_state,
            chi=a.chi,
            sweeps=a.sweeps,
            hot=a.hot,
            settings=chosen,
        )


if __name__ == "__main__":
    main()
