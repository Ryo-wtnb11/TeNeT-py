"""Imaginary-time TEBD, written out: the same ground state as ``dmrg.py``, from the same gates.

Run it standalone::

    uv run python examples/toy_codes/tebd.py

The simplest of the three algorithms here and the reason ``model.py`` states the
Hamiltonian twice: TEBD never sees an MPO. It exponentiates the two-site term into a gate,
applies the gate to a pair of neighbouring sites, and splits the pair back with a truncated
SVD. Imaginary time is what turns that into a ground-state method -- ``exp(-tau H)``
suppresses every excited state relative to the ground state -- so the state is normalized
after every split and the energy falls monotonically towards ``exact.ground_energy``.

The tensor operations it is built on: ``tenet.linalg.expm`` for the gate, ``tenet.einsum``
for the two contractions, ``tenet.linalg.svd_truncated`` for the split, and
``tenet.norm``. Nothing is imported from ``tenet.network``. The state and its measurements
come from ``mps.py``, the model from ``model.py``.

**The sweep is what keeps the truncation honest.** A truncated SVD discards the smallest
singular values of ``theta``, and that is the best rank-``chi`` approximation *of the
state* only when the rest of the chain is orthonormal on both sides of the cut -- which is
exactly the canonical form ``mps.canonicalize`` establishes and a left-to-right pass
preserves. So a step here is one left-to-right pass and one right-to-left pass over the
bonds, the same two-direction loop ``dmrg.sweep`` runs, with the gate carrying ``dt/2``
each way. Applying every gate forwards and then backwards is a symmetric product, so the
Trotter error is ``O(dt**3)`` per step without a second gate having to be built.

Simplification: **no stored Schmidt values, so no ``S^-1`` anywhere.** TeNPy's toy TEBD
keeps the singular values on every bond and divides them out to build ``theta``, which
makes a step ``O(1)`` in the chain length but puts an inverse of a truncated diagonal in
the hot path. Sweeping instead costs a pass over the chain and never inverts anything.

Simplification: **imaginary time only.** ``expm``'s ``alpha`` is where ``-1j * dt`` goes,
so real-time evolution is the same function with a complex step -- but it is a different
claim to check (there is no variational bound and the entropy grows until ``chi`` gives
out), and this file is here to reach a ground state that ``exact.py`` can confirm.
"""

import exact
import model
import mps
from mps import _as_site

import tenet

# The imaginary-time schedule: how long to evolve, and at what step. The first stage is
# where the convergence happens -- total imaginary time is what suppresses the excited
# states, and a coarse step buys the most of it per sweep -- and each stage after it
# removes the Trotter error the coarser one left, which is why the step shrinks and the
# stage lengthens. At N=12 this lands within 1e-11 of the exact energy.
SCHEDULE = ((0.5, 100), (0.05, 200), (0.01, 300))


def gate(h, dt: float):
    """``exp(-dt * h)`` for one bond, legs ``(P OUT, Q OUT, p IN, q IN)`` unchanged.

    ``tenet.linalg.expm`` lowers the two-site operator to a square map on the partition
    ``((0, 1), (2, 3))`` -- the pair's outputs against its inputs -- exponentiates one dense
    matrix per coupled sector, and hands back a tensor with ``h``'s structure exactly. The
    grading is what makes that cheap and what makes the gate ``S^z_tot``-conserving: a
    sector that cannot appear in ``h`` cannot appear in its exponential either.

    ``alpha`` is where the step goes, and ``-dt`` is the imaginary-time choice;
    ``-1j * dt`` would be the real-time gate.

    On the NumPy backend this needs SciPy, which is what ``autoray``'s ``linalg.expm``
    calls; the library declares it a dev dependency rather than a runtime one.
    """
    return tenet.linalg.expm(h, ((0, 1), (2, 3)), alpha=-dt)


def update_bond(psi, gates, schmidt, n: int, direction: str, *, chi: int, cutoff: float) -> float:
    """Apply the bond-``n`` gate to sites ``n, n+1`` and split them back. In place.

    Merge, apply, split: ``theta`` is the two-site tensor on
    ``(left bond OUT, p OUT, q OUT, right bond IN)`` -- the same object ``dmrg.sweep``
    hands to Lanczos -- the gate closes its two physical legs, and ``svd_truncated``
    re-decides the bond :class:`~tenet.GradedSpace` from the singular values it finds.

    ``s`` is normalized rather than the whole state, which is the same thing: ``u`` and
    ``vh`` are isometries, so ``norm(theta) = norm(s)``. The singular values go to whichever
    site the sweep is moving *away* from, which is what leaves the chain canonical behind
    the pass. Returns the bond's discarded weight, ``1 - norm(s)**2`` on the normalized
    two-site tensor, by Pythagoras, and writes the bond's Schmidt spectrum into ``schmidt``
    -- the cut is canonical on both sides at exactly this moment, which is the only moment
    those numbers mean what their name says.
    """
    theta = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
    theta = tenet.einsum("PQpq,apqr->aPQr", gates[n], theta)
    u, s, vh = tenet.linalg.svd_truncated(theta, ((0, 1), (2, 3)), max_bond=chi, cutoff=cutoff)
    norm_s = tenet.norm(s)
    discarded = 1.0 - float(norm_s / tenet.norm(theta)) ** 2
    s, vh = s / norm_s, _as_site(vh)
    schmidt[n] = mps.spectrum(s)
    if direction == "right":
        psi[n], psi[n + 1] = u, tenet.einsum("xy,yqr->xqr", s, vh)
    else:
        psi[n], psi[n + 1] = tenet.einsum("apx,xy->apy", u, s), vh
    return discarded


def step(psi, gates, schmidt, *, chi: int, cutoff: float) -> float:
    """One symmetric Trotter step: every bond left to right, then every bond right to left.

    ``dmrg.sweep``'s two-direction loop with the eigensolver replaced by a gate. Returns the
    worst discarded weight seen, which is the number that says whether ``chi`` was enough.
    """
    max_dw = 0.0
    for direction in ("right", "left"):
        bonds = range(len(psi) - 1) if direction == "right" else range(len(psi) - 2, -1, -1)
        for n in bonds:
            max_dw = max(
                max_dw, update_bond(psi, gates, schmidt, n, direction, chi=chi, cutoff=cutoff)
            )
    return max_dw


def energy(psi, bonds) -> float:
    """``<psi|H|psi>`` as the sum of the model's two-site terms, through ``mps.expectation``."""
    return sum(mps.expectation(psi, h, n) for n, h in enumerate(bonds))


def tebd(n_sites: int = 12, chi: int = 32, *, cutoff: float = 1e-14, schedule=SCHEDULE):
    """Evolve the Neel state in imaginary time and return ``(psi, history)``.

    ``history`` is one ``(dt, steps, energy, max_discarded_weight)`` tuple per schedule
    stage -- the same shape ``dmrg``'s per-sweep history has, and enough to see both
    convergence and whether the bond ever ran out. ``schmidt`` carries the last Schmidt
    spectrum written at each bond, which is what :func:`mps.entropy` reads.
    """
    psi = mps.canonicalize(mps.product_mps(n_sites))
    bonds = model.h_bonds(n_sites)
    schmidt: dict[int, list[float]] = {}
    history = []
    for dt, steps in schedule:
        gates = [gate(h, dt / 2) for h in bonds]
        max_dw = 0.0
        for _ in range(steps):
            max_dw = max(max_dw, step(psi, gates, schmidt, chi=chi, cutoff=cutoff))
        history.append((dt, steps, energy(psi, bonds), max_dw))
    return psi, history, schmidt


def main(n_sites: int = 12, chi: int = 32):
    """N=12 at chi=32 against ``exact.py``, plus the half-chain entanglement entropy.

    The energy is variational from above at every stage: imaginary time can only lower it
    and the truncation can only raise it, so a stage that came out *below* the exact value
    would be a bug and not a lucky run.
    """
    psi, history, schmidt = tebd(n_sites, chi)
    reference = exact.ground_energy(n_sites)
    for dt, steps, e, dw in history:
        print(f"  dt={dt:<7g} steps={steps:3d}  E={e:+.12f}  dw={dw:.3e}")
    print(
        f"N={n_sites} chi={chi}  E={history[-1][2]:+.12f}  exact={reference:+.12f}  "
        f"S(N/2)={mps.entropy(schmidt[n_sites // 2 - 1]):.6f}"
    )
    return psi, history, reference


if __name__ == "__main__":
    main()
