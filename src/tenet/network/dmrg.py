"""The driver: a Krylov step, a two-site sweep, and the loop that repeats it.

Promoted from ``examples/dmrg.py`` (#110) with no arithmetic change: ``lanczos``
:393-437, ``sweep_`` :443-484, ``_schmidt_change`` :487-503, ``DMRG_out`` :506-520 and
``dmrg_`` :524-557. Function-shaped drivers, YASTN's decomposition (``_dmrg.py``:42-128),
not TenPy's ``Sweep``/``EffectiveH`` hierarchy -- M11a ships **one** sweep, and a base
class with one subclass is the interface-with-one-implementation the repo's own rules
forbid.

:func:`lanczos` lives here rather than in ``tenet.linalg`` for a sharp reason:
``tenet.linalg`` is fixed-structure decompositions of a *tensor*, all traceable, and this
takes a ``matvec`` **callable** and breaks out of its loop on a float comparison. It is
not in a ``_krylov.py`` either: it is 45 lines whose only caller is the sweep below, and
the day ``expmv`` lands for TDVP, moving both next to each other is a two-file change and
*then* it is earned.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

import tenet
from tenet import SymmetricTensor
from tenet.network.common import spectrum
from tenet.network.env import Env
from tenet.network.mps import MPO, MPS

__all__ = ["DMRG_out", "Sweep", "dmrg_", "lanczos", "sweep_"]


class Sweep(NamedTuple):
    """One schedule entry: what a single sweep does.

    ``chi`` and ``cutoff`` are handed to ``tenet.linalg.svd_truncated`` at every bond of
    the sweep, and ``noise`` is the relative strength of the wavefunction perturbation
    :func:`sweep_` adds before each split. The defaults equal :func:`dmrg_`'s flat
    defaults, so ``Sweep()`` is today's sweep and ``schedule=[Sweep()]`` is the flat run.

    One record per sweep rather than parallel per-knob lists (block2's ``bond_dims`` /
    ``noises`` / ``thrds``), so a wrong-length list is impossible to write. The loop
    tolerances (``energy_tol``, ``schmidt_tol``, ``max_sweeps``, ``ncv``) are properties
    of :func:`dmrg_`'s loop, not of a sweep, and stay flat kwargs there.
    """

    chi: int = 64
    cutoff: float = 1e-14
    noise: float = 0.0


def lanczos(
    matvec: Callable[[SymmetricTensor], SymmetricTensor],
    v: SymmetricTensor,
    *,
    ncv: int = 3,
    tol: float = 1e-13,
) -> tuple[float, SymmetricTensor]:
    """Ground eigenpair ``(value, vector)`` of a Hermitian ``matvec`` over SymmetricTensors.

    YASTN's three-term recurrence (``yastn/tensor/_krylov.py``:34-42) and its happy
    breakdown (``H[(j+1,j)] < tol`` -> stop and drop the row, :39-43), then ``eigh`` of
    the ``(m, m)`` tridiagonal and one recombination (``yastn/krylov/_krylov.py``:226-239,
    a single iteration with no restart). ``hermitian=True, ncv=3, which='SR'`` are YASTN's
    own DMRG defaults (``_dmrg.py``:151-152) and are not knobs this layer tunes.

    The only tensor operations are ``tenet.add``/``subtract``, scalar multiply/divide,
    ``tenet.norm`` and :func:`tenet.inner` -- a Krylov solver needs a vector
    space and nothing else, and a ``SymmetricTensor`` is one.

    This is an inner solver inside an outer sweep, not a standalone eigensolver: the
    recurrence is **not reorthogonalized**, so ``ncv`` is meant to stay small.
    """
    # Simplification: no reorthogonalization, and neither has YASTN. At ``ncv=3`` the
    # recurrence has not had time to lose orthogonality, and the vector is reseeded from the
    # current MPS at every bond. Ceiling: raise ``ncv`` past ~10 and full reorthogonalization
    # against the stored ``V`` becomes the two-line addition.
    # Simplification: numpy ``eigh`` on the ``(3, 3)`` tridiagonal, not
    # ``tenet.linalg.eigh``. The projected matrix has no symmetry structure -- 9 floats.
    vecs = [v / tenet.norm(v)]
    alphas: list[float] = []
    betas: list[float] = []
    for j in range(ncv):
        w = matvec(vecs[j])
        alphas.append(float(tenet.inner(vecs[j], w)))
        w = tenet.subtract(w, vecs[j] * alphas[j])
        if j:
            w = tenet.subtract(w, vecs[j - 1] * betas[j - 1])
        beta = float(tenet.norm(w))
        if j + 1 == ncv or beta < tol:  # happy breakdown: drop the row, keep the space
            break
        betas.append(beta)
        vecs.append(w / beta)
    tri = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    values, states = np.linalg.eigh(tri)
    ground = states[:, 0]
    out = vecs[0] * float(ground[0])
    for k in range(1, len(vecs)):
        out = tenet.add(out, vecs[k] * float(ground[k]))
    return float(values[0]), out / tenet.norm(out)


def sweep_(
    psi: MPS,
    h: MPO,
    env: Env,
    schmidt: dict[int, list[float]],
    *,
    chi: int,
    cutoff: float,
    ncv: int = 3,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[float, float]:
    """One left-to-right then right-to-left two-site sweep. ``psi`` and ``env`` mutate.

    YASTN's ``_dmrg_sweep_2site_`` (``_dmrg.py``:222-249) and its
    ``(('last', 0), ('first', 1))`` two-direction loop, five steps per bond: merge,
    ``eigs``, split, ``clear_site_``, ``update_env_``.

    ``svd_truncated`` decides the bond :class:`~tenet.GradedSpace` here, every bond and
    every sweep, and the discarded weight is Pythagoras exactly as its docstring
    prescribes: ``U S Vh`` is isometric on both sides, so ``norm(U S Vh) = norm(S)`` and
    the dropped fraction of the (unit-norm) two-site tensor is ``1 - norm(S)**2``.

    ``vh`` comes back on the *map*'s partition and is stored straight into ``psi``: the
    :meth:`MPS.__setitem__` write barrier is what puts it back on ``(l, p | r)``, which is
    why no caller in this package ever spells a ``repartition``.

    Returns ``(energy, max_discarded_weight)``; ``schmidt`` is updated in place with the
    per-bond Schmidt spectra, which is the second convergence criterion's input. The
    discarded weight here is the **maximum** over bonds, because it feeds a per-sweep
    convergence report where the worst bond is the diagnostic;
    :meth:`~tenet.network.MPS.compress_` returns the **total** instead, because its caller
    is asking how much of the state was thrown away. Two conventions, two names.

    With ``noise > 0`` a random symmetric tensor over the two-site tensor's own legs is
    added after the eigensolver and before the split, at relative strength ``noise``
    (block2's ``NoiseTypes::Wavefunction`` scaling, ``operator_functions.hpp``:777-815, so
    ``noise`` is dimensionless and block2's 1e-4..1e-5 range transfers), and the two-site
    tensor is renormalized so the Pythagoras discarded weight stays a fraction of a
    unit-norm tensor. The perturbation fills every **structurally allowed** coupled sector
    of the ``(l, p | q, r)`` map -- including the ones the eigensolver left numerically
    empty and which ``svd_truncated`` therefore omits from the bond, which is the local
    minimum a symmetric DMRG falls into: a sector that is zero stays zero forever, because
    nothing else in the sweep can create it. It cannot reach outside ``bond_l (x) phys``
    -- no wavefunction noise can, and neither can two-site DMRG itself. Noise is not
    variational: a noisy sweep's energy may sit above the same sweep at ``noise=0.0``.
    ``seed`` makes the draw at bond ``n`` reproducible as ``seed + n``; with ``noise=0.0``
    no random number is drawn and the sweep is bit-identical to a sweep without the
    keyword.
    """
    # Simplification: wavefunction noise only, block2's cheapest (NoiseTypes::Wavefunction
    # = 1) and its own docs call the cheap end "not very effective". Every stronger mixer
    # (DensityMatrix, ReducedPerturbativeCollected, White's alpha) perturbs a density
    # matrix, and this sweep has none: it splits with svd_truncated, not an eigh of
    # rho = tr aa aa^dag. Upgrade path: perturbative noise is a different split and its
    # own issue.
    n_sites = len(psi)
    energy, max_dw = 0.0, 0.0
    for direction in ("right", "left"):
        bonds = range(n_sites - 1) if direction == "right" else range(n_sites - 2, -1, -1)
        for n in bonds:
            aa = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
            energy, aa = lanczos(lambda v, n=n: env.heff2(n, v), aa, ncv=ncv)
            if noise:
                r = SymmetricTensor.random(aa.legs, seed=seed + n)
                aa = tenet.add(aa, r * float(noise * tenet.norm(aa) / tenet.norm(r)))
                aa = aa / tenet.norm(aa)
            u, s, vh = tenet.linalg.svd_truncated(aa, ((0, 1), (2, 3)), max_bond=chi, cutoff=cutoff)
            norm_s = tenet.norm(s)
            max_dw = max(max_dw, 1.0 - float(norm_s / tenet.norm(aa)) ** 2)
            s = s / norm_s  # the two-site tensor is normalized; keep the MPS so
            psi[n + 1] = vh  # through the write barrier, so ``vh`` is on (l, p | r) now
            if direction == "right":
                psi[n] = u
                psi[n + 1] = tenet.einsum("xy,yqr->xqr", s, psi[n + 1])
            else:
                psi[n] = tenet.einsum("apx,xy->apy", u, s)
            psi.center = n if direction == "left" else n + 1
            schmidt[n] = spectrum(s)
            env.clear_(n, n + 1)
            if direction == "right":
                env.update_(n, to="last")
            else:
                env.update_(n + 1, to="first")
    return energy, max_dw


def _schmidt_change(old: dict[int, list[float]], new: dict[int, list[float]]) -> float:
    """``max_k ||S_k - S_k^old||`` over bonds, zero-padded -- YASTN ``_dmrg.py``:154-195.

    A bond present in only one of the two, or a spectrum whose length changed because
    ``svd_truncated`` moved the bond space, counts as a large change rather than an error,
    which is what it is. Infinite before the first sweep has any history.
    """
    if not old:
        return float("inf")
    worst = 0.0
    for n, current in new.items():
        previous = old.get(n, [])
        m = max(len(previous), len(current))
        a = previous + [0.0] * (m - len(previous))
        b = current + [0.0] * (m - len(current))
        worst = max(worst, sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5)
    return worst


class DMRG_out(NamedTuple):
    """YASTN's ``DMRG_out`` (``_dmrg.py``:33-39), plus the two things a test needs.

    ``history`` is one ``(energy, denergy, dSchmidt, discarded)`` tuple per sweep, and it
    says everything YASTN's ``iterator=True`` generator protocol says to a test without
    the protocol. ``schedule`` is the **realized** schedule, one :class:`Sweep` per sweep
    run -- ``zip(out.schedule, out.history)`` is exact, and ``out.schedule`` alone answers
    whether a run actually reached its final ``chi`` or converged earlier. ``psi`` is the
    converged :class:`MPS`.
    """

    sweeps: int
    energy: float
    denergy: float
    max_dSchmidt: float
    max_discarded_weight: float
    history: list[tuple[float, float, float, float]]
    schedule: list[Sweep]
    psi: MPS


def dmrg_(
    psi: MPS,
    h: MPO,
    *,
    schedule: Sequence[Sweep] | None = None,
    chi: int | None = None,
    cutoff: float | None = None,
    energy_tol: float = 1e-12,
    schmidt_tol: float = 1e-8,
    max_sweeps: int = 40,
    ncv: int = 3,
    seed: int = 0,
    callback: Callable[[DMRG_out], None] | None = None,
) -> DMRG_out:
    """Sweep ``psi`` to the ground state of ``h`` in place and return a :class:`DMRG_out`.

    ``psi`` is right-canonicalized first (:meth:`MPS.canonize_`), so a freshly seeded
    random MPS is the expected input and a caller keeping the returned ``out.psi`` and the
    one it passed in holds the same object.

    Two spellings of what the sweeps do, exclusive: the flat ``chi`` / ``cutoff`` kwargs
    (defaults 64 and 1e-14), or ``schedule``, a non-empty sequence of :class:`Sweep`
    entries whose **last entry repeats** until convergence or ``max_sweeps`` -- so
    ``schedule=[Sweep(chi=64)]`` is exactly the flat run, and
    ``schedule=[Sweep(32, noise=1e-4)] * 4 + [Sweep(64)]`` is a ramp that cools down at
    ``chi=64`` for as long as ``max_sweeps`` allows. Passing ``schedule`` together with
    ``chi`` or ``cutoff`` raises, because silently letting one win is how a run reports a
    ``chi`` it did not use. ``seed`` feeds :func:`sweep_`'s noise draw, distinctly per
    sweep; a schedule with ``noise=0.0`` everywhere draws nothing.

    Convergence uses **both** of YASTN's criteria (``_dmrg.py``:180-195): the energy
    change ``|E_old - E| < energy_tol`` *and* the worst-cut Schmidt change
    ``max_k ||S_k - S_k^old|| < schmidt_tol``, and the loop stops only when both are met in
    one sweep. The Schmidt criterion is the sensitive one, and it is what catches a run
    whose energy has plateaued on a wrong bond structure. **Convergence is never declared
    on a sweep that is still noisy or still inside the schedule**: the loop exits only
    when the sweep just run used the schedule's last entry and that entry's ``noise`` is
    ``0.0`` (block2's guard, ``sweep_algorithm.hpp``:3103-3105, and its docstring's "and
    the ``noise`` for the current sweep is zero") -- an energy that stopped moving under
    noise at a ramp's intermediate ``chi`` has converged to the wrong thing, and reporting
    it as converged is worse than sweeping on.

    ``callback``, if given, is invoked once per sweep with the :class:`DMRG_out` built for
    that sweep, after ``history`` is appended, so it sees the sweep that just finished.
    Its return value is ignored: there is no early-stop protocol.
    """
    if schedule is not None:
        if chi is not None or cutoff is not None:
            raise ValueError(
                "pass either schedule=[Sweep(chi=..., cutoff=...), ...] or the flat "
                "chi= / cutoff= kwargs, not both: schedule=[Sweep(chi=64)] is the "
                "flat chi=64 run, so letting one spelling win would misreport the other"
            )
        plan = list(schedule)
        if not plan:
            raise ValueError("schedule is empty; a run needs at least one Sweep entry")
    else:
        plan = [Sweep(64 if chi is None else chi, 1e-14 if cutoff is None else cutoff)]
    psi.canonize_(0)
    env = Env(psi, h).setup_(0)
    schmidt: dict[int, list[float]] = {}
    energy: float = float("inf")
    history: list[tuple[float, float, float, float]] = []
    realized: list[Sweep] = []
    out: Any = None
    for it in range(1, max_sweeps + 1):
        entry = plan[it - 1] if it <= len(plan) else plan[-1]
        old_energy, old_schmidt = energy, dict(schmidt)
        energy, max_dw = sweep_(
            psi,
            h,
            env,
            schmidt,
            chi=entry.chi,
            cutoff=entry.cutoff,
            ncv=ncv,
            noise=entry.noise,
            seed=seed + 977 * it,
        )
        denergy = abs(old_energy - energy)
        d_schmidt = _schmidt_change(old_schmidt, schmidt)
        history.append((energy, denergy, d_schmidt, max_dw))
        realized.append(entry)
        out = DMRG_out(it, energy, denergy, d_schmidt, max_dw, history, realized, psi)
        if callback:
            callback(out)
        if (
            denergy < energy_tol
            and d_schmidt < schmidt_tol
            and it >= len(plan)  # the sweep just run used the schedule's last entry
            and entry.noise == 0.0
        ):
            break
    return out
