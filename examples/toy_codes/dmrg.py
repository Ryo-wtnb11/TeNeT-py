"""Finite-chain two-site DMRG, written out: the U(1) Heisenberg chain against exact diagonalization.

Run it standalone::

    uv run python examples/toy_codes/dmrg.py

The algorithm is the file: the directed-bond environment cache and its invalidation, a
Lanczos step over the two-site tensor, and the sweep whose truncation re-decides each bond
space. The state it sweeps comes from ``mps.py`` and the Hamiltonian from ``mpo.py``, each
of which carries its own leg convention. Nothing is imported from ``tenet.network``, which
ships all of it; ``examples/heisenberg_walkthrough.py`` is the same physics through the
library.

The tensor operations it is built on: ``SymmetricTensor.from_blocks`` for the boundary
environments, ``tenet.einsum`` for every contraction, ``tenet.repartition`` for the leg
bends, ``tenet.linalg.svd_truncated`` for the bond-deciding factorization, and
``tenet.add``, ``tenet.subtract``, ``tenet.norm`` and ``tenet.inner`` for the Krylov
vector space.

**Leg convention** for the piece this file owns: environment ``F[(n, n+1)]`` is
``(ket IN, mpo OUT, bra OUT)``, built from sites ``<= n``; environment ``F[(n, n-1)]`` is
``(ket OUT, mpo IN, bra IN)``, built from sites ``>= n``.

**Operand order is part of the arithmetic, not a style choice.** Every ``tenet.einsum``
below is a *composition*: operand 1 supplies the ``IN`` end of every shared wire. Meeting
``IN`` against ``OUT`` is not enough -- that condition is symmetric, while the cap
direction, and hence the Koszul sign a fermionic provider pays, depends on which operand
supplies which end. The wires that genuinely turn around are bent *explicitly* by
:func:`_composed`. This chain is U(1), where every such sign is ``+1``; the orders are
still written correctly, because a reader copying this file for a fermionic model would
otherwise copy a silent sign error.

There is no ``jit`` and no ``grad`` here, and that is a decision: DMRG's control flow is
data-dependent at every level -- the truncation re-decides a bond space each sweep,
:func:`lanczos` tests a norm against a tolerance, :func:`dmrg` exits on a measured energy
change -- so this module runs on the eager NumPy backend and makes no differentiability
claim. ``ctmrg.py`` is the half of the library that lives under a trace.

Simplification: **two-site DMRG only.** It is what makes ``svd_truncated`` the
bond-deciding step, and it grows a bond by a factor of ``d`` per site with no extra
concept. Single-site DMRG cannot grow a bond at all, so it is only honest with subspace
expansion (Hubig-McCulloch-Schollwoeck-Wall, PRB 91, 155115 (2015)), which wants
``tenet.linalg.left_null``, a mixing factor and a second contraction chain.

Simplification: **hand-written pairwise contraction orders, not ``optimize=`` on a
five-operand einsum.** ``opt_einsum`` costs a graded network from *physical* leg sizes,
and a U(1) MPS bond whose sectors are unevenly filled is exactly where that estimate is
wrong. The orders here are YASTN's own (``yastn/tn/mps/_env.py``:496-518), documented as
``O(D^3 M d + D^2 M^2 d^2)`` per matvec -- optimal for *one* matvec, which is all a
Krylov step ever wants.
"""

from typing import NamedTuple

import numpy as np

# ``mps.py`` holds the state and ``mpo.py`` the Hamiltonian; this file holds the
# algorithm. Their names are re-exported here, so ``import dmrg`` still reaches the whole
# example. The names marked ``noqa`` below are re-exports only: this file never calls
# them, and dropping them would break every caller of the single-file version.
from mpo import MPO_BOND, mpo, mpo_blocks  # noqa: F401
from mps import (
    BOUNDARY,
    PHYS,  # noqa: F401
    _as_site,
    bond_spaces,  # noqa: F401
    canonicalize,
    random_mps,
    spectrum,
)

import tenet
from tenet import IN, OUT, Leg, SymmetricTensor, TensorStructure

# The thermodynamic limit, 1/4 - ln 2 (Bethe 1931; Hulthen 1938), for main()'s report.
E_INF = -0.4431471805599453


def _composed(equation: str, a: SymmetricTensor, b: SymmetricTensor, bend: str = ""):
    """A two-operand ``tenet.einsum`` with the wires named in ``bend`` bent first.

    Operand 1 must supply the ``IN`` end of every shared wire (the module docstring's
    composition rule). A wire that turns around in the intended planar diagram -- one
    running through an environment's cap -- cannot meet that rule as drawn, and letting
    ``einsum`` bend it implicitly would leave the cap direction to operand order. So the
    bend is spelled: both ends of each named wire move to the other side with
    ``tenet.repartition``, which pays the categorical bend coefficient by construction,
    and the einsum that follows is a plain composition again. ``bend=""`` is a straight
    composition and could as well be ``tenet.einsum``.
    """
    if bend:
        lhs, out = equation.split("->")
        ta, tb = lhs.split(",")

        def bent(t: SymmetricTensor, term: str) -> tuple[SymmetricTensor, str]:
            flip = set(bend)
            outs = tuple(i for i, c in enumerate(term) if (t.legs[i].side is OUT) != (c in flip))
            ins = tuple(i for i in range(len(term)) if i not in outs)
            return tenet.repartition(t, outs, ins), "".join(term[i] for i in (*outs, *ins))

        a, ta = bent(a, ta)
        b, tb = bent(b, tb)
        equation = f"{ta},{tb}->{out}"
    return tenet.einsum(equation, a, b)


# --- the environment: YASTN's directed-bond dict ------------------------------------


def _ones(legs) -> SymmetricTensor:
    """A tensor with every structurally allowed entry equal to 1.

    ``TensorStructure`` already knows which blocks the grading allows and how big each one
    is, so the seed is "fill the blocks that exist": there is no dense array here to build
    and project.
    """
    structure = TensorStructure(tuple(legs))
    blocks = {key: np.ones(structure.block_shape(key)) for key in structure.block_order}
    return SymmetricTensor.from_blocks(legs, blocks)


def boundary_envs(n_sites: int) -> dict[tuple[int, int], SymmetricTensor]:
    """``{(-1, 0): left, (n, n-1): right}``, both the trivial 1x1x1 tensor.

    The environment is a plain ``dict`` keyed by *directed* bond, exactly YASTN's ``Env``
    (``yastn/tn/mps/_env.py``:104-125 ``setup_``): ``F[(n, n+1)]`` is built from sites
    ``<= n`` and ``F[(n, n-1)]`` from sites ``>= n``. A list-of-left / list-of-right would
    hide the invalidation discipline, which is the entire correctness content of an
    environment cache -- a stale ``F[(n, n+1)]`` after site ``n`` changed gives an energy
    that is *plausible and wrong*, the worst failure mode a DMRG has.
    """
    return {
        (-1, 0): _ones((Leg(BOUNDARY, IN), Leg(BOUNDARY, OUT), Leg(BOUNDARY, OUT))),
        (n_sites, n_sites - 1): _ones((Leg(BOUNDARY, OUT), Leg(BOUNDARY, IN), Leg(BOUNDARY, IN))),
    }


def update_env(envs, psi, w, n: int, to: str) -> None:
    """Write one directed-bond entry from its neighbour -- YASTN ``_env.py``:152-168.

    ``to='last'`` writes ``F[(n, n+1)]`` from ``F[(n-1, n)]``; ``to='first'`` writes
    ``F[(n, n-1)]`` from ``F[(n+1, n)]``. Three pairwise contractions each: environment
    first, then the ket, then the MPO, then the bra. The ``'first'`` direction runs
    against the arrows -- the physical wire ``p`` and the bra wire ``P`` each turn around
    in the cap -- so those two are :func:`_composed` with the bend named.
    """
    a, bra = psi[n], tenet.adjoint(psi[n])
    if to == "last":
        t = tenet.einsum("axB,apr->xBpr", envs[n - 1, n], a)
        t = tenet.einsum("xPpm,xBpr->BrPm", w[n], t)
        envs[n, n + 1] = tenet.einsum("BPs,BrPm->rms", bra, t)
    else:
        t = tenet.einsum("apr,rys->apys", a, envs[n + 1, n])
        t = _composed("apys,xPpy->axPs", t, w[n], bend="p")
        envs[n, n - 1] = _composed("axPs,BPs->axB", t, bra, bend="P")


def invalidate(envs, *sites: int) -> None:
    """Pop every entry a changed site invalidates -- YASTN ``clear_site_``, :127-134.

    Both directed bonds touching each site go, and they go *before* the replacement is
    written, so a missed update is a ``KeyError`` rather than a wrong number.
    """
    for n in sites:
        envs.pop((n, n - 1), None)
        envs.pop((n, n + 1), None)


def setup_envs(psi, w) -> dict[tuple[int, int], SymmetricTensor]:
    """Every right-directed environment, for a right-canonical ``psi`` -- ``setup_(to='first')``."""
    envs = boundary_envs(len(psi))
    for n in range(len(psi) - 1, 0, -1):
        update_env(envs, psi, w, n, "first")
    return envs


# --- the local problem -------------------------------------------------------------


def heff2(envs, w1, w2, n: int, aa: SymmetricTensor) -> SymmetricTensor:
    """``H_eff`` on the two-site tensor at bond ``(n, n+1)``. Four pairwise contractions.

    Right environment, then ``W2``, then ``W1``, then the left environment: YASTN's
    ``Env_mps_mpo_mps.Heff2`` order (``_env.py``:496-518) with ``precompute=False``,
    which ``_dmrg.py``:102-108 documents as ``O(D^3 M d + D^2 M^2 d^2)`` -- optimal for
    a single matvec, which is all a Krylov step ever wants.

    In and out on ``(left bond OUT, p OUT, q OUT, right bond IN)``: the *bra* legs of the
    two environments become the output's bonds while the *ket* legs close against the
    input's, which is why the result has ``aa``'s structure exactly and
    :func:`lanczos` can add the two. Three of the four contractions run through a cap and
    name their bent wire; the first, which only rides the right environment, does not.
    """
    t = tenet.einsum("apqr,rys->apqys", aa, envs[n + 2, n + 1])
    t = _composed("apqys,mQqy->apQms", t, w2, bend="q")
    t = _composed("apQms,xPpm->aPQxs", t, w1, bend="p")
    return _composed("aPQxs,axB->BPQs", t, envs[n - 1, n], bend="a")


def lanczos(matvec, v: SymmetricTensor, ncv: int = 3, tol: float = 1e-13):
    """Ground eigenpair ``(value, vector)`` of a Hermitian ``matvec`` over SymmetricTensors.

    YASTN's three-term recurrence (``yastn/tensor/_krylov.py``:34-42) and its happy
    breakdown (``H[(j+1,j)] < tol`` -> stop and drop the row, :39-43), then ``eigh`` of
    the ``(m, m)`` tridiagonal and one recombination
    (``yastn/krylov/_krylov.py``:226-239, a single iteration with no restart at :217-219).
    ``hermitian=True, ncv=3, which='SR'`` are YASTN's own DMRG defaults
    (``_dmrg.py``:151-152) and are not knobs this example tunes.

    The only tensor operations are ``tenet.add``/``subtract``, scalar multiply/divide,
    ``tenet.norm`` and ``tenet.inner`` -- a Krylov solver needs a vector space and nothing
    else, and a ``SymmetricTensor`` is one.

    Simplification: **no reorthogonalization**, and neither has YASTN. At ``ncv=3`` the
    recurrence has not had time to lose orthogonality, and the vector is reseeded from the
    current MPS at every bond -- this is an inner solver inside an outer sweep, not a
    standalone eigensolver. Ceiling: raise ``ncv`` past ~10 and full reorthogonalization
    against the stored ``vecs`` becomes the two-line addition.

    Simplification: numpy ``eigh`` on the ``(3, 3)`` tridiagonal, not ``tenet.linalg.eigh``. The
    projected matrix has no symmetry structure to respect -- it is 9 floats.
    """
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


# --- the sweep ---------------------------------------------------------------------


def sweep(psi, w, envs, schmidt, *, chi: int, cutoff: float, ncv: int = 3):
    """One left-to-right then right-to-left two-site sweep. ``psi`` is updated in place.

    YASTN's ``_dmrg_sweep_2site_`` (``_dmrg.py``:222-249) and its
    ``(('last', 0), ('first', 1))`` two-direction loop, five steps per bond:
    merge, solve, split, invalidate, update the environment.

    ``svd_truncated`` decides the bond :class:`~tenet.GradedSpace` here, every bond and
    every sweep, and the discarded weight is Pythagoras exactly as its docstring
    prescribes: ``U S Vh`` is isometric on both sides, so ``norm(U S Vh) = norm(S)`` and
    the dropped fraction of the (unit-norm) two-site tensor is ``1 - norm(S)**2``.

    Returns ``(energy, max_discarded_weight)``; ``schmidt`` is updated in place with the
    per-bond Schmidt spectra, which is the second convergence criterion's input.
    """
    n_sites = len(psi)
    energy, max_dw = 0.0, 0.0
    for direction in ("right", "left"):
        bonds = range(n_sites - 1) if direction == "right" else range(n_sites - 2, -1, -1)
        for n in bonds:
            aa = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
            w1, w2 = w[n], w[n + 1]
            energy, aa = lanczos(
                lambda v, w1=w1, w2=w2, n=n: heff2(envs, w1, w2, n, v), aa, ncv=ncv
            )
            u, s, vh = tenet.linalg.svd_truncated(aa, ((0, 1), (2, 3)), max_bond=chi, cutoff=cutoff)
            vh = _as_site(vh)
            norm_s = tenet.norm(s)
            max_dw = max(max_dw, 1.0 - float(norm_s / tenet.norm(aa)) ** 2)
            s = s / norm_s  # the two-site tensor is normalized; keep the MPS so
            if direction == "right":
                psi[n], psi[n + 1] = u, tenet.einsum("xy,yqr->xqr", s, vh)
            else:
                psi[n], psi[n + 1] = tenet.einsum("apx,xy->apy", u, s), vh
            schmidt[n] = spectrum(s)
            invalidate(envs, n, n + 1)
            if direction == "right":
                update_env(envs, psi, w, n, "last")
            else:
                update_env(envs, psi, w, n + 1, "first")
    return energy, max_dw


def _schmidt_change(old: dict, new: dict) -> float:
    """``max_k ||S_k - S_k^old||`` over bonds, zero-padded -- YASTN ``_dmrg.py``:154-195.

    A bond present in only one of the two, or a spectrum whose length changed because
    ``svd_truncated`` moved the bond space, counts as a large change rather than an
    error, which is what it is. Infinite before the first sweep has any history.
    """
    if not old:
        return float("inf")
    worst = 0.0
    for n in new:
        previous, current = old.get(n, []), new[n]
        m = max(len(previous), len(current))
        a = previous + [0.0] * (m - len(previous))
        b = current + [0.0] * (m - len(current))
        worst = max(worst, sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5)
    return worst


class DMRG_out(NamedTuple):
    """YASTN's ``DMRG_out`` (``_dmrg.py``:33-39), plus the two things a test needs.

    ``history`` is one ``(energy, denergy, dSchmidt, discarded)`` tuple per sweep -- the
    ``ctmrg.converge`` precedent, and it says everything YASTN's ``iterator=True``
    generator protocol (:124-128, :196-198) says to a test without the protocol. ``psi``
    is the converged MPS, as a plain list of site tensors.
    """

    sweeps: int
    energy: float
    denergy: float
    max_dSchmidt: float
    max_discarded_weight: float
    history: list
    psi: list


def dmrg(
    n_sites: int,
    chi: int = 64,
    *,
    cutoff: float = 1e-14,
    energy_tol: float = 1e-12,
    schmidt_tol: float = 1e-8,
    max_sweeps: int = 40,
    seed: int = 0,
    ncv: int = 3,
) -> DMRG_out:
    """Sweep to the ground state and return a :class:`DMRG_out`.

    Convergence uses **both** of YASTN's criteria (``_dmrg.py``:180-195): the energy
    change ``|E_old - E| < energy_tol`` *and* the worst-cut Schmidt change
    ``max_k ||S_k - S_k^old|| < schmidt_tol``, and the loop stops only when both are met
    in one sweep. The Schmidt criterion is the sensitive one, and it is what catches a run
    whose energy has plateaued on a wrong bond structure.
    """
    psi = canonicalize(random_mps(n_sites, seed=seed))
    w = mpo(n_sites)
    envs = setup_envs(psi, w)
    schmidt: dict[int, list[float]] = {}
    energy, history, out = float("inf"), [], None
    for it in range(1, max_sweeps + 1):
        old_energy, old_schmidt = energy, dict(schmidt)
        energy, max_dw = sweep(psi, w, envs, schmidt, chi=chi, cutoff=cutoff, ncv=ncv)
        denergy = abs(old_energy - energy)
        d_schmidt = _schmidt_change(old_schmidt, schmidt)
        history.append((energy, denergy, d_schmidt, max_dw))
        out = DMRG_out(it, energy, denergy, d_schmidt, max_dw, history, psi)
        if denergy < energy_tol and d_schmidt < schmidt_tol:
            break
    return out


def main(n_sites: int = 12, chi: int = 64, big_sites: int = 32, big_chi: int = 64):
    """N=12 at chi=64 against the exact ground state, then N=32 at chi=64 against ``e_inf``.

    The N=12 reference printed here is the **open**-boundary energy
    ``-5.142090632840532``; the periodic chain's ``-5.387390917445203`` is a different
    number for a different model and an OBC MPS cannot reproduce it.
    ``tests/integration/test_dmrg.py`` computes the OBC value rather than trusting it.
    """
    small = dmrg(n_sites, chi)
    print(f"N={n_sites} chi={chi}  E={small.energy:+.12f}  exact=-5.142090632840532")
    for i, (e, de, ds, dw) in enumerate(small.history, start=1):
        print(f"  sweep {i:2d}  E={e:+.12f}  dE={de:.3e}  dS={ds:.3e}  dw={dw:.3e}")

    big = dmrg(big_sites, big_chi)
    print(
        f"N={big_sites} chi={big_chi}  E={big.energy:+.12f}  "
        f"E/N={big.energy / big_sites:+.12f}  e_inf={E_INF:+.12f}  "
        f"sweeps={big.sweeps}  max_dw={big.max_discarded_weight:.3e}"
    )
    return small, big


if __name__ == "__main__":
    main()
