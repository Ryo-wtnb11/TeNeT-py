"""Finite-chain two-site DMRG, written out: the U(1) Heisenberg chain against exact diagonalization.

Run it standalone::

    uv run python examples/toy_codes/dmrg.py

The algorithm is the file: the directed-bond environment cache and its invalidation, the
two-site effective Hamiltonian, and the sweep whose truncation re-decides each bond space.
The state comes from ``mps.py``, the Hamiltonian from ``model.py`` as an MPO, and the
eigensolver from ``lanczos.py``. Nothing is imported from ``tenet.network``, which ships
all of it; ``examples/heisenberg_walkthrough.py`` is the same physics through the library.
``tebd.py`` reaches the same ground state from the same model's two-site gates, and
``exact.py`` is the number both are judged against.

The tensor operations it is built on: ``tenet.einsum`` for every contraction,
``tenet.repartition`` for the leg bends, and ``tenet.linalg.svd_truncated`` for the
bond-deciding factorization.

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

from lanczos import lanczos

# ``mps.py`` holds the state, ``model.py`` the Hamiltonian and ``lanczos.py`` the
# eigensolver; this file holds the algorithm. The names marked ``noqa`` below are
# re-exports only: this file never calls them, and dropping them would break every caller
# that imports this module as the whole example.
from model import BOUNDARY, E_INF, MPO_BOND, PHYS, mpo, mpo_blocks  # noqa: F401
from mps import (
    _as_site,
    bond_spaces,  # noqa: F401
    canonicalize,
    ones,
    random_mps,
    spectrum,
)

import tenet
from tenet import IN, OUT, Leg, SymmetricTensor


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
            # A leg keeps its side unless its label is named in bend, in which case it
            # crosses: the xor is that sentence. Legs are then regrouped OUT-first.
            outs = tuple(i for i, c in enumerate(term) if (t.legs[i].side is OUT) != (c in flip))
            ins = tuple(i for i in range(len(term)) if i not in outs)
            # repartition is what actually pays the bend coefficient; the label string is
            # permuted to match so the einsum equation still names the right axes.
            return tenet.repartition(t, outs, ins), "".join(term[i] for i in (*outs, *ins))

        a, ta = bent(a, ta)
        b, tb = bent(b, tb)
        equation = f"{ta},{tb}->{out}"
    return tenet.einsum(equation, a, b)


# --- the environment: YASTN's directed-bond dict ------------------------------------


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
        # Every leg is BOUNDARY, the unit sector at degeneracy 1, so both of these are the
        # number 1 wearing three legs: past the end of an open chain there is nothing to
        # contract, and starting from the unit sector is what says so.
        (-1, 0): ones((Leg(BOUNDARY, IN), Leg(BOUNDARY, OUT), Leg(BOUNDARY, OUT))),
        # Mirrored sides, because a right-directed environment meets the chain the other
        # way round: (ket OUT, mpo IN, bra IN).
        (n_sites, n_sites - 1): ones((Leg(BOUNDARY, OUT), Leg(BOUNDARY, IN), Leg(BOUNDARY, IN))),
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
        # Growing the left environment by one site. a/x/B are its (ket, mpo, bra) legs;
        # absorbing the ket leaves p (physical) and r (the ket's new right bond) open.
        t = tenet.einsum("axB,apr->xBpr", envs[n - 1, n], a)
        # The MPO next: it eats the old mpo bond x and the physical p, and emits the new
        # mpo bond m and the physical P that the bra will close against.
        t = tenet.einsum("xPpm,xBpr->BrPm", w[n], t)
        # The bra closes B and P, leaving (r, m, s) -- ket, mpo, bra one site further on.
        envs[n, n + 1] = tenet.einsum("BPs,BrPm->rms", bra, t)
    else:
        # The mirror image, growing the right environment leftwards. This direction runs
        # against the arrows, so p and P each turn around in the cap and are bent by name.
        t = tenet.einsum("apr,rys->apys", a, envs[n + 1, n])
        t = _composed("apys,xPpy->axPs", t, w[n], bend="p")
        envs[n, n - 1] = _composed("axPs,BPs->axB", t, bra, bend="P")


def invalidate(envs, *sites: int) -> None:
    """Pop every entry a changed site invalidates -- YASTN ``clear_site_``, :127-134.

    Both directed bonds touching each site go, and they go *before* the replacement is
    written, so a missed update is a ``KeyError`` rather than a wrong number.
    """
    for n in sites:
        # Both directions: a changed site tensor is inside every environment built from
        # it, whichever way that environment was grown.
        envs.pop((n, n - 1), None)
        envs.pop((n, n + 1), None)


def setup_envs(psi, w) -> dict[tuple[int, int], SymmetricTensor]:
    """Every right-directed environment, for a right-canonical ``psi`` -- ``setup_(to='first')``."""
    envs = boundary_envs(len(psi))
    # Right to left, so the first bond the sweep visits already has its right environment
    # built from every site beyond it. The left side needs nothing: it is the boundary.
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
    # Right environment first: r is the two-site tensor's right bond, y the mpo bond it
    # brings in, s the bra bond that will become the output's right bond. Riding the
    # environment costs D^2 M chi and opens nothing that has to be bent.
    t = tenet.einsum("apqr,rys->apqys", aa, envs[n + 2, n + 1])
    # Then the right site's MPO: it eats the physical q and the mpo bond y, and emits Q
    # and the internal mpo bond m that the left W will meet.
    t = _composed("apqys,mQqy->apQms", t, w2, bend="q")
    # Then the left site's MPO: eats p and m, emits P and the mpo bond x.
    t = _composed("apQms,xPpm->aPQxs", t, w1, bend="p")
    # Left environment last, closing a and x. What comes back is on (B, P, Q, s) -- the
    # environments' *bra* legs became the bonds -- so it has aa's structure exactly, which
    # is what lets Lanczos add the input and the output as vectors of one space.
    return _composed("aPQxs,axB->BPQs", t, envs[n - 1, n], bend="a")


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
    # Both directions each sweep: a left-to-right pass optimizes each bond against a right
    # environment built from the *previous* sweep's tensors, and the return pass is what
    # lets the information that moved right come back.
    for direction in ("right", "left"):
        bonds = range(n_sites - 1) if direction == "right" else range(n_sites - 2, -1, -1)
        for n in bonds:
            # Merge the two sites: a = left bond, p and q the physicals, r = right bond.
            # Two sites at once is what lets the bond between them be re-decided below.
            aa = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
            w1, w2 = w[n], w[n + 1]
            # Solve the local eigenproblem. The rest of the chain enters only through the
            # two environments, which is why this is O(chi^3) and not exponential; aa is
            # both the seed and the answer's shape, and starting from the current state is
            # what makes three Krylov vectors enough.
            energy, aa = lanczos(
                lambda v, w1=w1, w2=w2, n=n: heff2(envs, w1, w2, n, v), aa, ncv=ncv
            )
            # Split along (a, p) against (q, r) -- the same cut the merge closed. This is
            # where the bond space is decided: svd_truncated keeps the largest singular
            # values within each sector and returns whatever grading survives, so the bond
            # both grows (by up to d per sweep) and re-sorts its charges.
            u, s, vh = tenet.linalg.svd_truncated(aa, ((0, 1), (2, 3)), max_bond=chi, cutoff=cutoff)
            vh = _as_site(vh)
            norm_s = tenet.norm(s)
            # Pythagoras: u and vh are isometries, so norm(aa) is the norm of the full
            # spectrum and the missing fraction is what the truncation cost. This is the
            # variational error bar on the printed energy.
            max_dw = max(max_dw, 1.0 - float(norm_s / tenet.norm(aa)) ** 2)
            s = s / norm_s  # the two-site tensor is normalized; keep the MPS so
            # The singular values go to the site the sweep is leaving behind, so the
            # trailing site stays a bare isometry and the chain remains canonical in the
            # direction of travel -- which is the condition that makes the *next* bond's
            # truncation optimal for the state and not merely for the local tensor.
            if direction == "right":
                psi[n], psi[n + 1] = u, tenet.einsum("xy,yqr->xqr", s, vh)
            else:
                psi[n], psi[n + 1] = tenet.einsum("apx,xy->apy", u, s), vh
            schmidt[n] = spectrum(s)
            # Drop the caches both changed sites appear in, before writing the new one, so
            # a missed update raises a KeyError instead of returning a plausible energy.
            invalidate(envs, n, n + 1)
            # Rebuild only the one environment the next bond will read: the site behind
            # the direction of travel is final for this pass.
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
        # Zero-pad the shorter spectrum: a bond that grew has genuinely new Schmidt
        # weight, and comparing it against zero is exactly the change it represents.
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
    # Random rather than Neel: a product state has zero overlap with most of the sectors
    # the ground state uses, and there is no noise term here to put them back.
    psi = canonicalize(random_mps(n_sites, seed=seed))
    w = mpo(n_sites)
    envs = setup_envs(psi, w)
    schmidt: dict[int, list[float]] = {}
    # Infinite starting energy so the first sweep's denergy cannot accidentally pass.
    energy, history, out = float("inf"), [], None
    for it in range(1, max_sweeps + 1):
        # Snapshot before the sweep overwrites both in place -- the two convergence
        # measures are differences against the previous sweep.
        old_energy, old_schmidt = energy, dict(schmidt)
        energy, max_dw = sweep(psi, w, envs, schmidt, chi=chi, cutoff=cutoff, ncv=ncv)
        denergy = abs(old_energy - energy)
        d_schmidt = _schmidt_change(old_schmidt, schmidt)
        history.append((energy, denergy, d_schmidt, max_dw))
        out = DMRG_out(it, energy, denergy, d_schmidt, max_dw, history, psi)
        # Both criteria, and in the same sweep. The energy is stationary at the minimum,
        # so it stops moving well before the state does -- energy_tol at 1e-12 is near
        # what float64 can resolve on a number of order 1, while schmidt_tol at 1e-8 is
        # the one that catches a run plateaued on the wrong bond structure.
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

    # N=32 is past dense ED, so the check changes: E/N has to sit above e_inf, and close
    # to it, the gap being the open chain's two missing bonds and its finite length.
    big = dmrg(big_sites, big_chi)
    print(
        f"N={big_sites} chi={big_chi}  E={big.energy:+.12f}  "
        f"E/N={big.energy / big_sites:+.12f}  e_inf={E_INF:+.12f}  "
        f"sweeps=~{big.sweeps}  max_dw={big.max_discarded_weight:.3e}"
    )
    return small, big


if __name__ == "__main__":
    main()
