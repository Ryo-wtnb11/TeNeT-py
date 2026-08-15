"""The environment cache: ``<psi|H|psi>`` partial contractions, keyed by *directed* bond.

Promoted from ``examples/dmrg.py`` (#110) with no arithmetic change: ``boundary_envs``
:318-331, ``update_env`` :334-349, ``invalidate`` :352-360, ``setup_envs`` :363-368 and
``heff2`` :374-390. :meth:`Env.measure` is the one genuinely new capability in M11a.
"""

from typing import Any

import autoray as ar

import tenet
from tenet import IN, OUT, Leg, SymmetricTensor
from tenet.network.mps import MPO, MPS, scalar

__all__ = ["Env"]


def _ones(legs: tuple[Leg, ...]) -> SymmetricTensor:
    """A tensor of ones on ``legs`` -- ``examples/ctmrg.py::init_env``'s seed spelling."""
    t = SymmetricTensor.zeros(legs)
    return t.apply_blocks(lambda b: ar.do("ones_like", b))


class Env:
    """``<psi|H|psi>`` partial contractions for one ``(psi, h)`` pair.

    ``F[(n, n + 1)]``: ``(ket IN, mpo OUT, bra OUT)``, built from sites ``<= n``;
    ``F[(n, n - 1)]``: ``(ket OUT, mpo IN, bra IN)``, built from sites ``>= n``.
    The two orientations are what makes every contraction in :meth:`update_` and
    :meth:`heff2` meet IN against OUT with no leg bend anywhere in this module.

    A plain ``dict`` keyed by *directed* bond, exactly YASTN's ``Env``
    (``yastn/tn/mps/_env.py``:94-125). A list-of-left / list-of-right would hide the
    invalidation discipline, which is the entire correctness content of an environment
    cache -- a stale ``F[(n, n+1)]`` after site ``n`` changed gives an energy that is
    *plausible and wrong*, the worst failure mode a DMRG has. :meth:`clear_` therefore
    pops **both** directed bonds per site, and it runs *before* the replacement is
    written, so a missed update is a ``KeyError`` rather than a wrong number.

    **One class, not YASTN's factory over eight.** ``yastn.tn.mps.Env`` is a function
    dispatching into ``Env2``, ``Env_mps_mpo_mps``, ``…_precompute``, ``Env_mpo_mpo_mpo``,
    ``Env_mps_mpopbc_mps``, ``Env_sum``, ``Env_project`` (``_env.py``:26-89), and every
    one of those serves a feature M11a does not ship -- MPO products, PBC, sums of
    Hamiltonians, excited-state penalties. The dispatch arrives if and when a second
    target does.
    """

    F: dict[tuple[int, int], SymmetricTensor]

    def __init__(self, psi: MPS, h: MPO) -> None:
        self.psi = psi
        self.h = h
        n = len(psi)
        bond_l, bond_r = psi[0].legs[0].space, psi[n - 1].legs[2].space
        mpo_l, mpo_r = h[0].legs[0].space, h[n - 1].legs[3].space
        self.F = {
            (-1, 0): _ones((Leg(bond_l, IN), Leg(mpo_l, OUT), Leg(bond_l, OUT))),
            (n, n - 1): _ones((Leg(bond_r, OUT), Leg(mpo_r, IN), Leg(bond_r, IN))),
        }

    def setup_(self, to: int = 0) -> "Env":
        """Build every environment directed towards site ``to``, and return ``self``.

        ``to=0`` is YASTN's ``setup_(to='first')`` (``_env.py``:104-125): for a
        right-canonical ``psi`` this is every right-directed environment, and it is the
        state a left-to-right sweep starts from.
        """
        if to != 0:
            raise NotImplementedError("only to=0 is implemented; canonize_ has the same note")
        for n in range(len(self.psi) - 1, 0, -1):
            self.update_(n, to="first")
        return self

    def update_(self, n: int, *, to: str) -> None:
        """Write one directed-bond entry from its neighbour -- YASTN ``_env.py``:152-168.

        ``to='last'`` writes ``F[(n, n+1)]`` from ``F[(n-1, n)]``; ``to='first'`` writes
        ``F[(n, n-1)]`` from ``F[(n+1, n)]``. Three pairwise ``tenet.einsum`` calls each:
        environment first, then the ket, then the MPO, then the bra.
        """
        a, bra = self.psi[n], tenet.adjoint(self.psi[n])
        if to == "last":
            t = tenet.einsum("axB,apr->xBpr", self.F[n - 1, n], a)
            t = tenet.einsum("xBpr,xPpm->BrPm", t, self.h[n])
            self.F[n, n + 1] = tenet.einsum("BrPm,BPs->rms", t, bra)
        else:
            t = tenet.einsum("apr,rys->apys", a, self.F[n + 1, n])
            t = tenet.einsum("apys,xPpy->axPs", t, self.h[n])
            self.F[n, n - 1] = tenet.einsum("axPs,BPs->axB", t, bra)

    def clear_(self, *sites: int) -> None:
        """Pop **both** directed bonds touching each changed site -- YASTN ``clear_site_``."""
        for n in sites:
            self.F.pop((n, n - 1), None)
            self.F.pop((n, n + 1), None)

    def heff2(self, n: int, aa: SymmetricTensor) -> SymmetricTensor:
        """``H_eff`` on the two-site tensor at bond ``(n, n+1)``. Four pairwise contractions.

        Right environment, then ``W2``, then ``W1``, then the left environment: YASTN's
        ``Env_mps_mpo_mps.Heff2`` order (``_env.py``:496-518) with ``precompute=False``,
        which ``_dmrg.py``:102-108 documents as ``O(D^3 M d + D^2 M^2 d^2)`` -- optimal
        for a single matvec, which is all a Krylov step ever wants.

        In and out on ``(left bond OUT, p OUT, q OUT, right bond IN)``: the *bra* legs of
        the two environments become the output's bonds while the *ket* legs close against
        the input's, which is why the result has ``aa``'s structure exactly and
        :func:`~tenet.network.lanczos` can add the two.
        """
        t = tenet.einsum("apqr,rys->apqys", aa, self.F[n + 2, n + 1])
        t = tenet.einsum("apqys,mQqy->apQms", t, self.h[n + 1])
        t = tenet.einsum("apQms,xPpm->aPQxs", t, self.h[n])
        return tenet.einsum("aPQxs,axB->BPQs", t, self.F[n - 1, n])

    def measure(self) -> float:
        """``<psi|H|psi>`` without the eigensolver, on a private left-to-right pass.

        The first thing in this repository that measures a converged energy independently
        of the ``lanczos`` Rayleigh quotient that produced it. YASTN's ``measure`` is the
        same closing contraction one level down (``_env.py``:462-468, ``vdot(vecL,
        vecR)``); the pass is built in a fresh :class:`Env` so a measurement never writes
        into a sweep's cache.
        """
        n = len(self.psi)
        env = Env(self.psi, self.h)
        for site in range(n):
            env.update_(site, to="last")
        return float(scalar(tenet.einsum("rms,Rms->Rr", env.F[n - 1, n], env.F[n, n - 1])))

    def __repr__(self) -> str:
        return f"Env(sites={len(self.psi)}, bonds={sorted(self.F)})"

    def __contains__(self, key: Any) -> bool:
        return key in self.F
