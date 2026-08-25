"""2D classical Ising through ``tenet.network.EnvCTMc4v``, on a core install -- no JAX.

Run it standalone::

    uv run python examples/ising2d.py

The library owns the environment -- the seed, the move, the sweep, the truncation -- and
it must not own the physics, so the Boltzmann tensor, Baxter's telescoping and the Onsager
oracle live here. The Boltzmann tensor is symmetric under every permutation of its four
legs, which is the full C4v point group, so one corner and one edge describe its whole
environment: the C4v lane fits it exactly. The ordered-phase corner spectrum prints
exactly two-fold degenerate across the Z2 parity sectors.
"""

import autoray as ar
import numpy as np

import tenet
from tenet import OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import EnvCTMc4v, Peps, SquareLattice, flip, spectrum
from tenet.symmetry import Z2, Z2Sector

BETA_C = 0.4406867935097714  # ln(1 + sqrt(2)) / 2


def ising_bulk(beta: float) -> SymmetricTensor:
    """The Boltzmann tensor on four *identical* legs -- the C4v ansatz's signature.

    ``a[t,l,b,r] = sum_s W[s,t] W[s,l] W[s,b] W[s,r]`` with ``W W^T`` the bond Boltzmann
    matrix, i.e. the symmetric splitting ``W = [[sqrt cosh b, sqrt sinh b],
    [sqrt cosh b, -sqrt sinh b]]``. That ``W`` *is* the parity basis, so the ``Z2`` legs
    are the statement rather than a claim checked afterwards: summing over ``s``
    annihilates every entry with an odd number of odd legs, and those eight entries have
    no block to live in.
    """
    # W W^T = [[e^b, e^-b], [e^-b, e^b]] is the bond Boltzmann weight, so splitting it
    # symmetrically puts half a bond on each of the two sites it joins.
    c, s = np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))
    w = np.array([[c, s], [c, -s]])
    # s is the site spin, summed over; t/l/b/r are the four half-bonds leaving it. Each
    # W[s,.] carries one half-bond, so the einsum *is* "one Ising spin with four legs".
    block = np.einsum("st,sl,sb,sr->tlbr", w, w, w, w)
    space = GradedSpace.new(Z2, {Z2Sector(0): 1, Z2Sector(1): 1})
    # All four legs OUT and identical: a leg direction only matters where two are joined,
    # and the C4v lane contracts this tensor against itself, so the four are one leg type.
    return SymmetricTensor.from_dense(block, (Leg(space, OUT),) * 4)


def onsager(beta: float, points: int = 200_001) -> float:
    """``beta f`` from Onsager's closed form, by direct quadrature. NumPy, no ``scipy``.

    ``-beta f = ln 2 / 2 + (1/2pi) int_0^pi dtheta ln[cosh^2(2b) + (1/k) sqrt(1 + k^2
    - 2k cos 2theta)]``, ``k = 1/sinh^2(2b)``.
    """
    kk = 1.0 / np.sinh(2.0 * beta) ** 2
    # 200k points on a smooth periodic integrand: the trapezoid rule is spectrally
    # accurate here, which is why 1e-12 agreement with the CTM run is a fair test.
    theta = np.linspace(0.0, np.pi, points)
    integrand = np.log(
        np.cosh(2.0 * beta) ** 2 + np.sqrt(1.0 + kk**2 - 2.0 * kk * np.cos(2.0 * theta)) / kk
    )
    return -(np.log(2.0) / 2.0 + np.trapezoid(integrand, theta) / (2.0 * np.pi))


def log_kappa(env: EnvCTMc4v, site=(0, 0)):
    """``ln`` of the partition function per site, Baxter's corner-transfer telescoping.

    ``kappa = z_a z_c / z_h**2``: four corners cover an ``L x L`` patch, four corners with
    four edges and the bulk tensor cover ``(L + 1) x (L + 1)``, and four corners with two
    edges cover ``L x (L + 1)``. Every contracted pair meets ``IN`` against ``OUT``, which
    for the four-corner and four-corner-two-edge objects means one of each pair crosses a
    sublattice boundary and enters flipped; the eight-tensor ring around a site does not,
    because corners and edges already alternate around it.
    """
    e, a = env[site], env.psi[site]
    # One corner and one edge is the whole C4v environment; ``flip`` reverses a tensor's
    # leg directions so an OUT leg can meet an OUT leg where the ring changes sublattice.
    c, cf, t, tf = e.tl, flip(e.tl), e.t, flip(e.t)
    # Four corners in a ring: each shares one leg with the next, and the trace closes the
    # ring. This is Z on an L x L patch.
    z_c = tenet.full_trace(tenet.einsum("ab,ac,dc,eb->de", c, cf, c, cf))
    # Four corners with two opposite edges wedged in: an L x (L+1) patch. ``f`` is the
    # edge's physical-bond leg, shared between the two edges facing each other.
    z_h = tenet.full_trace(tenet.einsum("ab,ac,dfc,ed,eg,gfh->hb", c, cf, tf, cf, c, t))
    # Corner, edge, corner, edge, ... around one bulk tensor: an (L+1) x (L+1) patch.
    # a-h are the ring's virtual bonds, p/q/r/s the four half-bonds each edge hands to
    # the bulk tensor ``a``. Corners and edges already alternate, so nothing is flipped.
    z_a = tenet.full_trace(
        tenet.einsum("ab,apc,cd,eqd,fe,grf,gh,hsk,spqr->kb", c, t, c, t, c, t, c, t, a)
    )
    # The patches telescope: (L+1)^2 + L^2 - 2 L(L+1) = 1, so every environment tensor
    # cancels and one site's worth of partition function is left, gauge and norm included.
    return ar.do("log", z_a * z_c / z_h**2)


def corner_spectrum(env: EnvCTMc4v, site=(0, 0)) -> list[float]:
    """The corner's singular values. The renormalized corner is ``V^dagger U S`` -- the
    correction between the two index groups is kept, so the spectrum is read with an SVD
    rather than off a diagonal."""
    return spectrum(tenet.linalg.svd(env[site].tl, ((0,), (1,)))[1])


def main(chi: int = 24):
    """Free energy against Onsager at three betas; returns {beta: (beta*f, rel_err)}."""
    results = {}
    # Disordered, critical, ordered. Only the critical point is hard: correlations there
    # are longer than any finite chi can hold, so its relative error is the loose one.
    for beta in (0.3, BETA_C, 0.5):
        # A 1x1 unit cell -- the Boltzmann tensor is the same on every site, so one
        # tensor tiles the whole lattice and one corner/edge pair is its environment.
        env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising_bulk(beta)))
        out = env.iterate_(max_bond=chi, max_sweeps=100)
        bf = -float(log_kappa(env))
        rel = abs(bf / onsager(beta) - 1)
        print(f"beta={beta:.4f}  {out.sweeps:3d} sweeps  beta*f = {bf:+.10f}  rel {rel:.1e}")
        results[beta] = (bf, rel)
    # The exact cross-sector doublet is a sharper question than the free energy, so the
    # ordered environment is swept to the float64 floor for it (as the integration suite
    # does) rather than to the default 1e-10 the loop above uses.
    ordered = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), ising_bulk(0.5)))
    ordered.iterate_(max_bond=chi, max_sweeps=200, corner_tol=1e-14)
    # Below T_c the two ordered states are degenerate, and the corner spectrum sees it:
    # each singular value appears once in the even and once in the odd Z2 sector.
    corner = corner_spectrum(ordered)
    print("corner spectrum at beta=0.5:", " ".join(f"{v:.4f}" for v in corner[:6]))
    return results, corner


if __name__ == "__main__":
    main()
