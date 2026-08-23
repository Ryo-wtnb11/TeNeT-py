"""The MPS: the list of site tensors, how to seed it, its canonical form, and what it measures.

The state ``tebd.py`` evolves and ``dmrg.py`` sweeps. The physical space comes from
``model.py``; nothing here knows the Hamiltonian, which is why the same container serves
both algorithms and both measurements below.

**MPS leg convention**, the part worth reading before the code: site ``A_n`` is
``(left bond OUT, physical OUT, right bond IN)``, the ``examples/toy_codes/vmc_mps.py``
convention. Charge flows left to right, ``bond_n (x) phys_n -> bond_{n+1}``, and both end
bonds are ``model.BOUNDARY``, the unit sector with degeneracy 1 -- which forces
``Sum_i 2 S^z_i = 0``, i.e. ``S^z_tot = 0``, structurally and for free.

The tensor operations it is built on: ``SymmetricTensor.from_blocks`` and
``SymmetricTensor.random`` for the seeds, ``tenet.einsum`` for every contraction,
``tenet.adjoint`` for the bra, ``tenet.repartition`` for the leg bends,
``tenet.linalg.lq`` for the canonical form, ``tenet.norm``, ``tenet.full_trace`` to close
the measured network, and ``tenet.to_matrices`` to read the Schmidt values off a bond.
"""

import math

import numpy as np
from model import BOUNDARY, PHYS

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.symmetry import U1, U1Sector

# --- the state ---------------------------------------------------------------------


def ones(legs) -> SymmetricTensor:
    """A tensor with every structurally allowed entry equal to 1.

    ``TensorStructure`` already knows which blocks the grading allows and how big each one
    is, so the seed is "fill the blocks that exist": there is no dense array here to build
    and project. Where the grading allows exactly one entry -- a ``D=1`` bond either side
    of a site, as in :func:`product_mps` -- that is a basis state written without naming a
    basis.
    """
    structure = TensorStructure(tuple(legs))
    blocks = {key: np.ones(structure.block_shape(key)) for key in structure.block_order}
    return SymmetricTensor.from_blocks(legs, blocks)


def bond_spaces(n_sites: int) -> list[GradedSpace]:
    """The ``n_sites + 1`` virtual spaces, degeneracy 1 in every reachable sector.

    The charge on bond ``i`` is the sum of ``i`` physical charges (each ``+-1``) and must
    still be able to reach 0 by site ``n_sites``, so it runs over
    ``-w, -w+2, ..., w`` with ``w = min(i, n_sites - i)``; bonds 0 and ``n_sites`` are
    :data:`BOUNDARY`, which is the whole ``S^z_tot = 0`` statement. Which charges are
    reachable is physics, and it is the only thing about this chain a generic MPS
    container cannot know -- which is why the library takes bond *spaces* and not a chi.

    Simplification: degeneracy 1 per sector, not a full-rank seed. Two-site DMRG multiplies a
    bond by ``d = 2`` per sweep, so a chi=64 bond is three sweeps away and the first
    sweeps are correspondingly cheap. A larger seed is one ``min(..., cap)`` here if a
    workload ever wants the bond full on sweep one.
    """
    spaces = []
    for i in range(n_sites + 1):
        w = min(i, n_sites - i)
        spaces.append(GradedSpace.new(U1, {U1Sector(q): 1 for q in range(-w, w + 1, 2)}))
    return spaces


def random_mps(n_sites: int, seed: int = 0) -> list[SymmetricTensor]:
    """A random U(1) MPS, ``A_n`` on ``(left bond OUT, phys OUT, right bond IN)``."""
    spaces = bond_spaces(n_sites)
    return [
        SymmetricTensor.random(
            (Leg(spaces[i], OUT), Leg(PHYS, OUT), Leg(spaces[i + 1], IN)), seed=seed + i
        )
        for i in range(n_sites)
    ]


def product_mps(n_sites: int) -> list[SymmetricTensor]:
    """The Neel state ``|up down up down ...>``, every bond ``D=1``.

    The charge on bond ``i`` is fixed -- it is the running sum of the alternating physical
    charges -- so each bond space holds one sector of degeneracy 1, and then each site has
    exactly one structurally allowed entry: :func:`ones` fills it and the result *is* the
    product state, with no dense basis written anywhere. ``n_sites`` must be even, or the
    chain does not close on ``S^z_tot = 0``.

    This is ``tebd.py``'s starting state: an unentangled state at the right total charge,
    which imaginary time then has to do all the work on.
    """
    charges = [sum(1 if k % 2 == 0 else -1 for k in range(i)) for i in range(n_sites + 1)]
    spaces = [GradedSpace.new(U1, {U1Sector(q): 1}) for q in charges]
    return [
        ones((Leg(spaces[i], OUT), Leg(PHYS, OUT), Leg(spaces[i + 1], IN))) for i in range(n_sites)
    ]


def _as_site(t: SymmetricTensor) -> SymmetricTensor:
    """Put a rank-3 factor back on the MPS partition ``(l, p | r)``.

    Every factorization in ``tenet.linalg`` lowers its input to a *map* first, so the
    legs it hands to the domain come back bent -- a physical leg that was ``OUT`` in the
    codomain is ``IN dual`` in the domain, which is the leg bend spelled out rather than
    hidden. One :func:`tenet.repartition` puts it back, and only then do a site tensor
    from :func:`canonicalize` and one from :func:`sweep` share a structure. (``MPS``
    does this in a write barrier on ``__setitem__``; here it is a named call, because
    seeing it is the lesson.)
    """
    return tenet.repartition(t, (0, 1), (2,))


def canonicalize(psi: list[SymmetricTensor]) -> list[SymmetricTensor]:
    """Right-canonicalize in place of YASTN's ``canonize_(to='first')`` (``_mps_obc.py``:390).

    One ``tenet.linalg.lq`` per site from the right, mirroring ``orthogonalize_site_``
    (:245-300): ``A_n = L . Q`` with ``Q`` on ``(bond OUT, phys OUT, right IN)`` -- the
    MPS leg convention unchanged -- and ``L`` absorbed into ``A_{n-1}``. ``lq`` rather
    than ``qr`` because ``qr`` would put the new bond on the *right* of the factor and
    leave the site tensor's left leg IN, which is not this file's convention.

    Setup only. A two-site sweep leaves the state canonical by construction on the side
    it came from, so this runs once, before the first environment is built.
    """
    psi = list(psi)
    for n in range(len(psi) - 1, 0, -1):
        left, q = tenet.linalg.lq(psi[n], ((0,), (1, 2)))
        psi[n] = _as_site(q)
        psi[n - 1] = tenet.einsum("apx,xy->apy", psi[n - 1], left)
    return [psi[0] / tenet.norm(psi[0]), *psi[1:]]


# --- reading numbers off a tensor --------------------------------------------------


def spectrum(s: SymmetricTensor) -> list[float]:
    """The Schmidt values on a bond, descending.

    ``s`` comes from :func:`tenet.linalg.svd_truncated` and is diagonal by construction,
    so this reads the diagonal of each coupled-sector matrix ``tenet.to_matrices`` hands
    back -- the public way to read block values. The ``sqrt(qdim)`` weight is the same one
    :func:`tenet.norm` carries, and it is 1 throughout for U(1).
    """
    qdim = s.provider.qdim
    out = [
        float(m[i, i]) * qdim(sector) ** 0.5
        for sector, m in tenet.to_matrices(s).items()
        for i in range(m.shape[0])
    ]
    return sorted(out, reverse=True)


def entropy(schmidt: list[float]) -> float:
    """Von Neumann entanglement entropy of a cut, from its Schmidt values.

    ``S = -Sum_k p_k ln p_k`` with ``p_k = s_k**2``, the standard measure of how much a
    bond has to carry. Values at or below zero after truncation are dropped rather than
    fed to ``log``: they are the discarded tail, and they contribute nothing.
    """
    return -sum(s**2 * math.log(s**2) for s in schmidt if s > 0.0)


def expectation(psi: list[SymmetricTensor], op: SymmetricTensor, n: int) -> float:
    """``<psi|op|psi> / <psi|psi>`` for ``op`` on site ``n`` (rank 2) or bond ``(n, n+1)`` (rank 4).

    One left-to-right pass of the transfer matrix, environment ``(ket IN, bra OUT)``,
    absorbing the ket, then the operator where there is one, then the bra -- the same three
    steps ``dmrg.update_env`` takes with an MPO in the middle, minus the MPO. Sweeping only
    left to right is what keeps every contraction a plain composition: operand 1 supplies
    the ``IN`` end of every shared wire and no wire turns around, so no bend and no
    :func:`dmrg._composed` is needed here.

    A two-site ``op`` is applied to the merged ``theta``; the split is never undone,
    because the merged tensor is thrown away with the environment.

    Simplification: **the whole chain is contracted, per measurement.** That is ``O(N D^3)``
    for a number the canonical form could give in ``O(D^3)`` if this container stored its
    Schmidt values on every bond the way ``tenet.network.MPS`` does. At the sizes here that
    trade buys nothing and costs the reader a second invariant to hold; the upgrade path is
    to keep the singular values from :func:`canonicalize` and cut the sweep short.
    """
    sites = list(psi)
    two_site = op.ndim == 4
    if two_site:
        sites[n : n + 2] = [tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])]
    env = ones((Leg(BOUNDARY, IN), Leg(BOUNDARY, OUT)))
    for i, a in enumerate(sites):
        bra = tenet.adjoint(a)
        if i != n:
            t = tenet.einsum("aB,apr->Bpr", env, a)
            env = tenet.einsum("Bps,Bpr->rs", bra, t)
        elif two_site:
            t = tenet.einsum("aB,apqr->Bpqr", env, a)
            t = tenet.einsum("PQpq,Bpqr->BPQr", op, t)
            env = tenet.einsum("BPQs,BPQr->rs", bra, t)
        else:
            t = tenet.einsum("aB,apr->Bpr", env, a)
            t = tenet.einsum("Pp,Bpr->BPr", op, t)
            env = tenet.einsum("BPs,BPr->rs", bra, t)
    return float(tenet.full_trace(env))
