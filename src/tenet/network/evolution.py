"""Time evolution on the 2D layer: Trotter gates, the bond metric, and the truncation.

YASTN's ``fpeps/_evolution.py``, ``gates.py``, ``_gates_auxiliary.py`` and
``envs/_env_ntu.py`` (b0187c4), adopted by M79/#277 and re-spelled on M79a's
[Peps][tenet.network.Peps] and the twelve contraction primitives.

**One step.** A two-site gate ``exp(-step * h)`` is split across the bond
([gate_nn][tenet.network.gate_nn]), applied to the two sites -- which multiplies the
bond dimension by the gate's rank -- and the enlarged bond is then reduced back
([truncate_][tenet.network.truncate_]) in the metric the environment supplies.
[evolution_step_][tenet.network.evolution_step_] runs that over a list of gates and
[accumulated_truncation_error][tenet.network.accumulated_truncation_error] adds the
per-bond errors up.

**The gate's auxiliary leg is never fused into a virtual leg.** YASTN fuses it (
``apply_gate_onsite``) so that the state stays rank 5 between the two halves of a step,
and unfuses it one line later inside ``truncate_``'s ``qr``. Here the two enlarged
tensors are rank 6 and live only between
[apply_gate][tenet.network.apply_gate] and the ``qr`` that consumes them: tenet's
[fuse][tenet.fuse] wants its group to be a prefix of one side, so the fuse would need a
repartition on both sides of a round trip that buys nothing. The enlargement is the same
enlargement -- the bond the ``qr`` hands to the truncation carries ``r`` *and* the
auxiliary wire -- and no caller ever sees a rank-6 site.

**The fermionic order is the geometry's, so the gate needs no swap gate.**
YASTN's ``gate_fix_swap_gate`` repairs a gate built for the fermionic order ``0 -> 1``
when the lattice hands the pair the other way round.
[bonds][tenet.network.SquareLattice.bonds] already emits every bond oriented
left-before-right and top-before-bottom, which *is* the fermionic order on every
geometry but the cylinder's wrap-around, so ``g0`` always lands on the earlier site and
the repair is the identity. Each site's application is then **one two-operand
composition**: the gate half supplies ``IN`` on the physical wire (its ``p`` leg) and the
site supplies ``OUT``, so **the gate is operand 1 and nothing bends**, on both sites and
in both directions. Where the auxiliary leg is placed in the result does not move the
tensor -- a leg permutation is an isomorphism the next contraction undoes -- which is
measured in ``tests/network/test_evolution.py::test_the_auxiliary_leg_may_sit_anywhere``
rather than assumed.

**The metric is measured and repaired, never trusted.** ``g`` comes back from
``env.bond_metric``; ``truncate_`` symmetrizes it to ``(g + g^dagger) / 2`` and reports
the norm of what it removed (``nonhermitian_part``), the most negative eigenvalue and
the fraction of eigenvalues below the resulting error scale, in
[Evolution_out][tenet.network.Evolution_out]. That is the whole point of #243 carried
into the evolution: nothing here assumes the environment produced a positive form.

**Two environments, one interface.** [EnvCTM.bond_metric][tenet.network.EnvCTM.bond_metric]
closes the six surrounding environment tensors around the bond;
[EnvNTU][tenet.network.EnvNTU] closes the eight sites of the ``'NN'`` cluster instead --
cheap, positive by construction, and needing no converged environment.

**What is deliberately not transcribed** (recorded in ``docs/design.md``, M79d): YASTN's
EAT and ZMT initializations of the truncation (the SVD initialization plus the
least-squares sweep is what the oracles reach); the bipartite bond metric and
``EnvApproximate``; ``EnvNTU``'s larger clusters (``'NN+'``, ``'NN++'``, ``'NNN'`` and
the rest) and its ladder mode; multi-site gates and the MPO gate form, and with them
``split_gate_2site`` and ``fill_eye_in_gate``; and local (one-site) gates, which need no
truncation and are one composition a caller can write.
"""

from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import autoray as ar

import tenet
from tenet import IN, OUT, SymmetricTensor
from tenet.network.common import composed, ones
from tenet.network.lattice import Bond, Site
from tenet.network.peps import (
    DoubleLayer,
    Peps,
    cor_bl,
    cor_br,
    cor_tl,
    cor_tr,
    edge_b,
    edge_l,
    edge_r,
    edge_t,
)

__all__ = [
    "EnvNTU",
    "Evolution_out",
    "Gate",
    "accumulated_truncation_error",
    "apply_gate",
    "evolution_step_",
    "gate_nn",
    "gates_nn",
    "truncate_",
]

#: The pseudo-inverse cutoffs the least-squares solve walks, smallest first. One of them
#: is chosen per solve by the truncation error it produces, which is YASTN's rule.
PINV_CUTOFFS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)


class Gate(NamedTuple):
    """A nearest-neighbour two-site gate, already split across its bond.

    Attributes
    ----------
    g0 : SymmetricTensor
        The half acting on ``bond.site0``: ``(phys OUT, phys IN, aux IN)``.
    g1 : SymmetricTensor
        The half acting on ``bond.site1``: ``(aux OUT, phys OUT, phys IN)``.
    bond : Bond
        The bond, oriented in the fermionic order -- ``site0`` before ``site1``.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import GradedSpace
    >>> from tenet.network import Bond, Site, gate_nn, local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> sz = np.diag([-0.5, 0.5])
    >>> h = local_op(np.kron(sz, sz), phys=phys)
    >>> g = gate_nn(h, 0.1, Bond(Site(0, 0), Site(0, 1)))
    >>> g.g0.ndim, g.g1.ndim
    (3, 3)
    """

    g0: SymmetricTensor
    g1: SymmetricTensor
    bond: Bond


class Evolution_out(NamedTuple):
    """What one bond's truncation measured. Every error and eigenvalue is relative.

    Attributes
    ----------
    bond : Bond or None
        The bond that was truncated.
    truncation_error : float
        ``||RR - MM||_g / ||RR||_g``: the norm of what the truncation discarded,
        measured in the environment's own metric rather than in the plain one.
    nonhermitian_part : float
        ``||g - g^dagger|| / (2 ||g||)`` before the symmetrization. An estimate of the
        environment's error, not of the truncation's.
    min_eigenvalue : float or None
        The smallest eigenvalue of the symmetrized metric over ``||g||``. Negative means
        the environment produced something that is not a form; ``None`` when
        ``fix_metric`` was ``None`` and no eigendecomposition was taken.
    wrong_eigenvalues : float or None
        The fraction of eigenvalues below the error scale, which ``fix_metric`` replaced.
    iterations : int
        Least-squares sweeps taken, ``0`` when the SVD initialization was kept.
    pinv_cutoff : float or None
        The pseudo-inverse cutoff the last solve chose off
        ``PINV_CUTOFFS``.

    Examples
    --------
    >>> from tenet.network import Evolution_out
    >>> Evolution_out(truncation_error=1e-7).iterations
    0
    """

    bond: Bond | None = None
    truncation_error: float = 0.0
    nonhermitian_part: float = 0.0
    min_eigenvalue: float | None = None
    wrong_eigenvalues: float | None = None
    iterations: int = 0
    pinv_cutoff: float | None = None


# -- gates ---------------------------------------------------------------------------


def gate_nn(h: SymmetricTensor, step: float, bond: Any) -> Gate:
    """``exp(-step * h)`` for one bond, split into the two halves a PEPS wants.

    Parameters
    ----------
    h : SymmetricTensor
        The bond Hamiltonian, rank 4 on ``(phys OUT, phys OUT, phys IN, phys IN)`` --
        [local_op][tenet.network.local_op]'s invariant two-site form, the same object
        ``expectation_2site`` reads.
    step : float
        The Trotter step. ``exp(-step h)`` is the imaginary-time gate; a complex
        ``step`` is the real-time one, and nothing here assumes otherwise.
    bond : Bond or tuple[Site, Site]
        The bond the gate belongs to, ``site0`` before ``site1`` in the fermionic order.

    Returns
    -------
    Gate
        ``g0``, ``g1`` and the bond.

    Raises
    ------
    ValueError
        If ``h`` is not rank 4.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import GradedSpace
    >>> from tenet.network import Bond, Site, gate_nn, local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> sz = np.diag([-0.5, 0.5])
    >>> h = local_op(np.kron(sz, sz), phys=phys)
    >>> gate_nn(h, 0.1, Bond(Site(0, 0), Site(1, 0))).g0.shape
    (2, 2, 2)

    Notes
    -----
    [tenet.linalg.expm][tenet.ops.linalg.expm] lowers the pair to a square map on the
    partition ``((0, 1), (2, 3))`` -- the pair's outputs against its inputs -- and
    exponentiates one dense matrix per coupled sector, which is the spelling
    ``examples/toy_codes/tebd.py`` uses at 1D. The split is YASTN's
    ``decompose_nn_gate``: an SVD across ``((0, 2), (1, 3))``, the first site's pair of
    legs against the second's, with the singular values shared as ``sqrt(s)`` so neither
    half carries the whole scale. YASTN's ``gate_nn_hopping`` / ``gate_nn_Ising`` build
    the same object in closed form for two particular models; the exponential is the
    general one and the two closed forms are an optimization this layer does not need.
    """
    if h.ndim != 4:
        raise ValueError(f"gate_nn: the bond Hamiltonian should be rank 4, got rank {h.ndim}")
    g = tenet.linalg.expm(h, ((0, 1), (2, 3)), alpha=-step)
    u, s, vh = tenet.linalg.svd_truncated(g, ((0, 2), (1, 3)), cutoff=1e-14)
    sq = tenet.block_sqrt(s)
    g0 = composed("Ppx,xy->Ppy", u, sq)
    g1 = composed("xy,yQq->xQq", sq, vh)
    return Gate(g0, g1, Bond(Site(*bond[0]), Site(*bond[1])))


def gates_nn(
    geometry: Any, h: SymmetricTensor, step: float, *, symmetrize: bool = True
) -> tuple[Gate, ...]:
    """One [gate_nn][tenet.network.gate_nn] per nearest-neighbour bond of a lattice.

    Parameters
    ----------
    geometry : SquareLattice or Lattice or Peps
        Anything with a ``bonds()``; the gates come out in its order.
    h : SymmetricTensor
        The bond Hamiltonian, the same on every bond.
    step : float
        The Trotter step of the *whole* sequence.
    symmetrize : bool, optional
        ``True`` (the default) halves the step and appends the reversed sequence, so a
        step is a symmetric product and the Trotter error is ``O(step**3)``. ``False``
        gives one pass at the full step.

    Returns
    -------
    tuple[Gate, ...]
        The gates, in the order to apply them.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import GradedSpace
    >>> from tenet.network import SquareLattice, gates_nn, local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> h = local_op(np.kron(np.diag([-0.5, 0.5]), np.diag([-0.5, 0.5])), phys=phys)
    >>> len(gates_nn(SquareLattice(dims=(2, 2), boundary="obc"), h, 0.1))
    8

    Notes
    -----
    YASTN's ``distribute``, minus the local gates and minus the per-site Hamiltonian: a
    lattice whose bonds do not all carry the same term builds its own list, one
    ``gate_nn`` per bond, and hands it to
    [evolution_step_][tenet.network.evolution_step_].
    """
    bonds = geometry.bonds()
    gates = tuple(gate_nn(h, step / 2 if symmetrize else step, b) for b in bonds)
    return gates + gates[::-1] if symmetrize else gates


#: Per bond direction and per end of the bond: the axis that site splits off, and the
#: ``(outputs, inputs)`` that put the ``qr`` factor back on the PEPS partition -- the
#: three ``OUT`` legs (``l``, ``b``, ``phys``, with the new bond leg standing in for
#: whichever of them it replaces) against the two ``IN`` ones. The ``qr`` leaves the
#: kept legs in the order they were asked for, followed by the bond, whether or not a
#: gate's auxiliary leg rode along -- so one table serves both, and
#: [_AS_SITE][tenet.network.evolution._AS_SITE] is then the same transpose every time.
_SPLIT = {
    "lr": ((3, (1, 2, 3), (0, 4)), (1, (4, 1, 3), (0, 2))),
    "tb": ((2, (1, 4, 3), (0, 2)), (0, (0, 1, 3), (4, 2))),
}

#: ``repartition`` leaves ``(out0, out1, phys, in0, in1)``; the PEPS order is this.
_AS_SITE = (3, 0, 1, 4, 2)


def apply_gate(a0: SymmetricTensor, a1: SymmetricTensor, gate: Gate) -> tuple[Any, Any]:
    """The two site tensors with the gate's halves on their physical legs.

    Parameters
    ----------
    a0, a1 : SymmetricTensor
        The rank-5 site tensors of ``gate.bond``, in the fermionic order.
    gate : Gate
        The gate.

    Returns
    -------
    tuple[SymmetricTensor, SymmetricTensor]
        Rank 6 each: ``(t, l, b, r, aux, phys)``. The auxiliary wire is the bond's
        enlargement and the ``qr`` in [truncate_][tenet.network.truncate_] consumes it.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import Bond, Site, apply_gate, gate_nn, local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1})
    >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(phys, OUT))
    >>> a = SymmetricTensor.random(legs, seed=0)
    >>> sz = np.diag([-0.5, 0.5])
    >>> g = gate_nn(local_op(np.kron(sz, sz), phys=phys), 0.1, Bond(Site(0, 0), Site(0, 1)))
    >>> b0, b1 = apply_gate(a, a, g)
    >>> b0.ndim, b1.ndim
    (6, 6)

    Notes
    -----
    One composition per site, and the operand order is the physical wire's: the gate
    half supplies ``IN`` there and the site supplies ``OUT``, so **the gate is operand 1
    and nothing bends**. The module docstring says why no swap gate is needed and why
    the auxiliary leg's position is free.
    """
    b0 = composed("Ppc,tlbrp->tlbrcP", gate.g0, a0)
    b1 = composed("cQq,tlbrq->tlbrcQ", gate.g1, a1)
    return b0, b1


def _split(a: SymmetricTensor, dirn: str, which: int) -> tuple[SymmetricTensor, SymmetricTensor]:
    """``qr`` a site into the isometry that stays and the factor that is truncated.

    ``which`` is 0 for the bond's first site and 1 for its second. The right group is the
    bond leg plus, when the tensor is rank 6, the gate's auxiliary leg. The isometry comes
    back a rank-5 tensor on the PEPS convention, so the environment primitives take it
    without knowing a ``qr`` happened.
    """
    axis, outputs, inputs = _SPLIT[dirn][which]
    right = (axis, 4) if a.ndim == 6 else (axis,)
    left = tuple(i for i in range(a.ndim) if i not in right)
    q, r = tenet.linalg.qr(a, (left, right))
    return tenet.transpose(tenet.repartition(q, outputs, inputs), _AS_SITE), r


# -- the metric, measured and repaired ------------------------------------------------


def _dagger(g: SymmetricTensor) -> SymmetricTensor:
    """``g^dagger`` laid out like ``g``: the adjoint with its two index groups swapped.

    [tenet.adjoint][] keeps the public axis order and flips every ``side``, so on a
    rank-4 metric it returns the map whose *first* pair is the ket pair. Swapping the
    pairs back is what makes ``g - g^dagger`` a subtraction of two tensors with the same
    legs, which is the only form in which the question "how Hermitian is it?" has an
    answer.
    """
    return tenet.transpose(tenet.adjoint(g), (2, 3, 0, 1))


def _fix_metric(
    g: SymmetricTensor, fix_metric: float | None
) -> tuple[SymmetricTensor, float, float | None, float | None]:
    """Symmetrize the metric and report what that cost. YASTN ``truncate_optimize_``:549.

    Returns the repaired metric, ``||g - g^dagger|| / (2 ||g||)``, the smallest
    eigenvalue over ``||g||`` and the fraction of eigenvalues below the error scale.
    """
    gh = _dagger(g)
    scale = float(tenet.norm(g))
    nonhermitian = float(tenet.norm(g - gh)) / 2
    g = (g + gh) / 2
    if fix_metric is None:
        return g, nonhermitian / scale, None, None
    w, v = tenet.linalg.eigh(g, ((0, 1), (2, 3)))
    values = [float(x) for m in tenet.to_matrices(w).values() for x in ar.do("diag", m)]
    smin = min(values)
    error = max(-smin, 0.0) + nonhermitian
    wrong = sum(1 for x in values if x < error) / len(values)

    def clamp(m: Any) -> Any:
        return ar.do("where", m < error, ar.do("full_like", m, error * (fix_metric or 0.0)), m)

    w = tenet.apply_blocks(w, clamp)
    g = composed("ABx,xy->ABy", v, w)
    g = composed("ABy,aby->ABab", g, tenet.adjoint(v))
    return g, nonhermitian / scale, smin / scale, wrong


# -- the truncation -------------------------------------------------------------------


def _bond_matrix(t: SymmetricTensor) -> SymmetricTensor:
    """A bond matrix on one partition, so two of them can be subtracted.

    ``R0 @ R1`` and ``M0 @ M1`` are the same map, but a ``qr`` and an ``svd`` hand their
    bond legs back on different sides of the partition -- the same wire in the opposite
    spelling. [tenet.subtract][] compares legs and not wires, so both are put on the
    all-outputs partition before they meet.
    """
    return tenet.repartition(t, (0, 1), ())


def _apply(g: SymmetricTensor, x: SymmetricTensor) -> SymmetricTensor:
    """``g`` acting on a bond matrix: ``(bra0, bra1) <- (ket0, ket1)``."""
    return _bond_matrix(composed("ABab,ab->AB", g, x))


def _error2(g: SymmetricTensor, rr: SymmetricTensor, mm: SymmetricTensor, scale: float) -> float:
    """``<RR - MM|g|RR - MM> / <RR|g|RR>``, the squared relative truncation error."""
    delta = rr - _bond_matrix(mm)
    return abs(float(tenet.inner(delta, _apply(g, delta)))) / scale


def _svd_split(
    rr: SymmetricTensor, max_bond: int | None, cutoff: float | None
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """YASTN ``symmetrized_svd``: split a bond matrix, the singular values shared.

    ``max_bond=None`` and ``cutoff=None`` together mean *do not truncate*, which is the
    exact [svd][tenet.ops.linalg.svd] rather than a ``svd_truncated`` with no cap: it is
    the one setting under which a bond may be re-decided by nothing, and the oracles use
    it to check the gate itself against a dense contraction.
    """
    if max_bond is None and cutoff is None:
        u, s, vh = tenet.linalg.svd(rr, ((0,), (1,)))
    else:
        u, s, vh = tenet.linalg.svd_truncated(rr, ((0,), (1,)), max_bond=max_bond, cutoff=cutoff)
    sq = tenet.block_sqrt(s)
    return composed("ax,xy->ay", u, sq), composed("xy,yb->xb", sq, vh)


def _reciprocal(w: SymmetricTensor, cutoff: float, scale: float) -> SymmetricTensor:
    """``1/w`` where ``w > cutoff * scale``, zero elsewhere."""
    threshold = cutoff * scale

    def inv(m: Any) -> Any:
        keep = ar.do("greater", m, threshold)
        safe = ar.do("where", keep, m, ar.do("ones_like", m))
        return ar.do("where", keep, 1.0 / safe, ar.do("zeros_like", m))

    return tenet.apply_blocks(w, inv)


def _solve(
    n: SymmetricTensor, j: SymmetricTensor, error_fun: Any, cutoffs: Sequence[float]
) -> tuple[SymmetricTensor, float, float]:
    """``pinv(n) @ j`` with the cutoff chosen by the truncation error it produces.

    YASTN's ``optimal_pinv``: the normal-equation matrix is near-singular by
    construction -- a bond metric has a null space wherever the environment does -- so
    the inverse is a pseudo-inverse and *which* pseudo-inverse is decided by measuring,
    not by a fixed tolerance.
    """
    w, v = tenet.linalg.eigh(n, ((0, 1), (2, 3)))
    scale = max(abs(float(x)) for m in tenet.to_matrices(w).values() for x in ar.do("diag", m))
    rhs = composed("ijx,ij->x", tenet.adjoint(v), j)
    best: tuple[SymmetricTensor, float, float] | None = None
    for cutoff in cutoffs:
        y = composed("xy,y->x", _reciprocal(w, cutoff, scale), rhs)
        m = composed("ijx,x->ij", v, y)
        err = error_fun(m)
        if best is None or err < best[1]:
            best = (m, err, cutoff)
    assert best is not None
    return best


def _optimize(
    g: SymmetricTensor,
    rr: SymmetricTensor,
    m0: SymmetricTensor,
    m1: SymmetricTensor,
    scale: float,
    error2: float,
    max_iter: int,
    tol_iter: float,
    cutoffs: Sequence[float],
) -> tuple[SymmetricTensor, SymmetricTensor, float, int, float | None]:
    """Alternating least squares on ``(M0, M1)`` in the metric ``g``.

    YASTN's ``optimize_truncation``: fix one factor, solve the normal equations for the
    other, swap. Each solve is a pseudo-inverse whose cutoff is picked by
    [_solve][tenet.network.evolution._solve].
    """
    grr = _apply(g, rr)
    cutoff = None
    sweeps = 0
    for sweeps in range(1, max_iter + 1):  # noqa: B007 -- the count is the loop's output
        # fix M1, solve for M0
        t = composed("ABab,lb->ABal", g, m1)
        n0 = composed("kB,ABal->Akal", tenet.adjoint(m1), t)
        j0 = composed("kB,AB->Ak", tenet.adjoint(m1), grr)
        m0, error2, cutoff = _solve(
            n0,
            j0,
            lambda x, fixed=m1: _error2(g, rr, composed("ak,kb->ab", x, fixed), scale),
            cutoffs,
        )
        # fix M0, solve for M1
        t = composed("ABab,al->ABlb", g, m0)
        n1 = composed("Ak,ABlb->kBlb", tenet.adjoint(m0), t)
        j1 = composed("Ak,AB->kB", tenet.adjoint(m0), grr)
        m1, error2, cutoff = _solve(
            n1,
            j1,
            lambda x, fixed=m0: _error2(g, rr, composed("ak,kb->ab", fixed, x), scale),
            cutoffs,
        )
        if abs(error2) < tol_iter:
            break
    return m0, m1, error2, sweeps, cutoff


def truncate_(
    env: Any,
    bond: Any,
    *,
    gate: Gate | None = None,
    max_bond: int | None = None,
    cutoff: float | None = 1e-14,
    fix_metric: float | None = 0.0,
    max_iter: int = 20,
    tol_iter: float = 1e-13,
    pinv_cutoffs: Sequence[float] = PINV_CUTOFFS,
) -> Evolution_out:
    """Reduce one bond of ``env``'s state, in the metric ``env`` supplies. In place.

    Parameters
    ----------
    env : EnvCTM or EnvNTU
        The environment. Its ``psi`` is the state being evolved and is written back into;
        its ``bond_metric`` is what "best" means here.
    bond : Bond or tuple[Site, Site]
        The bond to truncate, either orientation.
    gate : Gate or None, optional
        A gate to apply first, enlarging the bond. ``None`` (the default) truncates the
        bond as it stands, which is YASTN's ``truncate_`` without a gate.
    max_bond : int or None, optional
        The bond-dimension cap. Default ``None``, i.e. no cap.
    cutoff : float or None, optional
        Relative singular-value cutoff of the initializing SVD. Default ``1e-14``.
    fix_metric : float or None, optional
        Replace every eigenvalue below the metric's own error scale (the non-Hermitian
        norm plus the most negative eigenvalue) by ``fix_metric`` times that scale.
        ``0.0`` is the default and ``1.0`` the other sensible value; ``None`` skips the
        eigendecomposition altogether and reports no eigenvalues.
    max_iter : int, optional
        Least-squares sweeps. Default ``20``; ``0`` keeps the SVD initialization.
    tol_iter : float, optional
        Stop once the squared error falls below this. Default ``1e-13``.
    pinv_cutoffs : Sequence of float, optional
        The pseudo-inverse ladder each solve chooses from. Default
        ``PINV_CUTOFFS``.

    Returns
    -------
    Evolution_out
        The bond, the truncation error and what the metric was found to be.

    Raises
    ------
    ValueError
        If the two sites are not nearest neighbours (from
        [nn_bond_dirn][tenet.network.SquareLattice.nn_bond_dirn]).

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import EnvNTU, Peps, SquareLattice, gate_nn, local_op, truncate_
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1})
    >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(phys, OUT))
    >>> psi = Peps(SquareLattice(dims=(2, 2)), SymmetricTensor.random(legs, seed=0))
    >>> env = EnvNTU(psi)
    >>> sz = np.diag([-0.5, 0.5])
    >>> h = local_op(np.kron(sz, sz), phys=phys)
    >>> g = gate_nn(h, 0.05, psi.bonds()[0])
    >>> out = truncate_(env, g.bond, gate=g, max_bond=2)
    >>> out.truncation_error < 1e-8
    True

    Notes
    -----
    The shape is YASTN's: ``qr`` each site into the isometry that stays and the small
    factor that moves, take the metric on the two reduced legs, truncate the product of
    the factors, and put the survivors back. The isometries are what makes the metric a
    metric on a *small* space -- ``D`` by ``D`` rather than the whole site.
    """
    psi = getattr(env.psi, "ket", env.psi)
    dirn = psi.nn_bond_dirn(*bond)
    if dirn in ("rl", "bt"):
        bond, dirn = (bond[1], bond[0]), dirn[::-1]
    s0, s1 = Site(*bond[0]), Site(*bond[1])
    a0, a1 = psi[s0], psi[s1]
    if gate is not None:
        a0, a1 = apply_gate(a0, a1, gate)
    q0, r0 = _split(a0, dirn, 0)
    q1, r1 = _split(a1, dirn, 1)
    rr = composed("Axc,Bxc->AB", r0, r1) if a0.ndim == 6 else composed("Ax,Bx->AB", r0, r1)
    rr = _bond_matrix(rr)

    g, nonherm, smin, wrong = _fix_metric(env.bond_metric(q0, q1, s0, s1, dirn), fix_metric)
    scale = abs(float(tenet.inner(rr, _apply(g, rr))))
    m0, m1 = _svd_split(rr, max_bond, cutoff)
    iterations, pinv = 0, None
    if scale == 0.0:
        # ``fix_metric`` replaced every eigenvalue, which happens when the metric's error
        # scale reaches its largest eigenvalue: the environment measured nothing about
        # this bond. The plain SVD stands and the error is reported as ``nan`` rather than
        # as a zero that would read as "lossless".
        error2 = float("nan")
    else:
        error2 = _error2(g, rr, composed("ak,kb->ab", m0, m1), scale)
        if max_iter > 0 and error2 > tol_iter:
            m0, m1, error2, iterations, pinv = _optimize(
                g, rr, m0, m1, scale, error2, max_iter, tol_iter, pinv_cutoffs
            )
    m0, m1 = _rebalance(m0, m1)

    if dirn == "lr":
        psi[s0] = composed("tlbAs,Ak->tlbks", q0, m0)
        psi[s1] = composed("tBbrs,kB->tkbrs", q1, m1)
    else:
        psi[s0] = composed("tlArs,Ak->tlkrs", q0, m0)
        psi[s1] = composed("Blbrs,kB->klbrs", q1, m1)
    return Evolution_out(
        bond=Bond(s0, s1),
        truncation_error=error2**0.5 if error2 != error2 else max(error2, 0.0) ** 0.5,
        nonhermitian_part=nonherm,
        min_eigenvalue=smin,
        wrong_eigenvalues=wrong,
        iterations=iterations,
        pinv_cutoff=pinv,
    )


def _rebalance(m0: SymmetricTensor, m1: SymmetricTensor) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Share the bond's weight between the two factors and normalize it.

    YASTN's closing ``symmetrized_svd(..., normalize=True)``: the least-squares solve
    leaves the scale on whichever factor it happened to touch last, and an evolution
    that keeps it there loses digits within a few steps.
    """
    q0, x0 = tenet.linalg.qr(m0, ((0,), (1,)))
    x1, q1 = tenet.linalg.lq(m1, ((0,), (1,)))
    u, s, vh = tenet.linalg.svd(composed("xk,ky->xy", x0, x1), ((0,), (1,)))
    sq = tenet.block_sqrt(s / tenet.norm(s))
    left = composed("ax,xw->aw", q0, composed("xz,zw->xw", u, sq))
    right = composed("wz,zb->wb", sq, composed("zy,yb->zb", vh, q1))
    return left, right


def evolution_step_(env: Any, gates: Iterable[Gate], **kwargs: Any) -> list[Evolution_out]:
    """Apply every gate to ``env``'s state, truncating after each one. In place.

    Parameters
    ----------
    env : EnvCTM or EnvNTU
        The environment; its ``psi`` is evolved in place.
    gates : Iterable of Gate
        The Trotter gates, in the order to apply them --
        [gates_nn][tenet.network.gates_nn] builds the homogeneous list.
    **kwargs
        Passed to [truncate_][tenet.network.truncate_]: ``max_bond``, ``cutoff``,
        ``fix_metric``, ``max_iter``, ``tol_iter``, ``pinv_cutoffs``.

    Returns
    -------
    list[Evolution_out]
        One record per gate, in the order applied.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import EnvNTU, Peps, SquareLattice, evolution_step_, gates_nn, local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1})
    >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(phys, OUT))
    >>> psi = Peps(SquareLattice(dims=(2, 2)), SymmetricTensor.random(legs, seed=0))
    >>> sz = np.diag([-0.5, 0.5])
    >>> h = local_op(np.kron(sz, sz), phys=phys)
    >>> infos = evolution_step_(EnvNTU(psi), gates_nn(psi.geometry, h, 0.1), max_bond=2)
    >>> len(infos)
    16

    Notes
    -----
    YASTN's ``evolution_step_``, minus the patch mechanism (a provisional per-site update
    that lets an ``EnvCTM`` postpone rebuilding its corners) and minus multi-site gates.
    An ``EnvCTM`` here is *not* re-converged between gates: the caller decides how often
    to call ``update_``, which is the same choice YASTN's ``post_truncation_`` makes for
    it and one this layer has no policy about yet.
    """
    return [truncate_(env, gate.bond, gate=gate, **kwargs) for gate in gates]


def accumulated_truncation_error(
    infoss: Sequence[Sequence[Evolution_out]], statistics: str = "mean"
) -> float:
    """The truncation error accumulated over a sequence of evolution steps.

    Parameters
    ----------
    infoss : Sequence of Sequence of Evolution_out
        One [evolution_step_][tenet.network.evolution_step_] output per step.
    statistics : str, optional
        ``'mean'`` (the default) or ``'max'`` over the bonds of a step.

    Returns
    -------
    float
        ``sum_steps statistics_bond [ sum_gates truncation_error ]``.

    Raises
    ------
    ValueError
        If ``statistics`` is neither ``'mean'`` nor ``'max'``.

    Examples
    --------
    >>> from tenet.network import Evolution_out, accumulated_truncation_error
    >>> from tenet.network import Bond, Site
    >>> b = Bond(Site(0, 0), Site(0, 1))
    >>> step = [Evolution_out(bond=b, truncation_error=0.1)] * 2
    >>> accumulated_truncation_error([step, step])
    0.4

    Notes
    -----
    YASTN's ``accumulated_truncation_error`` verbatim. It is an *estimate*: the errors
    are measured in different metrics at different steps and adding them assumes they do
    not cancel, which is the conservative direction.
    """
    if statistics not in ("mean", "max"):
        raise ValueError(f"accumulated_truncation_error: {statistics=} should be 'mean' or 'max'")
    total = 0.0
    for infos in infoss:
        per_bond: dict[Any, float] = {}
        for info in infos:
            per_bond[info.bond] = per_bond.get(info.bond, 0.0) + info.truncation_error
        values = list(per_bond.values())
        if values:
            total += max(values) if statistics == "max" else sum(values) / len(values)
    return total


# -- the cheap environment: NTU ------------------------------------------------------

#: The ``Peps`` axis each direction names, the step that walks it, and the leg it meets.
_AXIS = {"t": 0, "l": 1, "b": 2, "r": 3}
_SHIFT = {"t": (-1, 0), "l": (0, -1), "b": (1, 0), "r": (0, 1)}
_OPPOSITE = {"t": "b", "l": "r", "b": "t", "r": "l"}


def _unit(leg: Any) -> Any:
    """The one-dimensional space carrying only the leg's provider's unit sector."""
    provider = leg.space.provider
    return tenet.GradedSpace.new(provider, {provider.unit: 1})


#: Per hair, the equation that closes the neighbour's other three virtual legs and its
#: physical one, leaving the pair that faces the bond, and whether the **ket** leads.
#: ``composed`` derives the bends, and for ``'t'`` and ``'r'`` it also decides the order:
#: three of the four closed wires take their ``IN`` end from the bra, so the bra leads and
#: one wire turns. ``'l'`` and ``'b'`` close two ket-``IN`` wires against two bra-``IN``
#: ones and **tie at two bends either way** -- the same tie M79a recorded for ``edge_t``
#: and ``edge_r`` -- and there the ket leads. That is not a preference: it is what the
#: dense oracle in ``tests/network/test_evolution.py`` measures, and the other reading
#: gives a different metric under fermionic parity while agreeing on every grading that
#: does not braid.
_HAIR = {
    "t": ("Tlbrs,tlbrs->tT", False),
    "r": ("tlbRs,tlbrs->rR", False),
    "l": ("tlbrs,tLbrs->lL", True),
    "b": ("tlbrs,tlBrs->bB", True),
}

#: Per hair direction, the axis of the ket that the hair is contracted into and the
#: equation that does it.
_HAIR_ONTO = {
    "t": "pq,plbrs->qlbrs",
    "l": "pq,tpbrs->tqbrs",
    "b": "pq,tlprs->tlqrs",
    "r": "pq,tlbps->tlbqs",
}


class EnvNTU:
    """The neighbourhood-tensor-update environment: a bond metric from the local cluster.

    Parameters
    ----------
    psi : Peps
        The state being evolved. Held, not copied: ``truncate_`` writes into it.
    which : str, optional
        The cluster. ``'NN'`` (the default) is the only one transcribed -- the six sites
        around the bond, exactly, with the boundary closed by rank-1 hairs.

    Raises
    ------
    ValueError
        If ``which`` is not ``'NN'``, or if ``psi`` has no physical leg.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import EnvNTU, Peps, SquareLattice
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(V, OUT))
    >>> env = EnvNTU(Peps(SquareLattice(dims=(2, 2)), SymmetricTensor.random(legs, seed=0)))
    >>> env.which
    'NN'

    Notes
    -----
    The cluster is::

            (-1 +0)==(-1 +1)
               |        |
        (+0 -1)==Q0== ==Q1==(+0 +2)
               |        |
            (+1 +0)==(+1 +1)

    for a horizontal bond and its transpose for a vertical one. The four corner sites
    enter through M79a's ``cor_*`` and the two ends through ``edge_l``/``edge_r`` with a
    *hair* -- the far neighbour closed on its other three virtual legs -- which is
    YASTN's ``hair_l``/``hair_r``. Every contraction is exact, so the metric is
    Hermitian and positive up to floating point; ``truncate_`` measures that rather than
    relying on it.

    Simplification: the larger clusters are not here. ``'NN+'`` and ``'NN++'`` add rings
    approximated by rank-1 SVDs of the boundary, ``'NNN'`` adds the four diagonal sites
    exactly, and the ladder mode is a different geometry again; each is a different
    contraction of the same primitives, and the ``'NN'`` cluster is the one that makes
    the metric-vs-CTM comparison in ``tests/network/test_evolution.py``.
    """

    def __init__(self, psi: Peps, which: str = "NN") -> None:
        if which != "NN":
            raise ValueError(f"EnvNTU: which={which!r}; only 'NN' is implemented")
        if not psi.has_physical():
            raise ValueError("EnvNTU: the state needs a physical leg (rank-5 tensors)")
        self.psi = psi
        self.which = which
        self.geometry = psi.geometry

    def __repr__(self) -> str:
        return f"EnvNTU(geometry={self.geometry!r}, which={self.which!r})"

    def _at(self, site: Any, d: tuple[int, int], facing: str) -> DoubleLayer:
        """The neighbour at the shift ``d``, or the one-dimensional stand-in for an edge.

        On an open boundary the missing neighbour is replaced by YASTN's
        ``trivial_peps_tensor``: a tensor of ones whose ``facing`` leg meets the leg it
        would have met -- one-dimensional, because that is what an ``obc`` boundary leg
        is -- and whose other four legs are the trivial sector.
        """
        neighbour = self.geometry.nn_site(site, d)
        if neighbour is not None:
            ket = self.psi[neighbour]
            return DoubleLayer(ket, tenet.adjoint(ket))
        axis = _AXIS[facing]
        # the site the stand-in faces: one step past the missing position
        step = _SHIFT[facing]
        ref = (site[0] + d[0] + step[0], site[1] + d[1] + step[1])
        met = self.psi[ref].legs[_AXIS[_OPPOSITE[facing]]]
        legs = [tenet.Leg(_unit(met), s) for s in (IN, OUT, OUT, IN, OUT)]
        legs[axis] = tenet.Leg(met.space, IN if met.side is OUT else OUT, dual=met.dual)
        ket = ones(tuple(legs))
        return DoubleLayer(ket, tenet.adjoint(ket))

    def _haired(self, a: DoubleLayer, d: str, neighbour: DoubleLayer) -> DoubleLayer:
        """``a`` with the neighbour's hair contracted into the ket's ``d`` leg."""
        equation, ket_first = _HAIR[_OPPOSITE[d]]
        first, second = (
            (neighbour.ket, neighbour.bra) if ket_first else (neighbour.bra, neighbour.ket)
        )
        hair = composed(equation, first, second)
        return DoubleLayer(composed(_HAIR_ONTO[d], hair, a.ket), a.bra)

    def bond_metric(
        self, q0: SymmetricTensor, q1: SymmetricTensor, s0: Any, s1: Any, dirn: str
    ) -> SymmetricTensor:
        """The ``'NN'`` cluster closed around the bond. YASTN ``EnvNTU._g_NN``.

        Parameters
        ----------
        q0, q1 : SymmetricTensor
            The two reduced site tensors, rank 5, whose bond legs the metric is on.
        s0, s1 : Site
            Their sites, in the fermionic order.
        dirn : str
            ``'lr'`` or ``'tb'``.

        Returns
        -------
        SymmetricTensor
            Rank 4, ``(bra0, bra1, ket0, ket1)``.

        Examples
        --------
        >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
        >>> from tenet.network import EnvNTU, Peps, SquareLattice
        >>> from tenet.symmetry import U1, U1Sector
        >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
        >>> legs = (Leg(V, IN), Leg(V, OUT), Leg(V, OUT), Leg(V, IN), Leg(V, OUT))
        >>> psi = Peps(SquareLattice(dims=(2, 2)), SymmetricTensor.random(legs, seed=0))
        >>> env = EnvNTU(psi)
        >>> g = env.bond_metric(psi[0, 0], psi[0, 1], (0, 0), (0, 1), "lr")
        >>> g.ndim
        4
        """
        a0 = DoubleLayer(q0, tenet.adjoint(q0))
        a1 = DoubleLayer(q1, tenet.adjoint(q1))
        if dirn == "lr":
            left = edge_l(self._haired(a0, "l", self._at(s0, (0, -1), "r")))
            right = edge_r(self._haired(a1, "r", self._at(s0, (0, 2), "l")))
            ctl = cor_tl(self._at(s0, (-1, 0), "b"))
            ctr = cor_tr(self._at(s0, (-1, 1), "b"))
            cbr = cor_br(self._at(s0, (1, 1), "t"))
            cbl = cor_bl(self._at(s0, (1, 0), "t"))
            lower = composed("tTlL,lLuU->tTuU", cbr, cbl)
            lower = composed("tTuU,uUrRvV->tTrRvV", lower, left)
            upper = composed("bBrR,rRcC->bBcC", ctl, ctr)
            upper = composed("bBcC,cCmMdD->bBmMdD", upper, right)
            g = composed("tTrRvV,vVmMtT->rRmM", lower, upper)
        else:
            top = edge_t(self._haired(a0, "t", self._at(s0, (-1, 0), "b")))
            bottom = edge_b(self._haired(a1, "b", self._at(s0, (2, 0), "t")))
            cbl = cor_bl(self._at(s0, (1, -1), "r"))
            ctl = cor_tl(self._at(s0, (0, -1), "r"))
            ctr = cor_tr(self._at(s0, (0, 1), "l"))
            cbr = cor_br(self._at(s0, (1, 1), "l"))
            leftc = composed("rRtT,tTcC->rRcC", cbl, ctl)
            leftc = composed("rRcC,cCvVdD->rRvVdD", leftc, top)
            rightc = composed("lLbB,bBmM->lLmM", ctr, cbr)
            rightc = composed("lLmM,mMuUdD->lLuUdD", rightc, bottom)
            g = composed("rRvVdD,dDuUrR->vVuU", leftc, rightc)
        return tenet.transpose(g, (1, 3, 0, 2))
