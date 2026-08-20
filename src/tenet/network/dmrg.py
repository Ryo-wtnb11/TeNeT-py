"""The driver: a Krylov step, a two-site sweep, and the loop that repeats it.

Promoted from ``examples/toy_codes/dmrg.py`` (#110) with no arithmetic change: ``lanczos``
:393-437, ``sweep_`` :443-484, ``_schmidt_change`` :487-503, ``DMRG_out`` :506-520 and
``dmrg_`` :524-557. Function-shaped drivers, YASTN's decomposition (``_dmrg.py``:42-128),
not TenPy's ``Sweep``/``EffectiveH`` hierarchy -- M11a ships **one** sweep, and a base
class with one subclass is the interface-with-one-implementation the repo's own rules
forbid.

[lanczos][tenet.network.lanczos] lives here rather than in ``tenet.linalg`` for a sharp reason:
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

    Attributes
    ----------
    chi : int, optional
        The bond-dimension cap handed to
        [svd_truncated][tenet.ops.linalg.svd_truncated] at every bond of the
        sweep. Default ``64``.
    cutoff : float, optional
        The singular-value cutoff handed to the same SVD. Default ``1e-14``.
    noise : float, optional
        Relative strength of the perturbation [sweep_][tenet.network.sweep_]
        mixes in at each split. Default ``0.0`` (no perturbation, no random
        number drawn, and the plain SVD split).
    noise_type : str, optional
        Which perturbation ``noise`` means -- ``"wavefunction"`` (the default)
        or ``"perturbative"``. It also decides the **decimation**: see
        [sweep_][tenet.network.sweep_]'s table. Ignored when ``noise`` is
        ``0.0``.

    Notes
    -----
    The defaults equal [dmrg_][tenet.network.dmrg_]'s flat defaults, so ``Sweep()`` is
    today's sweep and ``schedule=[Sweep()]`` is the flat run.

    One record per sweep rather than parallel per-knob lists (block2's ``bond_dims`` /
    ``noises`` / ``thrds``), so a wrong-length list is impossible to write. The loop
    tolerances (``energy_tol``, ``schmidt_tol``, ``max_sweeps``, ``ncv``) are properties
    of [dmrg_][tenet.network.dmrg_]'s loop, not of a sweep, and stay flat kwargs there.
    """

    chi: int = 64
    cutoff: float = 1e-14
    noise: float = 0.0
    noise_type: str = "wavefunction"


def _dot(a: SymmetricTensor, b: SymmetricTensor) -> float:
    """``<a|b>`` as the Hilbert-Schmidt pairing, through ``compose``/``full_trace``.

    Not [tenet.inner][], and the difference is measurable rather than stylistic: on a
    **graded** provider ``inner(t, t)`` and ``norm(t)**2`` disagree for a two-site tensor
    whose left bond carries an odd sector -- ``inner`` leaves axis 0 open and closes it
    with ``full_trace``, which puts a twist on that wire. A projector has to be built
    from the pairing the *state* is normalized in, which is ``norm``'s, so this spells
    that one directly: one map composed with another's adjoint, closed by the qdim-weighted
    trace -- the same two primitives ``_rho`` uses, for the same reason (no operand order
    is left to state). It agrees with ``inner`` on every ungraded provider.
    """
    m = tenet.repartition(a, (0, 1), (2, 3))
    return float(
        tenet.full_trace(tenet.compose(tenet.adjoint(m), tenet.repartition(b, (0, 1), (2, 3))))
    )


def _orthonormal(
    vectors: Sequence[SymmetricTensor],
) -> tuple[tuple[SymmetricTensor, float], ...]:
    """Gram-Schmidt the given vectors against each other, dropping the null ones.

    block2's Davidson does the same to its ``ors`` before it projects anything with them
    (``iterative_matrix_functions.hpp``:1219-1225): the projector ``1 - sum_k |p_k><p_k|``
    is only that projector on an orthonormal set. A vector that comes back numerically
    zero is dropped rather than divided by -- two converged states in *different* charge
    sectors give an identically empty projection vector, which is structurally right and
    would otherwise be a division by zero.
    """
    basis: list[tuple[SymmetricTensor, float]] = []
    for t in vectors:
        t = _project_out(t, basis)
        gram = _dot(t, t)
        if abs(gram) > 1e-24 and float(tenet.norm(t)) > 1e-12:
            basis.append((t, gram))
    return tuple(basis)


def _project_out(
    t: SymmetricTensor, basis: Sequence[tuple[SymmetricTensor, float]]
) -> SymmetricTensor:
    """``t - sum_k b_k <b_k,t>/<b_k,b_k>`` in [tenet.inner][]'s own pairing.

    Divided by ``<b_k, b_k>`` rather than pre-normalized by
    [tenet.norm][]: the two are the same number on an ungraded provider and are
    **not** on a graded one, where ``inner`` carries the string the sweep's own
    solve carries -- ``lanczos`` builds its tridiagonal from ``inner`` and
    ``Env.heff2`` returns its image in the same pairing, so the eigenproblem is
    self-consistent in that pairing and the projector has to be too. Mixing the two
    (normalizing with ``norm`` and projecting with ``inner``) makes the projector
    non-idempotent exactly on the odd-parity bonds.
    """
    for b, gram in basis:
        t = tenet.subtract(t, b * (_dot(b, t) / gram))
    return t


def lanczos(
    matvec: Callable[[SymmetricTensor], SymmetricTensor],
    v: SymmetricTensor,
    *,
    ncv: int = 3,
    tol: float = 1e-13,
    orthogonal_to: Sequence[SymmetricTensor] = (),
) -> tuple[float, SymmetricTensor]:
    """Ground eigenpair ``(value, vector)`` of a Hermitian ``matvec`` over SymmetricTensors.

    Parameters
    ----------
    matvec : Callable[[SymmetricTensor], SymmetricTensor]
        The Hermitian operator, as a function applying it to one vector.
    v : SymmetricTensor
        The starting vector; any tensor with ``matvec``'s input structure.
    ncv : int, optional
        Krylov-space dimension. Default ``3``, and meant to stay small (see
        Notes). Keyword-only.
    tol : float, optional
        The happy-breakdown threshold on the recurrence norm ``beta``.
        Default ``1e-13``. Keyword-only.
    orthogonal_to : Sequence of SymmetricTensor, optional
        Vectors on ``v``'s structure to hold the solve orthogonal to. Default
        ``()``, which projects nothing, allocates nothing and leaves the
        recurrence exactly as it was. Keyword-only.

    Returns
    -------
    value : float
        The smallest ('SR') Ritz value.
    vector : SymmetricTensor
        The matching normalized Ritz vector, on ``v``'s structure.

    Raises
    ------
    ValueError
        If ``orthogonal_to`` spans the whole space ``v`` lives in, so that the
        projected start vector is numerically zero.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import lanczos
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> v = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> value, vector = lanczos(lambda t: t * 2.0, v)  # matvec = 2 * identity
    >>> round(value, 6)
    2.0

    Notes
    -----
    YASTN's three-term recurrence (``yastn/tensor/_krylov.py``:34-42) and its happy
    breakdown (``H[(j+1,j)] < tol`` -> stop and drop the row, :39-43), then ``eigh`` of
    the ``(m, m)`` tridiagonal and one recombination (``yastn/krylov/_krylov.py``:226-239,
    a single iteration with no restart). ``hermitian=True, ncv=3, which='SR'`` are YASTN's
    own DMRG defaults (``_dmrg.py``:151-152) and are not knobs this layer tunes.

    The only tensor operations are ``tenet.add``/``subtract``, scalar multiply/divide,
    ``tenet.norm`` and [tenet.inner][] -- a Krylov solver needs a vector
    space and nothing else, and a ``SymmetricTensor`` is one.

    This is an inner solver inside an outer sweep, not a standalone eigensolver: the
    recurrence is **not reorthogonalized**, so ``ncv`` is meant to stay small.

    **``orthogonal_to`` is hard projection, not a level shift**, and both are block2's:
    with ``ors`` and no weights the basis vectors are projected by ``1 - |v><v|``
    (``iterative_matrix_functions.hpp``:1198-1200, :1226-1237), while a non-empty
    ``projection_weights`` instead replaces ``H`` by ``H + sum_k w_k |v_k><v_k|`` (:1201-1204,
    :1250-1253) -- the level-shift approach its own documentation names as such
    (``docs/source/user/keywords.rst``, ``proj_mps_tags``), and which it warns reports
    unphysical eigenvalues ``E_k + w_k`` when a weight is smaller than the gap. Hard
    projection is what ``statespecific`` alone does, it has no parameter to get wrong, and
    the eigenvalue it returns is the projected operator's own -- so it is what is adopted
    here and no weight argument exists.

    The projector is applied to the start vector and to every ``matvec`` result, which is
    plain Lanczos on ``P H P`` restricted to ``range(P)``: the recurrence stays a valid
    three-term one for a Hermitian operator, rather than a perturbed one for ``H``.
    """
    # Simplification: no reorthogonalization, and neither has YASTN. At ``ncv=3`` the
    # recurrence has not had time to lose orthogonality, and the vector is reseeded from the
    # current MPS at every bond. Ceiling: raise ``ncv`` past ~10 and full reorthogonalization
    # against the stored ``V`` becomes the two-line addition.
    # Simplification: numpy ``eigh`` on the ``(3, 3)`` tridiagonal, not
    # ``tenet.linalg.eigh``. The projected matrix has no symmetry structure -- 9 floats.
    basis = _orthonormal(orthogonal_to)
    if basis:
        v = _project_out(v, basis)
        if float(tenet.norm(v)) < 1e-12:
            raise ValueError(
                "the start vector lies in the span of orthogonal_to, so there is no "
                "vector left to iterate on at this bond; the two-site space is too "
                "small to hold another state (block2 asserts on the same condition)"
            )
        full = matvec

        def matvec(x: SymmetricTensor) -> SymmetricTensor:
            return _project_out(full(x), basis)

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


#: The two spellings of ``noise``, and the decimation each one implies. Stated once,
#: read by [Sweep][tenet.network.Sweep], [sweep_][tenet.network.sweep_] and the refusal.
_NOISE_TYPES = ("wavefunction", "perturbative")


def _pair(left: SymmetricTensor, right: SymmetricTensor) -> SymmetricTensor:
    """Merge two adjacent site tensors into one ``(l, p, q, r)`` two-site tensor.

    One spelling for both chains a projected sweep holds: the state being optimized and
    each converged state whose reduced form [Env.project2][tenet.network.Env.project2]
    then carries into the sweeping state's gauge.
    """
    return tenet.einsum("apx,xqr->apqr", left, right)


def _rho(m: SymmetricTensor, forward: bool) -> SymmetricTensor:
    """``m m^dag`` (forward) or ``m^dag m`` (backward), for ``m`` on ``(l, p | q, r)``.

    Composition, not ``einsum``: a density matrix *is* a map composed with its adjoint,
    both directions, so the shared wires are the composed side by construction and there
    is no operand order to state. The trace is over the legs the composition consumes --
    the right pair going forward, the left pair coming back.
    """
    return tenet.compose(m, tenet.adjoint(m)) if forward else tenet.compose(tenet.adjoint(m), m)


def _perturbations(
    env: Env, n: int, aa: SymmetricTensor, noise: float
) -> tuple[SymmetricTensor, ...]:
    """block2's perturbation vectors: the operator's term families on ``aa``, scaled.

    [Env.heff2_families][tenet.network.Env.heff2_families] resolves ``H_eff aa`` by term
    family -- tenet's analogue of block2's per-sub-label resolution
    (``effective_hamiltonian.hpp``:263-360) -- and this scales them the way
    ``scale_perturbative_noise`` does (``moving_environment.hpp``:3698-3713): each vector
    is first normalized to unit norm, then the whole collection is scaled so its total
    squared norm is ``noise``. A unit-norm ``aa`` contributes 1 to the density matrix's
    trace and the perturbation contributes ``noise``, which is what keeps ``noise``
    dimensionless and block2's 1e-4..1e-5 range meaningful. A family that comes back
    numerically zero is dropped rather than divided by, and drops out of the count.
    """
    parts = [(p, float(tenet.norm(p))) for p in env.heff2_families(n, aa)]
    live = [(p, w) for p, w in parts if w > 0.0]
    if not live:
        return ()
    scale = (noise / len(live)) ** 0.5
    return tuple(p * (scale / w) for p, w in live)


def _split_dm(
    aa: SymmetricTensor,
    forward: bool,
    *,
    chi: int,
    cutoff: float,
    perturbations: tuple[SymmetricTensor, ...] = (),
) -> tuple[SymmetricTensor, SymmetricTensor, SymmetricTensor]:
    """block2's default decimation: ``eigh`` of ``rho = tr aa aa^dag``, perturbation folded in.

    Returns ``(left, right, s)`` on the truncated bond, ``left`` for site ``n`` and
    ``right`` for site ``n + 1``: going forward ``left`` is the isometry and ``right``
    carries the weight, coming back the mirror. ``s`` is the singular-value tensor
    [svd_truncated][tenet.ops.linalg.svd_truncated] would have returned.

    The truncation is the existing one, not a second one:
    ``svd_truncated`` of the (Hermitian, positive) ``rho`` selects on ``sigma**2``, and
    ``cutoff_mode="rsum1"`` on that spectrum is the same rule as the default
    ``"rsum2"`` on ``aa``'s -- both drop the largest set whose ``qdim``-weighted
    ``sum sigma**2`` stays under ``cutoff`` times the total. ``max_bond`` walks the same
    order, because squaring is monotone. So the kept bond space is the SVD split's, and
    the agreement is by construction rather than by luck.

    ``rho`` is formed on the *bent* partition ``(l, p | q, r)`` -- the one
    ``svd_truncated`` lowers to anyway -- and every contraction here is a
    [tenet.compose][], so no wire's direction is left to operand order.
    """
    m = tenet.repartition(aa, (0, 1), (2, 3))
    rho = _rho(m, forward)
    for p in perturbations:
        rho = tenet.add(rho, _rho(tenet.repartition(p, (0, 1), (2, 3)), forward))
    v, w, _ = tenet.linalg.svd_truncated(rho, max_bond=chi, cutoff=cutoff, cutoff_mode="rsum1")
    s = tenet.block_sqrt(w)  # rho's spectrum is aa's, squared
    if forward:
        return v, tenet.compose(tenet.adjoint(v), m), s
    # ``adjoint`` keeps ``v``'s public axis order, so the new bond comes back *last*;
    # ``svd_truncated``'s ``vh`` spells the same map with the bond first, which is the
    # order [MPS.__setitem__][tenet.network.MPS.__setitem__] and the sweep's next merge
    # read. One ``transpose`` says so, and pays whatever braid the provider charges.
    return tenet.compose(m, v), tenet.transpose(tenet.adjoint(v), (2, 0, 1)), s


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
    noise_type: str = "wavefunction",
    orthogonal_to: Sequence[Env] = (),
    seed: int = 0,
) -> tuple[float, float]:
    """One left-to-right then right-to-left two-site sweep. ``psi`` and ``env`` mutate.

    Parameters
    ----------
    psi : MPS
        The state, mutated in place; expected mixed-canonical the way
        [dmrg_][tenet.network.dmrg_] prepares it.
    h : MPO
        The Hamiltonian.
    env : Env
        The environment cache for ``(psi, h)``, mutated in place.
    schmidt : dict[int, list[float]]
        Per-bond Schmidt spectra, updated in place -- the second convergence
        criterion's input.
    chi : int
        The bond-dimension cap handed to
        [svd_truncated][tenet.ops.linalg.svd_truncated] at every bond.
        Keyword-only.
    cutoff : float
        The singular-value cutoff handed to the same SVD. Keyword-only.
    ncv : int, optional
        Krylov-space dimension for [lanczos][tenet.network.lanczos].
        Default ``3``. Keyword-only.
    noise : float, optional
        Relative strength of the perturbation mixed in after the eigensolver
        and before each split; ``0.0`` (the default) draws no random number,
        builds no density matrix, and the sweep is bit-identical to a sweep
        without the keyword. Keyword-only.
    noise_type : {"wavefunction", "perturbative"}, optional
        Which perturbation ``noise`` means, and therefore which decimation the
        sweep runs -- the table in Notes. Default ``"wavefunction"``, today's
        behaviour. Read only when ``noise`` is nonzero. Keyword-only.
    orthogonal_to : Sequence of Env, optional
        Two-state environments, one per converged state to hold ``psi``
        orthogonal to -- each an ``Env(phi, MPO.identity(...), bra=psi)``, built
        and handed over the way ``env`` is, and **mutated in place** alongside
        it. Default ``()``, which projects nothing.
        [dmrg_][tenet.network.dmrg_] builds them from a list of
        [MPS][tenet.network.MPS] and is the spelling a caller wants.
        Keyword-only.
    seed : int, optional
        Makes the noise draw at bond ``n`` reproducible as ``seed + n``.
        Unused by ``noise_type="perturbative"``, which draws nothing.
        Default ``0``. Keyword-only.

    Returns
    -------
    energy : float
        The last ``lanczos`` Ritz value of the sweep.
    max_discarded_weight : float
        The **maximum** per-bond discarded weight (see Notes for why the
        maximum rather than the total).

    Raises
    ------
    ValueError
        If ``noise_type`` is neither ``"wavefunction"`` nor ``"perturbative"``
        -- a typo that silently ran the default would misreport the mixer.

    Notes
    -----
    **Which decimation runs is decided by the two keywords and by nothing else.** No
    bond width, no ``chi``, no runtime probe (#218):

    ======================================== ==================================
    ``(noise, noise_type)``                  the split
    ======================================== ==================================
    ``noise == 0.0``, any ``noise_type``     ``svd_truncated`` of ``aa``
    ``> 0``, ``"wavefunction"``              ``svd_truncated`` of a perturbed ``aa``
    ``> 0``, ``"perturbative"``              ``eigh`` of a perturbed ``rho``
    ======================================== ==================================

    A caller reads the rule off the [Sweep][tenet.network.Sweep] entry: the density-matrix
    split engages exactly when perturbative noise is asked for, and a noiseless sweep --
    including the cooling tail of a ramp, and every sweep of a run that never mentions
    noise -- takes the SVD split. That is deliberate rather than a default falling out:
    squaring the two-site tensor into ``rho`` resolves a singular value ``sigma`` through
    ``sigma**2``, so the split's own accuracy floor moves from machine epsilon to its
    square root, and a converged noiseless sweep is exactly where that costs something.
    block2 makes the same pairing in the other direction -- its wavefunction noise exists
    only on its SVD branch (``sweep_algorithm.hpp``:964-978) and its density-matrix branch
    (:930-953) is where the perturbative noise goes.

    YASTN's ``_dmrg_sweep_2site_`` (``_dmrg.py``:222-249) and its
    ``(('last', 0), ('first', 1))`` two-direction loop, five steps per bond: merge,
    ``eigs``, split, ``clear_site_``, ``update_env_``.

    ``svd_truncated`` decides the bond [GradedSpace][tenet.GradedSpace] here, every bond and
    every sweep, and the discarded weight is Pythagoras exactly as its docstring
    prescribes: ``U S Vh`` is isometric on both sides, so ``norm(U S Vh) = norm(S)`` and
    the dropped fraction of the (unit-norm) two-site tensor is ``1 - norm(S)**2``.

    ``vh`` comes back on the *map*'s partition and is stored straight into ``psi``: the
    ``MPS.__setitem__`` write barrier is what puts it back on ``(l, p | r)``, which is
    why no caller in this package ever spells a ``repartition``.

    The discarded weight here is the **maximum** over bonds, because it feeds a per-sweep
    convergence report where the worst bond is the diagnostic;
    [MPS.compress_][tenet.network.MPS.compress_] returns the **total** instead, because
    its caller is asking how much of the state was thrown away. Two conventions, two
    names.

    **Wavefunction noise** (``noise > 0``, ``noise_type="wavefunction"``): a random
    symmetric tensor over the two-site tensor's own legs is
    added after the eigensolver and before the split, at relative strength ``noise``
    (block2's ``NoiseTypes::Wavefunction`` scaling, ``operator_functions.hpp``:777-815, so
    ``noise`` is dimensionless and block2's 1e-4..1e-5 range transfers), and the two-site
    tensor is renormalized so the Pythagoras discarded weight stays a fraction of a
    unit-norm tensor. The perturbation fills every **structurally allowed** coupled sector
    of the ``(l, p | q, r)`` map -- including the ones the eigensolver left numerically
    empty and which ``svd_truncated`` therefore omits from the bond, which is the local
    minimum a symmetric DMRG falls into: a sector that is zero stays zero forever, because
    nothing else in the sweep can create it. It cannot reach outside ``bond_l (x) phys``
    -- no wavefunction noise can, and neither can two-site DMRG itself.

    **Perturbative noise** (``noise > 0``, ``noise_type="perturbative"``): block2's
    default (``noise_type = NoiseTypes::DensityMatrix`` with
    ``decomp_type = DecompositionTypes::DensityMatrix``, ``sweep_algorithm.hpp``:104-106),
    and **not randomness**: the perturbation vectors are the operator's own action on the
    current two-site tensor, resolved by term family through
    [Env.heff2_families][tenet.network.Env.heff2_families]. The split becomes an ``eigh``
    of ``rho = tr aa aa^dag`` with ``rho += noise * sum_k p_k p_k^dag`` folded in
    (``moving_environment.hpp``:3554, :3636, :4250), each ``p_k`` normalized and the
    collection scaled to total squared norm ``noise`` (:3698-3713), so ``noise`` stays
    dimensionless in the same 1e-4..1e-5 range. This reaches the sectors *the Hamiltonian
    couples to*, which is the difference from the random draw: it cannot waste the
    perturbation on a direction the operator never visits, and on a sector-poor bond it
    fills what ``H`` can actually populate.

    Neither noise is variational: a noisy sweep's energy may sit above the same sweep at
    ``noise=0.0``.

    **Excited states** (``orthogonal_to``): at each bond, every handed-over two-state
    environment produces one projection vector -- its converged state's two-site reduced
    form in ``psi``'s environment gauge -- and the collection is handed to
    [lanczos][tenet.network.lanczos] as *arguments of the solve*, which is the shape
    block2's ``eigs(..., ortho_bra, projection_weights)`` has
    (``sweep_algorithm.hpp``:1190-1206, :1244-1249). The converged states are **held
    fixed**: their per-bond reduced forms are recomputed at every bond, and their
    environments follow ``psi`` exactly as ``env`` does. block2 instead canonicalizes and
    propagates its ``ext_mpss`` alongside the sweep (:893-917), which its ``eff_ham``
    machinery needs; the contraction does not, because a gauge transformation on any bond
    of the converged state cancels between the two environments and its two-site tensor.
    What *is* required is that ``psi`` be mixed-canonical at the bond, which this sweep
    maintains anyway and which the projection's meaning rests on.
    """
    if noise_type not in _NOISE_TYPES:
        raise ValueError(
            f"noise_type={noise_type!r} is not one of {_NOISE_TYPES}; it decides the "
            "decimation as well as the perturbation, so a typo would run the SVD split "
            "and report a mixer that never ran"
        )
    n_sites = len(psi)
    energy, max_dw = 0.0, 0.0
    for direction in ("right", "left"):
        forward = direction == "right"
        bonds = range(n_sites - 1) if forward else range(n_sites - 2, -1, -1)
        for n in bonds:
            aa = _pair(psi[n], psi[n + 1])
            energy, aa = lanczos(
                lambda v, n=n: env.heff2(n, v),
                aa,
                ncv=ncv,
                orthogonal_to=[o.project2(n, _pair(o.psi[n], o.psi[n + 1])) for o in orthogonal_to],
            )
            if noise and noise_type == "perturbative":
                left, right, s = _split_dm(
                    aa,
                    forward,
                    chi=chi,
                    cutoff=cutoff,
                    perturbations=_perturbations(env, n, aa, noise),
                )
                # The factor that carries the weight is the truncated state; the other is
                # an isometry. So its norm is the kept fraction, which is the same
                # Pythagoras the SVD split spells with ``norm(s)``.
                carrier = right if forward else left
                norm_c = tenet.norm(carrier)
                max_dw = max(max_dw, 1.0 - float(norm_c / tenet.norm(aa)) ** 2)
                psi[n] = left if forward else left / norm_c
                psi[n + 1] = right / norm_c if forward else right
                schmidt[n] = spectrum(s / tenet.norm(s))
            else:
                if noise:
                    r = SymmetricTensor.random(aa.legs, seed=seed + n)
                    aa = tenet.add(aa, r * float(noise * tenet.norm(aa) / tenet.norm(r)))
                    aa = aa / tenet.norm(aa)
                u, s, vh = tenet.linalg.svd_truncated(
                    aa, ((0, 1), (2, 3)), max_bond=chi, cutoff=cutoff
                )
                norm_s = tenet.norm(s)
                max_dw = max(max_dw, 1.0 - float(norm_s / tenet.norm(aa)) ** 2)
                s = s / norm_s  # the two-site tensor is normalized; keep the MPS so
                psi[n + 1] = vh  # through the write barrier: ``vh`` is on (l, p | r) now
                if forward:
                    psi[n] = u
                    psi[n + 1] = tenet.einsum("xy,yqr->xqr", s, psi[n + 1])
                else:
                    psi[n] = tenet.einsum("apx,xy->apy", u, s)
                schmidt[n] = spectrum(s)
            psi.center = n + 1 if forward else n
            for e in (env, *orthogonal_to):
                e.clear_(n, n + 1)
                if direction == "right":
                    e.update_(n, to="last")
                else:
                    e.update_(n + 1, to="first")
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

    Attributes
    ----------
    sweeps : int
        Number of sweeps run.
    energy : float
        The last sweep's energy.
    denergy : float
        The last sweep's energy change.
    max_dSchmidt : float
        The last sweep's worst-cut Schmidt change.
    max_discarded_weight : float
        The last sweep's maximum per-bond discarded weight.
    history : list of tuple
        One ``(energy, denergy, dSchmidt, discarded)`` tuple per sweep.
    schedule : list of Sweep
        The **realized** schedule, one [Sweep][tenet.network.Sweep] per sweep run.
    psi : MPS
        The converged state -- the same object the caller passed in.

    Notes
    -----
    ``history`` is one ``(energy, denergy, dSchmidt, discarded)`` tuple per sweep, and it
    says everything YASTN's ``iterator=True`` generator protocol says to a test without
    the protocol. ``schedule`` is the **realized** schedule, one
    [Sweep][tenet.network.Sweep] per sweep run -- ``zip(out.schedule, out.history)`` is
    exact, and ``out.schedule`` alone answers whether a run actually reached its final
    ``chi`` or converged earlier. ``psi`` is the converged [MPS][tenet.network.MPS].
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
    orthogonal_to: Sequence[MPS] | None = None,
    seed: int = 0,
    callback: Callable[[DMRG_out], None] | None = None,
    compile: Callable | None = None,
) -> DMRG_out:
    """Sweep ``psi`` to the ground state of ``h`` in place and return a
    [DMRG_out][tenet.network.DMRG_out].

    Parameters
    ----------
    psi : MPS
        The starting state, swept in place; a freshly seeded random MPS is the
        expected input.
    h : MPO
        The Hamiltonian.
    schedule : Sequence of Sweep or None, optional
        Per-sweep settings; the **last entry repeats** until convergence or
        ``max_sweeps``. Exclusive with ``chi``/``cutoff``. Default ``None``.
        Keyword-only, as are all the following.
    chi : int or None, optional
        Flat bond-dimension cap for every sweep. Default ``None``, meaning 64.
    cutoff : float or None, optional
        Flat singular-value cutoff for every sweep. Default ``None``, meaning
        ``1e-14``.
    energy_tol : float, optional
        Energy-change convergence threshold. Default ``1e-12``.
    schmidt_tol : float, optional
        Worst-cut Schmidt-change convergence threshold. Default ``1e-8``.
    max_sweeps : int, optional
        Sweep budget. Default ``40``.
    ncv : int, optional
        Krylov-space dimension for [lanczos][tenet.network.lanczos].
        Default ``3``.
    orthogonal_to : Sequence of MPS or None, optional
        Already-converged states to hold ``psi`` orthogonal to, turning the run
        from a ground-state search into an excited-state one: with the ground
        state ``psi1`` in hand, ``dmrg_(psi2, h, orthogonal_to=[psi1])``
        targets the first excited state. The states are **not** modified and
        need no particular gauge. Default ``None``, which projects nothing and
        is today's behaviour. See Notes.
    seed : int, optional
        Feeds [sweep_][tenet.network.sweep_]'s noise draw, distinctly per
        sweep; a schedule with ``noise=0.0`` everywhere draws nothing.
        Default ``0``.
    callback : Callable[[DMRG_out], None] or None, optional
        Invoked once per sweep with that sweep's [DMRG_out][tenet.network.DMRG_out].
        Default ``None``.
    compile : Callable or None, optional
        Handed verbatim to [Env][tenet.network.Env], which wraps the prepared
        two-site matvec with it once per structure key. ``jax.jit`` is the
        intended argument and this layer names no accelerator, so the caller
        supplies it and the ``jax`` extra. It changes the run's performance
        *regime* rather than its accuracy, and the measured payoff is a matvec
        one: with ``jax.jit`` the two-site matvec runs **10.6x** faster on a
        lattice model with ``D_w = 8`` and **1.8--3.0x** faster on ab initio
        integrals at ``K = 16``/``K = 26``, the factor shrinking as the bond
        widens and the work moves into BLAS. **The sweep around it is not
        traceable** -- the truncating SVD re-decides the bond space every sweep --
        so on the JAX backend a compiled run is today slower end to end than the
        plain NumPy one, and ``Env`` re-invokes ``compile`` at every bond visit.
        Default ``None``, which runs the plain Python function and is today's
        behaviour. The grid is ``docs/design.md``, M54.

    Returns
    -------
    DMRG_out
        The last sweep's record; its ``psi`` is the object passed in.

    Raises
    ------
    ValueError
        If ``schedule`` is passed together with ``chi`` or ``cutoff`` --
        silently letting one win is how a run reports a ``chi`` it did not use
        -- or if ``schedule`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import GradedSpace
    >>> from tenet.network import MPO, MPS, dmrg_, local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})  # 2 S^z
    >>> sz, sp = np.diag([-0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 0.0]])
    >>> op = {q: local_op(o, phys=phys, charge=U1Sector(q))
    ...       for q, o in ((0, sz), (-2, sp), (2, sp.T))}
    >>> terms = []
    >>> for i in range(3):  # 4-site Heisenberg chain
    ...     terms.append((1.0, [(op[0], i), (op[0], i + 1)]))
    ...     terms.append((0.5, [(op[-2], i), (op[2], i + 1)]))
    ...     terms.append((0.5, [(op[2], i), (op[-2], i + 1)]))
    >>> h = MPO.from_terms(4, terms)
    >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1)] * 2)  # Neel seed
    >>> out = dmrg_(psi, h, chi=8)
    >>> round(out.energy, 6)  # the exact open-chain ground energy, to 6 places
    -1.616025

    Notes
    -----
    ``psi`` is right-canonicalized first ([MPS.canonize_][tenet.network.MPS.canonize_]),
    so a freshly seeded random MPS is the expected input and a caller keeping the
    returned ``out.psi`` and the one it passed in holds the same object.

    Two spellings of what the sweeps do, exclusive: the flat ``chi`` / ``cutoff`` kwargs
    (defaults 64 and 1e-14), or ``schedule``, a non-empty sequence of
    [Sweep][tenet.network.Sweep] entries whose **last entry repeats** until convergence
    or ``max_sweeps`` -- so ``schedule=[Sweep(chi=64)]`` is exactly the flat run, and
    ``schedule=[Sweep(32, noise=1e-4)] * 4 + [Sweep(64)]`` is a ramp that cools down at
    ``chi=64`` for as long as ``max_sweeps`` allows.

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

    **``orthogonal_to``, and why the name.** YASTN spells the same argument ``project``
    and TenPy ``orthogonal_to``; the second is taken, because ``project`` names the
    mechanism and this argument is one of two mechanisms that implement it -- block2 has
    both, hard projection and a level shift (see [lanczos][tenet.network.lanczos]) -- while
    ``orthogonal_to`` names the *result*, which is the same under either. The machinery is
    one two-state [Env][tenet.network.Env] per given state over
    [MPO.identity][tenet.network.MPO.identity], set up here and swept alongside ``env``;
    what each contributes at a bond is a projection vector handed to ``lanczos`` as an
    argument of the solve. Sector targeting is unaffected and composes with it: a
    charged ``D=1`` boundary leg on bond 0 fixes the sector, orthogonality then walks up
    the spectrum inside it, and a converged state whose boundary legs put it in a
    *different* sector is dropped from the projection -- it is orthogonal to ``psi`` by
    the symmetry, before the sweep does anything, which is right rather than merely
    harmless.

    The reported energy is the projected operator's own Ritz value, so it is the excited
    energy directly and needs no shift subtracted.

    ``callback``, if given, is invoked once per sweep with the
    [DMRG_out][tenet.network.DMRG_out] built for that sweep, after ``history`` is
    appended, so it sees the sweep that just finished. Its return value is ignored:
    there is no early-stop protocol.
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
    env = Env(psi, h, compile=compile).setup_(0)
    # One two-state environment per converged state, over the identity operator: block2's
    # ext_mes, whose per-bond ``multiply`` is the projection vector (``core.py``:4817-4830
    # builds them with ``get_identity_mpo()``). They are set up *after* ``canonize_``,
    # because their bra side is this ``psi`` and they follow it through the sweep.
    orthos: tuple[Env, ...] = ()
    if orthogonal_to:
        ident = MPO.identity(len(psi), psi[0].legs[1].space)
        ends = (psi[0].legs[0], psi[len(psi) - 1].legs[2])
        orthos = tuple(
            Env(phi, ident, bra=psi).setup_(0)
            for phi in orthogonal_to
            # A converged state whose boundary legs put it in another sector is
            # orthogonal to ``psi`` already, by the symmetry and not by the sweep, and
            # its mixed transfer has no coupled sector to be contracted at all. Skipping
            # it is the statement that sector targeting and orthogonality compose.
            if (phi[0].legs[0].space, phi[0].legs[0].dual) == (ends[0].space, ends[0].dual)
            and (phi[len(phi) - 1].legs[2].space, phi[len(phi) - 1].legs[2].dual)
            == (ends[1].space, ends[1].dual)
        )
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
            noise_type=entry.noise_type,
            orthogonal_to=orthos,
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
