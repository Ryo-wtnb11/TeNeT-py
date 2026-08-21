"""What every driver in this package needs: a bond spectrum and a ones seed.

Moved here by #114, **bodies unchanged**, from ``network/mps.py`` (``scalar``, ``inner``,
``spectrum``) and ``network/env.py`` (``_ones``, now public as
[ones][tenet.network.ones]). #126 then took the two scalar exits the rest of the way out:
``scalar`` is now [tenet.full_trace][] and ``inner`` is [tenet.inner][], both in
``tenet.ops`` next to ``trace``, with the same arithmetic plus a square-map refusal. No
alias is kept.

Why a module rather than a second copy or a cross-driver import. ``network/ctmrg.py``
needs the same diagonal read as ``network/mps.py``; importing it *from* ``mps.py`` would
assert a dependency between two drivers that share no concept, and ``env._ones`` cannot
be imported at all -- the hygiene test
``test_no_module_reaches_into_another_modules_private_names`` forbids exactly that, and
correctly: it is what stops the package growing a private web. So the shared pair moves
down instead.

**Trace-neutral**: nothing here decides a structure. [spectrum][tenet.network.spectrum]
is nonetheless only ever called outside a ``jit``/``grad`` region, because its ``sorted``
Python list is driver output, not a tensor.
"""

import math
from collections.abc import Sequence
from typing import Any

import autoray as ar

import tenet
from tenet import Leg, SymmetricTensor
from tenet.symmetry import Sector

__all__ = ["entropy", "ones", "spectrum", "spectrum_sectors"]

#: How many bytes of cached tensor payload each of the sweep's per-bond caches may hold.
#: **This is the whole cache policy and it is one number in one place** (#202):
#: ``EdgeTable``'s block tables and group embeddings and ``Env``'s merged cores, prepared
#: operators and compiled matvecs all use it through [Recent][tenet.network.common.Recent],
#: so there is no per-cache flag to thread and no way for two of them to disagree.
#:
#: A **byte** budget rather than a count of entries, and the count was measured before it
#: was rejected: a bound of four entries costs +22 to +26 % wall time on the small models
#: this package is otherwise used for, because a small MPO's whole block table is a few
#: megabytes and evicting any of it buys nothing at all. A byte budget is never reached by
#: those models, so they keep every entry and pay exactly nothing, and it is only the
#: operator that is measured in gibibytes per site that ever evicts. That is the "the
#: common case must not get slower to fix the rare one" criterion, met by construction
#: instead of by tuning. ``docs/design.md`` "Milestone 38" carries the measurements.
CACHE_BUDGET = 1 << 30  # 1 GiB


def payload(obj: Any) -> int:
    """Bytes of backend-array storage reachable from ``obj``.

    Parameters
    ----------
    obj : object
        A cached value: a [SymmetricTensor][tenet.SymmetricTensor], a mapping, a tuple
        (``NamedTuple`` included) or anything else, which weighs nothing.

    Returns
    -------
    int
        The total ``nbytes`` of the arrays under ``obj``.

    Notes
    -----
    Duck-typed rather than dispatched on type, which keeps it to three cases: anything
    with ``nbytes`` is a backend array and is the leaf, anything with ``items`` is a
    mapping *or* a tensor -- ``SymmetricTensor.items`` yields the reduced blocks, so the
    two spell the same recursion -- and a tuple or list is walked. A shared array is
    counted once per cache and so may be counted twice across two caches, which
    over-estimates in the safe direction for a budget.
    """
    size = getattr(obj, "nbytes", None)  # a backend array: the leaf
    if size is not None:
        return int(size)
    pairs = getattr(obj, "items", None)  # a mapping, or a tensor's reduced blocks
    if pairs is not None:
        return sum(payload(v) for _, v in pairs())
    if getattr(obj, "_fields", None) is not None or type(obj) in (tuple, list):
        return sum(payload(v) for v in obj)
    return 0


class Recent[K, V](dict[K, V]):
    """A ``dict`` evicting its least recently used entries past ``CACHE_BUDGET`` bytes.

    Notes
    -----
    A DMRG sweep visits bonds in a sliding order, so a cache of the last few bonds hits on
    everything the sweep asks for twice while the entries behind it are dead weight -- at
    quantum-chemistry scale, weight measured in gibibytes (#202). Recency is refreshed on
    read as well as write, so a bond revisited immediately on the return leg of a sweep is
    a hit.

    Eviction is oldest-used first, and never below two entries: a two-site bond asks for
    site ``n`` and site ``n + 1`` in one breath, so a cache that can hold fewer than two
    thrashes by construction rather than by tuning. Below the budget nothing is evicted at
    all, which is what makes this free for every model whose whole operator is smaller
    than the budget.
    """

    def __init__(self) -> None:
        super().__init__()
        # Sizes are measured once, on insert. No call site deletes from these caches, so
        # ``__delitem__`` and ``pop`` are not overridden and cannot desynchronise the
        # running total; a future deleter has to keep this pair honest.
        self._sizes: dict[K, int] = {}
        self._total = 0

    def __getitem__(self, key: K) -> V:
        value = super().pop(key)
        super().__setitem__(key, value)  # most recent again
        return value

    def __setitem__(self, key: K, value: V) -> None:
        if key in self:
            self._total -= self._sizes.pop(key)
            super().pop(key)
        self._sizes[key] = size = payload(value)
        self._total += size
        super().__setitem__(key, value)
        while len(self) > 2 and self._total > CACHE_BUDGET:
            oldest = next(iter(self))
            self._total -= self._sizes.pop(oldest)
            super().__delitem__(oldest)

    def get(self, key: Any, default: Any = None, /) -> Any:
        """``dict.get``, but a hit counts as a use.

        Parameters
        ----------
        key : object
            The key to look up.
        default : object, optional
            Returned when ``key`` is absent. Default ``None``.

        Returns
        -------
        object
            The cached value, or ``default``.
        """
        return self[key] if key in self else default


def spectrum(s: SymmetricTensor) -> list[float]:
    """The singular values on a bond, descending -- the spectrum of an ``svd`` output.

    Spectrum *of what*: of the diagonal tensor an
    [svd_truncated][tenet.ops.linalg.svd_truncated] returned, and of nothing
    else. Both callers hand it exactly that, and read it for two different
    things -- ``network/dmrg.py`` for the Schmidt values of a bond,
    ``network/ctmrg.py`` for the corner spectrum whose convergence ends a sweep
    -- which is why the name stays the general one rather than either caller's
    (#120, reaffirmed #185).

    Parameters
    ----------
    s : SymmetricTensor
        The diagonal singular-value tensor a
        [tenet.linalg.svd_truncated][tenet.ops.linalg.svd_truncated] returned.

    Returns
    -------
    list of float
        Every diagonal value, ``sqrt(qdim)``-weighted, sorted descending.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import spectrum
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> _, s, _ = tenet.linalg.svd(t, ((0,), (1,)))
    >>> vals = spectrum(s)
    >>> len(vals)
    4
    >>> vals == sorted(vals, reverse=True)
    True

    Notes
    -----
    ``s`` comes from [tenet.linalg.svd_truncated][tenet.ops.linalg.svd_truncated] and is
    diagonal by construction, so this reads its diagonal; the ``sqrt(qdim)`` weight is
    the same one [tenet.norm][] carries, and it is 1 throughout for U(1).
    """
    return sorted(
        (v for vals in spectrum_sectors(s).values() for v in vals),
        reverse=True,
    )


def spectrum_sectors(s: SymmetricTensor) -> dict[Sector, list[float]]:
    """[spectrum][tenet.network.spectrum], resolved by the sector of the bond.

    The same values, unflattened: on a [GradedSpace][tenet.GradedSpace] bond the singular
    values arrive already labelled, and [spectrum][tenet.network.spectrum] sorts that label
    away because its two callers -- ``network/dmrg.py`` and ``network/ctmrg.py`` -- both
    want one flat convergence diagnostic (#120, reaffirmed #185). A user asking *which
    symmetry sector carries the entanglement* wants the label back, and TenPy spells that
    ``entanglement_spectrum(by_charge=True)``.

    Parameters
    ----------
    s : SymmetricTensor
        The diagonal singular-value tensor a
        [tenet.linalg.svd_truncated][tenet.ops.linalg.svd_truncated] returned.

    Returns
    -------
    dict of Sector to list of float
        Per coupled sector, its diagonal values ``sqrt(qdim)``-weighted and sorted
        descending. Concatenating and re-sorting the values reproduces
        [spectrum][tenet.network.spectrum] exactly, which is how that function is written.

    Examples
    --------
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import spectrum_sectors
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> _, s, _ = tenet.linalg.svd(t, ((0,), (1,)))
    >>> sorted((sector.charge, len(vals)) for sector, vals in spectrum_sectors(s).items())
    [(0, 2), (1, 2)]

    Notes
    -----
    **The ``sqrt(qdim)`` weight is applied here and nowhere else in this package.** Both
    [spectrum][tenet.network.spectrum] and [entropy][tenet.network.entropy] read it off this
    function, so there is one place where a non-Abelian bond's multiplet weight is decided.
    It is the weight [tenet.norm][] carries, and it is 1 throughout for U(1).
    """
    # QuantumDimensionData is checked by svd_truncated before ``s`` can exist
    qdim = s.provider.qdim  # ty: ignore[unresolved-attribute]
    return {
        sector: sorted((float(v) for v in ar.do("diag", m) * qdim(sector) ** 0.5), reverse=True)
        for sector, m in tenet.to_matrices(s).items()
    }


def entropy(s: SymmetricTensor, *, alpha: float = 1.0) -> float:
    """The entanglement entropy of a bond, in **nats** -- von Neumann at ``alpha=1``.

    Parameters
    ----------
    s : SymmetricTensor
        The diagonal singular-value tensor a
        [tenet.linalg.svd_truncated][tenet.ops.linalg.svd_truncated] returned, on a bond of
        a canonical state; its values are normalized here, so an unnormalized ``s`` is
        read the same way.
    alpha : float, optional
        The Renyi index. Default ``1.0``, the von Neumann entropy
        ``-sum_i p_i log p_i``; any other positive value gives
        ``log(sum_i p_i**alpha) / (1 - alpha)``. Keyword-only.

    Returns
    -------
    float
        The entropy across the cut, in nats.

    Raises
    ------
    ValueError
        If ``alpha`` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> import tenet
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.network import entropy
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    >>> s = SymmetricTensor.from_dense(
    ...     np.eye(2) / 2**0.5, (Leg(V, OUT), Leg(V, IN))
    ... )
    >>> round(entropy(s), 6)  # a maximally entangled pair across the cut
    0.693147

    Notes
    -----
    **Nats, and the callable says so because the two references disagree**: YASTN's
    ``get_entropy`` is base 2, TenPy's ``entanglement_entropy`` is natural. Natural is taken
    because it is what a central-charge fit wants -- ``S = (c/6) log(x)`` on an open chain --
    and because every other logarithm in this package is natural. Divide by ``log(2)`` for
    bits.

    **The multiplet weight is where a non-Abelian bond is easy to get wrong.** A sector of
    quantum dimension ``d`` holds ``d`` copies of each of its reduced values in the dense
    Schmidt spectrum, so the probability of one copy is ``p_i / d`` where ``p_i`` is the
    ``sqrt(qdim)``-weighted value squared, and the sum over copies restores the ``d``:

        ``S = -sum_i p_i log(p_i / d_i)``

        ``S_alpha = log(sum_i d_i (p_i / d_i)**alpha) / (1 - alpha)``

    Reading ``-sum p log p`` off the flattened [spectrum][tenet.network.spectrum] instead
    would report ``0`` for an SU(2) singlet, whose whole entanglement lives in one
    ``j = 1/2`` multiplet. That equality -- an SU(2) state and the same state under U(1)
    giving the same number -- is what pins the weight rather than merely making it
    consistent, and ``tests/network/test_entanglement.py`` is where it is pinned.
    """
    if alpha <= 0.0:
        raise ValueError(f"the Renyi index must be positive, got alpha={alpha}")
    qdim = s.provider.qdim  # ty: ignore[unresolved-attribute]
    weighted = [
        (float(qdim(sector)), v * v) for sector, vals in spectrum_sectors(s).items() for v in vals
    ]
    total = sum(p for _, p in weighted)
    if total <= 0.0:
        return 0.0
    live = [(d, p / total) for d, p in weighted if p > 0.0]
    if alpha == 1.0:
        return -sum(p * math.log(p / d) for d, p in live)
    return math.log(sum(d * (p / d) ** alpha for d, p in live)) / (1.0 - alpha)


def ones(legs: Sequence[Leg]) -> SymmetricTensor:
    """A tensor of ones on ``legs`` -- ``examples/toy_codes/ctmrg.py::init_env``'s seed spelling.

    Parameters
    ----------
    legs : Sequence of Leg
        The legs of the tensor to build.

    Returns
    -------
    SymmetricTensor
        A tensor over ``legs`` with every structurally allowed entry equal to 1.

    Examples
    --------
    >>> from tenet import OUT, GradedSpace, Leg
    >>> from tenet.network import ones
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2})
    >>> ones((Leg(V, OUT),)).to_dense()
    array([1., 1.])
    """
    t = SymmetricTensor.zeros(legs)
    return t.apply_blocks(lambda b: ar.do("ones_like", b))
