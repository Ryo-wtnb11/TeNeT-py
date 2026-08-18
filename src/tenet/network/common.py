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

from collections.abc import Sequence

import autoray as ar

import tenet
from tenet import Leg, SymmetricTensor

__all__ = ["ones", "spectrum"]


def spectrum(s: SymmetricTensor) -> list[float]:
    """The Schmidt values on a bond, descending.

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
    # QuantumDimensionData is checked by svd_truncated before ``s`` can exist
    qdim = s.provider.qdim  # ty: ignore[unresolved-attribute]
    out = [
        float(v)
        for sector, m in tenet.to_matrices(s).items()
        for v in ar.do("diag", m) * qdim(sector) ** 0.5
    ]
    return sorted(out, reverse=True)


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
