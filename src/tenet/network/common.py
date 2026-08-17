"""What every driver in this package needs: a bond spectrum and a ones seed.

Moved here by #114, **bodies unchanged**, from ``network/mps.py`` (``scalar``, ``inner``,
``spectrum``) and ``network/env.py`` (``_ones``, now public as :func:`ones`). #126 then
took the two scalar exits the rest of the way out: ``scalar`` is now
:func:`tenet.full_trace` and ``inner`` is :func:`tenet.inner`, both in ``tenet.ops``
next to ``trace``, with the same arithmetic plus a square-map refusal. No alias is kept.

Why a module rather than a second copy or a cross-driver import. ``network/ctmrg.py``
needs the same diagonal read as ``network/mps.py``; importing it *from* ``mps.py`` would
assert a dependency between two drivers that share no concept, and ``env._ones`` cannot
be imported at all -- the hygiene test
``test_no_module_reaches_into_another_modules_private_names`` forbids exactly that, and
correctly: it is what stops the package growing a private web. So the shared pair moves
down instead.

**Trace-neutral**: nothing here decides a structure. :func:`spectrum` is nonetheless only
ever called outside a ``jit``/``grad`` region, because its ``sorted`` Python list is
driver output, not a tensor.
"""

from collections.abc import Sequence

import autoray as ar

import tenet
from tenet import Leg, SymmetricTensor

__all__ = ["ones", "spectrum"]


def spectrum(s: SymmetricTensor) -> list[float]:
    """The Schmidt values on a bond, descending.

    ``s`` comes from :func:`tenet.linalg.svd_truncated` and is diagonal by construction,
    so this reads its diagonal; the ``sqrt(qdim)`` weight is the same one
    :func:`tenet.norm` carries, and it is 1 throughout for U(1).
    """
    # QuantumDimension is checked by svd_truncated before ``s`` can exist
    qdim = s.provider.qdim  # ty: ignore[unresolved-attribute]
    out = [
        float(v)
        for sector, m in tenet.to_matrices(s).items()
        for v in ar.do("diag", m) * qdim(sector) ** 0.5
    ]
    return sorted(out, reverse=True)


def ones(legs: Sequence[Leg]) -> SymmetricTensor:
    """A tensor of ones on ``legs`` -- ``examples/toy_codes/ctmrg.py::init_env``'s seed spelling."""
    t = SymmetricTensor.zeros(legs)
    return t.apply_blocks(lambda b: ar.do("ones_like", b))
