"""The driver layer: finite tensor-network algorithms over the public ``tenet`` API.

The dependency edge is one-way: ``network`` imports ``ops``/``tensor``, never the
reverse. Nothing under ``src/tenet`` outside this package may import it, exactly as
``ops`` imports ``tensor`` and not back (``src/tenet/ops/__init__.py``:1-5).

M11a is **entirely outside** ``jit``/``grad``, and that is the layer's own invariant
rather than an exception to README invariant 9. Invariant 9 says structure-changing
operations live outside compile boundaries and the library never hides the distinction;
this package is the complement of that sentence -- the place where data-dependent control
flow is *allowed to live*, because the layer is never traced.
``tenet.linalg.svd_truncated`` re-decides a bond
:class:`~tenet.GradedSpace` at every bond of every sweep, :func:`lanczos`'s happy
breakdown tests a norm against ``tol``, and :func:`dmrg`'s loop exits on a measured
energy change. Every one of those is what tenet refuses to trace, and correctly.

Contents (M11a): :class:`MPS`, :class:`MPO`, :class:`Env`, :func:`lanczos`,
:func:`sweep`, :func:`dmrg`, plus the two scalar exits :func:`scalar` and :func:`inner`
and the bond spectrum :func:`spectrum`. Promoted verbatim from ``examples/dmrg.py``
(#110) under the rule that no number may move.

Uses **public ``tenet`` API only**: no ``jax``/``torch``/``scipy``/``quimb``/
``opt_einsum``, no ``_``-prefixed reach into other modules, no numerical use of
``t.blocks``. The one named exception is reading ``t.provider`` / ``provider.qdim`` /
``provider.unit``, which :func:`scalar` and :func:`spectrum` need because there is no
public spelling of a qdim-weighted trace; that is symmetry-generic metadata, not a
provider branch. ``tests/network/test_hygiene.py`` enforces both halves.
"""

from tenet.network.dmrg import DMRG_out, dmrg, lanczos, sweep
from tenet.network.env import Env
from tenet.network.mps import MPO, MPS, inner, scalar, spectrum

__all__ = [
    "MPO",
    "MPS",
    "DMRG_out",
    "Env",
    "dmrg",
    "inner",
    "lanczos",
    "scalar",
    "spectrum",
    "sweep",
]
