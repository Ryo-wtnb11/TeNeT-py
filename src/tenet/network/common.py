"""What every driver in this package needs: the scalar exits, a spectrum, a ones seed.

Moved here by #114, **bodies unchanged**, from ``network/mps.py`` (:24-33 ``scalar``,
:36-45 ``inner``, :48-61 ``spectrum``) and ``network/env.py`` (:19-22 ``_ones``, now
public as :func:`ones`). The move is the whole of the change: no number can differ,
because no arithmetic was touched.

Why a module rather than a second copy or a cross-driver import. ``network/ctmrg.py``
needs the same qdim-weighted trace and the same diagonal read as ``network/mps.py``;
importing them *from* ``mps.py`` would assert a dependency between two drivers that
share no concept, and ``env._ones`` cannot be imported at all -- the hygiene test
``test_no_module_reaches_into_another_modules_private_names`` forbids exactly that, and
correctly: it is what stops the package growing a private web. So the shared four move
down instead.

**Trace-neutral**: nothing here decides a structure, so it is used on both sides of the
``jit``/``grad`` boundary -- :func:`scalar` runs inside ``examples/ctmrg.py``'s
differentiated ``log_kappa``, :func:`spectrum` only ever outside it.

The bigger step -- a qdim-weighted trace in ``tenet.ops`` next to ``trace`` -- is a
public-API design question (does it give ``SymmetricTensor`` the rank-0 exit it
deliberately does not have?) and gets its own issue. This module is a move.
"""

import string
from collections.abc import Sequence
from typing import Any

import autoray as ar

import tenet
from tenet import Leg, SymmetricTensor

__all__ = ["inner", "ones", "scalar", "spectrum"]


def scalar(t: SymmetricTensor) -> Any:
    """The categorical trace of a rank-2 map ``(X OUT, X IN)``: ``sum_c d_c tr(M_c)``.

    ``SymmetricTensor`` has no rank 0, so every fully closed network here contracts down
    to a rank-2 tensor with one bond left open, and closing that bond is a trace carrying
    the same ``qdim`` weight :func:`tenet.norm` carries. This is where the tensor world
    is left explicitly, and it is the one place this package reads ``t.provider``.
    """
    qdim = t.provider.qdim
    return sum(qdim(c) * ar.do("trace", m) for c, m in tenet.to_matrices(t).items())


def inner(a: SymmetricTensor, b: SymmetricTensor) -> Any:
    """``<a|b>``: contract every axis but the first, then :func:`scalar` the rest.

    Works at any rank, which is what lets :func:`~tenet.network.lanczos` be a plain
    vector-space algorithm: the adjoint flips every leg, so axis 0 of ``adjoint(a)`` is IN
    and axis 0 of ``b`` is OUT, and the leftover rank-2 map is exactly what :func:`scalar`
    traces.
    """
    rest = string.ascii_lowercase[1 : a.ndim]
    return scalar(tenet.einsum(f"L{rest},l{rest}->lL", tenet.adjoint(a), b))


def spectrum(s: SymmetricTensor) -> list[float]:
    """The Schmidt values on a bond, descending.

    ``s`` comes from :func:`tenet.linalg.svd_truncated` and is diagonal by construction,
    so this reads its diagonal; the ``sqrt(qdim)`` weight is the same one
    :func:`tenet.norm` carries, and it is 1 throughout for U(1).
    """
    qdim = s.provider.qdim
    out = [
        float(v)
        for sector, m in tenet.to_matrices(s).items()
        for v in ar.do("diag", m) * qdim(sector) ** 0.5
    ]
    return sorted(out, reverse=True)


def ones(legs: Sequence[Leg]) -> SymmetricTensor:
    """A tensor of ones on ``legs`` -- ``examples/ctmrg.py::init_env``'s seed spelling."""
    t = SymmetricTensor.zeros(legs)
    return t.apply_blocks(lambda b: ar.do("ones_like", b))
