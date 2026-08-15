"""Elementwise maps over the reduced blocks — issue #93.

**Coefficient space, not dense space.** This is the one module in ``ops`` that is
*non-linear* in the blocks, and non-linearity is exactly what does not commute
with dense expansion: ``T = Σ_τ A^(τ) ⊗ C^(τ)`` is a sum of tensor products, so
``apply_blocks(t, f).to_dense()`` and ``f(t.to_dense())`` are two different
operations whenever the Clebsch-Gordan factor is not all-ones. Measured on a
rank-3 tensor with positive blocks: the two agree to ``0.0`` for U(1), and differ
by ``1.673`` on a dense scale of ``3.82`` for SU(2).

That measurement is why nothing here is registered in ``array/dispatch.py``:
``ar.do("sqrt", t)`` must keep raising, because autoray's ``"sqrt"`` means the
dense elementwise one and this is not it.

As in :mod:`tenet.ops.basic`, there is no NumPy call anywhere in this module.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import autoray as ar

if TYPE_CHECKING:
    from tenet.tensor import SymmetricTensor

__all__ = ["apply_blocks", "power", "sqrt"]

Array = Any


def apply_blocks(t: "SymmetricTensor", fn: Callable[[Array], Array]) -> "SymmetricTensor":
    """``fn`` applied to each reduced block. **Coefficient space, not dense space.**

    Structure is untouched, so this is linear-algebra-free, backend-generic,
    traceable and differentiable: it is exactly ``t.set_params(map(fn,
    t.get_params()))``, which is quimb's ``Tensor.apply_to_arrays`` over the same
    parameter protocol.

    ``fn`` must be **elementwise and shape-preserving**. It is not checked (a
    shape change is caught by ``SymmetricTensor.__post_init__``, a value-dependent
    one is the caller's problem), and it is not required that ``fn(0) == 0``: the
    blocks hold only allowed fusion channels, so *any* ``fn`` returns a valid
    symmetric tensor and every symmetry-forbidden dense entry stays exactly zero.

    What it does **not** do is commute with dense expansion. ``T = Σ_τ A^(τ) ⊗
    C^(τ)``, so for a non-Abelian provider ``apply_blocks(t, f).to_dense() !=
    f(t.to_dense())`` — measured off by ``1.673`` on a dense scale of ``3.82`` for
    a rank-3 SU(2) tensor with ``f = sqrt``. For every shipped Abelian provider
    (all-ones CG, ``d_a == 1``) they agree exactly. If you want dense-elementwise
    semantics, densify explicitly.
    """
    from tenet.tensor import SymmetricTensor

    return SymmetricTensor(t.structure, tuple(fn(b) for b in t.blocks))


def sqrt(t: "SymmetricTensor") -> "SymmetricTensor":
    """Blockwise ``sqrt``. The ``svd`` splitter: ``u @ sqrt(s)``, ``sqrt(s) @ vh``.

    Blockwise, i.e. on the coefficients — see :func:`apply_blocks` for the
    blockwise/dense caveat, which for SU(2) is 44% of the array's own scale and
    completely silent.

    For the ``S`` returned by :func:`tenet.linalg.svd` this *is* the matrix square
    root, because ``svd`` builds ``S``'s blocks with ``ar.do("diag", ...)`` and a
    diagonal matrix's elementwise and matrix square roots coincide. That is a fact
    about ``S``, not about ``sqrt``: for a non-diagonal ``t``, ``sqrt(t) @ sqrt(t)
    != t``.

    Negative or complex entries are the backend's business: ``sqrt(-1.0)`` is
    ``nan`` under NumPy, JAX and torch alike, and nothing here clips, guards or
    regularizes.
    """
    return apply_blocks(t, lambda b: ar.do("sqrt", b))


def power(t: "SymmetricTensor", p: Any) -> "SymmetricTensor":
    """Blockwise ``t ** p``. ``p`` is a scalar exponent, never a tensor.

    Same coefficient-space semantics and the same backend-owned branch cuts and
    ``nan``s as :func:`sqrt`; ``p = -0.5`` is the inverse-√S of canonical-form and
    gauge-fixing loops.
    """
    return apply_blocks(t, lambda b: ar.do("power", b, p))
