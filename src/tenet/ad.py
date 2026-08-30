"""Opt-in stabilized VJPs for ``svd``/``eigh`` on the JAX backend. Call [install][tenet.ad.install].

JAX's own SVD/eigh VJPs carry ``1/(sigma_i - sigma_j)`` / ``1/(w_i - w_j)`` factors that
are ``NaN`` at exact degeneracy, and under a non-Abelian symmetry degeneracy inside a
coupled sector is generic rather than an edge case. ``install()`` replaces them with the
Lorentzian-broadened form, ``1/x -> x/(x**2 + eps)``; ``uninstall()`` restores autoray's
stock bindings.

Three things a caller must know:

* **It is process-global for the JAX backend, by design.** The seam is
  ``autoray.register_function("jax", "linalg.svd", ...)``, autoray's own extension point,
  so after ``install()`` *any* ``ar.do("linalg.svd", jax_array)`` in the process --
  quimb's included -- gets the broadened VJP. Installation is therefore an explicit
  **function call**, not an import side effect like [tenet.pytree][]'s: mutating another
  library's dispatch table is the user's act, not an import's.

* **The broadened gradient is correct, not merely finite, exactly when the objective is
  gauge-invariant on each degenerate subspace.** At exact degeneracy the singular vectors
  within a multiplet are only defined up to a unitary, so ``dU/dA`` genuinely does not
  exist; what exists is the derivative of gauge-invariant combinations such as
  ``U S Vh``, the singular values, or any projector onto the multiplet. Broadening
  returns the correct value for those and an ``eps``-suppressed arbitrary value for the
  rest. It is a **precondition**, not a hidden approximation: a gauge-*dependent*
  objective gets an ``eps``-dependent answer, which is to say a meaningless one.

* **``eps`` is in units of sigma squared** -- ``safe_inverse(x) = x/(x**2 + eps)`` with
  ``x = sigma_j - sigma_i`` -- so a default tuned for an ``O(1)``-normalized spectrum
  assumes a normalization our tensors do not guarantee. Hence the knob.

Broadening rather than a hard tolerance, because a hard cut makes the gradient
discontinuous in the spectrum. ``qr``, ``lq`` and ``polar`` need nothing of their own:
their backend rules are already well defined where their inputs are full-rank.

[tenet.enable_jax][]``(ad=True)`` is the one-call spelling of the three statements
below, and is what the docs teach. It is the same explicit act this docstring demands --
``ad`` is opted into by name there, never defaulted on, precisely because of the
process-global reach above -- and it runs this module's ``install()`` unchanged.

Usage::

    import tenet

    tenet.enable_jax(ad=True)  # pytree registration + the broadened VJPs

    def objective(t):
        u, s, vh = tenet.linalg.svd(t)   # per coupled sector
        return tenet.norm(u @ s @ vh)

    g = jax.grad(objective)(t)
"""

# Simplification: ``eps`` is an absolute number; scale it by ``sigma_max**2`` if a caller
# ever hits a badly scaled tensor, and add a criterion at that point.
# Simplification: no ``custom_jvp``; add one when something asks for ``jacfwd``.

from typing import Any

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # loud, actionable, names the extra -- as tenet.pytree does
    raise ImportError(
        "tenet.ad requires JAX, which is an optional dependency. "
        "Install it with `pip install 'symtenet[jax]'` or `uv sync --extra jax`. "
        "The core library never imports JAX; get_params/set_params work without it."
    ) from exc

import autoray as ar

__all__: list[str] = []

# Read at trace time by the backward rules below, so it lands in the trace as a
# constant. Changing it does not invalidate an already-compiled `jax.jit` cache entry;
# set it once, at install time, before the hot loop.
_EPSILON = 1e-12

_NAMES = ("linalg.svd", "linalg.eigh")


def _dag(x: jax.Array) -> jax.Array:
    """``x†`` for one dense coupled-sector matrix."""
    return jnp.conj(x).T


def _antiherm(x: jax.Array) -> jax.Array:
    """``(x - x†)/2``. The gauge-carrying part of ``U† dU``; the rest is unphysical."""
    return (x - _dag(x)) / 2


def _safe_inverse(x: jax.Array, eps: float) -> jax.Array:
    """``1/x``, Lorentzian-broadened: ``x/(x**2 + eps)``. tensorgrad's ``safe_inverse``.

    Note that it is exactly ``0`` at ``x == 0``, which is what zeroes the diagonal of
    the ``1/(sigma_j - sigma_i)`` matrix without a separate ``fill_(0)``, and what makes
    an exactly-zero singular value survivable without a rank branch -- ``_lower``'s
    ``min(rows, cols)`` bond rule guarantees we meet those.
    """
    return x / (x * x + eps)


# --- svd ---------------------------------------------------------------------------


@jax.custom_vjp
def _svd(b: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    # a plain tuple, not JAX's SVDResult: fwd must return the same treedef, and
    # ops/linalg.py only ever indexes it
    u, s, vh = jnp.linalg.svd(b, full_matrices=False)
    return u, s, vh


def _svd_fwd(
    b: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    u, s, vh = jnp.linalg.svd(b, full_matrices=False)
    return (u, s, vh), (u, s, vh)


def _svd_bwd(
    res: tuple[jax.Array, jax.Array, jax.Array],
    ct: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array]:
    """PRX 9, 031041 Eq. (4) with ``1/x -> safe_inverse(x)``, per coupled-sector matrix.

    The ``1/(sigma_i + sigma_j)`` **diagonal is kept**, and that is deliberate:
    tensorgrad drops it (``G.diagonal().fill_(inf)``), correctly, because that file is
    real-only; for a complex block the diagonal of ``(aU - aV)/(2 sigma)`` is precisely
    the singular-vector relative-phase term, Francuz-Schuch-Vanhecke Eq. (G5), and it is
    what MatrixAlgebraKit computes without excluding the diagonal. It vanishes
    identically for real input.

    The ``conj`` on the way in and on the way out is the convention change, and it is
    a no-op for real input: JAX's raw VJP cotangents are transpose-convention
    (``x_bar = J^T y_bar``, with antiholomorphy carried by the ``conj`` primitive),
    while the literature formula is written in the ``dX = dL/d conj(X)`` convention that
    PyTorch's ``.grad`` and ``jax.grad`` both hand back. Conjugating both ends converts
    once, rather than re-deriving the rule.
    """
    u, s, vh = res
    du, ds, dvh = (jnp.conj(x) for x in ct)
    eps = _EPSILON
    v, dv = _dag(vh), _dag(dvh)

    au, av = _antiherm(_dag(u) @ du), _antiherm(_dag(v) @ dv)
    fm = _safe_inverse(s[None, :] - s[:, None], eps)  # 1/(sigma_j - sigma_i)
    fp = _safe_inverse(s[None, :] + s[:, None], eps)  # 1/(sigma_j + sigma_i)
    core = (au + av) * fm + (au - av) * fp + jnp.diag(jnp.real(ds))
    da = u @ core @ vh

    # The rectangular complements, only when the matrix actually has the extra
    # directions: (I - U U†) and (I - V V†) are the zero map otherwise.
    rows, cols, k = u.shape[0], vh.shape[1], s.shape[0]
    sinv = _safe_inverse(s, eps)
    if rows > k:
        da = da + (du - u @ (_dag(u) @ du)) * sinv[None, :] @ vh
    if cols > k:
        da = da + u @ (sinv[:, None] * (dvh - (dvh @ v) @ vh))
    return (jnp.conj(da),)


_svd.defvjp(_svd_fwd, _svd_bwd)


# --- eigh --------------------------------------------------------------------------


@jax.custom_vjp
def _eigh(b: jax.Array) -> tuple[jax.Array, jax.Array]:
    w, v = jnp.linalg.eigh(b)
    return w, v


def _eigh_fwd(
    b: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    w, v = jnp.linalg.eigh(b)
    return (w, v), (w, v)


def _eigh_bwd(
    res: tuple[jax.Array, jax.Array], ct: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array]:
    """PRX 9, 031041 Eq. (3) with ``1/x -> safe_inverse(x)``. ``F``'s diagonal is zero
    for free (see ``_safe_inverse``), and the result is Hermitian by construction:
    ``F`` is antisymmetric and ``aV`` antihermitian. Same cotangent-convention ``conj``
    as ``_svd_bwd``."""
    w, v = res
    dw, dv = jnp.real(ct[0]), jnp.conj(ct[1])
    f = _safe_inverse(w[None, :] - w[:, None], _EPSILON)
    return (jnp.conj(v @ (jnp.diag(dw) + f * _antiherm(_dag(v) @ dv)) @ _dag(v)),)


_eigh.defvjp(_eigh_fwd, _eigh_bwd)


# --- the seam ----------------------------------------------------------------------


def _svd_dispatch(b: jax.Array, full_matrices: bool = True, **kwargs: Any):
    """What ``ar.do("linalg.svd", jax_array, ...)`` resolves to after [install][tenet.ad.install].

    Anything but the compact factorization is handed straight back to JAX with its stock
    VJP: ``full_matrices=True`` returns different shapes than the rule above assumes, and
    this registration is process-global, so a third-party caller must not silently get
    the wrong factorization. ``tenet.ops.linalg`` only ever asks for the compact one.
    """
    if full_matrices or kwargs:
        return jnp.linalg.svd(b, full_matrices=full_matrices, **kwargs)
    return _svd(b)


def _eigh_dispatch(b: jax.Array, **kwargs: Any):
    """As ``_svd_dispatch``: a non-default ``UPLO`` goes back to JAX untouched."""
    return jnp.linalg.eigh(b, **kwargs) if kwargs else _eigh(b)


def install(*, epsilon: float = 1e-12) -> None:
    """Swap JAX's stock ``linalg.svd``/``linalg.eigh`` VJPs for broadened ones.

    Parameters
    ----------
    epsilon : float, optional
        The Lorentzian broadening, in units of sigma squared; the default is
        the PRX value and assumes an ``O(1)``-normalized spectrum. Read at
        trace time, so set it before the hot loop.

    Notes
    -----
    Process-global for the JAX backend, by design -- see the module docstring.
    Idempotent.
    """
    global _EPSILON
    _EPSILON = epsilon
    ar.register_function("jax", "linalg.svd", _svd_dispatch)
    ar.register_function("jax", "linalg.eigh", _eigh_dispatch)


def uninstall() -> None:
    """Restore autoray's stock JAX bindings. Idempotent.

    ``register_function`` writes ``_FUNCS[backend, name]``; *deleting* that entry is what
    restores the stock binding, because the next lookup falls back through autoray's
    import machinery and reapplies autoray's own wrapper for the function. Registering a
    remembered "old" callable would instead leave a second override in place. Private
    attributes, knowingly: autoray ships an install seam and no matching uninstall one,
    and the only real caller is the test that needs stock and broadened gradients in the
    same process.
    """
    for name in _NAMES:
        ar.autoray._FUNCS.pop(("jax", name), None)
    ar.autoray._NAMESPACE_CACHE.clear()  # as register_function itself does
