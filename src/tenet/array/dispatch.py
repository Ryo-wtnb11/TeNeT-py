"""``autoray`` registration for :class:`~tenet.tensor.SymmetricTensor`.

Registration only — no numerical logic lives here, ever. Every entry is an
already-implemented ``tenet`` function; the list is **closed**, so
``ar.do("exp", ...)``, ``ar.do("svd", ...)`` and friends raise rather than leak
through to a backend (invariant 11). ``"reshape"`` is registered only to *refuse*
in our own voice — a shape tuple has no categorical meaning here — so the
defined-operation list is unchanged by its presence. Adding a
name here is the deliberate act of declaring its categorical meaning defined.
``"einsum"`` needs no ``ar.register_dispatch``: autoray ships an
``einsum_dispatcher`` that infers the backend from *all* arguments, and a
``str`` equation ranks below a ``SymmetricTensor``, so the string first argument
resolves here on its own.

autoray would find most of these anyway — it infers a class's backend from the
module it is defined in and then looks the function up in that module, and
``tenet.transpose`` / ``tenet.conj`` / ``tenet.norm`` are module-level. The
mapping is stated explicitly regardless: one line each, survives moving the
class between modules, and makes ``"tenet"`` a documented fact rather than a
heuristic's accident. This mirrors symmray's ``interface.py``.
"""

from collections.abc import Callable

import autoray as ar

from tenet.ops import (
    add,
    conj,
    einsum,
    fuse,
    multiply,
    negative,
    norm,
    subtract,
    tensordot,
    trace,
    transpose,
)
from tenet.tensor import SymmetricTensor

__all__: list[str] = []

ar.register_backend(SymmetricTensor, "tenet")


def _fuse(t: SymmetricTensor, *axes_groups: object) -> SymmetricTensor:
    """autoray's ``fuse(x, *axes_groups)`` convention over :func:`tenet.fuse`.

    Registering a name under autoray means agreeing to autoray's calling
    convention for it; ``tenet.fuse(t, axes)`` takes a single group, so callers
    such as quimb's ``tensor_split`` — ``do("fuse", x, left, right)`` — would
    otherwise get a bare ``TypeError``. Adapt here, and refuse the multi-group form
    the way every other undefined operation is refused. ``tenet.fuse``'s own
    signature is untouched.
    """
    if len(axes_groups) != 1:
        raise ValueError(
            "fuse: autoray's multi-group form fuse(t, g0, g1, ...) is a reshape in "
            "disguise: it asks for a matricisation whose leg order and sides are not "
            "determined by the groups alone. Fuse one group at a time: "
            "tenet.fuse(t, (0, 1))"
        )
    # autoray hands the groups through untyped varargs, so the single group
    # arrives as ``object``; the adapter contract above makes it a Sequence[int].
    return fuse(t, axes_groups[0])  # ty: ignore[invalid-argument-type]  # untyped autoray varargs


def _elementwise_refused(name: str) -> "Callable[..., SymmetricTensor]":
    """Refuse a dense-elementwise name in our own voice.

    The blockwise maps are spelled ``tenet.block_sqrt`` / ``tenet.block_power``, so
    autoray's module lookup finds nothing under ``"sqrt"`` or ``"power"`` and the
    *wrong* number cannot leak. These two registrations are therefore a courtesy:
    a better answer than autoray's ``ImportError`` for a name whose dense meaning
    genuinely does not exist here. autoray's ``"sqrt"`` is the dense elementwise
    one, and for a non-Abelian provider the two are different operations — off by
    ``1.673`` on a dense scale of ``3.82`` for a rank-3 SU(2) tensor. Same
    precedent as ``"reshape"``, applied the same way:
    the numpy name is refused and this package's own operation is spelled
    differently (``fuse``/``unfuse`` there, ``block_sqrt``/``block_power`` here)
    rather than rebound. A refusal about meaning, not about effort — and, like
    ``"reshape"``, no addition to the defined-operation list.
    """

    def refuse(t: SymmetricTensor, *args: object, **kwargs: object) -> SymmetricTensor:
        raise ValueError(
            f"{name} is not defined for a symmetric tensor: autoray's {name!r} is the dense "
            "elementwise operation, and no non-linear function commutes with "
            "T = sum_tau A^(tau) (x) C^(tau) (1.673 off on a dense scale of 3.82 "
            f"for SU(2)). The blockwise map on the coefficients is tenet.block_{name}(t) / "
            "tenet.apply_blocks(t, fn); for dense semantics, densify explicitly."
        )

    return refuse


def _reshape(t: SymmetricTensor, *args: object, **kwargs: object) -> SymmetricTensor:
    """Refuse ``reshape`` in our own voice rather than as autoray's ``ImportError``."""
    raise ValueError(
        "reshape by shape is not defined for a symmetric tensor; a tuple of physical "
        "dimensions does not say how graded spaces are to be fused or split. The "
        "categorical operation is fuse/unfuse over named axes: "
        "tenet.fuse(t, (0, 1)), tenet.unfuse(t, 0)."
    )


def _flip(t: SymmetricTensor, *args: object, **kwargs: object) -> SymmetricTensor:
    """Refuse ``flip`` in our own voice.

    ``numpy.flip`` reverses element order along an axis and takes ``(t, axes)``,
    which is also the shape of a duality toggle -- so a bare ``flip`` under
    autoray's module lookup would be a wrong answer with no error. The duality
    toggle is named ``tenet.flip_dual``, which closes that lookup; this
    registration replaces autoray's ``ImportError`` with the sentence that says
    which operation was meant.
    """
    raise ValueError(
        "flip is not defined for a symmetric tensor: autoray's 'flip' reverses element "
        "order along an axis, and no element of a symmetric tensor may move without its "
        "sector moving with it. The duality toggle — relabel a leg's space through "
        "provider.dual and pay the Z-isomorphism's scalar — is tenet.flip_dual(t, axes); "
        "for dense semantics, densify explicitly."
    )


for _name, _fn in {
    "transpose": transpose,
    "conj": conj,
    "conjugate": conj,
    "linalg.norm": norm,
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "negative": negative,
    "fuse": _fuse,
    # not an addition to the list: an explicit, categorical refusal in place of
    # autoray's "couldn't find function 'reshape' for backend 'tenet'".
    "reshape": _reshape,
    "flip": _flip,
    # likewise not additions: explicit refusals for three numpy names whose dense
    # meaning has no categorical counterpart here.
    "sqrt": _elementwise_refused("sqrt"),
    "power": _elementwise_refused("power"),
    "einsum": einsum,
    "tensordot": tensordot,
    "trace": trace,
    "shape": lambda t: t.shape,
    "ndim": lambda t: t.ndim,
    # the only densifying registration: an explicit request, unlike np.asarray
    # (which must still not densify — invariant 9). ``to_dense`` returns the
    # tensor's *own* backend as of #82, so the NumPy contract autoray attaches to
    # this name is honoured here, by one wrap and no new key.
    "to_numpy": lambda t: ar.to_numpy(t.to_dense()),
}.items():
    ar.register_function("tenet", _name, _fn)

del _name, _fn
