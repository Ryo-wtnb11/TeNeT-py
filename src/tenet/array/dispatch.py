"""``autoray`` registration for :class:`~tenet.tensor.SymmetricTensor`.

Registration only — no numerical logic lives here, ever. Every entry is an
already-implemented Milestone 2 function; the list is **closed**, so
``ar.do("exp", ...)``, ``ar.do("svd", ...)`` and friends raise rather than leak
through to a backend (invariant 11). ``"reshape"`` is registered only to *refuse*
in our own voice — a shape tuple has no categorical meaning here — so the
defined-operation list is unchanged by its presence. Adding a
name here is the deliberate act of declaring its categorical meaning defined —
``"tensordot"`` and ``"trace"`` joined the list with #51 and ``"einsum"`` with
#52. ``"einsum"`` needs no ``ar.register_dispatch``: autoray ships an
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
    such as quimb's ``tensor_split`` — ``do("fuse", x, left, right)`` — used to
    get a bare ``TypeError``. Adapt here, and refuse the multi-group form the
    way every other undefined operation is refused. ``tenet.fuse``'s own
    signature, the one the README documents, is untouched.
    """
    if len(axes_groups) != 1:
        raise ValueError(
            "fuse: autoray's multi-group form fuse(t, g0, g1, ...) is a reshape in "
            "disguise: it asks for a matricisation whose leg order and sides are not "
            "determined by the groups alone. Fuse one group at a time: "
            "tenet.fuse(t, (0, 1))"
        )
    return fuse(t, axes_groups[0])  # type: ignore[arg-type]


def _reshape(t: SymmetricTensor, *args: object, **kwargs: object) -> SymmetricTensor:
    """Refuse ``reshape`` in our own voice rather than as autoray's ``ImportError``."""
    raise ValueError(
        "reshape by shape is not defined for a symmetric tensor; a tuple of physical "
        "dimensions does not say how graded spaces are to be fused or split. The "
        "categorical operation is fuse/unfuse over named axes (README 'reshape and "
        "fusion'): tenet.fuse(t, (0, 1)), tenet.unfuse(t, 0)."
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
