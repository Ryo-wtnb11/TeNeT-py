"""``autoray`` registration for :class:`~tenet.tensor.SymmetricTensor`.

Registration only — no numerical logic lives here, ever. Every entry is an
already-implemented Milestone 2 function; the list is **closed**, so
``ar.do("einsum", ...)``, ``ar.do("reshape", ...)``, ``ar.do("exp", ...)``,
``ar.do("svd", ...)`` and friends raise rather than leak through to a backend
(invariant 11). Adding a name here is the deliberate act of declaring its
categorical meaning defined — ``"tensordot"`` and ``"trace"`` joined the list
with #51, and ``"einsum"`` waits for #52.

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

for _name, _fn in {
    "transpose": transpose,
    "conj": conj,
    "conjugate": conj,
    "linalg.norm": norm,
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "negative": negative,
    "fuse": fuse,
    "tensordot": tensordot,
    "trace": trace,
    "shape": lambda t: t.shape,
    "ndim": lambda t: t.ndim,
    # the only densifying registration: an explicit request, unlike np.asarray
    # (which must still not densify — invariant 9).
    "to_numpy": lambda t: t.to_dense(),
}.items():
    ar.register_function("tenet", _name, _fn)

del _name, _fn
