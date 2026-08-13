"""The array-protocol layer: ``autoray`` registration only.

Imported for its side effect from the end of ``tenet/__init__.py``. The
``get_params`` / ``set_params`` / ``copy`` half of the protocol lives as methods
on :class:`~tenet.tensor.SymmetricTensor` itself.
"""

from tenet.array import dispatch  # noqa: F401  (import side effect: registration)

__all__ = ["dispatch"]
