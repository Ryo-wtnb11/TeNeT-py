"""The public tensor type: a :class:`TensorStructure` plus ordered reduced blocks.

Fields are ``(structure, blocks)``, never ``(legs, blocks)``: the structure is the
static, hashable half (a JAX treedef in Milestone 6) and ``blocks`` is the clean
parameter tree of dynamic leaves, ordered by ``structure.block_order`` (invariant
8). ``T.legs``, ``T.domain``, ``T.codomain``, ``T.block(key)`` and ``T.items()``
are derived views; ``from_legs`` supplies the README's ergonomics.

:meth:`SymmetricTensor.to_dense` is the only NumPy-assuming code here and the only
way to densify — there is deliberately no ``__array__`` (invariant 9). Its layout
convention, fixed once and depended on downstream: axis ``i`` has length
``legs[i].space.dim``; sectors occupy contiguous slabs in the space's canonical
order; within sector ``a``'s slab the index is ``alpha * d_a + m``.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import autoray as ar
import numpy as np

from tenet.fusion_tree import FusionTree
from tenet.leg import OUT, Leg
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry.base import ClebschGordan, FusionProvider, requires

__all__ = ["SymmetricTensor"]

Array = Any
"""Milestone 1 keeps blocks NumPy; ``autoray`` dispatch is Milestone 2."""


@dataclass(frozen=True, slots=True, eq=False)
class SymmetricTensor:
    """A symmetric tensor: categorical structure plus one reduced block per key."""

    structure: TensorStructure
    blocks: tuple[Array, ...]

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        object.__setattr__(self, "blocks", blocks)
        order = self.structure.block_order
        if len(blocks) != len(order):
            raise ValueError(f"expected {len(order)} blocks, got {len(blocks)}")
        for i, (key, block) in enumerate(zip(order, blocks, strict=True)):
            expected = self.structure.block_shape(key)
            if tuple(block.shape) != expected:
                raise ValueError(
                    f"block {i} has shape {tuple(block.shape)}, expected {expected} for {key}"
                )
        dtypes = {block.dtype for block in blocks}
        if len(dtypes) > 1:
            raise ValueError(f"blocks must share one dtype, got {sorted(map(str, dtypes))}")

    # --- constructors ---------------------------------------------------------

    @classmethod
    def from_legs(cls, legs: Sequence[Leg], blocks: Sequence[Array]) -> "SymmetricTensor":
        """Build from public legs, in ``block_order``. The README's spelling."""
        return cls(TensorStructure(tuple(legs)), tuple(blocks))

    @classmethod
    def zeros(cls, legs: Sequence[Leg], dtype: Any = np.float64) -> "SymmetricTensor":
        structure = TensorStructure(tuple(legs))
        return cls(
            structure,
            tuple(np.zeros(structure.block_shape(k), dtype) for k in structure.block_order),
        )

    @classmethod
    def random(
        cls, legs: Sequence[Leg], *, seed: int | None = None, dtype: Any = np.float64
    ) -> "SymmetricTensor":
        """Standard-normal blocks from ``np.random.default_rng(seed)``, reproducible."""
        structure = TensorStructure(tuple(legs))
        rng = np.random.default_rng(seed)
        # ponytail: real draws cast to dtype; give complex dtypes a real+imag draw
        # if a test ever needs genuinely complex random data.
        return cls(
            structure,
            tuple(
                rng.standard_normal(structure.block_shape(k)).astype(dtype)
                for k in structure.block_order
            ),
        )

    # --- derived views --------------------------------------------------------

    @property
    def legs(self) -> tuple[Leg, ...]:
        return self.structure.legs

    @property
    def ndim(self) -> int:
        return self.structure.ndim

    @property
    def provider(self) -> FusionProvider:
        return self.structure.provider

    @property
    def codomain(self) -> tuple[Leg, ...]:
        """OUT legs in public axis order. A ``ProductSpace`` view arrives in M3."""
        return tuple(leg for leg in self.legs if leg.side is OUT)

    @property
    def domain(self) -> tuple[Leg, ...]:
        """IN legs in public axis order."""
        return tuple(leg for leg in self.legs if leg.side is not OUT)

    def block(self, key: FusionBlockKey) -> Array:
        """The stored block for ``key`` — the array itself, not a copy."""
        return self.blocks[self.structure.index_of(key)]

    def items(self) -> Iterator[tuple[FusionBlockKey, Array]]:
        return zip(self.structure.block_order, self.blocks, strict=True)

    # --- array-style properties -----------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Full **physical** dimension per public axis: ``Σ_a m_a d_a``.

        Equal to ``self.to_dense().shape``. Requires ``ClebschGordan`` (via
        ``GradedSpace.dim``) and raises ``CapabilityError`` without it — a
        provider with non-integer quantum dimensions has no physical shape, and
        silently returning :attr:`reduced_shape` would violate invariant 11.
        """
        return tuple(leg.space.dim for leg in self.legs)

    @property
    def reduced_shape(self) -> tuple[int, ...]:
        """Degeneracy dimension per public axis: ``Σ_a m_a``. Any provider.

        The storage-facing shape: what the reduced blocks are made of.
        """
        return tuple(leg.space.reduced_dim for leg in self.legs)

    @property
    def dtype(self) -> Any:
        """The single dtype shared by all blocks (``__post_init__`` validates it)."""
        return self._first_block().dtype

    @property
    def backend(self) -> str:
        """``"numpy"`` / ``"jax"`` / ``"torch"``, inferred from the first block.

        One tensor uses one backend; construction does not re-check every block,
        since ``to_backend`` is the only sanctioned way to move them.
        """
        return ar.infer_backend(self._first_block())

    @property
    def device(self) -> Any:
        """The first block's own ``.device`` (``None`` if it has none).

        A plain ``getattr``: autoray exposes no portable device accessor, and
        NumPy >= 2 arrays already carry ``.device == 'cpu'``.
        """
        return getattr(self._first_block(), "device", None)

    def _first_block(self) -> Array:
        if not self.blocks:
            raise ValueError(
                "tensor has no blocks: dtype/backend/device are undefined "
                f"(structure with legs {self.legs})"
            )
        return self.blocks[0]

    def to_backend(self, backend: str) -> "SymmetricTensor":
        """Same structure, blocks converted with ``ar.do("array", b, like=backend)``.

        The target backend's own dtype policy applies (JAX demotes float64 to
        float32 unless ``jax_enable_x64`` is set); no dtype is forced here.
        """
        return SymmetricTensor(
            self.structure, tuple(ar.do("array", b, like=backend) for b in self.blocks)
        )

    # --- parameter protocol (quimb / autoray) ---------------------------------

    def get_params(self) -> tuple[Array, ...]:
        """The blocks, in ``structure.block_order`` — a pytree of backend arrays.

        The identity, deliberately: ``blocks`` was chosen to be an ordered tuple
        so that no dict ordering or key hashing ever enters the dynamic data.
        """
        return self.blocks

    def set_params(self, params: Sequence[Array]) -> "SymmetricTensor":
        """Same structure, new numerical data. A **new** tensor; ``self`` is untouched.

        Goes through the ordinary constructor, so block count and per-block shape
        are validated by ``__post_init__``.

        ponytail: quimb's ``inject_variables`` may expect this to mutate in place.
        quimb is not a dependency, so this is not guessed at here — the frozen
        structure is what the JAX story rests on. If M8 finds quimb needs it, a
        thin mutable adapter belongs there, not in the core type.
        """
        return SymmetricTensor(self.structure, tuple(params))

    def copy(self) -> "SymmetricTensor":
        """A new instance sharing the same structure and block objects."""
        return SymmetricTensor(self.structure, self.blocks)

    # --- value semantics ------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Exact equality: same ``structure`` and every block exactly equal.

        Never raises on a structure mismatch. Under a JAX trace ``bool()`` of a
        traced comparison does raise, correctly: ``==`` is a concrete-value
        question (invariants 9/10).
        """
        if not isinstance(other, SymmetricTensor):
            return NotImplemented
        if self.structure != other.structure:
            return False
        return all(
            bool(ar.do("all", a == b)) for a, b in zip(self.blocks, other.blocks, strict=True)
        )

    __hash__ = None  # type: ignore[assignment]  # holds arrays; T.structure is the hashable half

    # --- arithmetic -----------------------------------------------------------
    # One-line wrappers over tenet.ops.basic, with function-local imports so the
    # dependency edge stays one-way (ops -> tensor). No __iadd__: the tensor is
    # frozen and Python's rebinding fallback is the intended behaviour.

    def _ops(self):
        from tenet.ops import basic

        return basic

    def __add__(self, other: "SymmetricTensor") -> "SymmetricTensor":
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"cannot add {type(other).__name__} to a SymmetricTensor: adding a scalar "
                "is not equivariant and would break the symmetry"
            )
        return self._ops().add(self, other)

    def __sub__(self, other: "SymmetricTensor") -> "SymmetricTensor":
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"cannot subtract {type(other).__name__} from a SymmetricTensor: "
                "scalar shifts are not equivariant"
            )
        return self._ops().subtract(self, other)

    def __neg__(self) -> "SymmetricTensor":
        return self._ops().negative(self)

    def __mul__(self, s: Any) -> "SymmetricTensor":
        """Scalar multiplication only; a ``SymmetricTensor`` operand is a ``TypeError``."""
        return self._ops().multiply(self, s)

    __rmul__ = __mul__

    def __truediv__(self, s: Any) -> "SymmetricTensor":
        return self._ops().divide(self, s)

    def conj(self) -> "SymmetricTensor":
        """Conjugate the blocks; ``legs`` unchanged. See :func:`tenet.conj`."""
        return self._ops().conj(self)

    def norm(self) -> float:
        """qdim-weighted Frobenius norm. See :func:`tenet.norm`."""
        return self._ops().norm(self)

    def transpose(self, *axes: Any) -> "SymmetricTensor":
        """``T.transpose(2, 0, 1)``, ``T.transpose((2, 0, 1))`` or ``T.transpose()``.

        The last form reverses all axes (NumPy convention). See
        :func:`tenet.transpose`; no leg changes ``side``.
        """
        from tenet.ops import permutation

        if len(axes) == 1 and (axes[0] is None or isinstance(axes[0], Sequence)):
            axes = tuple(axes[0] or ())
        return permutation.transpose(self, axes or None)

    # --- fusion ---------------------------------------------------------------

    def fuse(self, *axes: Any) -> "SymmetricTensor":
        """``T.fuse(0, 1)`` or ``T.fuse((0, 1))``. See :func:`tenet.fuse`."""
        from tenet.ops import fusion

        if len(axes) == 1 and not isinstance(axes[0], int):
            axes = tuple(axes[0])
        return fusion.fuse(self, axes)

    def unfuse(self, axis: int, legs: Sequence[Leg]) -> "SymmetricTensor":
        """Split ``axis`` into ``legs``. See :func:`tenet.unfuse`."""
        from tenet.ops import fusion

        return fusion.unfuse(self, axis, legs)

    def __repr__(self) -> str:
        def safe(get) -> Any:
            try:
                return get()
            except Exception:
                return "?"

        return (
            f"SymmetricTensor(ndim={self.ndim}, shape={safe(lambda: self.shape)}, "
            f"dtype={safe(lambda: self.dtype)}, backend={safe(lambda: self.backend)!r}, "
            f"blocks={len(self.blocks)})"
        )

    # --- dense expansion ------------------------------------------------------

    def to_dense(self) -> np.ndarray:
        """``T = Σ_τ A^(τ) ⊗ C^(τ)`` expanded into a plain NumPy array.

        Explicit by design (invariant 9). Requires ``ClebschGordan``; legs with
        ``dual=True`` need the Z-isomorphism and Frobenius-Schur signs (M4).
        """
        provider = self.provider
        requires(provider, ClebschGordan)
        for i, leg in enumerate(self.legs):
            if leg.dual:
                raise NotImplementedError(
                    f"to_dense: axis {i} has dual=True; the Z-isomorphism is Milestone 4"
                )

        n = self.ndim
        order = (*self.structure.out_axes, *self.structure.in_axes)
        dtype = np.result_type(self.blocks[0].dtype, np.float64) if self.blocks else np.float64
        dense = np.zeros(tuple(leg.space.dim for leg in self.legs), dtype=dtype)

        # einsum subscripts: A carries the degeneracy index of each public axis,
        # C carries the irrep index but in (out..., in...) axis order.
        a_sub = list(range(n))
        c_sub = [n + ax for ax in order]
        out_sub = [x for i in range(n) for x in (i, n + i)]

        for key, block in self.items():
            if not block.any():
                continue
            xout = _tree_cgt(provider, key.output_tree)
            xin = _tree_cgt(provider, key.input_tree)
            cgt = np.tensordot(xout, xin.conj(), axes=([-1], [-1]))
            full = np.einsum(block, a_sub, cgt, c_sub, out_sub)

            sectors = self.structure.axis_sectors(key)
            slabs = []
            shape = []
            for leg, a in zip(self.legs, sectors, strict=True):
                size = leg.degeneracy(a) * provider.irrep_dim(a)
                start = leg.space.sector_offset(a)
                slabs.append(slice(start, start + size))
                shape.append(size)
            # (m_0, d_0, m_1, d_1, ...) -> within-slab index alpha * d_a + m
            dense[tuple(slabs)] += full.reshape(shape)

        return dense


def _tree_cgt(provider: ClebschGordan, tree: FusionTree) -> np.ndarray:
    """A tree's CG tensor, shape ``(d_u0, ..., d_u{N-1}, d_coupled)``.

    Left-associated contraction along the spine. Rank 0 is the unit's ``(1,)``
    (so an empty side contracts like any other) and rank 1 is the identity.
    """
    dim = provider.irrep_dim
    if tree.rank == 0:
        return np.ones((dim(tree.coupled),))
    x = np.eye(dim(tree.uncoupled[0]))
    for e, u, f, mu in tree.vertices():
        x = np.tensordot(x, provider.cgc(e, u, f)[..., mu], axes=([-1], [0]))
    return x
