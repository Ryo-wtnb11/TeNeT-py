"""The public tensor type: a [TensorStructure][tenet.TensorStructure] plus ordered reduced blocks.

Fields are ``(structure, blocks)``, never ``(legs, blocks)``: the structure is the
static, hashable half (a JAX treedef in Milestone 6) and ``blocks`` is the clean
parameter tree of dynamic leaves, ordered by ``structure.block_order`` (invariant
8). ``T.legs``, ``T.domain``, ``T.codomain``, ``T.block(key)`` and ``T.items()``
are derived views; ``from_legs`` supplies the docs/design.md's ergonomics.

[to_dense][tenet.SymmetricTensor.to_dense] and [from_dense][tenet.SymmetricTensor.from_dense]
are the only way to cross into the dense basis — there is deliberately no ``__array__``
(invariant 9) — and both are thin delegations to ``tenet.ops.dense``, which
owns the layout convention, the plan cache and the only NumPy in the boundary.
That convention, fixed once and depended on downstream: axis ``i`` has length
``legs[i].space.dim``; sectors occupy contiguous slabs in the space's canonical
order; within sector ``a``'s slab the index is ``alpha * d_a + m``.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import autoray as ar
import numpy as np

from tenet.leg import OUT, Leg
from tenet.structure import FusionBlockKey, TensorStructure
from tenet.symmetry.base import _DualFusionRules

if TYPE_CHECKING:
    from types import ModuleType

    from tenet.map_view import TensorMapView

__all__ = ["SymmetricTensor"]

Array = Any
"""Milestone 1 keeps blocks NumPy; ``autoray`` dispatch is Milestone 2."""


@dataclass(frozen=True, slots=True, eq=False)
class SymmetricTensor:
    """A symmetric tensor: categorical structure plus one reduced block per key.

    Parameters
    ----------
    structure : TensorStructure
        The static, hashable half: legs and everything derived from them.
    blocks : tuple of array
        One reduced block per key, in ``structure.block_order``, all sharing
        one dtype.

    Raises
    ------
    ValueError
        If the number of blocks does not match ``block_order``, a block's shape
        does not match ``structure.block_shape(key)``, or the blocks do not
        share one dtype.

    Examples
    --------
    >>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    >>> from tenet.symmetry import U1, U1Sector
    >>> V = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 1})
    >>> t = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    >>> t.ndim, len(t.blocks)
    (2, 2)
    >>> t.shape, t.reduced_shape
    ((3, 3), (3, 3))
    >>> t.backend
    'numpy'
    """

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
        """Build from public legs, in ``block_order``. The docs/design.md's spelling.

        Parameters
        ----------
        legs : sequence of Leg
            The legs, in public axis order.
        blocks : sequence of array
            One block per key, in ``TensorStructure(legs).block_order``.

        Returns
        -------
        SymmetricTensor
            The assembled tensor; the constructor validates counts and shapes.
        """
        return cls(TensorStructure(tuple(legs)), tuple(blocks))

    @classmethod
    def zeros(cls, legs: Sequence[Leg], dtype: Any = np.float64) -> "SymmetricTensor":
        """All-zero blocks over ``legs``.

        Parameters
        ----------
        legs : sequence of Leg
            The legs, in public axis order.
        dtype : dtype, optional
            The blocks' dtype. Default ``np.float64``.

        Returns
        -------
        SymmetricTensor
            The zero tensor, NumPy blocks.
        """
        structure = TensorStructure(tuple(legs))
        return cls(
            structure,
            tuple(np.zeros(structure.block_shape(k), dtype) for k in structure.block_order),
        )

    @classmethod
    def random(
        cls, legs: Sequence[Leg], *, seed: int | None = None, dtype: Any = np.float64
    ) -> "SymmetricTensor":
        """Standard-normal blocks from ``np.random.default_rng(seed)``, reproducible.

        Parameters
        ----------
        legs : sequence of Leg
            The legs, in public axis order.
        seed : int or None, optional
            The RNG seed; ``None`` (the default) draws fresh entropy.
        dtype : dtype, optional
            The blocks' dtype. Default ``np.float64``.

        Returns
        -------
        SymmetricTensor
            The random tensor, NumPy blocks.
        """
        structure = TensorStructure(tuple(legs))
        rng = np.random.default_rng(seed)
        # Simplification: real draws cast to dtype; give complex dtypes a real+imag draw
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
        """The structure's legs, in public axis order.

        Returns
        -------
        tuple of Leg
            ``self.structure.legs``.
        """
        return self.structure.legs

    @property
    def ndim(self) -> int:
        """Number of legs.

        Returns
        -------
        int
            ``self.structure.ndim``.
        """
        return self.structure.ndim

    @property
    def provider(self) -> _DualFusionRules:
        """The legs' shared symmetry provider.

        Returns
        -------
        provider
            ``self.structure.provider``.
        """
        return self.structure.provider

    @property
    def codomain(self) -> tuple[Leg, ...]:
        """OUT legs in public axis order. A ``ProductSpace`` view arrives in M3.

        Returns
        -------
        tuple of Leg
            The OUT legs.
        """
        return tuple(leg for leg in self.legs if leg.side is OUT)

    @property
    def domain(self) -> tuple[Leg, ...]:
        """IN legs in public axis order.

        Returns
        -------
        tuple of Leg
            The IN legs.
        """
        return tuple(leg for leg in self.legs if leg.side is not OUT)

    def block(self, key: FusionBlockKey) -> Array:
        """The stored block for ``key`` — the array itself, not a copy.

        Parameters
        ----------
        key : FusionBlockKey
            A key of this tensor's structure.

        Returns
        -------
        array
            The block, in public axis order.

        Raises
        ------
        KeyError
            If ``key`` is foreign to the structure.
        """
        return self.blocks[self.structure.index_of(key)]

    def items(self) -> Iterator[tuple[FusionBlockKey, Array]]:
        """Iterate ``(key, block)`` pairs in ``block_order``.

        Yields
        ------
        tuple of (FusionBlockKey, array)
            Each key with its stored block.
        """
        return zip(self.structure.block_order, self.blocks, strict=True)

    # --- array-style properties -----------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Full **physical** dimension per public axis: ``Σ_a m_a d_a``.

        Equal to ``self.to_dense().shape``.

        Returns
        -------
        tuple of int
            One dense dimension per public axis.

        Raises
        ------
        CapabilityError
            If the provider lacks ``ClebschGordanData`` (via ``GradedSpace.dim``)
            — a provider with non-integer quantum dimensions has no physical
            shape, and silently returning
            [reduced_shape][tenet.SymmetricTensor.reduced_shape] would violate
            invariant 11.
        """
        return tuple(leg.space.dim for leg in self.legs)

    @property
    def reduced_shape(self) -> tuple[int, ...]:
        """Degeneracy dimension per public axis: ``Σ_a m_a``. Any provider.

        The storage-facing shape: what the reduced blocks are made of.

        Returns
        -------
        tuple of int
            One degeneracy dimension per public axis.
        """
        return tuple(leg.space.reduced_dim for leg in self.legs)

    @property
    def dtype(self) -> Any:
        """The single dtype shared by all blocks (``__post_init__`` validates it).

        Returns
        -------
        dtype
            The first block's dtype.

        Raises
        ------
        ValueError
            If the tensor has no blocks — the dtype is then undefined.
        """
        return self._first_block().dtype

    @property
    def backend(self) -> str:
        """``"numpy"`` / ``"jax"`` / ``"torch"``, inferred from the first block.

        Returns
        -------
        str
            The autoray backend name.

        Raises
        ------
        ValueError
            If the tensor has no blocks — the backend is then undefined.

        Notes
        -----
        One tensor uses one backend; construction does not re-check every block,
        since ``to_backend`` is the only sanctioned way to move them.
        """
        return ar.infer_backend(self._first_block())

    @property
    def device(self) -> Any:
        """The first block's own ``.device`` (``None`` if it has none).

        Returns
        -------
        device or None
            Whatever the backend exposes.

        Raises
        ------
        ValueError
            If the tensor has no blocks — the device is then undefined.

        Notes
        -----
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

        Parameters
        ----------
        backend : str
            The target autoray backend, e.g. ``"jax"``.

        Returns
        -------
        SymmetricTensor
            A new tensor on ``backend``.

        Notes
        -----
        The target backend's own dtype policy applies (JAX demotes float64 to
        float32 unless ``jax_enable_x64`` is set); no dtype is forced here.
        """
        return SymmetricTensor(
            self.structure, tuple(ar.do("array", b, like=backend) for b in self.blocks)
        )

    # --- parameter protocol (quimb / autoray) ---------------------------------

    def get_params(self) -> tuple[Array, ...]:
        """The blocks, in ``structure.block_order`` — a pytree of backend arrays.

        Returns
        -------
        tuple of array
            ``self.blocks``, the identity.

        Notes
        -----
        The identity, deliberately: ``blocks`` was chosen to be an ordered tuple
        so that no dict ordering or key hashing ever enters the dynamic data.
        """
        return self.blocks

    def set_params(self, params: Sequence[Array]) -> "SymmetricTensor":
        """Same structure, new numerical data. A **new** tensor; ``self`` is untouched.

        Parameters
        ----------
        params : sequence of array
            The new blocks, in ``structure.block_order``.

        Returns
        -------
        SymmetricTensor
            A new tensor over the same structure.

        Notes
        -----
        Goes through the ordinary constructor, so block count and per-block shape
        are validated by ``__post_init__``.

        """
        # Simplification: quimb's ``inject_variables`` may expect this to mutate in place.
        # quimb is not a dependency, so this is not guessed at here — the frozen structure
        # is what the JAX story rests on. If M8 finds quimb needs it, a thin mutable
        # adapter belongs there, not in the core type.
        return SymmetricTensor(self.structure, tuple(params))

    def copy(self) -> "SymmetricTensor":
        """A new instance sharing the same structure and block objects.

        Returns
        -------
        SymmetricTensor
            The shallow copy.
        """
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

    def _ops(self) -> "ModuleType":
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

    def __matmul__(self, other: "SymmetricTensor") -> "SymmetricTensor":
        """Morphism composition ``a ∘ b``. See [tenet.compose][]."""
        if not isinstance(other, SymmetricTensor):
            raise TypeError(
                f"@ composes two SymmetricTensors, got {type(other).__name__}: "
                "for scalar multiplication write `t * s`"
            )
        from tenet.ops import map as map_ops

        return map_ops.compose(self, other)

    # --- map view -------------------------------------------------------------

    def as_map(self) -> "TensorMapView":
        """View this tensor as a morphism. Zero-copy.

        Returns
        -------
        TensorMapView
            The semantic view; see [TensorMapView][tenet.TensorMapView].
        """
        from tenet.map_view import as_map

        return as_map(self)

    def conj(self) -> "SymmetricTensor":
        """Conjugate the blocks; ``legs`` unchanged. See [tenet.conj][].

        Returns
        -------
        SymmetricTensor
            The blockwise complex conjugate.
        """
        return self._ops().conj(self)

    def adjoint(self) -> "SymmetricTensor":
        """``T†``: every leg's ``side`` flips, blocks are conjugated and key-swapped.

        Returns
        -------
        SymmetricTensor
            The adjoint. Not ``conj()`` (which touches no leg) and not a
            dualization. See [tenet.adjoint][].
        """
        from tenet.ops import map as map_ops

        return map_ops.adjoint(self)

    def norm(self) -> Any:
        """qdim-weighted Frobenius norm (a backend scalar). See [tenet.norm][].

        Returns
        -------
        scalar
            The norm, on this tensor's backend.
        """
        return self._ops().norm(self)

    # --- elementwise block maps (coefficient space, not dense space) -----------

    def apply_blocks(self, fn: Any) -> "SymmetricTensor":
        """``fn`` on each reduced block. See [tenet.apply_blocks][] for the caveat.

        Parameters
        ----------
        fn : callable
            Applied to each block; must preserve shape and be backend-generic.

        Returns
        -------
        SymmetricTensor
            The mapped tensor, same structure.
        """
        from tenet.ops import blocks

        return blocks.apply_blocks(self, fn)

    def block_sqrt(self) -> "SymmetricTensor":
        """Blockwise ``sqrt`` — *not* ``sqrt(self.to_dense())``. See [tenet.block_sqrt][].

        Returns
        -------
        SymmetricTensor
            The blockwise square root.
        """
        from tenet.ops import blocks

        return blocks.block_sqrt(self)

    def block_power(self, p: Any) -> "SymmetricTensor":
        """Blockwise ``self ** p`` for a scalar ``p``. See [tenet.block_power][].

        Parameters
        ----------
        p : scalar
            The exponent.

        Returns
        -------
        SymmetricTensor
            The blockwise power.
        """
        from tenet.ops import blocks

        return blocks.block_power(self, p)

    # --- serialization --------------------------------------------------------

    def save(self, path: Any, *, compress: bool = False) -> None:
        """Write to ``path`` as a single ``.npz``. See [tenet.save][].

        Parameters
        ----------
        path : path-like
            Where to write.
        compress : bool, optional
            Compress the archive. Default ``False``.
        """
        from tenet.serialize import save

        save(self, path, compress=compress)

    @classmethod
    def load(cls, path: Any) -> "SymmetricTensor":
        """Read a file written by [save][tenet.SymmetricTensor.save]; NumPy blocks.

        See [tenet.load][].

        Parameters
        ----------
        path : path-like
            The ``.npz`` file to read.

        Returns
        -------
        SymmetricTensor
            The loaded tensor, NumPy blocks.
        """
        from tenet.serialize import load

        return load(path)

    def transpose(self, *axes: Any) -> "SymmetricTensor":
        """``T.transpose(2, 0, 1)``, ``T.transpose((2, 0, 1))`` or ``T.transpose()``.

        Parameters
        ----------
        *axes : int or sequence of int
            The new axis order; empty (or ``None``) reverses all axes (NumPy
            convention).

        Returns
        -------
        SymmetricTensor
            The permuted tensor; see [tenet.transpose][] — no leg changes
            ``side``.
        """
        from tenet.ops import permutation

        if len(axes) == 1 and (axes[0] is None or isinstance(axes[0], Sequence)):
            axes = tuple(axes[0] or ())
        return permutation.transpose(self, axes or None)

    def repartition(self, outputs: Sequence[int], inputs: Sequence[int]) -> "SymmetricTensor":
        """``T.repartition(outputs=(0, 1), inputs=(2,))``. See ``tenet.repartition``.

        Parameters
        ----------
        outputs : sequence of int
            The public axes to place on the OUT side.
        inputs : sequence of int
            The public axes to place on the IN side.

        Returns
        -------
        SymmetricTensor
            The repartitioned tensor.

        Notes
        -----
        Every leg that crosses sides is *bent*: its ``side`` and its ``dual`` both
        flip. Requires ``BendingCoefficients`` unless no leg crosses.
        """
        from tenet.ops.repartition import repartition

        return repartition(self, outputs, inputs)

    # --- fusion ---------------------------------------------------------------

    def fuse(self, *axes: Any) -> "SymmetricTensor":
        """``T.fuse(0, 1)`` or ``T.fuse((0, 1))``. See [tenet.fuse][].

        Parameters
        ----------
        *axes : int or sequence of int
            The adjacent axes to fuse into one.

        Returns
        -------
        SymmetricTensor
            The fused tensor.
        """
        from tenet.ops import fusion

        if len(axes) == 1 and not isinstance(axes[0], int):
            axes = tuple(axes[0])
        return fusion.fuse(self, axes)

    def unfuse(self, axis: int, legs: Sequence[Leg]) -> "SymmetricTensor":
        """Split ``axis`` into ``legs``. See [tenet.unfuse][].

        Parameters
        ----------
        axis : int
            The fused axis to split.
        legs : sequence of Leg
            The constituent legs the axis splits into.

        Returns
        -------
        SymmetricTensor
            The unfused tensor.
        """
        from tenet.ops import fusion

        return fusion.unfuse(self, axis, legs)

    def embed(self, legs: Sequence[Leg]) -> "SymmetricTensor":
        """Zero-pad into larger, containing legs. See ``tenet.embed``.

        Parameters
        ----------
        legs : sequence of Leg
            The target legs, one per axis, each containing the current leg.

        Returns
        -------
        SymmetricTensor
            The embedded tensor.
        """
        from tenet.ops.embed import embed

        return embed(self, legs)

    def to_symmetry(
        self, target: _DualFusionRules, *, atol: float | None = None
    ) -> "SymmetricTensor":
        """Restrict to a smaller symmetry, e.g. SU(2) -> U(1). See ``tenet.to_symmetry``.

        Parameters
        ----------
        target : provider
            The smaller symmetry's provider.
        atol : float or None, optional
            Symmetry-check tolerance; ``None`` (the default) uses the default.

        Returns
        -------
        SymmetricTensor
            The tensor over ``target``.
        """
        from tenet.ops.cast import to_symmetry

        return to_symmetry(self, target, atol=atol)

    def restrict(self, legs: Sequence[Leg], *, atol: float | None = None) -> "SymmetricTensor":
        """Slice down to smaller, contained legs. See [tenet.restrict][].

        Parameters
        ----------
        legs : sequence of Leg
            The target legs, one per axis, each contained in the current leg.
        atol : float or None, optional
            Tolerance for the discarded weight check; ``None`` (the default)
            skips it.

        Returns
        -------
        SymmetricTensor
            The restricted tensor.
        """
        from tenet.ops.embed import restrict

        return restrict(self, legs, atol=atol)

    def direct_sum(self, other: "SymmetricTensor", axes: int | Sequence[int]) -> "SymmetricTensor":
        """``self ⊕ other`` along ``axes``. See [tenet.direct_sum][].

        Parameters
        ----------
        other : SymmetricTensor
            The other summand.
        axes : int or sequence of int
            The axes along which the spaces are summed.

        Returns
        -------
        SymmetricTensor
            The direct sum.
        """
        from tenet.ops.embed import direct_sum

        return direct_sum(self, other, axes)

    def __repr__(self) -> str:
        def safe(get: Callable[[], Any]) -> Any:
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

    def to_dense(self) -> Array:
        """``T = Σ_τ A^(τ) ⊗ C^(τ)`` expanded into a dense array of ``self``'s backend.

        Returns
        -------
        array
            The dense carrier-basis array, shape
            [shape][tenet.SymmetricTensor.shape].

        Raises
        ------
        CapabilityError
            If the provider lacks ``ClebschGordanData``; a leg with
            ``dual=True`` additionally requires ``DualBasis``.

        Notes
        -----
        Explicit by design (invariant 9). See
        ``tenet.ops.dense.to_dense`` — traceable and differentiable as of #82.
        """
        from tenet.ops.dense import to_dense

        return to_dense(self)

    @classmethod
    def from_dense(
        cls, dense: Array, legs: Sequence[Leg], *, atol: float | None = None
    ) -> "SymmetricTensor":
        """Project a dense carrier-basis array onto the symmetric subspace of ``legs``.

        Parameters
        ----------
        dense : array
            The dense array, in the layout convention of the module docstring.
        legs : sequence of Leg
            The legs describing each dense axis.
        atol : float or None, optional
            Tolerance for the symmetry check; ``None`` (the default) uses the
            default.

        Returns
        -------
        SymmetricTensor
            The projected tensor.

        Notes
        -----
        The inverse of [to_dense][tenet.SymmetricTensor.to_dense]; non-symmetric
        input is refused rather than silently projected. See
        ``tenet.ops.dense.from_dense``.
        """
        from tenet.ops.dense import from_dense

        return from_dense(dense, legs, atol=atol)
