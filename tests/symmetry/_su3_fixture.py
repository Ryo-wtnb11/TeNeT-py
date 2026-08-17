"""A test-only SU(3) provider reading the vendored ``tests/fixtures/su3_*.txt``.

SU(3) is the first symmetry tenet can reach with ``N^c_ab > 1`` (``8 x 8 -> 8``
has ``N = 2``), and #127's M12a plumbing has to be driven by *real* pentagon- and
hexagon-satisfying coefficients — synthetic ones would test the expansion against
data that satisfies no identity. The shipped provider lands in M12c on the racah
FFI; this one exists so the plumbing is testable with no wheel in CI, and the
fixtures it reads are reused verbatim as M12c's oracle.

**Truncated, and only valid because of it.** The fixtures cover the eleven
sectors up to ``27``; ``fusion`` returns the true SU(3) decomposition *filtered
to that set*. That is not a fusion category — it is a lookup table that happens
to be complete for the tensors the tests build. The ``to_dense`` cross-checks are
what prove the truncation loses nothing for those tensors: ``to_dense`` goes
through untouched CG code, so agreement is a genuine check of the new braid /
bend expansion and of the sector set at the same time.
"""

from dataclasses import dataclass
from functools import cache
from math import sqrt
from pathlib import Path

import numpy as np

from tenet.fusion_tree import FusionTree
from tenet.structure import FusionBlockKey
from tenet.symmetry.base import Sector, bend_braided, permute_braided_tree

FIXTURES = Path(__file__).parent.parent / "fixtures"

Dynkin = tuple[int, int]


@dataclass(frozen=True, slots=True, order=True)
class SU3Sector(Sector):
    """An SU(3) irrep, labelled by its Dynkin pair ``(p, q)``."""

    dynkin: Dynkin


def _rows(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURES / name, comments="#", ndmin=2)


def _load() -> tuple[dict, dict, dict, dict]:
    f: dict[tuple[Dynkin, ...], np.ndarray] = {}
    for row in _rows("su3_f.txt"):
        labels = tuple(tuple(int(x) for x in row[2 * i : 2 * i + 2]) for i in range(6))
        idx = tuple(int(x) for x in row[12:16])
        f.setdefault(labels, {})[idx] = row[16]

    r: dict[tuple[Dynkin, ...], dict] = {}
    for row in _rows("su3_r.txt"):
        labels = tuple(tuple(int(x) for x in row[2 * i : 2 * i + 2]) for i in range(3))
        r.setdefault(labels, {})[(int(row[6]), int(row[7]))] = row[8]

    g: dict[tuple[Dynkin, ...], dict] = {}
    for row in _rows("su3_cg.txt"):
        labels = tuple(tuple(int(x) for x in row[2 * i : 2 * i + 2]) for i in range(3))
        g.setdefault(labels, {})[tuple(int(x) for x in row[6:10])] = row[10]

    # The R table lists every triple with N > 0 over the fixture sector set, so it
    # is also the fusion rule and the N-symbol.
    n = {k: max(i for i, _ in v) + 1 for k, v in r.items()}
    return f, r, g, n


_F, _R, _CG, _N = _load()

_CHANNELS: dict[tuple[Dynkin, Dynkin], list[SU3Sector]] = {}
for _a, _b, _c in _N:
    _CHANNELS.setdefault((_a, _b), []).append(SU3Sector(_c))
_FUSION = {k: tuple(sorted(v)) for k, v in _CHANNELS.items()}


def _dense(entries: dict, shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(shape)
    for idx, val in entries.items():
        out[idx] = val
    out.flags.writeable = False
    return out


def _dim(d: Dynkin) -> int:
    p, q = d
    return (p + 1) * (q + 1) * (p + q + 2) // 2


@cache
def _f_block(labels: tuple[Dynkin, ...]) -> np.ndarray:
    a, b, c, d, e, f = labels
    shape = (_N.get((a, b, e), 0), _N.get((e, c, d), 0), _N.get((b, c, f), 0), _N.get((a, f, d), 0))
    if labels not in _F:
        if 0 in shape:
            raise KeyError(f"F^{a}{b}{c}_{d}[{e},{f}] has an empty vertex")
        raise KeyError(f"F^{a}{b}{c}_{d}[{e},{f}] is outside the vendored sector set")
    return _dense(_F[labels], shape)


@cache
def _cg_block(labels: tuple[Dynkin, ...]) -> np.ndarray:
    a, b, c = labels
    return _dense(_CG[labels], (_dim(a), _dim(b), _dim(c), _N[labels]))


@dataclass(frozen=True, slots=True)
class SU3FixtureProvider:
    """SU(3) over the vendored fixture sector set. Frozen, hashable, array-free."""

    @property
    def name(self) -> str:
        return "SU3-fixture"

    @property
    def unit(self) -> SU3Sector:
        return SU3Sector((0, 0))

    def dual(self, a: SU3Sector) -> SU3Sector:
        return SU3Sector((a.dynkin[1], a.dynkin[0]))

    def fusion(self, a: SU3Sector, b: SU3Sector) -> tuple[SU3Sector, ...]:
        return _FUSION.get((a.dynkin, b.dynkin), ())

    def n_symbol(self, a: SU3Sector, b: SU3Sector, c: SU3Sector) -> int:
        return _N.get((a.dynkin, b.dynkin, c.dynkin), 0)

    def qdim(self, a: SU3Sector) -> float:
        return float(_dim(a.dynkin))

    def irrep_dim(self, a: SU3Sector) -> int:
        return _dim(a.dynkin)

    def cgc(self, a: SU3Sector, b: SU3Sector, c: SU3Sector) -> np.ndarray:
        return _cg_block((a.dynkin, b.dynkin, c.dynkin))

    def z_matrix(self, a: SU3Sector) -> np.ndarray:
        """``Z_a: V_a -> V_a^*``, shape ``(d_a, d_dual(a))``.

        Derived exactly as SU(2)'s is: the singlet in ``V_a (x) V_dual(a)`` *is* the
        duality pairing, so it cannot drift out of gauge with the CG tensors
        ``to_dense`` contracts it against.
        """
        out = sqrt(_dim(a.dynkin)) * self.cgc(a, self.dual(a), self.unit)[:, :, 0, 0]
        out.flags.writeable = False
        return out

    # --- MultiplicityRecoupling ------------------------------------------

    def f_matrix(
        self,
        a: SU3Sector,
        b: SU3Sector,
        c: SU3Sector,
        d: SU3Sector,
        e: SU3Sector,
        f: SU3Sector,
    ) -> np.ndarray:
        return _f_block((a.dynkin, b.dynkin, c.dynkin, d.dynkin, e.dynkin, f.dynkin))

    def r_matrix(self, a: SU3Sector, b: SU3Sector, c: SU3Sector) -> np.ndarray:
        labels = (a.dynkin, b.dynkin, c.dynkin)
        return _r_block(labels)

    def b_matrix(self, a: SU3Sector, b: SU3Sector, c: SU3Sector) -> np.ndarray:
        """``B^{ab}_c = sqrt(d_a d_b / d_c) [F^{a b dual(b)}_a]_{c, 1}``.

        The same ``Bsymbol_from_Fsymbol`` derivation SU(2) uses, with the two
        one-dimensional vertices ``(b, dual(b), 1)`` and ``(a, 1, a)`` dropped —
        which leaves exactly the ``(N^c_ab, N^a_{c,dual(b)})`` shape.
        """
        block = self.f_matrix(a, b, self.dual(b), a, c, self.unit)
        scale = sqrt(_dim(a.dynkin) * _dim(b.dynkin) / _dim(c.dynkin))
        return scale * block[:, :, 0, 0]

    # --- RecouplingData (scalar shadows, valid where every vertex is N == 1) ---

    def f_symbol(
        self,
        a: SU3Sector,
        b: SU3Sector,
        c: SU3Sector,
        d: SU3Sector,
        e: SU3Sector,
        f: SU3Sector,
    ) -> complex:
        return complex(_scalar(self.f_matrix(a, b, c, d, e, f)))

    def r_symbol(self, a: SU3Sector, b: SU3Sector, c: SU3Sector) -> complex:
        return complex(_scalar(self.r_matrix(a, b, c)))

    def b_symbol(self, a: SU3Sector, b: SU3Sector, c: SU3Sector) -> complex:
        return complex(_scalar(self.b_matrix(a, b, c)))

    def frobenius_schur(self, a: SU3Sector) -> int:
        """``chi_a = sign([F^{a dual(a) a}_a]_{1,1})``, the TensorKit definition."""
        v = self.f_matrix(a, self.dual(a), a, a, self.unit, self.unit)[0, 0, 0, 0]
        return 1 if v > 0 else -1

    def flip_phase(self, a: SU3Sector) -> int:
        """``chi_a * theta_a = chi_a``: SU(3) braiding is symmetric, so the twist is 1."""
        return self.frobenius_schur(a)

    # --- the capabilities under test -------------------------------------

    def permute_tree(
        self, tree: FusionTree, perm: tuple[int, ...]
    ) -> tuple[tuple[FusionTree, complex], ...]:
        return permute_braided_tree(self, tree, perm)

    def bend_right(
        self, key: FusionBlockKey, *, dual: bool
    ) -> tuple[tuple[FusionBlockKey, complex], ...]:
        return bend_braided(self, key, right=True, dual=dual)

    def bend_left(
        self, key: FusionBlockKey, *, dual: bool
    ) -> tuple[tuple[FusionBlockKey, complex], ...]:
        return bend_braided(self, key, right=False, dual=dual)


@cache
def _r_block(labels: tuple[Dynkin, ...]) -> np.ndarray:
    n = _N[labels]
    return _dense(_R[labels], (n, n))


def _scalar(block: np.ndarray) -> float:
    if block.size != 1:
        raise AssertionError(f"scalar symbol asked for a {block.shape} multiplicity block")
    return float(block.reshape(()))


SU3 = SU3FixtureProvider()

# Dynkin labels of the vendored sector set.
ONE = SU3Sector((0, 0))
THREE = SU3Sector((1, 0))
THREEBAR = SU3Sector((0, 1))
SIX = SU3Sector((2, 0))
SIXBAR = SU3Sector((0, 2))
EIGHT = SU3Sector((1, 1))
