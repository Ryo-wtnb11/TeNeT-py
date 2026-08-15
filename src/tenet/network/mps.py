"""The containers: :class:`MPS`, :class:`MPO`, and the scalar exits they need.

Promoted from ``examples/dmrg.py`` (#110) with no arithmetic change: ``scalar`` :136-146,
``inner`` :149-157, ``spectrum`` :160-173, ``random_mps`` :266-274, ``_as_site`` :277-286
(now the :meth:`MPS.__setitem__` write barrier), ``canonicalize`` :289-306 (now
:meth:`MPS.canonize_`) and ``mpo`` :220-240 (now :meth:`MPO.from_w`).
"""

import string
from collections.abc import Iterable, Sequence
from typing import Any

import autoray as ar

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor

__all__ = ["MPO", "MPS", "inner", "scalar", "spectrum"]


# --- leaving the tensor world ------------------------------------------------------


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


# --- the state ----------------------------------------------------------------------


class MPS:
    """A finite open-boundary MPS: a mutable list of frozen ``SymmetricTensor``s.

    **Site convention**, pinned here once and enforced on every write::

        A_n : (left bond OUT, physical OUT, right bond IN)

    Charge flows left to right, ``bond_n (x) phys_n -> bond_{n+1}``, and both end bonds
    have ``D=1``; a non-unit sector on bond 0 targets that total charge (YASTN's
    charged-first-virtual-leg recipe, ``_initialize.py``:194). The convention is not
    invented here -- ``examples/dmrg.py``:57-59 and ``examples/vmc_mps.py``:69-74 both
    chose it independently and ``tests/integration/test_dmrg.py`` already pins it.

    **Why a mutable container does not violate REPOSITORY_RULES:30.** That rule protects
    *categorical* objects -- ``Leg``, ``GradedSpace``, ``TensorStructure``,
    ``SymmetricTensor`` -- whose identity is their metadata, and every tensor this class
    holds is still frozen. An MPS is a container of those plus an orthogonality centre
    that *moves*: a state machine, not a category. In-place methods therefore carry
    YASTN's trailing underscore (``canonize_``), because the invalidation discipline is
    the entire correctness content of a sweep and ``env.clear_(n, n + 1)`` reading as a
    mutation at the call site is worth the character it costs.

    ``center`` is one ``int | None``, ``None`` meaning "no claim made". Deliberately
    *not* TenPy's per-site ``form`` table plus singular values on ``L + 1`` bonds
    (``tenpy/networks/mps.py``:64-79, :2882-2933) and not YASTN's central block ``pC``
    (``_mps_parent.py``:39-40): those exist to serve ``get_B(form=)``, mixers and 1-site
    DMRG, none of which M11a ships. Both are the named upgrade paths, with their specs at
    those line numbers, if TDVP or 1-site DMRG ever lands.
    """

    sites: list[SymmetricTensor]
    center: int | None

    def __init__(self, sites: Iterable[SymmetricTensor], center: int | None = None) -> None:
        self.sites = []
        self.center = center
        for t in sites:
            self.sites.append(_as_site(t))

    # --- container ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sites)

    def __getitem__(self, n: int) -> SymmetricTensor:
        return self.sites[n]

    def __setitem__(self, n: int, t: SymmetricTensor) -> None:
        """The write barrier: normalize the partition, then validate.

        Every factorization in ``tenet.linalg`` lowers its input to a *map* first, so a
        rank-3 factor comes back on the map's partition -- a physical leg that was OUT in
        the codomain returns IN-dual in the domain. One :func:`tenet.repartition` to
        ``((0, 1), (2,))`` puts it back, and doing it *here* is what lets a factor from
        ``lq``, ``qr`` or ``svd_truncated`` be stored directly: no caller ever writes the
        bend out again, and forgetting it is no longer a silent structure mismatch.

        Both references do this at the same boundary -- TenPy transposes into
        ``['vL', 'p', 'vR']`` inside ``MPS.__init__`` (``mps.py``:1616, :1650) and YASTN's
        ``__setitem__`` rejects a non-int index and a wrong ``ndim``
        (``_mps_parent.py``:88-105).
        """
        if not isinstance(n, int):  # YASTN _mps_parent.py:88-105
            raise TypeError(f"MPS index must be an int, got {type(n).__name__}")
        self.sites[n] = _as_site(t)

    def __iter__(self):
        return iter(self.sites)

    def copy(self) -> "MPS":
        """A new container over the same frozen tensors."""
        return MPS(self.sites, self.center)

    # --- constructors ---------------------------------------------------------------

    @classmethod
    def random(cls, phys: GradedSpace, bonds: Sequence[GradedSpace], *, seed: int = 0) -> "MPS":
        """A random MPS on ``len(bonds) - 1`` sites over the given bond spaces.

        The library takes bond *spaces*; deciding which are reachable for a given
        symmetry and target charge is physics and stays in the caller
        (``examples/dmrg.py::bond_spaces``).
        """
        return cls(
            SymmetricTensor.random(
                (Leg(bonds[i], OUT), Leg(phys, OUT), Leg(bonds[i + 1], IN)), seed=seed + i
            )
            for i in range(len(bonds) - 1)
        )

    @classmethod
    def from_tensors(cls, tensors: Iterable[SymmetricTensor]) -> "MPS":
        """An MPS over already-built site tensors, each through the write barrier."""
        return cls(tensors)

    # --- state machine --------------------------------------------------------------

    def canonize_(self, to: int = 0) -> "MPS":
        """Right-canonicalize in place and return ``self`` -- YASTN ``canonize_(to='first')``.

        One ``tenet.linalg.lq`` per site from the right, mirroring ``orthogonalize_site_``
        (``_mps_obc.py``:245-300): ``A_n = L . Q`` with ``Q`` on the MPS convention and
        ``L`` absorbed into ``A_{n-1}``. ``lq`` rather than ``qr`` because ``qr`` would put
        the new bond on the *right* of the factor and leave the site tensor's left leg IN.

        Setup only: a two-site sweep leaves the state canonical by construction on the
        side it came from, which is precisely what an ``int`` centre records.

        ponytail: only ``to=0`` (fully right-canonical) is implemented, because that is
        the one form the sweep's setup wants. Ceiling: a general ``to`` is the same loop
        run from both ends, and YASTN's ``canonize_(to='last')`` (``_mps_obc.py``:390) is
        the spelling to copy the day a caller needs it.
        """
        if to != 0:
            raise NotImplementedError("only to=0 (right-canonical) is implemented; see the note")
        for n in range(len(self) - 1, 0, -1):
            left, q = tenet.linalg.lq(self[n], ((0,), (1, 2)))
            self[n] = q
            self[n - 1] = tenet.einsum("apx,xy->apy", self[n - 1], left)
        self[0] = self[0] / tenet.norm(self[0])
        self.center = to
        return self

    def norm(self) -> float:
        """``sqrt(<psi|psi>)`` by one bra-ket transfer pass, closed with :func:`scalar`.

        No dense expansion and no environment object: two ``tenet.einsum`` calls per site,
        the same pairwise shape :meth:`Env.update_` uses with the MPO row removed.
        """
        t = tenet.einsum("apR,apr->Rr", tenet.adjoint(self[0]), self[0])
        for n in range(1, len(self)):
            t = tenet.einsum("Rr,rps->Rps", t, self[n])
            t = tenet.einsum("RpS,Rps->Ss", tenet.adjoint(self[n]), t)
        return float(scalar(t)) ** 0.5

    def to_dense(self) -> Any:
        """The full ``d**N`` amplitude array, ``D=1`` boundaries dropped.

        Exponential in ``N``: an oracle exit for tests, and nothing an algorithm calls.
        """
        out = self[0]
        for n in range(1, len(self)):
            body = string.ascii_uppercase[:n]
            out = tenet.einsum(f"a{body}x,xpr->a{body}pr", out, self[n])
        return out.to_dense()[0, ..., 0]


def _as_site(t: SymmetricTensor) -> SymmetricTensor:
    """Put a rank-3 tensor on the MPS partition ``(l, p | r)``, or refuse it."""
    if t.ndim != 3:
        raise ValueError(f"an MPS site tensor is rank 3, got rank {t.ndim}")
    site = tenet.repartition(t, (0, 1), (2,))
    sides = tuple(leg.side for leg in site.legs)
    if sides != (OUT, OUT, IN):  # repartition guarantees it; the claim is stated anyway
        raise ValueError(f"an MPS site is (bond OUT, phys OUT, bond IN), got {sides}")
    return site


# --- the Hamiltonian ----------------------------------------------------------------


class MPO:
    """A finite MPO: one rank-4 ``SymmetricTensor`` per site, ``(wl IN, p OUT, p IN, wr OUT)``.

    Invariance reads ``q(p_out) + q(wr) = q(wl) + q(p_in)``. The first and last sites
    carry a ``D=1`` boundary MPO bond, which is what makes *every* ``W_n`` rank 4 and
    removes the boundary-vector special case.

    **A separate class from :class:`MPS`, with no shape flag.** YASTN unifies the two
    behind ``_nr_phys in {1, 2}`` (``_mps_obc.py``:223-225) and pays a runtime branch on
    that flag at :284, :291, :438, :443 and :90-100 -- inside the code whose whole job is
    structural bookkeeping, which is the pattern tenet is typed to avoid. TenPy agrees for
    its own reason (``mpo.py``:16-18: "unlike for an MPS, this doesn't simplify
    calculations. Thus, an MPO has no ``form``"). Two classes, no branch.
    """

    sites: list[SymmetricTensor]

    def __init__(self, sites: Iterable[SymmetricTensor]) -> None:
        self.sites = list(sites)

    def __len__(self) -> int:
        return len(self.sites)

    def __getitem__(self, n: int) -> SymmetricTensor:
        return self.sites[n]

    def __iter__(self):
        return iter(self.sites)

    @classmethod
    def from_w(
        cls,
        w: Any,
        n_sites: int,
        *,
        phys: GradedSpace,
        bond: GradedSpace,
        boundary: GradedSpace,
        start: int,
        end: int,
    ) -> "MPO":
        """One dense bulk ``W`` plus a graded MPO bond -> first / bulk / last.

        ``w`` is indexed ``[wl, p_out, p_in, wr]``. The first site keeps only row
        ``start`` and the last only column ``end``, each on a ``D=1`` ``boundary`` MPO leg.

        ``SymmetricTensor.from_dense`` is called at its **default** relative ``atol``
        (``src/tenet/ops/dense.py``:301), so a wrong grading *raises* rather than
        projecting -- and that refusal is the proof the grading is right in a way a
        passing ``allclose`` is not.

        This is the only builder. A term-list front end (YASTN's ``Hterm`` /
        ``generate_mpo``, ``_generate_mpo.py``:73-298, a 186-line body with fermionic sign
        canonicalization and its own compressing SVD sweep) is refused, not deferred: it
        is the right API for a library accepting arbitrary Hamiltonians from users it has
        never met, and it becomes a real issue the day a second Hamiltonian appears.
        """

        def make(array: Any, left: GradedSpace, right: GradedSpace) -> SymmetricTensor:
            legs = (Leg(left, IN), Leg(phys, OUT), Leg(phys, IN), Leg(right, OUT))
            return SymmetricTensor.from_dense(array, legs)

        first = make(w[start : start + 1], boundary, bond)
        bulk = make(w, bond, bond)
        last = make(w[:, :, :, end : end + 1], bond, boundary)
        return cls([first, *[bulk] * (n_sites - 2), last])
