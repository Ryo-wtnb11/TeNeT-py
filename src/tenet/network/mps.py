"""The containers: :class:`MPS` and :class:`MPO`.

Promoted from ``examples/dmrg.py`` (#110) with no arithmetic change: ``random_mps``
:266-274, ``_as_site`` :277-286 (now the :meth:`MPS.__setitem__` write barrier),
``canonicalize`` :289-306 (now :meth:`MPS.canonize_`) and ``mpo`` :220-240 (now
:meth:`MPO.from_w`). ``scalar``, ``inner`` and ``spectrum`` lived here until #114 moved
them to :mod:`tenet.network.common`, where ``network/ctmrg.py`` can reach them without
importing a driver it shares no concept with; #126 then promoted the first two out of the
driver layer entirely, as :func:`tenet.full_trace` and :func:`tenet.inner`.
"""

import json
import pathlib
import string
from collections.abc import Iterable, Sequence
from typing import Any

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network.common import spectrum

__all__ = [
    "MPO",
    "MPS",
    "MPS_FORMAT_VERSION",
    "expectation_1site",
    "expectation_2site",
    "spectrum",
]

#: Version of the :meth:`MPS.save` *directory* layout -- ``mps.json`` plus the ``NNN.npz``
#: naming. Distinct from ``tenet.serialize.FORMAT_VERSION``, which versions the per-tensor
#: ``.npz`` header: two formats, two owners, so bumping one is not a lie about the other.
MPS_FORMAT_VERSION = 1


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

        Simplification: only ``to=0`` (fully right-canonical) is implemented, because that is
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
        """``sqrt(<psi|psi>)`` by one bra-ket transfer pass, closed with :func:`tenet.full_trace`.

        No dense expansion and no environment object: two ``tenet.einsum`` calls per site,
        the same pairwise shape :meth:`Env.update_` uses with the MPO row removed.
        """
        t = tenet.einsum("apR,apr->Rr", tenet.adjoint(self[0]), self[0])
        for n in range(1, len(self)):
            t = tenet.einsum("Rr,rps->Rps", t, self[n])
            t = tenet.einsum("RpS,Rps->Ss", tenet.adjoint(self[n]), t)
        return float(tenet.full_trace(t)) ** 0.5

    def to_dense(self) -> Any:
        """The full ``d**N`` amplitude array, ``D=1`` boundaries dropped.

        Exponential in ``N``: an oracle exit for tests, and nothing an algorithm calls.
        """
        out = self[0]
        for n in range(1, len(self)):
            body = string.ascii_uppercase[:n]
            out = tenet.einsum(f"a{body}x,xpr->a{body}pr", out, self[n])
        return out.to_dense()[0, ..., 0]

    def compress_(self, *, chi: int, cutoff: float = 0.0) -> float:
        """Truncate to bond ``chi`` in place; return the **total** discarded weight.

        :meth:`canonize_` then one left-to-right ``svd_truncated`` sweep -- the per-bond
        body of :func:`~tenet.network.sweep_` with the eigensolver removed -- leaving
        ``center = len(self) - 1``. YASTN's ``truncate_`` (``_mps_obc.py``:379-413)
        instead *assumes* a canonical input and takes ``to=``; canonizing here costs an
        ``lq`` pass the caller usually needs anyway and removes the silently-wrong result
        on a non-canonical state.

        **The returned convention differs from** :func:`~tenet.network.sweep_`'s **on
        purpose.** ``sweep_`` returns the per-bond *maximum*, because it feeds a per-sweep
        convergence report where the worst bond is the diagnostic; this returns
        ``sqrt(sum_bond dw)`` (YASTN's "norm of the truncated elements normalized by the
        norm of the untruncated state"), because its caller is asking how much of the
        state it just threw away. Two conventions, two names -- which is the mitigation.
        """
        self.canonize_(0)
        total = 0.0
        for n in range(len(self) - 1):
            aa = tenet.einsum("apx,xqr->apqr", self[n], self[n + 1])
            u, s, vh = tenet.linalg.svd_truncated(aa, ((0, 1), (2, 3)), max_bond=chi, cutoff=cutoff)
            norm_s = tenet.norm(s)
            total += 1.0 - float(norm_s / tenet.norm(aa)) ** 2
            s = s / norm_s  # the two-site tensor is normalized; keep the MPS so
            self[n], self[n + 1] = u, vh  # the write barrier bends ``vh`` back
            self[n + 1] = tenet.einsum("xy,yqr->xqr", s, self[n + 1])
        self.center = len(self) - 1
        return max(total, 0.0) ** 0.5  # an untruncated bond lands on -1e-17, not on 0

    # --- serialization --------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Write a **directory**: one ``NNN.npz`` per site plus ``mps.json``.

        Per-tensor through :func:`tenet.save`, and that is the whole reason for the shape:
        :func:`tenet.load` *verifies* the SU(2) and fermionic-parity coefficient gauges
        (``serialize.py``:196-206), so a serializer that bypassed it would be the one place
        gauge-mismatched coefficients could enter silently. A directory rather than a
        zip-of-npz because ``np.load`` refuses nesting.

        ``mps.json`` carries exactly ``format`` (:data:`MPS_FORMAT_VERSION`), ``n_sites``
        and ``center``; ``center=None`` is JSON ``null``. Blocks save as NumPy whatever the
        backend, so ``MPS.load(...)`` then ``to_backend("jax")`` per site is the restore --
        a device placement is not a property of a tensor.

        A non-empty destination is refused **before anything is written**: writing an
        8-site MPS over a 12-site directory would leave ``008.npz`` onwards behind and the
        loader would then reject the result, destroying the previous good checkpoint and
        producing an unreadable new one.
        """
        path = pathlib.Path(path)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"{path}: destination is not empty; delete it or choose another")
        path.mkdir(parents=True, exist_ok=True)
        for n, t in enumerate(self.sites):
            tenet.save(t, path / f"{n:03d}.npz")
        meta = {"format": MPS_FORMAT_VERSION, "n_sites": len(self), "center": self.center}
        (path / "mps.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "MPS":
        """Read a directory written by :meth:`save`. NumPy blocks; structures exactly equal.

        Per-tensor version, gauge, block-count and member checks stay :func:`tenet.load`'s
        job, unmodified. Only what the directory owns is added here: a present and
        readable ``mps.json``, an exact file set, an in-range ``center``, and consecutive
        sites whose bond spaces agree.

        **The neighbour-bond check lives here and deliberately not in**
        :meth:`__setitem__`. A half-written directory is a corrupt file and this is the
        trust boundary; a sweeping MPS is transiently inconsistent *by construction* --
        :func:`~tenet.network.sweep_` writes ``psi[n + 1] = vh`` before ``psi[n] = u``
        (``dmrg.py``:120-127) -- so the same check in the write barrier would fire on
        correct code. It is not to be "fixed" upward.
        """
        path = pathlib.Path(path)
        meta_path = path / "mps.json"
        if not meta_path.is_file():
            raise ValueError(f"{meta_path}: not found; an MPS directory must contain mps.json")
        meta = json.loads(meta_path.read_text())
        if meta["format"] > MPS_FORMAT_VERSION:
            raise ValueError(
                f"{meta_path}: MPS directory format {meta['format']} is newer than this "
                f"tenet's format {MPS_FORMAT_VERSION}; upgrade tenet to read it"
            )
        n_sites, center = meta["n_sites"], meta["center"]
        expected = {f"{n:03d}.npz" for n in range(n_sites)}
        found = {p.name for p in path.iterdir()} - {"mps.json"}
        if found != expected:
            raise ValueError(
                f"{path}: mps.json says n_sites={n_sites}, but the directory is missing "
                f"{sorted(expected - found)} and has unexpected {sorted(found - expected)}"
            )
        if center is not None and center not in range(n_sites):
            raise ValueError(
                f"{meta_path}: center {center} is neither null nor in range({n_sites})"
            )
        sites = [tenet.load(path / f"{n:03d}.npz") for n in range(n_sites)]
        for n in range(n_sites - 1):
            if sites[n].legs[2].space != sites[n + 1].legs[0].space:
                raise ValueError(
                    f"{path}: site {n}'s right bond space does not match site {n + 1}'s left "
                    "bond space; the directory is corrupt"
                )
        return cls(sites, center)


def _as_site(t: SymmetricTensor) -> SymmetricTensor:
    """Put a rank-3 tensor on the MPS partition ``(l, p | r)``, or refuse it."""
    if t.ndim != 3:
        raise ValueError(f"an MPS site tensor is rank 3, got rank {t.ndim}")
    site = tenet.repartition(t, (0, 1), (2,))
    sides = tuple(leg.side for leg in site.legs)
    if sides != (OUT, OUT, IN):  # repartition guarantees it; the claim is stated anyway
        raise ValueError(f"an MPS site is (bond OUT, phys OUT, bond IN), got {sides}")
    return site


# --- measurement --------------------------------------------------------------------
#
# Module-level, not methods and not a ``network/measure.py``: a measurement is not
# container state, and a new module for ~30 lines buys a second entry in
# ``tests/network/test_hygiene.py``'s module list and nothing else. Simplification: the day
# ``correlation``, ``sample`` or ``rdm`` land, ``network/measure.py`` is the module and
# YASTN's ``_measure.py`` the design -- ``measure_1site`` (:76) has a ~40-line body and
# ``measure_2site`` (:130) a ~75-line one, which is the argument for not starting it now.


def _braket(bra: Sequence[SymmetricTensor], ket: Sequence[SymmetricTensor]) -> Any:
    """``<bra|ket>`` by one transfer pass -- :meth:`MPS.norm`'s body with two lists.

    The two chains may carry *different* bond spaces: the transfer tensor holds one index
    from each, which is what lets an operator-applied ket close against a plain bra.
    """
    t = tenet.einsum("apR,apr->Rr", tenet.adjoint(bra[0]), ket[0])
    for n in range(1, len(ket)):
        t = tenet.einsum("Rr,rps->Rps", t, ket[n])
        t = tenet.einsum("RpS,Rps->Ss", tenet.adjoint(bra[n]), t)
    return tenet.full_trace(t)


def _check(psi: MPS, o: SymmetricTensor, n: int, ndim: int, last: int) -> None:
    if not 0 <= n <= last:
        raise ValueError(f"site index {n} is out of range; expected 0 <= n <= {last}")
    if o.ndim != ndim:
        raise ValueError(f"the operator must be rank {ndim}, got rank {o.ndim}")


def expectation_1site(psi: MPS, o: SymmetricTensor, n: int) -> float:
    """``<psi|o_n|psi> / <psi|psi>``, with ``o`` rank 2 on ``(phys OUT, phys IN)``.

    **Divided by the norm, and the name carries that.** YASTN spells the pair
    ``measure_1site``/``measure_2site`` and does *not* divide; tenpy spells it
    ``expectation_value`` and does. The references disagree, so conformance decides
    nothing: in a package whose :meth:`Env.measure` also returns ``<psi|H|psi>`` undivided,
    a second ``measure_*`` that quietly did the same would make one verb mean two things.
    No ``normalize=`` opt-out -- a config for a value that never changes.
    """
    _check(psi, o, n, 2, len(psi) - 1)
    ket = list(psi)
    ket[n] = tenet.einsum("Pq,aqr->aPr", o, ket[n])
    return float(_braket(psi.sites, ket) / _braket(psi.sites, psi.sites))


def expectation_2site(psi: MPS, o: SymmetricTensor, n: int) -> float:
    """``<psi|o_{n,n+1}|psi> / <psi|psi>``, ``o`` rank 4 on ``(p OUT, p OUT, p IN, p IN)``.

    Divided by ``<psi|psi>`` for :func:`expectation_1site`'s reason, stated there.
    Adjacent sites only: at arbitrary separation this becomes a transfer-matrix walk with
    a caching strategy (YASTN's ``measure_2site(bonds=)``, a 75-line body; tenpy's
    ``correlation_function``), and every Hamiltonian here is nearest-neighbour. The
    operator-applied pair is split by the exact ``tenet.linalg.svd`` purely to hand
    :func:`_braket` two rank-3 sites again.
    """
    _check(psi, o, n, 4, len(psi) - 2)
    ket = list(psi)
    pair = tenet.einsum("apx,xqr->apqr", ket[n], ket[n + 1])
    pair = tenet.einsum("PQpq,apqr->aPQr", o, pair)
    u, s, vh = tenet.linalg.svd(pair, ((0, 1), (2, 3)))
    ket[n], ket[n + 1] = _as_site(u), tenet.einsum("xy,yqr->xqr", s, _as_site(vh))
    return float(_braket(psi.sites, ket) / _braket(psi.sites, psi.sites))


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
