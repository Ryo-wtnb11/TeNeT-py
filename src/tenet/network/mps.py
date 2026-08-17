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

import numpy as np

import tenet
from tenet import IN, OUT, FusionTree, GradedSpace, Leg, SymmetricTensor
from tenet.network.common import spectrum
from tenet.symmetry import Sector

__all__ = [
    "MPO",
    "MPS",
    "MPS_FORMAT_VERSION",
    "expectation_1site",
    "expectation_2site",
    "local_op",
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

        Only ``to=0`` (fully right-canonical) is supported; any other ``to`` raises
        :exc:`NotImplementedError`.
        """
        # Simplification: only ``to=0``, because that is the one form the sweep's setup
        # wants. Ceiling: a general ``to`` is the same loop run from both ends, and YASTN's
        # ``canonize_(to='last')`` (``_mps_obc.py``:390) is the spelling to copy.
        if to != 0:
            raise NotImplementedError("only to=0 (right-canonical) is implemented")
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


def local_op(dense: Any, *, phys: GradedSpace, charge: Sector) -> SymmetricTensor:
    """A ``(d, d)`` dense operator as rank 3 on ``(phys OUT, phys IN, charge OUT)``.

    The third leg is why this exists. ``S^+`` raises ``2 S^z`` by 2, so as a rank-2 tensor
    it is symmetry-forbidden and ``from_dense`` refuses it, correctly; the charge has to
    live on a leg. Invariance reads ``q(p_out) + q(charge) = q(p_in)``, so ``charge`` is
    literally the MPO bond the operator emits. Built at ``from_dense``'s **default**
    relative ``atol``, so a ``charge`` that does not match the array *raises*.

    The matrices themselves are physics and stay in the caller; turning one into a
    charge-carrying tensor is the step the caller cannot get right by hand.
    """
    d = phys.dim
    shape = tuple(np.shape(dense))
    if shape != (d, d):
        raise ValueError(f"local_op: expected a ({d}, {d}) array on this phys, got {shape}")
    emitted = Leg(GradedSpace.new(phys.provider, {charge: 1}), OUT)
    return SymmetricTensor.from_dense(
        np.reshape(dense, (d, d, 1)), (Leg(phys, OUT), Leg(phys, IN), emitted)
    )


def _as_w(t: SymmetricTensor) -> SymmetricTensor:
    """Put a rank-4 tensor on the MPO partition ``(wl IN, p OUT, p IN, wr OUT)``.

    :meth:`MPS.__setitem__`'s job for sites, as a function: ``svd_truncated`` lowers its
    input to a map, so its factors come back bent and one ``repartition`` puts them back.
    Deliberately **not** ``MPO.__setitem__`` -- one internal compression sweep is not a
    reason to grow the class a mutation API with a single caller.
    """
    if t.ndim != 4:
        raise ValueError(f"an MPO site tensor is rank 4, got rank {t.ndim}")
    # ``repartition``'s result order is ``(*outputs, *inputs)``, so the transpose puts the
    # public axes back into MPO order; for an already-correct tensor the pair is a no-op.
    return tenet.transpose(tenet.repartition(t, (1, 3), (0, 2)), (2, 0, 3, 1))


def _braids_with_signs(space: GradedSpace) -> bool:
    """Whether exchanging two lines of ``space`` can carry a minus sign (super-vector spaces).

    Asked of the braiding rather than of the provider's identity -- a category is fermionic
    exactly when some sector braids past itself with coefficient ``-1``, which is
    symmetry-generic recoupling metadata and not a provider branch. Called only after the
    Abelian check below, because SU(2)'s ``(-1)^(j_a + j_b - j_c)`` wears the same minus.
    """
    sym = space.provider
    for a, _ in space.sectors:
        tree = FusionTree((a, a), (), (0,), sym.fusion(a, a)[0])
        if sym.permute_tree(tree, (1, 0))[0][1].real < 0:
            return True
    return False


def _check_op(op: SymmetricTensor, phys: GradedSpace | None) -> GradedSpace:
    """Validate one term operator and return the physical space it declares."""
    if op.ndim != 3:
        raise ValueError(
            f"a term operator is rank 3 on (phys OUT, phys IN, charge OUT), got rank "
            f"{op.ndim}; build it with tenet.network.local_op, which adds the charge leg"
        )
    got, emitted = op.legs[0].space, op.legs[2].space
    if phys is not None and got != phys:
        raise ValueError(f"term operators disagree about the physical space: {phys} vs {got}")
    if emitted.reduced_dim != 1:
        raise ValueError(
            f"a term operator's charge leg carries one sector at degeneracy 1, got "
            f"{emitted}; it names the MPO bond the operator emits, so it cannot be wider"
        )
    # ``space.dim`` is ``Σ m_a irrep_dim(a)`` and ``reduced_dim`` is ``Σ m_a``, so the two
    # differ exactly when some irrep is more than one-dimensional. Symmetry-generic
    # metadata off the space, of the kind ``network/__init__.py`` names -- no provider read.
    if got.dim != got.reduced_dim or emitted.dim != emitted.reduced_dim:
        raise ValueError(
            "from_terms is Abelian-only: this operator's charge leg has irrep_dim > 1, and "
            "a list of non-Abelian operators does not determine a term -- three of them "
            "fuse through several channels and the DSL has no slot for a coupling tree. "
            "The 2-site case is the named follow-up (tenet.ops.fusion.fused_leg)"
        )
    if _braids_with_signs(got) or _braids_with_signs(emitted):
        raise ValueError(
            "from_terms refuses fermionic braiding: the Jordan-Wigner string needs a swap "
            "gate between an odd MPO bond and a physical line, which tenet does not have "
            "(docs/design.md:2888 lists fermionic swap gates as not planned). Env and "
            "sweep_ would contract the result silently and wrongly rather than refuse it"
        )
    return got


def _term_string(n_sites: int, coeff: float, ops, phys: GradedSpace) -> list[SymmetricTensor]:
    """One term as a bond-1 MPO: identities on the untouched sites, charge accumulated.

    The running MPO bond at each cut carries exactly the sum of the charges emitted to its
    left, which is the whole of ``q(p_out) + q(wr) = q(wl) + q(p_in)`` for a ``D=1`` bond.
    """
    placed: dict[int, SymmetricTensor] = {}
    for op, site in ops:
        if site not in range(n_sites):
            raise ValueError(f"term site index {site} is outside range({n_sites})")
        if site in placed:
            raise ValueError(f"two operators of one term sit on site {site}; multiply them first")
        placed[site] = op
    sym, d = phys.provider, phys.dim
    running, out = sym.unit, []
    for n in range(n_sites):
        left = GradedSpace.new(sym, {running: 1})
        op = placed.get(n)
        if op is None:
            mat = np.eye(d)
        else:
            mat = np.asarray(op.to_dense())[:, :, 0]
            running = sym.fusion(running, op.legs[2].space.sectors[0][0])[0]
        legs = (
            Leg(left, IN),
            Leg(phys, OUT),
            Leg(phys, IN),
            Leg(GradedSpace.new(sym, {running: 1}), OUT),
        )
        block = np.reshape(coeff * mat if n == 0 else mat, (1, d, d, 1))
        out.append(_as_w(SymmetricTensor.from_dense(block, legs)))
    if running != sym.unit:
        raise ValueError(
            f"a term's operator charges must sum to the unit sector, got {running}; both "
            "MPO boundaries are the trivial D=1 leg, so a charged term has nowhere to end"
        )
    return out


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

        The builder that shows what an MPO *is*, and the one a reader needs before they
        can debug one. :meth:`from_terms` is the other route (#133 reversed this
        docstring's refusal of it, on a direct request rather than on new evidence); it is
        not a replacement, and neither is deprecated or aliased to the other.
        """

        def make(array: Any, left: GradedSpace, right: GradedSpace) -> SymmetricTensor:
            legs = (Leg(left, IN), Leg(phys, OUT), Leg(phys, IN), Leg(right, OUT))
            return SymmetricTensor.from_dense(array, legs)

        first = make(w[start : start + 1], boundary, bond)
        bulk = make(w, bond, bond)
        last = make(w[:, :, :, end : end + 1], bond, boundary)
        return cls([first, *[bulk] * (n_sites - 2), last])

    @classmethod
    def from_terms(cls, n_sites: int, terms: Iterable, *, cutoff: float = 1e-13) -> "MPO":
        """A term list ``[(coeff, [(op, site), ...]), ...]`` as a compressed graded MPO.

        Each ``op`` is rank 3 from :func:`local_op`; a term is a coefficient and a list of
        ``(operator, site)`` pairs, with identities implied on every untouched site. Every
        term is *already* a bond-1 MPO, so the sum is one :func:`tenet.direct_sum` per site
        over the MPO-bond axes -- shared at the two ends, so the chain sums rather than
        block-diagonalizing -- and :func:`tenet.linalg.svd_truncated` then compresses the
        M-wide bond to the operator Schmidt rank.

        **The MPO bond spaces are derived, never declared.** Charges enter once, as
        :func:`local_op`'s ``charge``; from there the bond is arithmetic on legs and
        ``svd_truncated`` decides the sector-by-sector degeneracies, omitting a sector
        entirely when nothing is kept there. There is no place left to write a wrong
        grading down.

        **There is no** ``phys=`` **argument**: the operators carry the physical space and
        a second source of truth could disagree with them, which would surface as a
        structure error instead of a message about ``phys``. Uniform physical space only,
        and every term's charges must sum to the unit sector.

        Fermionic and non-Abelian terms are refused with a message rather than accepted;
        ``cutoff`` is a keyword so a test can tighten or loosen the compression. Outside
        ``jit``/``grad`` like the rest of this module, because ``svd_truncated`` re-decides
        a :class:`~tenet.GradedSpace`.
        """
        phys = None
        strings = []
        for coeff, ops in terms:
            ops = list(ops)
            for op, _ in ops:
                phys = _check_op(op, phys)
            strings.append((coeff, ops))
        if phys is None:
            raise ValueError("from_terms: no terms; the physical space is read off an operator")
        sites = _term_string(n_sites, *strings[0], phys)
        for coeff, ops in strings[1:]:
            other = _term_string(n_sites, coeff, ops, phys)
            for n in range(n_sites):
                # The end bonds stay shared and D=1: summed there, the chain would become a
                # direct sum of scalars instead of a sum of terms.
                axes = ([0] if n else []) + ([3] if n < n_sites - 1 else [])
                sites[n] = (
                    tenet.direct_sum(sites[n], other[n], axes) if axes else sites[n] + other[n]
                )
        # Right to left first, exactly as ``MPS.compress_`` canonizes before it truncates:
        # a forward sweep alone measures the rank of the *left* part against the raw
        # M-wide channel bond, which overshoots wherever the redundancy is only visible
        # from the right. After this pass the forward rank is the operator Schmidt rank.
        for n in range(n_sites - 1, 0, -1):
            u, s, vh = tenet.linalg.svd_truncated(sites[n], ((0,), (1, 2, 3)), cutoff=cutoff)
            sites[n] = _as_w(vh)
            carry = tenet.einsum("xy,yz->xz", u, s)
            sites[n - 1] = _as_w(tenet.einsum("apqx,xy->apqy", sites[n - 1], carry))
        for n in range(n_sites - 1):
            u, s, vh = tenet.linalg.svd_truncated(sites[n], ((0, 1, 2), (3,)), cutoff=cutoff)
            sites[n] = _as_w(u)
            carry = tenet.einsum("xy,yz->xz", s, vh)
            sites[n + 1] = _as_w(tenet.einsum("xy,ypqr->xpqr", carry, sites[n + 1]))
        return cls(sites)

    def to_dense(self) -> Any:
        """The full ``d**N x d**N`` operator, ``D=1`` boundaries dropped.

        :meth:`MPS.to_dense`'s twin, with its warning: exponential in ``N``, an oracle exit
        for tests, and nothing an algorithm calls.
        """
        out = self[0]
        for n in range(1, len(self)):
            body = string.ascii_uppercase[: 2 * n]
            out = tenet.einsum(f"a{body}x,xpqr->a{body}pqr", out, self[n])
        n_sites, d = len(self), self[0].legs[1].space.dim
        order = list(range(0, 2 * n_sites, 2)) + list(range(1, 2 * n_sites, 2))
        return out.to_dense()[0, ..., 0].transpose(order).reshape(d**n_sites, d**n_sites)
