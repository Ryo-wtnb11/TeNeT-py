"""The containers: [MPS][tenet.network.MPS] and [MPO][tenet.network.MPO].

Promoted from ``examples/toy_codes/dmrg.py`` (#110) with no arithmetic change: ``random_mps``
:266-274, ``_as_site`` :277-286 (now the ``MPS.__setitem__`` write barrier),
``canonicalize`` :289-306 (now [MPS.canonize_][tenet.network.MPS.canonize_]) and ``mpo`` :220-240
(now
[MPO.from_w][tenet.network.MPO.from_w]). ``scalar``, ``inner`` and ``spectrum`` lived here until
#114 moved
them to ``tenet.network.common``, where ``network/ctmrg.py`` can reach them without
importing a driver it shares no concept with; #126 then promoted the first two out of the
driver layer entirely, as [tenet.full_trace][] and [tenet.inner][].

Every two-operand ``tenet.einsum`` in this module follows the package's composition rule
-- operand 1 supplies ``IN`` on every shared wire; stated in ``docs/design.md``
"Milestone 11", pinned by ``tests/network/test_hygiene.py`` (#160).
"""

import json
import pathlib
import string
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

import tenet
from tenet import IN, OUT, FusionTree, GradedSpace, Leg, SymmetricTensor
from tenet.network.common import spectrum
from tenet.symmetry import Sector

__all__ = [
    "MPO",
    "MPS",
    "MPS_FORMAT_VERSION",
    "EdgeBlocks",
    "expectation_1site",
    "expectation_2site",
    "local_op",
    "spectrum",
]

#: Version of the [MPS.save][tenet.network.MPS.save] *directory* layout -- ``mps.json`` plus
#: the ``NNN.npz``
#: naming. Distinct from ``tenet.serialize.FORMAT_VERSION``, which versions the per-tensor
#: ``.npz`` header: two formats, two owners, so bumping one is not a lie about the other.
MPS_FORMAT_VERSION = 1


# --- the state ----------------------------------------------------------------------


class MPS:
    """A finite open-boundary MPS: a mutable list of frozen ``SymmetricTensor``s.

    Parameters
    ----------
    sites : Iterable of SymmetricTensor
        The site tensors, each passed through the ``__setitem__`` write
        barrier onto ``(l OUT, p OUT | r IN)``.
    center : int or None, optional
        The orthogonality centre; ``None`` (the default) means "no claim
        made".

    Examples
    --------
    >>> from tenet import GradedSpace
    >>> from tenet.network import MPS
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1), U1Sector(1)])
    >>> len(psi)
    3
    >>> round(psi.norm(), 6)
    1.0

    Notes
    -----
    **Site convention**, pinned here once and enforced on every write::

        A_n : (left bond OUT, physical OUT, right bond IN)

    Charge flows left to right, ``bond_n (x) phys_n -> bond_{n+1}``, and both end bonds
    have ``D=1``; a non-unit sector on bond 0 targets that total charge (YASTN's
    charged-first-virtual-leg recipe, ``_initialize.py``:194). The convention is not
    invented here -- ``examples/toy_codes/dmrg.py``:57-59 and
    ``examples/toy_codes/vmc_mps.py``:69-74 both chose it independently and
    ``tests/integration/test_dmrg.py`` already pins it.

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
        the codomain returns IN-dual in the domain. One [tenet.repartition][] to
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

    def __iter__(self) -> Iterator[SymmetricTensor]:
        return iter(self.sites)

    def copy(self) -> "MPS":
        """A new container over the same frozen tensors.

        Returns
        -------
        MPS
            A fresh [MPS][tenet.network.MPS] holding the same tensors and
            ``center``.
        """
        return MPS(self.sites, self.center)

    # --- constructors ---------------------------------------------------------------

    @classmethod
    def random(cls, phys: GradedSpace, bonds: Sequence[GradedSpace], *, seed: int = 0) -> "MPS":
        """A random MPS on ``len(bonds) - 1`` sites over the given bond spaces.

        Parameters
        ----------
        phys : GradedSpace
            The physical space of every site.
        bonds : Sequence of GradedSpace
            The ``n_sites + 1`` virtual spaces, both ends included.
        seed : int, optional
            Site ``i`` draws with ``seed + i``. Default ``0``. Keyword-only.

        Returns
        -------
        MPS
            A random state with ``center=None``.

        Notes
        -----
        The library takes bond *spaces*; deciding which are reachable for a given
        symmetry and target charge is physics and stays in the caller
        (``examples/toy_codes/dmrg.py::bond_spaces``).
        """
        return cls(
            SymmetricTensor.random(
                (Leg(bonds[i], OUT), Leg(phys, OUT), Leg(bonds[i + 1], IN)), seed=seed + i
            )
            for i in range(len(bonds) - 1)
        )

    @classmethod
    def product(cls, phys: GradedSpace, states: Sequence[Sector]) -> "MPS":
        """A product state: one physical sector per site, bonds derived rather than declared.

        Parameters
        ----------
        phys : GradedSpace
            The physical space of every site.
        states : Sequence of Sector
            One sector of ``phys`` per site, each at degeneracy 1, naming the
            basis vector that site carries.

        Returns
        -------
        MPS
            A norm-1 product state whose bond 0 carries the total charge.

        Raises
        ------
        ValueError
            If a sector is not in ``phys``; if a sector has degeneracy > 1 in
            ``phys`` (this constructor has no slot for the degeneracy index);
            or if a fusion along the backwards bond derivation has more than
            one channel -- the constructor is Abelian-only and refuses rather
            than picking a channel.

        Examples
        --------
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPS
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(1)])
        >>> psi[0].legs[0].space.sectors  # bond 0 carries the total charge
        ((U1Sector(charge=2), 1),)
        >>> round(psi.norm(), 6)
        1.0

        Notes
        -----
        ``states[n]`` names the basis vector site ``n`` carries -- a sector of ``phys``
        at degeneracy 1 -- and the bond spaces fall out of the charges: bond
        ``len(states)`` is the unit sector and the loop runs backwards through the
        provider's ``dual``, so **bond 0 carries the total charge**, which is where the
        target-sector statement lives (YASTN's charged-first-virtual-leg recipe,
        ``_initialize.py``:194). ``psi[0].legs[0].space`` is then printable and
        assertable: on the U(1) spin chain a ``D=1`` boundary leg carrying ``U1Sector(q)``
        targets ``S^z_tot = q/2`` (the leg is dual there when ``q`` is non-unit, which is
        what puts the recipe's sign on the space). The result has norm 1 and a single
        dense amplitude of exactly 1.0.

        Abelian-only, by construction and permanently: when a fusion has more than one
        channel the bonds are not determined -- a single sector is not a non-Abelian
        multiplet -- and the constructor refuses rather than picking a channel. To target
        a sector under a non-Abelian symmetry, seed with [MPS.random][tenet.network.MPS.random] and
        put the
        target on a charged ``D=1`` boundary leg of bond 0.
        """
        states = list(states)
        sym = phys.provider
        for n, a in enumerate(states):
            if a not in phys:
                raise ValueError(
                    f"sector {a!r} at site {n} is not in the physical space; available: "
                    f"{[b for b, _ in phys.sectors]}"
                )
            if phys.degeneracy(a) != 1:
                raise ValueError(
                    f"sector {a!r} has degeneracy {phys.degeneracy(a)} in the physical "
                    "space, and a product state names a basis vector: this constructor "
                    "has no slot for the degeneracy index (a states: "
                    "Sequence[tuple[Sector, int]] spelling would be the upgrade)"
                )
        bonds = [GradedSpace.new(sym, {sym.unit: 1})]  # bond len(states), built backwards
        for n in range(len(states) - 1, -1, -1):
            channels = sym.fusion(bonds[0].sectors[0][0], sym.dual(states[n]))
            if len(channels) != 1:
                raise ValueError(
                    f"MPS.product is Abelian-only: fusing bond {n + 1} with the dual of "
                    f"{states[n]!r} has {len(channels)} channels, so the bonds are not "
                    "determined -- a single sector is not a non-Abelian multiplet (a "
                    "single spin-up is not an SU(2) multiplet). Seed with MPS.random and "
                    "a charged D=1 boundary leg on bond 0 to target a sector instead"
                )
            bonds.insert(0, GradedSpace.new(sym, {channels[0]: 1}))
        # Bond 0's *space* shows the total charge with the recipe's sign -- +q targets
        # S^z_tot = q/2 -- so the leg is dual there (Leg.fused_sector then feeds the
        # accumulated sector to the invariance check unchanged). On the unit sector the
        # flag is a pure label and stays False, so Env's plain boundary legs match and a
        # charge-0 product state feeds dmrg_ directly.
        total = sym.dual(bonds[0].sectors[0][0])
        first = Leg(GradedSpace.new(sym, {total: 1}), OUT, total != sym.unit)
        sites = []
        for n, a in enumerate(states):
            dense = np.zeros((1, phys.dim, 1))
            dense[0, phys.sector_offset(a), 0] = 1.0
            left = first if n == 0 else Leg(bonds[n], OUT)
            sites.append(
                SymmetricTensor.from_dense(dense, (left, Leg(phys, OUT), Leg(bonds[n + 1], IN)))
            )
        return cls(sites)

    @classmethod
    def from_tensors(cls, tensors: Iterable[SymmetricTensor]) -> "MPS":
        """An MPS over already-built site tensors, each through the write barrier.

        Parameters
        ----------
        tensors : Iterable of SymmetricTensor
            The rank-3 site tensors, left to right.

        Returns
        -------
        MPS
            The state, with ``center=None``.
        """
        return cls(tensors)

    # --- state machine --------------------------------------------------------------

    def canonize_(self, to: int = 0) -> "MPS":
        """Right-canonicalize in place and return ``self`` -- YASTN ``canonize_(to='first')``.

        Parameters
        ----------
        to : int, optional
            The target centre. Only ``0`` (fully right-canonical) is
            supported. Default ``0``.

        Returns
        -------
        MPS
            ``self``, normalized, with ``center = to``.

        Raises
        ------
        NotImplementedError
            If ``to`` is not ``0``.

        Notes
        -----
        One ``tenet.linalg.lq`` per site from the right, mirroring ``orthogonalize_site_``
        (``_mps_obc.py``:245-300): ``A_n = L . Q`` with ``Q`` on the MPS convention and
        ``L`` absorbed into ``A_{n-1}``. ``lq`` rather than ``qr`` because ``qr`` would put
        the new bond on the *right* of the factor and leave the site tensor's left leg IN.

        Setup only: a two-site sweep leaves the state canonical by construction on the
        side it came from, which is precisely what an ``int`` centre records.
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
        """``sqrt(<psi|psi>)`` by one bra-ket transfer pass, closed with [tenet.full_trace][].

        Returns
        -------
        float
            The 2-norm of the state.

        Notes
        -----
        No dense expansion and no environment object: two ``tenet.einsum`` calls per site,
        the same pairwise shape [Env.update_][tenet.network.Env.update_] uses with the MPO row
        removed.
        """
        t = tenet.einsum("apR,apr->Rr", tenet.adjoint(self[0]), self[0])
        for n in range(1, len(self)):
            t = tenet.einsum("Rr,rps->Rps", t, self[n])
            t = tenet.einsum("RpS,Rps->Ss", tenet.adjoint(self[n]), t)
        return float(tenet.full_trace(t)) ** 0.5

    def to_dense(self) -> Any:
        """The full ``d**N`` amplitude array, ``D=1`` boundaries dropped.

        Returns
        -------
        array
            The backend's dense amplitude array of shape ``(d,) * N``.

        Notes
        -----
        Exponential in ``N``: an oracle exit for tests, and nothing an algorithm calls.
        """
        out = self[0]
        for n in range(1, len(self)):
            body = string.ascii_uppercase[:n]
            out = tenet.einsum(f"a{body}x,xpr->a{body}pr", out, self[n])
        return out.to_dense()[0, ..., 0]

    def compress_(self, *, chi: int, cutoff: float = 0.0) -> float:
        """Truncate to bond ``chi`` in place; return the **total** discarded weight.

        Parameters
        ----------
        chi : int
            The bond-dimension cap handed to
            [svd_truncated][tenet.ops.linalg.svd_truncated] at every bond.
            Keyword-only.
        cutoff : float, optional
            The singular-value cutoff handed to the same SVD. Default ``0.0``.
            Keyword-only.

        Returns
        -------
        float
            ``sqrt(sum_bond dw)``, the total discarded weight of the sweep.

        Notes
        -----
        [canonize_][tenet.network.MPS.canonize_] then one left-to-right ``svd_truncated`` sweep --
        the per-bond
        body of [sweep_][tenet.network.sweep_] with the eigensolver removed -- leaving
        ``center = len(self) - 1``. YASTN's ``truncate_`` (``_mps_obc.py``:379-413)
        instead *assumes* a canonical input and takes ``to=``; canonizing here costs an
        ``lq`` pass the caller usually needs anyway and removes the silently-wrong result
        on a non-canonical state.

        **The returned convention differs from** [sweep_][tenet.network.sweep_]'s **on
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

        Parameters
        ----------
        path : str or pathlib.Path
            The destination directory; created if absent.

        Raises
        ------
        FileExistsError
            If ``path`` exists and is not empty -- refused **before anything
            is written** (see Notes for why).

        Notes
        -----
        Per-tensor through [tenet.save][], and that is the whole reason for the shape:
        [tenet.load][] *verifies* the SU(2) and fermionic-parity coefficient gauges
        (``serialize.py``:196-206), so a serializer that bypassed it would be the one place
        gauge-mismatched coefficients could enter silently. A directory rather than a
        zip-of-npz because ``np.load`` refuses nesting.

        ``mps.json`` carries exactly ``format`` (``MPS_FORMAT_VERSION``), ``n_sites``
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
        """Read a directory written by [save][tenet.network.MPS.save]. NumPy blocks; structures
        exactly equal.

        Parameters
        ----------
        path : str or pathlib.Path
            The MPS directory.

        Returns
        -------
        MPS
            The restored state, ``center`` included.

        Raises
        ------
        ValueError
            If ``mps.json`` is missing or names a newer format than this
            tenet's ``MPS_FORMAT_VERSION``; if the file set does not match
            ``n_sites``; if ``center`` is neither null nor in range; or if two
            consecutive sites' bond spaces disagree -- a corrupt directory.

        Notes
        -----
        Per-tensor version, gauge, block-count and member checks stay [tenet.load][]'s
        job, unmodified. Only what the directory owns is added here: a present and
        readable ``mps.json``, an exact file set, an in-range ``center``, and consecutive
        sites whose bond spaces agree.

        **The neighbour-bond check lives here and deliberately not in**
        ``__setitem__``. A half-written directory is a corrupt file and this is the
        trust boundary; a sweeping MPS is transiently inconsistent *by construction* --
        [sweep_][tenet.network.sweep_] writes ``psi[n + 1] = vh`` before ``psi[n] = u``
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
    """``<bra|ket>`` by one transfer pass -- [MPS.norm][tenet.network.MPS.norm]'s body with two
    lists.

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

    Parameters
    ----------
    psi : MPS
        The state; any gauge, any norm.
    o : SymmetricTensor
        The operator, rank 2 on ``(phys OUT, phys IN)`` --
        [local_op][tenet.network.local_op]'s invariant one-site form.
    n : int
        The site, ``0 <= n <= len(psi) - 1``.

    Returns
    -------
    float
        The normalized expectation value.

    Raises
    ------
    ValueError
        If ``n`` is out of range, or if ``o`` is not rank 2.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import GradedSpace
    >>> from tenet.network import MPS, expectation_1site, local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1)])
    >>> sz = local_op(np.diag([-0.5, 0.5]), phys=phys)
    >>> round(expectation_1site(psi, sz, 0), 6)
    0.5

    Notes
    -----
    **Divided by the norm, and the name carries that.** YASTN spells the pair
    ``measure_1site``/``measure_2site`` and does *not* divide; tenpy spells it
    ``expectation_value`` and does. The references disagree, so conformance decides
    nothing: in a package whose [Env.measure][tenet.network.Env.measure] also returns
    ``<psi|H|psi>`` undivided,
    a second ``measure_*`` that quietly did the same would make one verb mean two things.
    No ``normalize=`` opt-out -- a config for a value that never changes.
    """
    _check(psi, o, n, 2, len(psi) - 1)
    ket = list(psi)
    ket[n] = tenet.einsum("Pq,aqr->aPr", o, ket[n])
    return float(_braket(psi.sites, ket) / _braket(psi.sites, psi.sites))


def expectation_2site(psi: MPS, o: SymmetricTensor, n: int) -> float:
    """``<psi|o_{n,n+1}|psi> / <psi|psi>``, ``o`` rank 4 on ``(p OUT, p OUT, p IN, p IN)``.

    Parameters
    ----------
    psi : MPS
        The state; any gauge, any norm.
    o : SymmetricTensor
        The two-site operator, rank 4 --
        [local_op][tenet.network.local_op]'s invariant form on ``np.kron(a, b)``.
    n : int
        The pair's left site, ``0 <= n <= len(psi) - 2``.

    Returns
    -------
    float
        The normalized expectation value on the adjacent pair.

    Raises
    ------
    ValueError
        If ``n`` is out of range, or if ``o`` is not rank 4.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import GradedSpace
    >>> from tenet.network import MPS, expectation_2site, local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1)])
    >>> sz = np.diag([-0.5, 0.5])
    >>> szsz = local_op(np.kron(sz, sz), phys=phys)
    >>> round(expectation_2site(psi, szsz, 0), 6)
    -0.25

    Notes
    -----
    Divided by ``<psi|psi>`` for [expectation_1site][tenet.network.expectation_1site]'s reason,
    stated there.
    Adjacent sites only: at arbitrary separation this becomes a transfer-matrix walk with
    a caching strategy (YASTN's ``measure_2site(bonds=)``, a 75-line body; tenpy's
    ``correlation_function``), and every Hamiltonian here is nearest-neighbour. The
    operator-applied pair is split by the exact ``tenet.linalg.svd`` purely to hand
    ``_braket`` two rank-3 sites again.
    """
    _check(psi, o, n, 4, len(psi) - 2)
    ket = list(psi)
    pair = tenet.einsum("apx,xqr->apqr", ket[n], ket[n + 1])
    pair = tenet.einsum("PQpq,apqr->aPQr", o, pair)
    u, s, vh = tenet.linalg.svd(pair, ((0, 1), (2, 3)))
    ket[n], ket[n + 1] = _as_site(u), tenet.einsum("xy,yqr->xqr", s, _as_site(vh))
    return float(_braket(psi.sites, ket) / _braket(psi.sites, psi.sites))


# --- the Hamiltonian ----------------------------------------------------------------


def local_op(dense: Any, *, phys: GradedSpace, charge: Sector | None = None) -> SymmetricTensor:
    """A dense operator as a term operator: rank 3 with a charge leg, or invariant on *k* sites.

    Parameters
    ----------
    dense : array_like
        The operator's dense matrix: ``(d, d)`` with ``charge``, or
        ``(d**k, d**k)`` / ``(d,) * 2k`` without one, ``k`` inferred.
    phys : GradedSpace
        The physical space (``d = phys.dim``). Keyword-only.
    charge : Sector or None, optional
        The sector the operator emits onto its MPO bond; ``None`` (the
        default) builds the invariant *k*-site form. Keyword-only.

    Returns
    -------
    SymmetricTensor
        Rank 3 on ``(phys OUT, phys IN, charge OUT)`` with ``charge``, rank
        2*k* on ``(phys OUT)*k`` then ``(phys IN)*k`` without.

    Raises
    ------
    ValueError
        With ``charge``, if the array is not ``(d, d)`` on this ``phys``; with
        ``charge=None``, if no integer *k* makes the shape ``(d**k, d**k)`` or
        ``(d,) * 2k``. A symmetry-forbidden array *raises* inside
        ``from_dense`` rather than being projected (see Notes).

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import GradedSpace
    >>> from tenet.network import local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> sp = np.array([[0.0, 0.0], [1.0, 0.0]])  # S^+ raises 2 S^z by 2
    >>> local_op(sp, phys=phys, charge=U1Sector(-2)).ndim  # the charge-leg form
    3
    >>> sz = np.diag([-0.5, 0.5])
    >>> local_op(np.kron(sz, sz), phys=phys).ndim  # one invariant 2-site term
    4

    Notes
    -----
    With ``charge``, a ``(d, d)`` array becomes rank 3 on ``(phys OUT, phys IN, charge
    OUT)``. The third leg is why that form exists: ``S^+`` raises ``2 S^z`` by 2, so as a
    rank-2 tensor it is symmetry-forbidden and ``from_dense`` refuses it, correctly; the
    charge has to live on a leg. Invariance reads ``q(p_out) + q(charge) = q(p_in)``, so
    ``charge`` is literally the MPO bond the operator emits.

    With ``charge=None`` the array is **one whole term** spanning *k* sites -- ``(d**k,
    d**k)`` or ``(d,)*2k``, *k* inferred from ``d``, the layout ``np.kron(a, b)`` already
    has -- and the result is rank 2*k* on ``(phys OUT)*k`` then ``(phys IN)*k`` with no
    auxiliary leg at all. A term is a scalar under the symmetry, so this form **cannot
    express a symmetry-breaking term**: on SU(2) legs ``Sz (x) Sz`` alone raises and
    ``S.S`` builds. A non-Abelian term's coupling lives inside the array's own blocks,
    which is why it needs no coupling-tree argument; [MPO.from_terms][tenet.network.MPO.from_terms]
    splits it with
    ``svd_truncated`` and the MPO bond comes out of that SVD.

    Both forms are built at ``from_dense``'s **default** relative ``atol``, so an array
    that does not match what it was declared to be *raises*. The matrices themselves are
    physics and stay in the caller.
    """
    d = phys.dim
    shape = tuple(np.shape(dense))
    if charge is not None:
        if shape != (d, d):
            raise ValueError(f"local_op: expected a ({d}, {d}) array on this phys, got {shape}")
        emitted = Leg(GradedSpace.new(phys.provider, {charge: 1}), OUT)
        return SymmetricTensor.from_dense(
            np.reshape(dense, (d, d, 1)), (Leg(phys, OUT), Leg(phys, IN), emitted)
        )
    k = next((k for k in range(1, 9) if shape in ((d**k, d**k), (d,) * (2 * k))), None)
    if k is None:
        raise ValueError(
            f"local_op: with charge=None the array is an invariant k-site term, so it is "
            f"({d}**k, {d}**k) or ({d},)*2k on this phys (d={d}); got {shape}, from which no "
            f"integer k follows"
        )
    legs = (Leg(phys, OUT),) * k + (Leg(phys, IN),) * k
    return SymmetricTensor.from_dense(np.reshape(dense, (d,) * (2 * k)), legs)


def _as_w(t: SymmetricTensor) -> SymmetricTensor:
    """Put a rank-4 tensor on the MPO partition ``(wl IN, p OUT, p IN, wr OUT)``.

    ``MPS.__setitem__``'s job for sites, as a function: ``svd_truncated`` lowers its
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
    symmetry-generic recoupling metadata and not a provider branch. The sign is demanded
    in **every** channel, because in a super-vector space it depends on parity alone:
    SU(2) wears a minus in some channels only (``(-1)^(j_a + j_b - j_c)``, ``-1`` on the
    singlet and ``+1`` on the triplet), and it is not fermionic.
    """
    sym = space.provider
    for a, _ in space.sectors:
        signs = [
            # permute_tree is the opt-in PermutationCoefficients capability;
            # every provider that reaches this braiding question implements it
            sym.permute_tree(FusionTree((a, a), (), (0,), c), (1, 0))[0][1].real  # ty: ignore[unresolved-attribute]
            for c in sym.fusion(a, a)
        ]
        if signs and all(sign < 0 for sign in signs):
            return True
    return False


def _check_op(op: SymmetricTensor, phys: GradedSpace | None) -> GradedSpace:
    """Validate one term operator and return the physical space it declares."""
    if op.ndim != 3 and (op.ndim < 4 or op.ndim % 2):
        raise ValueError(
            f"a term operator is rank 3 on (phys OUT, phys IN, charge OUT) or rank 2k on "
            f"(phys OUT)*k then (phys IN)*k for an invariant k-site term (k >= 2), got rank "
            f"{op.ndim}; build it with tenet.network.local_op"
        )
    got = op.legs[0].space
    if phys is not None and got != phys:
        raise ValueError(f"term operators disagree about the physical space: {phys} vs {got}")
    if op.ndim != 3:
        k = op.ndim // 2
        sides = (OUT,) * k + (IN,) * k
        if any(leg.side != s or leg.space != got for leg, s in zip(op.legs, sides, strict=True)):
            raise ValueError(
                f"an invariant {k}-site term operator is (phys OUT)*{k} then (phys IN)*{k} on "
                f"one space, got {[(leg.space, leg.side) for leg in op.legs]}; an MPO site "
                "tensor, with its bond legs, is not a term operator"
            )
        return got
    emitted = op.legs[2].space
    if emitted.reduced_dim != 1:
        raise ValueError(
            f"a term operator's charge leg carries one sector at degeneracy 1, got "
            f"{emitted}; it names the MPO bond the operator emits, so it cannot be wider"
        )
    # ``space.dim`` is ``Σ m_a irrep_dim(a)`` and ``reduced_dim`` is ``Σ m_a``, so the two
    # differ exactly when some irrep is more than one-dimensional. Symmetry-generic
    # metadata off the space, of the kind ``tests/network/test_hygiene.py`` allows -- no
    # provider read.
    if got.dim != got.reduced_dim or emitted.dim != emitted.reduced_dim:
        raise ValueError(
            "from_terms is Abelian-only: this operator's charge leg has irrep_dim > 1, and "
            "a list of non-Abelian operators does not determine a term -- three of them "
            "fuse through several channels and the DSL has no slot for a coupling tree. "
            "Hand the whole term over instead: one invariant k-site operator from "
            "local_op(dense, phys=...) with no charge, on a tuple of sites"
        )
    return got


def _unit_leg(sym, dual: bool) -> Leg:
    """The trivial ``D=1`` MPO boundary leg, ``IN``."""
    return Leg(GradedSpace.new(sym, {sym.unit: 1}), IN, dual)


def _identity_w(aux: Leg, phys: GradedSpace) -> SymmetricTensor:
    """A spectator site: ``id(aux) (x) id(phys)`` on ``(wl IN, p OUT, p IN, wr OUT)``.

    Symmetry-generic, so the MPO bond a k-site term derives may be *graded* and still run
    through the sites the term does not touch -- which is what makes a non-adjacent term
    work, and what the old ``np.eye`` spectator could not do. ``id(aux)``'s legs are
    ``(OUT, IN)``, so the subscripts put its IN on the left of the ``W`` and its OUT on
    the right; ``aux`` is a whole [Leg][tenet.Leg] because the derived bond's ``dual``
    flag would be dropped by a space alone.
    """
    return _as_w(
        tenet.einsum("ba,pq->apqb", tenet.identity((aux,)), tenet.identity((Leg(phys, IN),)))
    )


def _split(op: SymmetricTensor, cutoff: float) -> list[SymmetricTensor]:
    """One invariant rank-2k operator as k MPO tensors -- MPSKit's ``decompose_localmpo``.

    A trivial leg is glued to each end (``add_util_leg``) so that every peeled factor is
    already rank 4, then one site at a time is peeled off by ``svd_truncated``. **The aux
    bonds are never declared**: they come out of the SVD sector by sector, empty sectors
    omitted, which is why nothing here needs a coupling tree or a multiplicity label.

    The util legs are ``dual`` because ``_as_w``'s bend makes the *derived* bonds dual, and
    an MPO bond carries one ``dual`` convention — ``V (+) V*`` is not a graded space; on
    the unit sector the flag is free. **Except on a sign-braiding grading**: for a bond
    sector with a ``-1`` self-braiding the two caps differ by exactly that twist, so a
    dual odd bond contracted in #160's composition order pays one Koszul sign per cut —
    measured as ``(-1)^(number of odd internal bonds)`` on the chain (#147 gate 2). The
    factors are therefore flipped to the non-dual convention the rank-3 route already
    uses, with the twist paid once per bond (``inv`` on exactly one end -- ``flip_dual`` twice
    is ``chi_a * theta_a``, ``-1`` on an odd line); bosonic splits are byte-identical.
    """
    sym = op.legs[0].space.provider
    fermionic = _braids_with_signs(op.legs[0].space)
    body = string.ascii_uppercase[: op.ndim]
    util = _unit_leg(sym, not fermionic)
    carry = tenet.einsum(f"ba,{body}->a{body}b", tenet.identity((util,)), op)
    out = []
    while carry.ndim > 4:
        m = (carry.ndim - 2) // 2
        rest = (*range(2, 1 + m), *range(2 + m, 2 * m + 2))
        u, s, vh = tenet.linalg.svd_truncated(carry, ((0, 1, 1 + m), rest), cutoff=cutoff)
        out.append(_as_w(u))
        body = string.ascii_uppercase[: vh.ndim - 1]
        carry = tenet.einsum(f"xy,y{body}->x{body}", s, vh)
    out.append(_as_w(carry))
    if fermionic:
        for k in range(len(out) - 1):
            out[k] = tenet.flip_dual(out[k], 3)
            out[k + 1] = tenet.flip_dual(out[k + 1], 0, inv=True)
    return out


# --- the symbolic layer -------------------------------------------------------------
#
# ``from_terms`` assembles a finite-state machine over the MPO bond before it touches a
# single wide tensor. A bond *state* is ``_IDL`` (identity to the left, the term not yet
# begun), ``_IDR`` (the term already closed), or the tuple of ``(operator, site)`` events
# a term has placed so far — its open left-partial-string. Terms sharing an opening share
# a state; the closing edge carries the coefficient. Each state carries its **own**
# ``GradedSpace``: the running fused charge for rank-3 operators, or the ``Leg``
# ``_split``'s ``svd_truncated`` derived for a rank-2k operator mid-split. The bond
# space at a cut is the direct sum, over that cut's states, of the states' spaces — no
# charge solver anywhere (MPSKit's ``mpohamiltonian.jl``:479-492 is the precedent).
# Simplification: operator identity is object identity (``id(op)``) — two equal-valued but
# distinct operator objects get two states, and the compressing sweep erases the
# difference; a symbolic ``Rule`` needs an operator vocabulary tenet refuses to grow.

_IDL: tuple = ()  # the empty prefix
_IDR = "IdR"


def _slabs(space: GradedSpace) -> Iterator[tuple[Sector, int, int, int]]:
    """``(sector, degeneracy, dense offset, dense extent)`` per sector, canonical order."""
    offsets = [space.sector_offset(a) for a in space]
    ends = [*offsets[1:], space.dim]
    for (a, m), o, e in zip(space.sectors, offsets, ends, strict=True):
        yield a, m, o, e - o


def _place(items, space_l, space_r, phys, dual_l, dual_r, carry=None):
    """Scatter every edge's dense block into one buffer; ``from_dense`` **once** per site.

    Placing a state at its slots of a bond is a 0/1 isometry, ``_merge`` hands every state
    a *disjoint* slab per sector, and the edge table is keyed ``(state_l, state_r)`` — so
    the write is disjoint **on the left index**: no two edges share a row slab. ``items``
    yields ``(dense block, left state space, its slots, right state space, its slots)``
    per live edge; the block is the edge's **full** rank-4 dense array, never capped or
    sliced on the way in (#147 gate 4), so the round trip is exact by
    [from_dense][tenet.SymmetricTensor.from_dense]'s own contract.

    With ``carry`` — the compressing sweep's already-truncated right leg as a dense
    ``(D_FSM(n+1), chi)`` map, ``space_r``/``dual_r`` then describing that truncated bond
    — the right end of every block is contracted on the way in, one small ``gemm`` per
    edge, so the buffer is ``D_FSM x d**2 x chi`` rather than quadratic in the FSM width.
    The column index is then no longer disjoint: two edges with different right states
    contract into overlapping columns of the truncated bond, so the write **accumulates**
    instead of assigning. Only the left slab assignment keeps the isometry argument; the
    accumulation's oracle is the round trip itself.

    Built at ``from_dense``'s **default** relative ``atol``, which keeps the refusal the
    per-state isometries used to carry, carry folded or not: a slot map that disagrees
    with its state puts mass in a symmetry-forbidden cell of the assembled site and
    construction *raises* rather than projecting. ``None`` when no edge survives the cut.

    Simplification: the buffer is dense across sectors where the tensor stores only the
    allowed blocks. #191 recorded per-coupled-sector assembly through ``to_matrices`` /
    ``from_matrices`` (``map_view.py``:238, :282) as the upgrade path if that ever stopped
    fitting. #193 took the measurement instead of assuming it: with the carry folded,
    ``from_dense`` is 9% of a K=26 ab initio build and on a two-sector fermionic grading
    the dense buffer is exactly 2x the stored blocks, so the ceiling on that route is
    **~4.5%**. It is not the upgrade path it was recorded as; folding the carry was.
    """
    d = phys.dim
    buf = None
    for blk, state_l, slots_l, state_r, slots_r in items:
        dtype = blk.dtype if carry is None else np.result_type(blk.dtype, carry.dtype)
        if buf is None:
            buf = np.zeros((space_l.dim, d, d, space_r.dim), dtype=dtype)
        elif buf.dtype != np.result_type(buf.dtype, dtype):
            buf = buf.astype(np.result_type(buf.dtype, dtype))
        for a, _, oa, ea in _slabs(state_l):
            ra = slots_l[a]
            for b, _, ob, eb in _slabs(state_r):
                rb = slots_r[b]
                sub = blk[oa : oa + ea, :, :, ob : ob + eb]
                if carry is None:
                    buf[ra : ra + ea, :, :, rb : rb + eb] = sub
                else:
                    buf[ra : ra + ea] += sub @ carry[rb : rb + eb]
    if buf is None:
        return None
    legs = (Leg(space_l, IN, dual_l), Leg(phys, OUT), Leg(phys, IN), Leg(space_r, OUT, dual_r))
    return SymmetricTensor.from_dense(buf, legs)


def _merge(sym, keys, states):
    """Direct-sum the given states' spaces; return the space and each state's dense slots.

    The bond space at a cut and every group subspace inside it are the same construction:
    per state, per sector, one slab whose dense offset is recorded so that ``_place`` can
    scatter into it. Allocation order is the caller's ``keys`` order.
    """
    merged: dict = {}
    for k in keys:
        for a, m in states[k].sectors:
            merged[a] = merged.get(a, 0) + m
    space = GradedSpace.new(sym, merged)
    seen: dict = {}
    per_state = {}
    for k in keys:
        slots = {}
        for a, m, _, ext in _slabs(states[k]):
            slots[a] = space.sector_offset(a) + seen.get(a, 0) * (ext // m)
            seen[a] = seen.get(a, 0) + m
        per_state[k] = slots
    return space, per_state


def _group_embedding(bond, bond_starts, group, group_starts, keys, states, *, left, dual, dual_b):
    """The 0/1 isometry between one *group* of states and its slots of the full bond.

    ``_place``'s slot map as a tensor, a whole group at once: the ``left`` orientation reads
    ``(bond IN, group OUT)`` and slices a left environment down to the group; the other
    reads ``(group IN, bond OUT)`` and does the same to a right environment, or embeds a
    group-resolved environment back into the full bond. ``dual_b`` is the bond side's
    ``dual`` flag, which differs from the states' at the two capped boundary cuts.
    """
    dense = np.zeros((bond.dim, group.dim))
    for k in keys:
        for a, _, _o, ext in _slabs(states[k]):
            r0, c0 = bond_starts[k][a], group_starts[k][a]
            dense[r0 : r0 + ext, c0 : c0 + ext] = np.eye(ext)
    if left:
        return SymmetricTensor.from_dense(dense, (Leg(bond, IN, dual_b), Leg(group, OUT, dual)))
    return SymmetricTensor.from_dense(dense.T, (Leg(group, IN, dual), Leg(bond, OUT, dual_b)))


class EdgeBlocks(NamedTuple):
    """One site of the edge-block table: the FSM edges, split the way the matvec eats them.

    MPSKit's ``(1 C D; . A B; . . 1)`` partition (``jordanmpotensor.jl``:1-42): ``a`` holds
    the open-to-open edges, ``b`` open-to-``IdR``, ``c`` ``IdL``-to-open and ``d`` the
    ``IdL``-to-``IdR`` closings, each keyed by its ``(state_l, state_r)`` labels with the
    per-edge rank-4 tensor on the *states' own* bond legs -- ``None`` meaning "identity",
    never a materialised identity tensor, which is MPSKit's ``tensors``/``scalars`` split
    with the scalar always 1. The two corner identities are implicit.

    ``a_op``/``b_op``/``c_op``/``d_op`` are the same blocks summed onto group-restricted
    bond legs (``IdL`` / open / ``IdR`` subspaces), and the six embeddings connect those
    groups to the full instantiated bond: ``*_l`` slice the site's left cut as
    ``(bond IN, group OUT)``, ``*_r`` its right cut as ``(group IN, bond OUT)``; a missing
    group is ``None``. [Env.heff2][tenet.network.Env.heff2] folds environments into these blocks
    once per
    bond, which is the whole reason the table survives instantiation.

    Three derived views serve the matvec's term merging: ``idmap`` is the rank-2 0/1 map
    of every identity channel of the site -- the two corners plus the free-riding ``a``
    spectators -- between the full left and right bonds, so all "identity through this
    site" paths ride one tensor; ``spec_op`` is the same spectator map restricted to the
    open subspaces; and ``a_real_op`` is ``a_op`` minus those spectators. **A spectator
    rides the rank-2 maps only if its state's space braids with no sign** (#160): the
    identity tensor interleaves the bond line with the two physical lines, and for a
    state that braids with signs that crossing is the Jordan-Wigner string -- it lives
    in the rank-4 tensor and in nothing cheaper, so such a state is classified into
    ``a_real_op`` instead, where [Env.heff2][tenet.network.Env.heff2]'s open-to-open chain contracts
    the
    full tensor. For every sign-free provider the classification is what it always was,
    at the cost it always had.

    MPSKit calls this partition the MPO's *Jordan form*; why this type is not spelled
    that way is in ``docs/design.md`` "Milestone 16".
    """

    a: dict
    b: dict
    c: dict
    d: dict
    a_op: SymmetricTensor | None
    b_op: SymmetricTensor | None
    c_op: SymmetricTensor | None
    d_op: SymmetricTensor | None
    idmap: SymmetricTensor | None
    spec_op: SymmetricTensor | None
    a_real_op: SymmetricTensor | None
    idl_l: SymmetricTensor | None
    open_l: SymmetricTensor | None
    idr_l: SymmetricTensor | None
    idl_r: SymmetricTensor | None
    open_r: SymmetricTensor | None
    idr_r: SymmetricTensor | None


_INTERLEAVE = (
    "the operators of one term interleave at site {}: each k-site operator's "
    "derived bond must close before the next one begins"
)


def _term_edges(
    n_sites, coeff, ops, phys, dual, split, rank3, states, order, moves, stops, spectators
):
    """Walk one term left to right: allocate prefix states, emit its edges.

    ``moves[n]`` collects the non-closing edges ``(state_l, state_r) -> W`` of site ``n``
    (set-once: terms sharing a prefix share the edge), ``stops[n]`` the closing edges into
    ``_IDR`` (accumulating — the coefficient rides here, so repeated strings sum), and
    ``spectators[n]`` the states whose space runs through site ``n`` untouched.
    """
    sym, d = phys.provider, phys.dim
    unit_space = GradedSpace.new(sym, {sym.unit: 1})
    # A term is the *ordered product* of its operators. Placement walks the sites left
    # to right, so putting the list into site order first costs one Koszul sign per
    # swap of two sign-braiding operators -- `c+_1 c_0` written as
    # ``[(c+, 1), (c, 0)]`` is ``-(c_0 c+_1)`` and both spellings build the same MPO.
    # Bosonic operators commute freely; a k-site operator is invariant, hence even.
    odd_sites = [
        sites if isinstance(sites, int) else sites[0]
        for op, sites in ops
        if op.ndim == 3 and _braids_with_signs(op.legs[2].space)
    ]
    inversions = sum(
        1
        for x in range(len(odd_sites))
        for y in range(x + 1, len(odd_sites))
        if odd_sites[x] > odd_sites[y]
    )
    coeff = coeff * (-1.0) ** inversions
    placed: dict[int, tuple] = {}
    for op, sites in ops:
        sites = (sites,) if isinstance(sites, int) else tuple(sites)
        span = [op] if op.ndim == 3 else split(op)
        if len(sites) != len(span):
            raise ValueError(
                f"a rank-{op.ndim} term operator spans {len(span)} site(s), got sites {sites}"
            )
        for j, (site, w) in enumerate(zip(sites, span, strict=True)):
            if site not in range(n_sites):
                raise ValueError(f"term site index {site} is outside range({n_sites})")
            if site in placed:
                raise ValueError(
                    f"two operators of one term sit on site {site}; multiply them first"
                )
            elem = (id(op), site) if op.ndim == 3 else (id(op), sites, j)
            placed[site] = (op, w, elem, j, len(span))
    walk = sorted(placed)
    if not walk:  # a constant shift: coeff times the identity, closed at the last site
        w = tenet.multiply(_identity_w(Leg(unit_space, IN, dual), phys), coeff)
        stops[n_sites - 1][_IDL] = stops[n_sites - 1][_IDL] + w if _IDL in stops[n_sites - 1] else w
        return
    prefix, c, open_k, prev = _IDL, sym.unit, None, -1
    for s in walk:
        op, w, elem, j, k = placed[s]
        for m in range(prev + 1, s):
            spectators[m].add(prefix)
        is3 = op.ndim == 3
        if open_k is not None and (is3 or (id(op), j) != open_k):
            raise ValueError(_INTERLEAVE.format(s))
        if not is3 and open_k is None and (j != 0 or c != sym.unit):
            raise ValueError(_INTERLEAVE.format(s))
        if is3:
            q = op.legs[2].space.sectors[0][0]
            c_next = sym.fusion(c, q)[0]
            lab = sym.dual(c_next) if dual else c_next
            space = GradedSpace.new(sym, {lab: 1})
            key = (id(op), c)
            if key not in rank3:
                left = GradedSpace.new(sym, {sym.dual(c) if dual else c: 1})
                legs = (Leg(left, IN, dual), Leg(phys, OUT), Leg(phys, IN), Leg(space, OUT, dual))
                block = np.reshape(np.asarray(op.to_dense())[:, :, 0], (1, d, d, 1))
                # The running charge to the *right* of this site crosses the site's
                # incoming physical line on its way out -- a braiding the dense
                # ``[:, :, 0]`` round trip cannot see. The R-coefficient is paid here,
                # per physical sector: ``+1`` everywhere for a bosonic grading, the
                # parity sign for a super one, which is exactly the Jordan-Wigner
                # string's foothold on the operator's own site (#147 gate 4 -- at
                # ``d=2`` it is invisible because ``a+ Z = a+`` and ``Z a = a``).
                if _braids_with_signs(space):
                    block = block.copy()
                    for a, _, o, e in _slabs(phys):
                        tree = FusionTree((lab, a), (), (0,), sym.fusion(lab, a)[0])
                        block[:, :, o : o + e, :] *= sym.permute_tree(tree, (1, 0))[0][1].real
                rank3[key] = SymmetricTensor.from_dense(block, legs)
            wt = rank3[key]
            open_next = None
        else:
            wt, space = w, w.legs[3].space
            c_next = sym.unit
            open_next = (id(op), j + 1) if j + 1 < k else None
        if s == walk[-1]:
            if is3 and c_next != sym.unit:
                raise ValueError(
                    f"a term's operator charges must sum to the unit sector, got {space}; "
                    "both MPO boundaries are the trivial D=1 leg, so a charged term has "
                    "nowhere to end"
                )
            wt = tenet.multiply(wt, coeff)
            stops[s][prefix] = stops[s][prefix] + wt if prefix in stops[s] else wt
        else:
            new_prefix = (*prefix, elem)
            if new_prefix not in states:
                states[new_prefix] = space
                order.append(new_prefix)
            moves[s].setdefault((prefix, new_prefix), wt)
            prefix, c = new_prefix, c_next
        open_k = open_next
        prev = s


def _expr_names(expr: str) -> list[str]:
    """One block's operator pattern as a list of names, block2's own spelling rule.

    Whitespace separates names; a pattern with no whitespace in it is one name per
    character. That is what makes ``"cdcd"`` and ``"cd c"`` both legal, and it is the
    line ``general_hamiltonian.hpp``:323-330 draws for the same reason: single
    characters concatenate, a long name needs a separator.
    """
    return expr.split() if any(ch.isspace() for ch in expr) else list(expr)


def _canonical_blocks(
    n_sites: int,
    ops: Mapping[str, SymmetricTensor],
    blocks: Iterable[tuple[str, Any, Any]],
    screen: float,
) -> tuple[GradedSpace, list[SymmetricTensor], list[tuple[Any, Any, Any]]]:
    """The three arrays as merged terms in site order, one numpy pass per block.

    Returns the physical space, the operator table (the caller's operators followed by
    whatever on-site products the coincidences needed), and one
    ``(labels, sites, coefficients)`` triple per surviving term *width*: ``labels`` and
    ``sites`` are ``(T, L)`` integer arrays and ``coefficients`` is length ``T``.

    Three things happen here and each is whole-array work, so none of it costs a Python
    object per term. Rows are sorted into site order by a stable ``argsort``, paying the
    Koszul sign of every inversion of two sign-braiding operators -- the strict ``>``
    rule ``_term_edges`` uses, so a repeated index never contributes a sign. Coincident
    sites are then pre-multiplied into one on-site operator, cached per run of names, and
    a run whose product vanishes drops its terms. Finally ``np.unique`` over
    ``(labels, sites)`` with ``np.add.at`` over the coefficients fuses terms that agree,
    and ``screen`` is applied to what the sum leaves -- after the merge, which is the
    only position that can see a cancellation.
    """
    table = list(ops.values())
    label = {name: i for i, name in enumerate(ops)}
    phys = None
    for name, op in ops.items():
        if op.ndim != 3:
            raise ValueError(
                f"from_arrays: operator {name!r} is rank {op.ndim}; a block names one site per "
                f"operator, so every entry of ops is local_op's rank-3 charge-leg form. An "
                f"invariant k-site term spans k sites and goes to from_terms instead"
            )
        phys = _check_op(op, phys)
    if phys is None:
        raise ValueError("from_arrays: ops is empty; the physical space is read off an operator")
    sym = phys.provider
    charges = [op.legs[2].space.sectors[0][0] for op in table]
    odd = [_braids_with_signs(op.legs[2].space) for op in table]
    matrices = [np.asarray(op.to_dense())[:, :, 0] for op in table]
    products: dict[tuple[int, ...], int] = {}

    def product(combo: tuple[int, ...]) -> int:
        """The on-site product of one run of coincident operators; ``-1`` if it vanishes."""
        if combo not in products:
            m, q = matrices[combo[0]], charges[combo[0]]
            for c in combo[1:]:
                m, q = m @ matrices[c], sym.fusion(q, charges[c])[0]
            # ``local_op`` rebuilds through ``from_dense`` at its default relative atol, so a
            # product whose charge is not the fused one *raises* rather than being projected.
            products[combo] = -1
            if np.any(m):
                table.append(local_op(m, phys=phys, charge=q))
                products[combo] = len(table) - 1
        return products[combo]

    groups: dict[int, list] = {}
    for expr, indices, data in blocks:
        names = _expr_names(expr)
        if not names:
            raise ValueError(f"from_arrays: block {expr!r} names no operator")
        unknown = sorted({nm for nm in names if nm not in label})
        if unknown:
            raise ValueError(
                f"from_arrays: block {expr!r} names {unknown}, which ops does not define; "
                f"the table has {sorted(label)}"
            )
        width = len(names)
        coeffs = np.asarray(data)
        idx = np.asarray(indices)
        if coeffs.ndim != 1:
            raise ValueError(
                f"from_arrays: block {expr!r} has a rank-{coeffs.ndim} data array; one "
                f"coefficient per term means a 1-D array"
            )
        if idx.size != width * coeffs.size:
            raise ValueError(
                f"from_arrays: block {expr!r} has {width} operator(s) and {coeffs.size} "
                f"coefficient(s), so indices holds {width * coeffs.size} site index(es), got "
                f"{idx.size}"
            )
        if not coeffs.size:
            continue
        idx = np.ascontiguousarray(idx).astype(np.intp, copy=False).reshape(coeffs.size, width)
        if idx.min() < 0 or idx.max() >= n_sites:
            raise ValueError(
                f"from_arrays: block {expr!r} has site indices [{idx.min()}, {idx.max()}], "
                f"outside range({n_sites})"
            )
        cols = np.array([label[nm] for nm in names], dtype=np.intp)
        # A term is the ordered product of its operators, so sorting it into site order
        # costs one Koszul sign per inversion of two sign-braiding operators -- strictly
        # ``>``, exactly ``_term_edges``' rule, so coincident sites cost nothing.
        inversions = np.zeros(coeffs.size, dtype=np.intp)
        for x in range(width):
            for y in range(x + 1, width):
                if odd[cols[x]] and odd[cols[y]]:
                    inversions += idx[:, x] > idx[:, y]
        coeffs = coeffs * (-1.0) ** inversions
        perm = np.argsort(idx, axis=1, kind="stable")
        sites = np.take_along_axis(idx, perm, axis=1)
        labels = cols[perm]
        # Rows are bucketed by *which* of their sites coincide -- at most 2**(L-1) buckets
        # -- because that is what fixes the merged width, and a bucket is again rectangular.
        if width == 1:
            signature = np.zeros(coeffs.size, dtype=np.intp)
        else:
            same = (sites[:, 1:] == sites[:, :-1]).astype(np.intp)
            signature = same @ (1 << np.arange(width - 1))
        for sig in np.unique(signature):
            runs: list[list[int]] = []
            for j in range(width):
                if j and (int(sig) >> (j - 1)) & 1:
                    runs[-1].append(j)
                else:
                    runs.append([j])
            rows = signature == sig
            row_labels, row_sites, row_coeffs = labels[rows], sites[rows], coeffs[rows]
            merged = np.empty((row_coeffs.size, len(runs)), dtype=np.intp)
            keep = np.ones(row_coeffs.size, dtype=bool)
            for c, run in enumerate(runs):
                if len(run) == 1:
                    merged[:, c] = row_labels[:, run[0]]
                    continue
                combos, back = np.unique(row_labels[:, run], axis=0, return_inverse=True)
                mapped = np.array([product(tuple(int(x) for x in combo)) for combo in combos])
                merged[:, c] = mapped[back.reshape(-1)]
                keep &= merged[:, c] >= 0
            starts = [run[0] for run in runs]
            groups.setdefault(len(runs), []).append(
                (merged[keep], row_sites[keep][:, starts], row_coeffs[keep])
            )

    out = []
    for width, parts in sorted(groups.items()):
        labels = np.concatenate([p[0] for p in parts])
        sites = np.concatenate([p[1] for p in parts])
        coeffs = np.concatenate([p[2] for p in parts])
        if not coeffs.size:
            continue
        uniq, back = np.unique(np.concatenate([labels, sites], axis=1), axis=0, return_inverse=True)
        summed = np.zeros(len(uniq), dtype=coeffs.dtype)
        np.add.at(summed, back.reshape(-1), coeffs)
        keep = np.abs(summed) > screen
        if keep.any():
            out.append((uniq[keep, :width], uniq[keep, width:], summed[keep]))
    return phys, table, out


class _EdgeTable(NamedTuple):
    """The pruned FSM, its bond spaces and its dense slot maps — built with no tensor.

    ``edges[n]`` is site ``n``'s surviving ``(state_l, state_r) -> W | None`` map (``None``
    meaning a spectator identity); ``ordered[i]`` the live state keys at cut ``i`` in
    allocation order, ``_IDL`` first and ``_IDR`` last; ``bonds[i]`` that cut's
    ``GradedSpace`` and ``starts[i][k][a]`` the dense offset state ``k``'s sector ``a``
    occupies in it; ``groups[i]`` the same, per ``IdL``/open/``IdR`` subset, as
    ``(space, slots, keys)`` or ``None``.

    It is the carrier every consumer scatters into: the dense sites and
    [EdgeBlocks][tenet.network.EdgeBlocks]' operators differ only in which slot map they
    place against, which is why one ``_place`` serves both.
    """

    edges: list[dict]
    ordered: list[list]
    bonds: list
    starts: list
    groups: list
    states: dict
    phys: GradedSpace
    dual: bool


def _edge_table(n_sites, phys, dual, states, order, moves, stops, spectators) -> _EdgeTable:
    """Prune dead states and derive every cut's bond space and slot map.

    Pruning intersects each cut's states with (reachable from ``_IDL``) and (co-reachable
    to ``_IDR``) — two passes over the edge tables, tenpy's ``add_missing_IdL_IdR`` and
    block2's zero-propagation doing the same job in their own spellings. The bond space at
    a cut is then the direct sum of the surviving states' spaces, in allocation order with
    the identities at the two ends.

    Symbolic throughout: not one tensor is built here. It is the half of the old
    ``_instantiate`` that was already cheap and was thrown away once the sites existed.
    """
    sym = phys.provider
    edges: list[dict] = []
    for n in range(n_sites):
        e = dict(moves[n])
        for l_key, w in stops[n].items():
            e[(l_key, _IDR)] = w
        for k in spectators[n]:
            e[(k, k)] = None  # a spectator identity, instantiated lazily by ``_place``
        edges.append(e)
    reach = [set() for _ in range(n_sites + 1)]
    reach[0] = {_IDL}
    for n in range(n_sites):
        reach[n + 1] = {r for (left, r) in edges[n] if left in reach[n]}
    live = [set() for _ in range(n_sites + 1)]
    live[n_sites] = reach[n_sites] & {_IDR}
    for n in reversed(range(n_sites)):
        live[n] = reach[n] & {left for (left, r) in edges[n] if r in live[n + 1]}
    ordered = [[k for k in (_IDL, *order, _IDR) if k in live[n]] for n in range(n_sites + 1)]
    edges = [
        {(lk, rk): w for (lk, rk), w in e.items() if lk in live[n] and rk in live[n + 1]}
        for n, e in enumerate(edges)
    ]

    bonds, starts, groups = [], [], []
    for i, cut in enumerate(ordered):
        bond, per_state = _merge(sym, cut, states)
        bonds.append(bond)
        starts.append(per_state)
        groups.append(
            {
                name: (*_merge(sym, keys, states), keys) if keys else None
                for name, keys in (
                    ("idl", [_IDL] if _IDL in live[i] else []),
                    ("open", [k for k in cut if k not in (_IDL, _IDR)]),
                    ("idr", [_IDR] if _IDR in live[i] else []),
                )
            }
        )
    return _EdgeTable(edges, ordered, bonds, starts, groups, states, phys, dual)


def _dense_w(tab: _EdgeTable, w, key, identities: dict):
    """One edge's full rank-4 dense block; ``None`` is the state's identity.

    Identity edges still go through ``_identity_w`` and its per-space cache: on a
    sign-braiding aux space that ``einsum`` is *not* a bare delta in the carrier basis, so
    the slab is read off the tensor rather than hand-derived.
    """
    if w is not None:
        return np.asarray(w.to_dense())
    space = tab.states[key]
    if space not in identities:
        identities[space] = np.asarray(_identity_w(Leg(space, IN, tab.dual), tab.phys).to_dense())
    return identities[space]


def _site(tab: _EdgeTable, n: int, identities: dict, carry=None) -> SymmetricTensor:
    """Site ``n``, scattered into one buffer against the bond, boundary caps included.

    The caps ride here rather than after a whole-MPO loop so that a site is finished the
    moment it is placed, which is what lets the compressing sweep consume them one at a
    time. ``carry`` is that sweep's rank-2 map onto the already-truncated right bond, both
    legs ``IN`` as ``_instantiate`` repartitions it; ``_place`` folds it into the scatter,
    so the site is born contracted with it and the buffer never widens past ``chi``. The
    site's right leg is the carry's second leg moved to the codomain, which is the same
    relabelling ``einsum("xy,apqx->apqy", carry, w)`` used to make: same space, ``OUT``,
    ``dual`` flipped.
    """
    if carry is None:
        space_r, dual_r, dense_c = tab.bonds[n + 1], tab.dual, None
    else:
        space_r, dual_r = carry.legs[1].space, not carry.legs[1].dual
        dense_c = np.asarray(carry.to_dense())
    items = (
        (
            _dense_w(tab, w, lk, identities),
            tab.states[lk],
            tab.starts[n][lk],
            tab.states[rk],
            tab.starts[n + 1][rk],
        )
        for (lk, rk), w in tab.edges[n].items()
    )
    w = _place(items, tab.bonds[n], space_r, tab.phys, tab.dual, dual_r, dense_c)
    if not tab.dual:
        return w
    # The two *outer* legs go back non-dual: ``Env`` builds its boundary environments from
    # the boundary legs' own flags (``Env.__init__``) and the other builder's ends are
    # non-dual too. Both ends are D=1 on the unit sector, where the flag is a label.
    sym = tab.phys.provider
    triv = GradedSpace.new(sym, {sym.unit: 1})
    if n == 0:
        cap = SymmetricTensor.from_dense(np.ones((1, 1)), (Leg(triv, IN), Leg(triv, OUT, True)))
        w = _as_w(tenet.einsum("xpqr,ax->apqr", w, cap))
    if n == len(tab.edges) - 1:
        cap = SymmetricTensor.from_dense(np.ones((1, 1)), (Leg(triv, IN, True), Leg(triv, OUT)))
        w = _as_w(tenet.einsum("xb,apqx->apqb", cap, w))
    return w


# Which group subspace each block of MPSKit's ``(1 C D; . A B; . . 1)`` partition places
# its two ends against — the only thing that separates the block table's scatter from the
# full site's.
_BLOCK_GROUPS = {
    "a": ("open", "open"),
    "b": ("open", "idr"),
    "c": ("idl", "open"),
    "d": ("idl", "idr"),
}


def _blocks(tab: _EdgeTable, identities: dict) -> list[EdgeBlocks]:
    """The per-site [EdgeBlocks][tenet.network.EdgeBlocks], from the same table.

    The a/b/c/d partition of the surviving edges, with each block's operator placed by the
    *same* ``_place`` against the group slot maps instead of the full-bond ones. The bond
    side of the two boundary cuts is non-dual, matching ``_site``'s caps and ``Env``'s
    boundary legs.
    """
    n_sites = len(tab.edges)
    states = tab.states

    def group_place(n, pairs, gl, gr):
        left, right = tab.groups[n][gl], tab.groups[n + 1][gr]
        if left is None or right is None:
            return None
        (space_l, slots_l, _), (space_r, slots_r, _) = left, right
        items = (
            (_dense_w(tab, w, lk, identities), states[lk], slots_l[lk], states[rk], slots_r[rk])
            for (lk, rk), w in pairs
        )
        return _place(items, space_l, space_r, tab.phys, tab.dual, tab.dual)

    def channel_map(keys, spaces, slots_l, slots_r, dual_l, dual_r):
        """The rank-2 0/1 map carrying each key's space identically across a site."""
        dense = np.zeros((spaces[0].dim, spaces[1].dim))
        for k in keys:
            for a, _, _o, ext in _slabs(states[k]):
                r0, c0 = slots_l[k][a], slots_r[k][a]
                dense[r0 : r0 + ext, c0 : c0 + ext] = np.eye(ext)
        legs = (Leg(spaces[0], IN, dual_l), Leg(spaces[1], OUT, dual_r))
        return SymmetricTensor.from_dense(dense, legs)

    embeds = []
    for i in range(n_sites + 1):
        dual_b = tab.dual and 0 < i < n_sites
        emb = {}
        for name, g in tab.groups[i].items():
            if g is None:
                emb[name] = (None, None)
                continue
            space, slots, keys = g
            emb[name] = tuple(
                _group_embedding(
                    tab.bonds[i],
                    tab.starts[i],
                    space,
                    slots,
                    keys,
                    states,
                    left=left,
                    dual=tab.dual,
                    dual_b=dual_b,
                )
                for left in (True, False)
            )
        embeds.append(emb)

    out = []
    for n in range(n_sites):
        dicts: dict[str, dict] = {"a": {}, "b": {}, "c": {}, "d": {}}
        id_channels = []  # states carried identically through this site, corners included
        for corner in (_IDL, _IDR):
            if corner in tab.ordered[n] and corner in tab.ordered[n + 1]:
                id_channels.append(corner)
        spec_keys = []
        for (lk, rk), w in tab.edges[n].items():
            if lk == rk and lk in (_IDL, _IDR):
                continue  # the two corner identities stay implicit
            name = (
                "d"
                if lk == _IDL and rk == _IDR
                else ("c" if lk == _IDL else ("b" if rk == _IDR else "a"))
            )
            dicts[name][lk, rk] = w
            # A spectator may ride the factorized rank-2 channel maps only if its state's
            # space braids with no sign: the identity tensor interleaves the bond line
            # with the two physical lines, and for a state that braids with signs that
            # crossing is the Jordan-Wigner string, which lives in the rank-4 tensor and
            # in nothing cheaper (#160). Such a state is classified as operator-carrying
            # and rides ``a_real_op``. Grading metadata off the space, not a provider
            # branch.
            if w is None and not _braids_with_signs(states[lk]):
                id_channels.append(lk)
                spec_keys.append(lk)
        ops = {
            name: group_place(n, dicts[name].items(), *where)
            for name, where in _BLOCK_GROUPS.items()
        }
        real = [(kk, w) for kk, w in dicts["a"].items() if not (w is None and kk[0] in spec_keys)]
        idmap = (
            channel_map(
                id_channels,
                (tab.bonds[n], tab.bonds[n + 1]),
                tab.starts[n],
                tab.starts[n + 1],
                tab.dual and n > 0,
                tab.dual and n + 1 < n_sites,
            )
            if id_channels
            else None
        )
        spec = None
        if spec_keys:
            gl_space, gl_slots, _ = tab.groups[n]["open"]
            gr_space, gr_slots, _ = tab.groups[n + 1]["open"]
            spec = channel_map(
                spec_keys, (gl_space, gr_space), gl_slots, gr_slots, tab.dual, tab.dual
            )
        out.append(
            EdgeBlocks(
                dicts["a"],
                dicts["b"],
                dicts["c"],
                dicts["d"],
                ops["a"],
                ops["b"],
                ops["c"],
                ops["d"],
                idmap,
                spec,
                group_place(n, real, "open", "open"),
                embeds[n]["idl"][0],
                embeds[n]["open"][0],
                embeds[n]["idr"][0],
                embeds[n + 1]["idl"][1],
                embeds[n + 1]["open"][1],
                embeds[n + 1]["idr"][1],
            )
        )
    return out


def _instantiate(n_sites, phys, dual, states, order, moves, stops, spectators, *, cutoff):
    """Prune dead states, derive the bond spaces, and place every edge numerically.

    ``_edge_table`` does the symbolic half; ``_site`` scatters each site's live edges into
    one dense buffer and calls ``from_dense`` once for it.

    At ``cutoff=None`` every site is placed and the surviving edges come back *also* as one
    [EdgeBlocks][tenet.network.EdgeBlocks] per site — the same pruned graph, in the a/b/c/d
    partition the two-site matvec consumes, placed by the same primitive against the group
    slot maps. That branch does not stream and its peak memory is unchanged, deliberately:
    the table needs every site.

    Given a float, materialisation runs **inside** the backward compressing sweep, which
    is the only consumer of a site there. Site ``N-1`` is placed and its left bond
    truncated; site ``n`` is placed **against the carry**, which ``_place`` folds into the
    scatter so that the site is born on the already-truncated right leg, and is truncated
    in turn. The forward sweep in ``from_terms`` then runs on tensors that are already at
    the operator Schmidt rank. Right to left first, exactly as ``MPS.compress_`` canonizes
    before it truncates: a forward sweep alone measures the rank of the *left* part against
    the raw FSM bond, which overshoots wherever the redundancy is only visible from the
    right. The widest object that ever exists is
    ``D_FSM x d**2 x chi_Schmidt``, so the full-width MPO never exists as a whole.
    """
    tab = _edge_table(n_sites, phys, dual, states, order, moves, stops, spectators)
    identities: dict = {}
    if cutoff is None:
        return [_site(tab, n, identities) for n in range(n_sites)], _blocks(tab, identities)
    out, carry = [], None
    for n in reversed(range(n_sites)):
        w = _site(tab, n, identities, carry)
        if n:
            u, s, vh = tenet.linalg.svd_truncated(w, ((0,), (1, 2, 3)), cutoff=cutoff)
            w = _as_w(vh)
            # ``u`` came back on the map partition, its ``wl`` leg bent; spell the bend
            # with ``repartition`` so the join above is a composition, not an implicit cap.
            carry = tenet.repartition(tenet.einsum("xy,yz->xz", u, s), (), (0, 1))
        out.append(w)
    out.reverse()
    return out, None


def _compress_forward(sites: list[SymmetricTensor], cutoff: float) -> list[SymmetricTensor]:
    """The left-to-right half of the compressing sweep, in place over ``sites``.

    ``_instantiate`` ran the right-to-left half while it placed the sites; this is the
    return leg, and it is the same eight lines for every builder that takes a float
    ``cutoff``, so it lives here rather than once per builder.
    """
    for n in range(len(sites) - 1):
        u, s, vh = tenet.linalg.svd_truncated(sites[n], ((0, 1, 2), (3,)), cutoff=cutoff)
        sites[n] = _as_w(u)
        carry = tenet.einsum("xy,yz->xz", s, vh)
        # ``vh``'s ``wr`` leg came back bent; spell the bend, as in ``_instantiate``'s sweep.
        carry = tenet.repartition(carry, (0, 1), ())
        sites[n + 1] = _as_w(tenet.einsum("ypqr,xy->xpqr", sites[n + 1], carry))
    return sites


class MPO:
    """A finite MPO: one rank-4 ``SymmetricTensor`` per site, ``(wl IN, p OUT, p IN, wr OUT)``.

    Parameters
    ----------
    sites : Iterable of SymmetricTensor
        The rank-4 site tensors, left to right.
    edge_blocks : Sequence of EdgeBlocks or None, optional
        The per-site block table [from_terms][tenet.network.MPO.from_terms]
        keeps at ``cutoff=None``; ``None`` (the default) for every other MPO.

    Notes
    -----
    Invariance reads ``q(p_out) + q(wr) = q(wl) + q(p_in)``. The first and last sites
    carry a ``D=1`` boundary MPO bond, which is what makes *every* ``W_n`` rank 4 and
    removes the boundary-vector special case.

    **A separate class from [MPS][tenet.network.MPS], with no shape flag** -- the comparison
    with YASTN's and TenPy's choices is in ``docs/design.md`` "Milestone 11".

    [edge_blocks][tenet.network.MPO.edge_blocks] is the one read-only accessor beyond
    the container protocol: the
    per-site ``EdgeBlocks`` table when [from_terms][tenet.network.MPO.from_terms] kept its
    finite-state
    machine (``cutoff=None``), ``None`` for every other MPO. It exists so that
    [Env][tenet.network.Env] can reach the symbolic edge structure without touching a
    private name -- #138 refused public exposure of the symbolic layer "if a caller ever
    needs to inspect it, that is a separate issue with an argument attached", and the
    prepared two-site matvec is that caller and that argument (#141).
    """

    sites: list[SymmetricTensor]

    def __init__(
        self, sites: Iterable[SymmetricTensor], edge_blocks: Sequence[EdgeBlocks] | None = None
    ) -> None:
        self.sites = list(sites)
        self._edge_blocks = None if edge_blocks is None else list(edge_blocks)

    def __len__(self) -> int:
        return len(self.sites)

    def __getitem__(self, n: int) -> SymmetricTensor:
        return self.sites[n]

    def __iter__(self) -> Iterator[SymmetricTensor]:
        return iter(self.sites)

    def edge_blocks(self, n: int) -> EdgeBlocks | None:
        """Site ``n``'s ``EdgeBlocks``, or ``None`` when no table survived.

        Parameters
        ----------
        n : int
            The site.

        Returns
        -------
        EdgeBlocks or None
            The site's block table, or ``None`` when the MPO carries none.

        Notes
        -----
        Only ``from_terms(..., cutoff=None)`` carries a table: the compressing sweep's
        SVD gauge mixes the FSM states, so a compressed ``W`` *has* no edge structure to
        recover -- measured in #141, the sweep leaves zero identity edges on every model
        -- and [from_w][tenet.network.MPO.from_w] never had one. ``None`` therefore also routes
        [Env.heff2][tenet.network.Env.heff2] onto its dense path.
        """
        return None if self._edge_blocks is None else self._edge_blocks[n]

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

        Parameters
        ----------
        w : array_like
            The dense bulk tensor, indexed ``[wl, p_out, p_in, wr]``.
        n_sites : int
            Chain length.
        phys : GradedSpace
            The physical space. Keyword-only, as are all the following.
        bond : GradedSpace
            The graded MPO bond of the bulk.
        boundary : GradedSpace
            The ``D=1`` space of the two boundary legs.
        start : int
            The row of ``w`` the first site keeps.
        end : int
            The column of ``w`` the last site keeps.

        Returns
        -------
        MPO
            ``[first, bulk * (n_sites - 2), last]``, no edge-block table.

        Raises
        ------
        ValueError
            From ``from_dense`` at its **default** relative ``atol``: a wrong
            grading *raises* rather than projecting (see Notes).

        Notes
        -----
        ``w`` is indexed ``[wl, p_out, p_in, wr]``. The first site keeps only row
        ``start`` and the last only column ``end``, each on a ``D=1`` ``boundary`` MPO leg.

        ``SymmetricTensor.from_dense`` is called at its **default** relative ``atol``
        (``src/tenet/ops/dense.py``:301), so a wrong grading *raises* rather than
        projecting -- and that refusal is the proof the grading is right in a way a
        passing ``allclose`` is not.

        The builder that shows what an MPO *is*, and the one a reader needs before they
        can debug one. [from_terms][tenet.network.MPO.from_terms] is the other route (#133 reversed
        this
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
    def from_terms(cls, n_sites: int, terms: Iterable, *, cutoff: float | None = 1e-13) -> "MPO":
        """A term list ``[(coeff, [(op, sites), ...]), ...]`` as a graded MPO.

        Parameters
        ----------
        n_sites : int
            Chain length.
        terms : Iterable
            ``[(coeff, [(op, sites), ...]), ...]``: each ``op`` comes from
            [local_op][tenet.network.local_op] in one of its two forms, and
            ``sites`` is an ``int`` or a tuple of that many site indices --
            ``i`` and ``(i,)`` are the same thing.
        cutoff : float or None, optional
            The two compressing SVD sweeps' cutoff. ``0.0`` keeps every
            singular value; ``None`` skips both sweeps entirely and keeps the
            finite-state machine's block table (see Notes for the three-way
            behaviour and what ``None`` does *not* affect). Default ``1e-13``.
            Keyword-only.

        Returns
        -------
        MPO
            The assembled operator; with ``cutoff=None`` it carries the
            per-site [edge_blocks][tenet.network.MPO.edge_blocks] table.

        Raises
        ------
        ValueError
            If ``terms`` is empty (the physical space is read off an
            operator); if an operator is neither rank 3 nor rank 2k, sits on
            the wrong number of sites, leaves ``range(n_sites)``, or shares a
            site with another operator of its term; if the operators disagree
            about the physical space; if two k-site operators of one term
            interleave; if a term's operator charges do not sum to the unit
            sector (both MPO boundaries are the trivial ``D=1`` leg, so a
            charged term has nowhere to end); if a charge leg carries more
            than one sector at degeneracy 1; or if a non-Abelian term is
            spelled as a *list* of charge-leg operators -- the DSL has no slot
            for a coupling tree, and the message says to hand the whole term
            over as one invariant k-site operator instead.

        Examples
        --------
        >>> import numpy as np
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPO, local_op
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> sp = np.array([[0.0, 0.0], [1.0, 0.0]])
        >>> opp = local_op(sp, phys=phys, charge=U1Sector(-2))
        >>> opm = local_op(sp.T, phys=phys, charge=U1Sector(2))
        >>> terms = []  # the 3-site XY chain
        >>> for i in range(2):
        ...     terms.append((0.5, [(opp, i), (opm, i + 1)]))
        ...     terms.append((0.5, [(opm, i), (opp, i + 1)]))
        >>> h = MPO.from_terms(3, terms)
        >>> len(h), h.to_dense().shape
        (3, (8, 8))

        Notes
        -----
        The dispatch is on ``op.ndim`` alone:

        * **rank 3**, the charge-leg form, one site, Abelian only;
        * **rank 2k**, an invariant *k*-site term on ``k`` sites, any symmetry.
          [tenet.linalg.svd_truncated][tenet.ops.linalg.svd_truncated] peels it into ``k`` MPO
          tensors and **the aux
          bond it runs through is the one the SVD found**, so a non-Abelian term needs no
          coupling tree and no multiplicity label: both live inside the operator's own
          blocks. The sites need not be adjacent -- the derived bond, graded or not, runs
          through the identities on the sites in between.

        A term is a coefficient and a list of ``(operator, sites)`` pairs, with identities
        implied on every untouched site. The sum is assembled **symbolically**, as a
        finite-state machine over labelled bond states: identity-left, identity-right,
        and one state per distinct open left-partial-string, so terms that share an
        opening share a state and the closing edge carries the coefficient. States
        unreachable from the left identity or unable to reach the right one are pruned,
        each edge is placed as one rank-4 tensor, and a site is the plain sum of its
        edges. The bond handed to the compressing sweep is therefore the FSM's — one
        state per open string — never one channel per term.

        **The MPO bond spaces are derived, never declared.** Charges enter once, as
        [local_op][tenet.network.local_op]'s ``charge``; from there every FSM state carries its own
        space —
        the running fused charge of the rank-3 operators to its left, or the graded
        ``Leg`` a k-site operator's internal SVD produced — and the bond at each cut is
        the direct sum of its states' spaces. There is no place left to write a wrong
        grading down, and nothing re-decides the grading afterwards.

        ``cutoff`` controls the two compressing SVD sweeps that run after assembly (right
        to left, then left to right), taking the FSM bond down to the operator Schmidt
        rank -- worth it exactly for couplings with numerical low rank the graph cannot
        see, such as power laws. ``cutoff=0.0`` keeps every singular value;
        ``cutoff=None`` skips **both sweeps entirely**: the MPO is the finite-state
        machine, its bond dimension is combinatorial, and no floating-point tolerance
        participates in the assembly. A k-site operator's *internal* SVD -- the one that
        peels it into ``k`` tensors -- is a different SVD and is **unaffected** by
        ``cutoff=None``; it runs at the default ``1e-13`` in that case.

        **For finite-range models the compressing sweeps reduce the bond dimension by
        exactly nothing**, while the SVD gauge mixes the FSM states and erases the
        identity edges, so only ``cutoff=None`` keeps the block table that
        [edge_blocks][tenet.network.MPO.edge_blocks] exposes and that routes
        [Env.heff2][tenet.network.Env.heff2] onto its prepared per-bond operator. Whether
        that trade wins depends on the backend and the bond dimension, and the default
        stays ``1e-13`` because power-law couplings are where the sweep earns its keep.
        The measurements behind both statements are in ``docs/design.md``
        "Milestone 16".

        **There is no** ``phys=`` **argument**: the operators carry the physical space and
        a second source of truth could disagree with them, which would surface as a
        structure error instead of a message about ``phys``. Uniform physical space only,
        and every term's charges must sum to the unit sector.

        Fermionic terms build like any other graded terms, and the braided route needs
        no Jordan-Wigner operator in the API: an odd FSM bond crossing a physical line
        *is* the string, paid by the Koszul braiding under #160's composition rule
        (#147). Two conventions follow from that and are pinned by the gate-4 oracle.
        A term's operator list is the **ordered product** of its operators --
        ``[(c, i), (c+, j)]`` is ``c_i c+_j``, which for ``i != j`` is ``-c+_j c_i`` --
        with the reordering-to-site-order sign paid on the coefficient. And intra-site
        ordering for a multi-flavour site (spinful ``d=4``) is a property of the
        *on-site matrices*, documented where they are defined, not of this assembler.
        Non-Abelian terms spelled as a *list* of charge-leg operators are refused with
        a message rather than accepted. Outside
        ``jit``/``grad`` like the rest of this module, because the assembly decides
        [GradedSpace][tenet.GradedSpace]\\ s.
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
        sym = phys.provider
        # One dual convention per MPO, set by whether any k-site split runs in it: the
        # split's derived bonds come back dual, and a bond hosts both kinds of state.
        # On a sign-braiding grading ``_split`` hands its bonds back non-dual (the dual
        # cap would cost the twist per odd cut; see its docstring), so the whole MPO
        # stays on the rank-3 route's non-dual convention there.
        dual = any(op.ndim != 3 for _, ops in strings for op, _ in ops) and not (
            _braids_with_signs(phys)
        )
        split_cutoff = 1e-13 if cutoff is None else cutoff
        splits: dict[int, list[SymmetricTensor]] = {}

        def split(op: SymmetricTensor) -> list[SymmetricTensor]:
            if id(op) not in splits:
                splits[id(op)] = _split(op, split_cutoff)
            return splits[id(op)]

        states: dict = {_IDL: GradedSpace.new(sym, {sym.unit: 1})}
        states[_IDR] = states[_IDL]
        order: list = []
        rank3: dict = {}
        moves: list[dict] = [{} for _ in range(n_sites)]
        stops: list[dict] = [{} for _ in range(n_sites)]
        spectators: list[set] = [{_IDL, _IDR} for _ in range(n_sites)]
        for coeff, ops in strings:
            _term_edges(
                n_sites,
                coeff,
                ops,
                phys,
                dual,
                split,
                rank3,
                states,
                order,
                moves,
                stops,
                spectators,
            )
        # ``_instantiate`` streams whenever it is given a float cutoff: it places one site
        # at a time in the backward sweep's order and truncates as it goes, so what comes
        # back is already at the operator Schmidt rank and the full-width MPO never exists.
        sites, tables = _instantiate(
            n_sites, phys, dual, states, order, moves, stops, spectators, cutoff=cutoff
        )
        if cutoff is None:
            return cls(sites, tables)
        return cls(_compress_forward(sites, cutoff))

    @classmethod
    def from_arrays(
        cls,
        n_sites: int,
        ops: Mapping[str, SymmetricTensor],
        blocks: Iterable[tuple[str, Any, Any]],
        *,
        cutoff: float | None = 1e-13,
        screen: float = 1e-12,
    ) -> "MPO":
        """Blocks of terms as arrays -- ``(expr, indices, data)`` -- as a graded MPO.

        Parameters
        ----------
        n_sites : int
            Chain length.
        ops : Mapping of str to SymmetricTensor
            The caller's operator table: a name to the rank-3 charge-leg form
            of [local_op][tenet.network.local_op]. The names are what an
            ``expr`` spells; nothing is registered anywhere and operator
            identity stays object identity.
        blocks : Iterable
            One ``(expr, indices, data)`` triple per operator pattern.
            ``expr`` is a string of names -- whitespace-separated, or one name
            per character when it holds no whitespace. ``indices`` is an
            integer array of shape ``(T, L)`` with ``L == len(expr)`` naming
            the site of each operator (a 1-D array of ``T * L`` entries is
            reshaped). ``data`` is the length-``T`` array of coefficients, and
            its dtype decides the MPO's.
        cutoff : float or None, optional
            The compressing SVD sweeps' cutoff, with
            [from_terms][tenet.network.MPO.from_terms]'s three-way meaning
            unchanged. Default ``1e-13``. Keyword-only.
        screen : float, optional
            Coefficient magnitude threshold, applied **after** the merge:
            a merged term survives when ``abs(coeff) > screen``. Default
            ``1e-12``. Keyword-only.

        Returns
        -------
        MPO
            The assembled operator; with ``cutoff=None`` it carries the
            per-site [edge_blocks][tenet.network.MPO.edge_blocks] table.

        Raises
        ------
        ValueError
            If an entry of ``ops`` is not rank 3, or the operators disagree
            about the physical space, or a charge leg is non-Abelian (the
            checks [from_terms][tenet.network.MPO.from_terms] makes); if
            ``ops`` is empty; if a block's ``expr`` names an operator the
            table does not define, or names none at all; if ``data`` is not
            1-D, or ``len(expr) * len(data) != indices.size``; if a site index
            leaves ``range(n_sites)``; or if no term survives the merge and
            the screen.

        Examples
        --------
        >>> import numpy as np
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPO, local_op
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> sp = np.array([[0.0, 0.0], [1.0, 0.0]])
        >>> ops = {
        ...     "+": local_op(sp, phys=phys, charge=U1Sector(-2)),
        ...     "-": local_op(sp.T, phys=phys, charge=U1Sector(2)),
        ... }
        >>> bonds = np.array([[0, 1], [1, 2]])  # the 3-site XY chain, two blocks
        >>> blocks = [("+-", bonds, np.full(2, 0.5)), ("-+", bonds, np.full(2, 0.5))]
        >>> h = MPO.from_arrays(3, ops, blocks)
        >>> len(h), h.to_dense().shape
        (3, (8, 8))

        Notes
        -----
        The same operator [from_terms][tenet.network.MPO.from_terms] builds, from the
        input shape block2 uses (``integral_general.hpp``:45-57): three parallel arrays
        per operator pattern, transposed into a triple because Python has no reason to
        keep them apart. It exists for the input where the term count is the wall -- an
        ab initio Hamiltonian is ``O(K^4)`` terms over a handful of patterns -- and the
        difference is that the pattern's work is done once per *block* in numpy instead
        of once per term in Python. ``from_terms``' list is the right shape for a lattice
        model and is unchanged; neither is deprecated or aliased to the other.

        **Every operator is rank 3.** A block gives one site index per name, so
        ``local_op``'s invariant *k*-site form -- which spans *k* sites through an SVD --
        has nowhere to put its extra indices and is refused with a message pointing at
        ``from_terms``. The MPO bond spaces are still derived and never declared, and the
        assembler is the same finite-state machine walk; only the way terms arrive is new.

        Three things happen before the walk, and all three are whole-array work:

        * each row is sorted into site order by a stable ``argsort``, paying the Koszul
          sign of every inversion of two sign-braiding operators -- the same strict-``>``
          rule ``from_terms`` applies to a term's operator list, so the two spellings of
          one fermionic term agree by construction;
        * operators that coincide on a site are **pre-multiplied** into one on-site
          operator, cached per run of names, and a term whose on-site product vanishes
          (``c c`` on one site, say) is dropped. This is the burden ``from_terms`` refuses
          with "two operators of one term sit on site N; multiply them first";
        * terms agreeing on ``(operator labels, sites)`` are fused, their coefficients
          summed. Permutational symmetry is therefore **expanded** by the caller and
          merged here, which is block2's own order and the only correct one: the eight
          images of ``(ij|kl)`` are eight different operator strings, so folding the
          orbit into one coefficient builds a different operator.

        ``screen`` runs on what the merge leaves, which is the only position that can see
        a cancellation, and it is one knob where block2 has four. At its default it
        removes the symmetry-forbidden ``~1e-15`` entries a real integral file carries and
        nothing else; it is an accuracy/size trade the caller can take deliberately at
        ``1e-4`` and above, not a performance lever.
        """
        phys, table, merged = _canonical_blocks(n_sites, ops, blocks, screen)
        sym = phys.provider

        def split(op: SymmetricTensor) -> list[SymmetricTensor]:
            # Unreachable: ``_canonical_blocks`` admits rank-3 operators only, so no term
            # here spans more than one site per operator and nothing is ever peeled.
            raise ValueError(f"from_arrays: a rank-{op.ndim} operator cannot reach the walk")

        states: dict = {_IDL: GradedSpace.new(sym, {sym.unit: 1})}
        states[_IDR] = states[_IDL]
        order: list = []
        rank3: dict = {}
        moves: list[dict] = [{} for _ in range(n_sites)]
        stops: list[dict] = [{} for _ in range(n_sites)]
        spectators: list[set] = [{_IDL, _IDR} for _ in range(n_sites)]
        seen = 0
        for labels, sites, coeffs in merged:
            for row, where, coeff in zip(labels, sites, coeffs, strict=True):
                _term_edges(
                    n_sites,
                    coeff,
                    [(table[k], int(s)) for k, s in zip(row, where, strict=True)],
                    phys,
                    False,  # no k-site split runs here, so the MPO stays non-dual
                    split,
                    rank3,
                    states,
                    order,
                    moves,
                    stops,
                    spectators,
                )
                seen += 1
        if not seen:
            raise ValueError(
                "from_arrays: no term survived the merge and screen "
                f"(screen={screen}); an MPO is read off its terms and there are none"
            )
        sites_out, tables = _instantiate(
            n_sites, phys, False, states, order, moves, stops, spectators, cutoff=cutoff
        )
        if cutoff is None:
            return cls(sites_out, tables)
        return cls(_compress_forward(sites_out, cutoff))

    def to_dense(self) -> Any:
        """The full ``d**N x d**N`` operator, ``D=1`` boundaries dropped.

        Returns
        -------
        array
            The backend's dense matrix of shape ``(d**N, d**N)``.

        Notes
        -----
        [MPS.to_dense][tenet.network.MPS.to_dense]'s twin, with its warning: exponential in ``N``,
        an oracle exit
        for tests, and nothing an algorithm calls.
        """
        out = self[0]
        for n in range(1, len(self)):
            body = string.ascii_uppercase[: 2 * n]
            out = tenet.einsum(f"xpqr,a{body}x->a{body}pqr", self[n], out)
        n_sites, d = len(self), self[0].legs[1].space.dim
        order = list(range(0, 2 * n_sites, 2)) + list(range(1, 2 * n_sites, 2))
        return out.to_dense()[0, ..., 0].transpose(order).reshape(d**n_sites, d**n_sites)
