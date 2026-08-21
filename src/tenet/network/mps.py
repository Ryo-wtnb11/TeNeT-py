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
from tenet.network.common import Recent, entropy, ones, spectrum, spectrum_sectors
from tenet.symmetry import Sector

__all__ = [
    "MPO",
    "MPS",
    "MPS_FORMAT_VERSION",
    "EdgeBlocks",
    "EdgeTable",
    "expectation_1site",
    "expectation_2site",
    "expectation_profile",
    "local_op",
    "overlap",
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
        removed. Written *through* [overlap][tenet.network.overlap] rather than beside it
        (#213), so the one-state and two-state readings of the same transfer pass cannot
        drift.
        """
        return overlap(self, self) ** 0.5

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

    # --- entanglement ---------------------------------------------------------------

    def _bond_svds(self) -> dict[int, SymmetricTensor]:
        """Every bond's normalized singular-value tensor, off a **canonical copy**.

        [compress_][tenet.network.MPS.compress_]'s body with the truncation removed, run on
        ``self.copy()``: a caller reading a state must not have it re-gauged underneath, and
        the values of a non-canonical gauge are not Schmidt values at all. So this canonizes
        first, always -- the choice ``compress_`` makes for the same reason -- and pays one
        ``lq`` pass for it whatever ``center`` says.
        """
        psi = self.copy().canonize_(0)
        out: dict[int, SymmetricTensor] = {}
        for n in range(len(psi) - 1):
            aa = tenet.einsum("apx,xqr->apqr", psi[n], psi[n + 1])
            u, s, vh = tenet.linalg.svd(aa, ((0, 1), (2, 3)))
            out[n] = s = s / tenet.norm(s)
            psi[n], psi[n + 1] = u, vh  # the write barrier bends ``vh`` back
            psi[n + 1] = tenet.einsum("xy,yqr->xqr", s, psi[n + 1])
        return out

    def schmidt_values(self) -> dict[int, list[float]]:
        """The Schmidt values across every bond, flattened and descending.

        Returns
        -------
        dict of int to list of float
            One entry per internal bond, keyed by the bond's **left site** -- the key
            [sweep_][tenet.network.sweep_]'s ``schmidt`` dict already uses -- holding that
            cut's values, ``sqrt(qdim)``-weighted, normalized and sorted descending. A
            one-site state has no internal bond and gives ``{}``. The SVD here is the
            exact [tenet.linalg.svd][tenet.ops.linalg.svd], so a sector the bond carries
            but the state does not occupy contributes an explicit ``0.0`` rather than
            being dropped the way a truncating sweep drops it.

        Examples
        --------
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPS
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1), U1Sector(1)])
        >>> {n: [round(v, 6) for v in vals] for n, vals in psi.schmidt_values().items()}
        {0: [1.0, 0.0], 1: [1.0, 0.0]}

        Notes
        -----
        YASTN's ``get_Schmidt_values`` and TenPy's ``entanglement_spectrum(by_charge=False)``.
        Both are methods on the state rather than outputs of an algorithm, and so is this:
        the sweep computes the same numbers at every bond of every sweep and reports only
        how much they *moved*, which is a convergence diagnostic and not an answer about the
        converged state (#215).

        A canonical **copy** is taken first, so this never re-gauges the state it reads and
        never reports the values of a non-canonical gauge. Each of the three readers here
        pays its own SVD sweep; a caller wanting two of them on a large state should keep
        the first result rather than call twice.
        """
        return {n: spectrum(s) for n, s in self._bond_svds().items()}

    def schmidt_sectors(self) -> dict[int, dict[Sector, list[float]]]:
        """[schmidt_values][tenet.network.MPS.schmidt_values], resolved by symmetry sector.

        Returns
        -------
        dict of int to (dict of Sector to list of float)
            One entry per internal bond, keyed by the bond's left site; each is that cut's
            spectrum split by coupled sector, values descending within a sector.

        Examples
        --------
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPS
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1), U1Sector(1)])
        >>> [(sector.charge, len(vals)) for sector, vals in psi.schmidt_sectors()[0].items()]
        [(-2, 1), (0, 1)]

        Notes
        -----
        TenPy's ``entanglement_spectrum(by_charge=True)``, and the read a graded bond is
        *for*: which sector carries the entanglement is a question a flat list cannot answer
        and a labelled bond answers for free. The ``sqrt(qdim)`` weight is applied in
        [spectrum_sectors][tenet.network.spectrum_sectors] and nowhere else.
        """
        return {n: spectrum_sectors(s) for n, s in self._bond_svds().items()}

    def entanglement_entropy(self, *, alpha: float = 1.0) -> dict[int, float]:
        """The entanglement entropy across every bond, in **nats**.

        Parameters
        ----------
        alpha : float, optional
            The Renyi index handed to [entropy][tenet.network.entropy]. Default ``1.0``,
            the von Neumann entropy. Keyword-only.

        Returns
        -------
        dict of int to float
            One entropy per internal bond, keyed by the bond's left site.

        Examples
        --------
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPS
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1)])
        >>> psi.entanglement_entropy()  # a product state is unentangled
        {0: -0.0}

        Notes
        -----
        Nats, not bits: the convention is stated on [entropy][tenet.network.entropy], which
        does the arithmetic, because the two references disagree about it. The multiplet
        weight that makes an SU(2) state agree with the same state under U(1) is stated
        there too.

        TenPy's ``entanglement_entropy(n=1)`` and YASTN's ``get_entropy(alpha=1)``. Both
        return one value per cut including the two trivial boundary cuts; this returns the
        ``N - 1`` internal bonds only, because a boundary cut of a finite open chain is
        zero by construction and the key here is a bond's left site, which a boundary cut
        does not have.
        """
        return {n: entropy(s, alpha=alpha) for n, s in self._bond_svds().items()}

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
# container state. #213 added ``overlap`` and ``expectation_profile`` here and put
# ``measure_mpo`` and ``correlation_function`` in ``env.py``, and the split is forced
# rather than chosen: those two read an ``Env`` and ``env.py`` imports this module, while
# ``MPS.norm`` is written through ``overlap``, so a single ``network/measure.py`` holding
# all four would be a cycle. Simplification: the day ``sample`` or ``rdm`` land -- neither
# of which touches ``MPS.norm`` -- ``network/measure.py`` is the module and YASTN's
# ``_measure.py`` the design.


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


def overlap(bra: MPS, ket: MPS) -> float:
    """``<bra|ket>`` by one transfer pass, **undivided** by either state's norm.

    Parameters
    ----------
    bra : MPS
        The state that is conjugated; any gauge, any norm.
    ket : MPS
        The state that is not; any gauge, any norm. The two may carry different bond
        spaces, and must have the same number of sites.

    Returns
    -------
    float
        ``<bra|ket>``.

    Raises
    ------
    ValueError
        If the two states have different lengths.

    Examples
    --------
    >>> from tenet import GradedSpace
    >>> from tenet.network import MPS, overlap
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1)])
    >>> phi = MPS.product(phys, [U1Sector(1), U1Sector(-1)])
    >>> round(overlap(phi, psi), 12)
    1.0
    >>> round(overlap(psi, psi) - psi.norm() ** 2, 12)
    0.0

    Notes
    -----
    **Undivided**, the convention [Env.measure][tenet.network.Env.measure] already keeps and
    the one both references keep for this function -- YASTN's ``measure_overlap(bra, ket)``
    and ``vdot``, TenPy's ``MPS.overlap(other)``. A fidelity is
    ``overlap(phi, psi) / (phi.norm() * psi.norm())`` and the caller spells the division,
    because the two states it needs are the caller's.
    [expectation_1site][tenet.network.expectation_1site] divides and says so in its own
    name; a ``measure_``-shaped name here would make one verb mean two things.

    The two chains may carry **different bond spaces**: the transfer tensor holds one index
    from each, which is the same fact ``Env(psi, h, bra=phi)`` rests on one level up. Two
    states whose *boundary* legs sit in different sectors have no coupled sector at all and
    the overlap is structurally zero rather than numerically small.

    [MPS.norm][tenet.network.MPS.norm] is ``overlap(psi, psi) ** 0.5``, and
    ``Env(psi, MPO.identity(len(psi), phys), bra=phi).measure()`` is this number computed
    the long way, through the environment cache; both agreements are tested.
    """
    if len(bra) != len(ket):
        raise ValueError(
            f"an overlap needs two states of the same length, got {len(bra)} and {len(ket)}"
        )
    return float(_braket(bra.sites, ket.sites))


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


def expectation_profile(psi: MPS, o: SymmetricTensor) -> list[float]:
    """``<psi|o_n|psi> / <psi|psi>`` at **every** site, in one pass over the chain.

    Parameters
    ----------
    psi : MPS
        The state; any gauge, any norm, and **not** modified -- the walk runs on a copy.
    o : SymmetricTensor
        The operator, rank 2 on ``(phys OUT, phys IN)`` --
        [local_op][tenet.network.local_op]'s invariant one-site form, exactly as
        [expectation_1site][tenet.network.expectation_1site] takes it.

    Returns
    -------
    list of float
        One normalized expectation value per site, in site order.

    Raises
    ------
    ValueError
        If ``o`` is not rank 2.

    Examples
    --------
    >>> import numpy as np
    >>> from tenet import GradedSpace
    >>> from tenet.network import MPS, expectation_profile, local_op
    >>> from tenet.symmetry import U1, U1Sector
    >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1), U1Sector(1)])
    >>> sz = local_op(np.diag([-0.5, 0.5]), phys=phys)
    >>> [round(v, 6) for v in expectation_profile(psi, sz)]
    [0.5, -0.5, 0.5]

    Notes
    -----
    **One pass, not one pass per site.** ``[expectation_1site(psi, o, n) for n in
    range(len(psi))]`` is the same numbers and costs two *full-chain* transfer passes per
    site, i.e. ``O(N**2)`` transfer contractions for the profile every DMRG user plots.
    This walks the chain once, moving the orthogonality centre right by a ``qr`` at each
    step and reading the operator off the centre -- which a canonical MPS makes exact,
    because everything left of the centre is left-orthonormal and everything right of it
    right-orthonormal, so both halves of the transfer close to the identity and
    ``<o_n> = <A_n|o|A_n>``. Both references do exactly this
    (YASTN ``measure_1site(..., sites=None)``, TenPy ``expectation_value(ops)``), and
    ``tests/network/test_measure.py`` counts the contractions rather than claiming them.

    ``psi.copy().canonize_(0)`` first, so no gauge is assumed of the input and the input is
    not re-gauged: the same choice
    [MPS.schmidt_values][tenet.network.MPS.schmidt_values] makes, for the same reason. The
    normalization is then free -- ``canonize_`` leaves a unit-norm state -- which is why no
    second transfer pass computes ``<psi|psi>``.

    Divided by ``<psi|psi>``, matching
    [expectation_1site][tenet.network.expectation_1site]; the reason the divided and
    undivided readings carry different names is stated there.
    """
    _check(psi, o, 0, 2, len(psi) - 1)
    walk = psi.copy().canonize_(0)
    out = []
    for n in range(len(walk)):
        a = walk[n]
        out.append(float(tenet.inner(a, tenet.einsum("Pq,aqr->aPr", o, a))))
        if n + 1 < len(walk):
            q, r = tenet.linalg.qr(a, ((0, 1), (2,)))
            walk[n] = q
            walk[n + 1] = tenet.einsum("xy,yqr->xqr", r, walk[n + 1])
    return out


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


def _w_entry(value: Any, key: Any, n: int) -> tuple[Any, SymmetricTensor | None]:
    """One ``W`` entry as ``(coefficient, operator)``; a ``None`` operator is the identity.

    The four spellings ``MPO.from_entries`` accepts, flattened to the one pair the walk
    eats. MPSKit's matrix entries are the same vocabulary in Julia's spelling --
    ``MPOTensor``, ``Missing`` or ``Number`` (``mpohamiltonian.jl``'s matrix constructor)
    -- with ``None`` for ``Missing`` and the pair form added because a ``W`` matrix is
    usually written as a coefficient times a named operator.
    """
    if value is None:
        return 1.0, None
    if isinstance(value, SymmetricTensor):
        return 1.0, value
    if isinstance(value, tuple):
        if len(value) != 2 or not isinstance(value[1], SymmetricTensor):
            raise ValueError(
                f"from_entries: entry {key} of site {n} is a tuple, so it is the pair "
                f"(coefficient, operator) with the operator from local_op; got {value!r}"
            )
        return value
    if np.ndim(value) == 0:
        return value, None
    raise ValueError(
        f"from_entries: entry {key} of site {n} is None (the identity), a number (that "
        f"multiple of the identity), a rank-3 operator from local_op, or the pair "
        f"(coefficient, operator); got a {type(value).__name__}. A dense matrix becomes an "
        f"operator through tenet.network.local_op(dense, phys=..., charge=...)"
    )


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
# Both builders assemble a finite-state machine over the MPO bond before either touches a
# single wide tensor. A bond *state* is ``_IDL`` (identity to the left, the term not yet
# begun), ``_IDR`` (the term already closed), or one open left-partial-string — the
# operators a term has placed so far. Terms sharing an opening share a state; the closing
# edge carries the coefficient. Each state carries its **own** ``GradedSpace``: the
# running fused charge for rank-3 operators, or the ``Leg`` ``_split``'s ``svd_truncated``
# derived for a rank-2k operator mid-split. The bond space at a cut is the direct sum,
# over that cut's states, of the states' spaces — no charge solver anywhere (MPSKit's
# ``mpohamiltonian.jl``:479-492 is the precedent).
#
# **A state is an integer, and so is everything the walk indexes by** (#197). A partial
# string used to be the tuple of its ``(operator, site)`` events, rebuilt and rehashed
# once per operator placed, which made the prefix key quadratic in the term length; it is
# now interned from ``(parent, slot, site)``, one dict lookup on three small ints.
# ``_IDL`` and ``_IDR`` are 0 and 1 for the same reason. Nothing outside this module reads
# a state label's *value* — ``_edge_table``, ``EdgeBlocks`` and ``env.py`` compare it
# against those two constants and otherwise carry it — so the labels are free to be
# whatever the walk is fastest on.
# Simplification: operator identity is object identity (``id(op)``) — two equal-valued but
# distinct operator objects get two slots, and the compressing sweep erases the
# difference; a symbolic ``Rule`` needs an operator vocabulary tenet refuses to grow.

_IDL = 0  # the empty prefix
_IDR = 1


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


class _Walk:
    """The finite-state machine under construction, in the integer form the walk eats.

    A **slot** is one unit of on-site work: a rank-3 charge-leg operator, or one piece of
    a rank-2k operator's split placed on a fixed tuple of sites. A canonicalized term is
    then two integer rows -- its slots and its sites, both in site order -- and one
    coefficient, which is the shape [MPO.from_arrays][tenet.network.MPO.from_arrays]
    receives from its caller and the shape [MPO.from_terms][tenet.network.MPO.from_terms]
    canonicalizes its term list into. One walk, two front ends.

    Everything that depends on a *pattern* rather than on a term lives here and is built
    once: ``ops`` per slot, and ``trans`` per ``(slot, running charge)`` -- the rank-4
    edge tensor, the charge after it and the state's space. That is what takes the
    per-operator cost in ``_term_edges`` down to two dict lookups, with the fusion, the
    ``GradedSpace``, the braiding probe and the dense round trip paid once per pattern
    instead of once per term (#197: at ab initio scale they measured 3.7
    ``_braids_with_signs`` and 4.7 ``GradedSpace.new`` calls *per term*).

    The five structures ``_edge_table`` consumes -- ``states``, ``order``, ``moves``,
    ``stops``, ``spectators`` -- are attributes, filled in place. Closing edges accumulate
    as plain coefficients in ``closing`` and become tensors in ``close``, once per
    surviving edge rather than once per term.
    """

    def __init__(self, n_sites: int, phys: GradedSpace, dual: bool) -> None:
        self.n_sites = n_sites
        self.phys = phys
        self.sym = phys.provider
        self.dual = dual
        unit = GradedSpace.new(self.sym, {self.sym.unit: 1})
        self.slots: dict = {}  # the front end's key -> slot
        self.ops: list = []  # slot -> (operator, its split piece or None)
        self.charges: dict = {self.sym.unit: 0}  # running charge -> index; the unit is 0
        self.sectors: list = [self.sym.unit]  # the inverse of ``charges``
        self.trans: dict = {}  # (slot, charge) -> (W, the charge after it, the state's space)
        self.states: dict = {_IDL: unit, _IDR: unit}
        self.order: list = []
        self.prefixes: dict = {}  # (parent, slot, site) -> state
        self.high: dict = {}  # state -> the highest site already marked as its spectator
        self.moves: list[dict] = [{} for _ in range(n_sites)]
        self.stops: list[dict] = [{} for _ in range(n_sites)]
        self.spectators: list[set] = [{_IDL, _IDR} for _ in range(n_sites)]
        self.closing: dict = {}  # (site, state, slot, charge) -> summed coefficient
        self.shift: Any = 0.0  # the constant terms, summed

    def slot(self, key: Any, op: SymmetricTensor, piece: SymmetricTensor | None = None) -> int:
        """Intern one unit of on-site work; ``piece`` is the split tensor of a k-site operator."""
        got = self.slots.get(key)
        if got is None:
            got = self.slots[key] = len(self.ops)
            self.ops.append((op, piece))
        return got

    def transition(self, slot: int, charge: int, site: int) -> tuple:
        """The edge one slot writes out of one running charge -- built once, then cached."""
        op, piece = self.ops[slot]
        sym, phys, dual = self.sym, self.phys, self.dual
        c = self.sectors[charge]
        if piece is not None:
            # A k-site operator's pieces run at unit charge and carry the split's own
            # derived bond, so one that begins on a charged string is the same interleaving
            # a term list is refused for, with the same message.
            if c != sym.unit:
                raise ValueError(_INTERLEAVE.format(site))
            out = (piece, 0, piece.legs[3].space)
        else:
            d = phys.dim
            q = op.legs[2].space.sectors[0][0]
            nxt = sym.fusion(c, q)[0]
            lab = sym.dual(nxt) if dual else nxt
            space = GradedSpace.new(sym, {lab: 1})
            left = GradedSpace.new(sym, {sym.dual(c) if dual else c: 1})
            legs = (Leg(left, IN, dual), Leg(phys, OUT), Leg(phys, IN), Leg(space, OUT, dual))
            block = np.reshape(np.asarray(op.to_dense())[:, :, 0], (1, d, d, 1))
            # The running charge to the *right* of this site crosses the site's incoming
            # physical line on its way out -- a braiding the dense ``[:, :, 0]`` round trip
            # cannot see. The R-coefficient is paid here, per physical sector: ``+1``
            # everywhere for a bosonic grading, the parity sign for a super one, which is
            # exactly the Jordan-Wigner string's foothold on the operator's own site (#147
            # gate 4 -- at ``d=2`` it is invisible because ``a+ Z = a+`` and ``Z a = a``).
            if _braids_with_signs(space):
                block = block.copy()
                for a, _, o, e in _slabs(phys):
                    tree = FusionTree((lab, a), (), (0,), sym.fusion(lab, a)[0])
                    # permute_tree is the opt-in PermutationCoefficients capability;
                    # every provider that reaches this braiding question implements it
                    braid = sym.permute_tree(tree, (1, 0))  # ty: ignore[unresolved-attribute]
                    block[:, :, o : o + e, :] *= braid[0][1].real
            if nxt not in self.charges:
                self.charges[nxt] = len(self.sectors)
                self.sectors.append(nxt)
            out = (SymmetricTensor.from_dense(block, legs), self.charges[nxt], space)
        self.trans[(slot, charge)] = out
        return out

    def close(self) -> None:
        """Turn the accumulated closing coefficients into edges: one multiply per edge.

        The coefficient is the only thing that differs between two terms closing the same
        string, so summing the coefficients and multiplying once is the same edge as
        multiplying per term and adding the tensors -- at one ``multiply`` and no
        ``SymmetricTensor.__add__`` per surviving edge, where the term-at-a-time form paid
        both per term.
        """
        stops, last = self.stops, self.n_sites - 1
        if self.shift:  # a constant shift: the identity at the last site, closed from IdL
            idw = _identity_w(Leg(self.states[_IDL], IN, self.dual), self.phys)
            w = tenet.multiply(idw, self.shift)
            stops[last][_IDL] = stops[last][_IDL] + w if _IDL in stops[last] else w
        for (site, state, slot, charge), coeff in self.closing.items():
            w = tenet.multiply(self.trans[(slot, charge)][0], coeff)
            stops[site][state] = stops[site][state] + w if state in stops[site] else w
        self.closing.clear()


def _term_edges(walk: _Walk, slots: Sequence, sites: Sequence, coeffs: Sequence) -> None:
    """Walk canonicalized terms left to right: intern their states, emit their edges.

    ``slots[t]`` and ``sites[t]`` are one term's operator slots and their site indices,
    the same length and both in site order; ``coeffs[t]`` is its coefficient, with the
    Koszul sign of the sort into site order already paid by whichever front end
    canonicalized it. An empty row is a constant shift.

    ``walk.moves[n]`` collects the non-closing edges ``(state_l, state_r) -> W`` of site
    ``n`` (set-once: terms sharing a prefix share the edge, so it is written exactly where
    the prefix is *new*), ``walk.closing`` the closing coefficients, and
    ``walk.spectators[n]`` the states whose space runs through site ``n`` untouched.

    Nothing in the loop builds anything: the prefix is interned rather than rebuilt, the
    edge comes off ``walk.trans`` keyed by ``(slot, running charge)`` rather than being
    constructed, the closing edge accumulates a coefficient rather than a tensor, and a
    state's spectator span is marked at most once, by a high-water mark, rather than
    re-walked by every term that passes through it.
    """
    trans, prefixes, states, order = walk.trans, walk.prefixes, walk.states, walk.order
    moves, spectators, high, closing = walk.moves, walk.spectators, walk.high, walk.closing
    transition = walk.transition
    for row, where, coeff in zip(slots, sites, coeffs, strict=True):
        if not row:
            walk.shift = walk.shift + coeff
            continue
        state, charge, last = _IDL, 0, len(row) - 1
        for i, slot in enumerate(row):
            site = where[i]
            if state != _IDL:
                mark = high[state]
                if mark < site - 1:
                    for m in range(mark + 1, site):
                        spectators[m].add(state)
                    high[state] = site - 1
            key = (slot, charge)
            edge = trans.get(key)
            if edge is None:
                edge = transition(slot, charge, site)
            w, charge, space = edge
            if i == last:
                if charge:
                    raise ValueError(
                        f"a term's operator charges must sum to the unit sector, got {space}; "
                        "both MPO boundaries are the trivial D=1 leg, so a charged term has "
                        "nowhere to end"
                    )
                shut = (site, state, slot, key[1])
                closing[shut] = closing[shut] + coeff if shut in closing else coeff
            else:
                pkey = (state, slot, site)
                nxt = prefixes.get(pkey)
                if nxt is None:
                    nxt = prefixes[pkey] = len(order) + 2
                    states[nxt] = space
                    high[nxt] = site
                    order.append(nxt)
                    moves[site][(state, nxt)] = w
                state = nxt


def _canonical_term(walk: _Walk, n_sites: int, coeff: Any, ops: list, split: Any) -> tuple:
    """One ``(coeff, [(op, sites), ...])`` term as the ``(slots, sites, coefficient)`` triple.

    ``from_terms``' half of the canonicalization ``_canonical_blocks`` does over whole
    arrays for ``from_arrays``: every validation a term list can fail, the sort into site
    order, and the Koszul sign that sort costs. What comes back is the integer rows
    ``_term_edges`` eats from either front end.
    """
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
            placed[site] = (op, w, sites, j, len(span))
    where = sorted(placed)
    row, open_k = [], None
    for site in where:
        op, w, sites, j, k = placed[site]
        is3 = op.ndim == 3
        if open_k is not None and (is3 or (id(op), j) != open_k):
            raise ValueError(_INTERLEAVE.format(site))
        if not is3 and open_k is None and j:
            raise ValueError(_INTERLEAVE.format(site))
        # The slot key is what state identity is made of: a rank-3 operator's states are
        # told apart by the site the prefix key already carries, while a k-site operator's
        # pieces belong to the tuple of sites they were placed on.
        row.append(walk.slot(id(op) if is3 else (id(op), sites, j), op, None if is3 else w))
        open_k = None if is3 or j + 1 == k else (id(op), j + 1)
    return row, where, coeff


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


class EdgeTable:
    """The pruned FSM, its bond spaces and its dense slot maps — and where they become tensors.

    Parameters
    ----------
    edges : list of dict
        Per site, the surviving ``(state_l, state_r) -> W | None`` map, ``None``
        meaning a spectator identity.
    ordered : list of list
        Per cut, the live state keys in allocation order, ``_IDL`` first and
        ``_IDR`` last.
    bonds : list
        Per cut, that cut's ``GradedSpace``.
    starts : list
        Per cut, ``starts[i][k][a]``: the dense offset state ``k``'s sector ``a``
        occupies in ``bonds[i]``.
    groups : list
        Per cut, the same as ``(space, slots, keys)`` per ``IdL`` / open / ``IdR``
        subset, or ``None`` where the subset is empty.
    states : dict
        State label to its ``GradedSpace``.
    phys : GradedSpace
        The physical space.
    dual : bool
        The MPO's one dual convention, set by whether any k-site split runs in it.

    Notes
    -----
    **This is the MPO's symbolic representation, and it is the instantiation boundary**
    (#200, staging #184 Part 2 (a)). Nothing above it is numeric: ``_edge_table`` builds
    the whole thing without a tensor. Everything below it is, and there are exactly two
    ways down — [site][tenet.network.EdgeTable.site], which materialises one
    dense-blocked rank-4 ``W`` on the full bond, and
    [edge_blocks][tenet.network.EdgeTable.edge_blocks], which materialises one site's
    [EdgeBlocks][tenet.network.EdgeBlocks] on the group-restricted bonds the two-site
    matvec eats. They are peers: ``_instantiate`` is the first one's consumer and
    ``env._cores2`` the second's, and neither is built out of the other.

    The group *embedding* — the 0/1 isometry between a group of states and its slots of
    the full bond — belongs to the second way down and is built there, which is the
    restructuring #184 named as the real cost of deferred instantiation: ``Env``'s cores
    used to be assembled from tensors ``_instantiate`` had already embedded.

    Both ways down are cached per site and neither runs at construction, so an MPO that
    only ever reaches the matvec never allocates a full-width site tensor. That is the
    memory claim, and ``tests/network/test_deferred.py`` is what fails if it stops
    holding.

    The per-site cache is **bounded by bytes** (M38, #202): it keeps its most recently
    used sites up to ``common.CACHE_BUDGET`` and evicts past it, because a sweep visits
    every site and a cache holding all of them is the whole operator again -- the object
    this boundary exists not to build. An operator small enough to sit under the budget
    keeps every site and never evicts. A rebuilt block table is bit-identical to the
    evicted one; ``edge_blocks`` is a pure function of the description.

    **Two producers, one interface** (#204). ``_edge_table`` builds the finite-state
    machine; ``_compressed_table`` builds the *compressed* description a float-``cutoff``
    builder returns -- the compressed sites plus, per cut, the three group slabs the
    pinned sweeps preserved, with one open state per cut where the FSM has one per open
    string and with the per-edge dicts empty. Both answer the same two ways down, because
    what the prepared matvec consumes of a bond is the ``IdL (+) open (+) IdR``
    decomposition and the four blocks placed against it, never the edges: no consumer of
    [EdgeBlocks][tenet.network.EdgeBlocks] reads ``a``/``b``/``c``/``d``. The one
    difference is inside [edge_blocks][tenet.network.EdgeTable.edge_blocks], which slices
    the compressed ``W'`` where it scatters the FSM's edges.

    The next stage (#184 candidate (d)) plugs in *here*: a per-cut assembler with an
    expression algebra replaces ``_edge_table`` as the producer and
    [edge_blocks][tenet.network.EdgeTable.edge_blocks] as the per-site core builder, with the
    a/b/c/d partition as the interface it has to keep hitting.
    """

    edges: list[dict]
    ordered: list[list]
    bonds: list
    starts: list
    groups: list
    states: dict
    phys: GradedSpace
    dual: bool

    def __init__(
        self,
        edges: list[dict],
        ordered: list[list],
        bonds: list,
        starts: list,
        groups: list,
        states: dict,
        phys: GradedSpace,
        dual: bool,
    ) -> None:
        self.edges = edges
        self.ordered = ordered
        self.bonds = bonds
        self.starts = starts
        self.groups = groups
        self.states = states
        self.phys = phys
        self.dual = dual
        # The numeric caches, and the only mutable state here. They are what makes the
        # two ways down peers rather than a pipeline: each is filled on request, per
        # site, so neither consumer pays for the other's tensors.
        #
        # The two per-site caches are bounded (#202): held to the whole chain they *are*
        # the operator, which is the object deferring instantiation exists to avoid, and
        # a sweep never asks for more than the last few sites at once. ``_identities`` is
        # keyed by state space rather than by site -- one small rank-4 slab per distinct
        # space, shared by every site that carries the state -- so it is not a per-bond
        # cache and is not bounded.
        self._identities: dict = {}
        self._table: dict = Recent()
        self._embeds: dict = Recent()
        # The compressed site tensors, or ``None`` for the finite-state machine itself
        # (#204). Set by ``_compressed_table``: a description whose cuts are already the
        # compressed ``IdL (+) open (+) IdR`` and whose sites are already numeric, so
        # ``site`` hands one back and ``edge_blocks`` slices one instead of scattering.
        self.compressed: list[SymmetricTensor] | None = None

    def __len__(self) -> int:
        return len(self.edges)

    def boundary_legs(self) -> tuple[Leg, Leg]:
        """The MPO's two ``D=1`` boundary legs, without materialising a site.

        Returns
        -------
        tuple of Leg
            The first site's left leg and the last site's right leg, both non-dual
            -- ``site``'s caps put them there and ``from_w``'s ends are non-dual too.
        """
        return Leg(self.bonds[0], IN), Leg(self.bonds[-1], OUT)

    def site(self, n: int, carry: SymmetricTensor | None = None) -> SymmetricTensor:
        """Site ``n``, scattered into one buffer against the bond, boundary caps included.

        Parameters
        ----------
        n : int
            The site.
        carry : SymmetricTensor or None, optional
            The compressing sweep's rank-2 map onto the already-truncated right
            bond, both legs ``IN``. ``None`` (the default) places against the
            full FSM bond.

        Returns
        -------
        SymmetricTensor
            The rank-4 ``(wl IN, p OUT, p IN, wr OUT)`` site tensor.

        Notes
        -----
        The caps ride here rather than after a whole-MPO loop so that a site is finished
        the moment it is placed, which is what lets the compressing sweep consume them one
        at a time. ``_place`` folds ``carry`` into the scatter, so the site is born
        contracted with it and the buffer never widens past ``chi``. The site's right leg
        is the carry's second leg moved to the codomain, which is the same relabelling
        ``einsum("xy,apqx->apqy", carry, w)`` used to make: same space, ``OUT``, ``dual``
        flipped.

        A **compressed** description (#204) is already numeric: its site is handed back
        as it is and ``carry`` has nowhere to go, because the two sweeps that would fold
        one are what produced the description in the first place.

        This is the full-width way down, and the one the prepared matvec does not take.
        """
        if self.compressed is not None:
            return self.compressed[n]
        if carry is None:
            space_r, dual_r, dense_c = self.bonds[n + 1], self.dual, None
        else:
            space_r, dual_r = carry.legs[1].space, not carry.legs[1].dual
            dense_c = np.asarray(carry.to_dense())
        items = (
            (
                self._dense_w(w, lk),
                self.states[lk],
                self.starts[n][lk],
                self.states[rk],
                self.starts[n + 1][rk],
            )
            for (lk, rk), w in self.edges[n].items()
        )
        w = _place(items, self.bonds[n], space_r, self.phys, self.dual, dual_r, dense_c)
        if not self.dual:
            return w
        # The two *outer* legs go back non-dual: ``Env`` builds its boundary environments
        # from the boundary legs' own flags (``Env.__init__``) and the other builder's ends
        # are non-dual too. Both ends are D=1 on the unit sector, where the flag is a label.
        sym = self.phys.provider
        triv = GradedSpace.new(sym, {sym.unit: 1})
        if n == 0:
            cap = SymmetricTensor.from_dense(np.ones((1, 1)), (Leg(triv, IN), Leg(triv, OUT, True)))
            w = _as_w(tenet.einsum("xpqr,ax->apqr", w, cap))
        if n == len(self.edges) - 1:
            cap = SymmetricTensor.from_dense(np.ones((1, 1)), (Leg(triv, IN, True), Leg(triv, OUT)))
            w = _as_w(tenet.einsum("xb,apqx->apqb", cap, w))
        return w

    def edge_blocks(self, n: int) -> EdgeBlocks:
        """Site ``n``'s [EdgeBlocks][tenet.network.EdgeBlocks] -- the per-site core builder.

        Parameters
        ----------
        n : int
            The site.

        Returns
        -------
        EdgeBlocks
            The a/b/c/d partition of the site's surviving edges, its four
            group-restricted operators, its three derived rank-2 views and the
            six group embeddings of its two cuts.

        Notes
        -----
        Each block's operator is placed by the *same* ``_place`` as a full site, against
        the group slot maps instead of the full-bond ones -- so no full-width rank-4
        tensor exists on this route. The bond side of the two boundary cuts is non-dual,
        matching ``site``'s caps and ``Env``'s boundary legs.

        The six group embeddings are built here, per cut and shared between the two sites
        that meet at it. Until #200 they were built by ``_instantiate`` for the whole MPO
        at once, which is what tied ``Env``'s cores to a materialised operator.

        For a **compressed** description (#204) there is nothing to scatter: the four
        blocks are the four sub-slabs of the compressed ``W'``, cut out by the same group
        embeddings, and ``spec_op``/``a_real_op`` say that in a rotated open basis a
        spectator's identity ride no longer separates -- everything open is
        operator-carrying. A spin chain that wants the spectator shortcut keeps
        ``cutoff=None``, where nothing changed.
        """
        got = self._table.get(n)
        if got is None:
            build = self._slice_table if self.compressed is not None else self._build_table
            got = self._table[n] = build(n)
        return got

    # --- the machinery under the two ways down ---------------------------------------

    def _dense_w(self, w, key):
        """One edge's full rank-4 dense block; ``None`` is the state's identity.

        Identity edges still go through ``_identity_w`` and its per-space cache: on a
        sign-braiding aux space that ``einsum`` is *not* a bare delta in the carrier
        basis, so the slab is read off the tensor rather than hand-derived.
        """
        if w is not None:
            return np.asarray(w.to_dense())
        space = self.states[key]
        if space not in self._identities:
            self._identities[space] = np.asarray(
                _identity_w(Leg(space, IN, self.dual), self.phys).to_dense()
            )
        return self._identities[space]

    def _group_place(self, n, pairs, gl, gr):
        """One a/b/c/d block's operator, placed against its two groups' slot maps."""
        left, right = self.groups[n][gl], self.groups[n + 1][gr]
        if left is None or right is None:
            return None
        (space_l, slots_l, _), (space_r, slots_r, _) = left, right
        items = (
            (self._dense_w(w, lk), self.states[lk], slots_l[lk], self.states[rk], slots_r[rk])
            for (lk, rk), w in pairs
        )
        return _place(items, space_l, space_r, self.phys, self.dual, self.dual)

    def _channel_map(self, keys, spaces, slots_l, slots_r, dual_l, dual_r):
        """The rank-2 0/1 map carrying each key's space identically across a site."""
        dense = np.zeros((spaces[0].dim, spaces[1].dim))
        for k in keys:
            for a, _, _o, ext in _slabs(self.states[k]):
                r0, c0 = slots_l[k][a], slots_r[k][a]
                dense[r0 : r0 + ext, c0 : c0 + ext] = np.eye(ext)
        legs = (Leg(spaces[0], IN, dual_l), Leg(spaces[1], OUT, dual_r))
        return SymmetricTensor.from_dense(dense, legs)

    def _embed(self, i):
        """Cut ``i``'s six group embeddings, ``{group: (left slicer, right slicer)}``."""
        got = self._embeds.get(i)
        if got is not None:
            return got
        dual_b = self.dual and 0 < i < len(self.edges)
        emb = {}
        for name, g in self.groups[i].items():
            if g is None:
                emb[name] = (None, None)
                continue
            space, slots, keys = g
            emb[name] = tuple(
                _group_embedding(
                    self.bonds[i],
                    self.starts[i],
                    space,
                    slots,
                    keys,
                    self.states,
                    left=left,
                    dual=self.dual,
                    dual_b=dual_b,
                )
                for left in (True, False)
            )
        self._embeds[i] = emb
        return emb

    def _slice_table(self, n):
        """``edge_blocks``' body for a compressed description: slice ``W'`` by slab.

        The pinned sweeps left the two corner channels exact, so ``idmap`` is those two
        channels and nothing else -- no free-riding spectator survives a rotation of the
        open basis, which is why ``spec_op`` is ``None`` and ``a_real_op`` is ``a_op``.
        """
        # ``_slice_table`` is reached only through ``edge_blocks``' compressed branch
        w = self.compressed[n]  # ty: ignore[not-subscriptable]
        left, right = self._embed(n), self._embed(n + 1)

        def block(gl, gr):
            # ``left[g][1]`` reads ``(group IN, bond OUT)`` and ``right[g][0]``
            # ``(bond IN, group OUT)``: the two orientations that meet ``W'``'s own legs.
            el, er = left[gl][1], right[gr][0]
            if el is None or er is None:
                return None
            return tenet.einsum("xw,vpqx->vpqw", er, tenet.einsum("xpqb,vx->vpqb", w, el))

        corners = [k for k in (_IDL, _IDR) if k in self.ordered[n] and k in self.ordered[n + 1]]
        ops = {name: block(*where) for name, where in _BLOCK_GROUPS.items()}
        n_sites = len(self.edges)
        idmap = (
            self._channel_map(
                corners,
                (self.bonds[n], self.bonds[n + 1]),
                self.starts[n],
                self.starts[n + 1],
                self.dual and n > 0,
                self.dual and n + 1 < n_sites,
            )
            if corners
            else None
        )
        return EdgeBlocks(
            {},
            {},
            {},
            {},
            ops["a"],
            ops["b"],
            ops["c"],
            ops["d"],
            idmap,
            None,
            ops["a"],
            left["idl"][0],
            left["open"][0],
            left["idr"][0],
            right["idl"][1],
            right["open"][1],
            right["idr"][1],
        )

    def _build_table(self, n):
        """``edge_blocks``' body: partition site ``n``'s edges, place them, embed its two cuts."""
        n_sites = len(self.edges)
        dicts: dict[str, dict] = {"a": {}, "b": {}, "c": {}, "d": {}}
        id_channels = []  # states carried identically through this site, corners included
        for corner in (_IDL, _IDR):
            if corner in self.ordered[n] and corner in self.ordered[n + 1]:
                id_channels.append(corner)
        spec_keys = []
        for (lk, rk), w in self.edges[n].items():
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
            if w is None and not _braids_with_signs(self.states[lk]):
                id_channels.append(lk)
                spec_keys.append(lk)
        ops = {
            name: self._group_place(n, dicts[name].items(), *where)
            for name, where in _BLOCK_GROUPS.items()
        }
        real = [(kk, w) for kk, w in dicts["a"].items() if not (w is None and kk[0] in spec_keys)]
        idmap = (
            self._channel_map(
                id_channels,
                (self.bonds[n], self.bonds[n + 1]),
                self.starts[n],
                self.starts[n + 1],
                self.dual and n > 0,
                self.dual and n + 1 < n_sites,
            )
            if id_channels
            else None
        )
        spec = None
        if spec_keys:
            gl_space, gl_slots, _ = self.groups[n]["open"]
            gr_space, gr_slots, _ = self.groups[n + 1]["open"]
            spec = self._channel_map(
                spec_keys, (gl_space, gr_space), gl_slots, gr_slots, self.dual, self.dual
            )
        left, right = self._embed(n), self._embed(n + 1)
        return EdgeBlocks(
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
            self._group_place(n, real, "open", "open"),
            left["idl"][0],
            left["open"][0],
            left["idr"][0],
            right["idl"][1],
            right["open"][1],
            right["idr"][1],
        )


# Which group subspace each block of MPSKit's ``(1 C D; . A B; . . 1)`` partition places
# its two ends against — the only thing that separates the block table's scatter from the
# full site's.
_BLOCK_GROUPS = {
    "a": ("open", "open"),
    "b": ("open", "idr"),
    "c": ("idl", "open"),
    "d": ("idl", "idr"),
}


def _edge_table(n_sites, phys, dual, states, order, moves, stops, spectators) -> EdgeTable:
    """Prune dead states and derive every cut's bond space and slot map.

    Pruning intersects each cut's states with (reachable from ``_IDL``) and (co-reachable
    to ``_IDR``) — two passes over the edge tables, tenpy's ``add_missing_IdL_IdR`` and
    block2's zero-propagation doing the same job in their own spellings. The bond space at
    a cut is then the direct sum of the surviving states' spaces, in allocation order with
    the identities at the two ends.

    Symbolic throughout: not one tensor is built here, and since #200 not one is built
    afterwards either unless a consumer asks for it. This is the producer of the
    description; [EdgeTable][tenet.network.EdgeTable] names its two consumers.
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
    return EdgeTable(edges, ordered, bonds, starts, groups, states, phys, dual)


# --- the corner-pinned compressing sweeps -------------------------------------------
#
# Both truncating sweeps rotate the **open** block of a cut only; the ``_IDL`` and
# ``_IDR`` channels pass through unrotated, so a compressed bond still decomposes as
# ``IdL (+) open (+) IdR`` and still carries the partition ``Env.heff2``'s prepared path
# eats (#204). That is a restriction of the gauge freedom, not a new algorithm:
# ``block-diag(1, U, 1)`` is a legal MPO gauge transformation and the truncation drops
# open directions under the same ``cutoff``. block2 pins the same way on its own SVD
# route (``general_mpo.hpp``:764-805 removes the delayed identity row from the matrix
# before the SVD and gives it a unit singular value); the ``IdR`` half is tenet's.
#
# The layout fact both halves ride on: a cut's bond is ``_merge``'s direct sum over
# ``ordered[i] = [_IDL, *open, _IDR]`` and both corner states carry the trivial ``D=1``
# unit space (``_Walk.__init__``), so the two corner channels are the **first** and
# **last** degeneracy slot of the bond's unit sector and nothing else. Pinning is then:
# take those two rows out, rotate what is left, put the three slabs back in that order --
# and ``tenet.direct_sum`` puts them back in exactly ``_merge``'s order, which is what
# lets the compressed description reuse ``_merge`` to describe its own cuts.


def _corner_slots(bond: GradedSpace, sym: Any) -> dict[str, int]:
    """The dense rows the two corner channels occupy; asked only where one is live.

    Both corner states carry the unit space, so a cut with no live corner need not have
    a unit sector at all -- an all-odd bond is what a chain of one fermionic operator
    leaves -- and this is not asked there.
    """
    off = bond.sector_offset(sym.unit)
    return {"idl": off, "idr": off + dict(bond.sectors)[sym.unit] - 1}


def _corner_map(bond: GradedSpace, row: int, legs: tuple, *, column: bool) -> SymmetricTensor:
    """The one-row (or one-column) selector of a corner channel, on the given legs.

    A dense ``(D, 1)`` indicator through ``from_dense`` with the legs declared, the way
    ``_channel_map`` and ``_group_embedding`` build theirs -- the slab is *read off* a
    tensor rather than hand-derived, which is what keeps the dual flags and any braiding
    sign honest. Cheap in the one place it matters: this never allocates the
    ``D_FSM x D_FSM`` object a whole-group embedding of the uncompressed bond would.
    """
    dense = np.zeros((bond.dim, 1))
    dense[row, 0] = 1.0
    return SymmetricTensor.from_dense(dense if column else dense.T, legs)


def _aligned(t: SymmetricTensor, dual: bool, axis: int) -> SymmetricTensor:
    """``t`` with ``axis``'s ``dual`` flag matching ``dual``; the corner spaces are the unit."""
    return t if t.legs[axis].dual == dual else tenet.flip_dual(t, axis)


def _joined(parts: list[SymmetricTensor], axes: Any) -> SymmetricTensor:
    """``parts[0] (+) parts[1] (+) ...`` along ``axes``, in ``_merge``'s allocation order."""
    out = parts[0]
    for p in parts[1:]:
        out = tenet.direct_sum(out, p, axes=axes)
    return out


def _instantiate(tab: EdgeTable, cutoff: float) -> tuple[list[SymmetricTensor], list]:
    """Place every site numerically, truncating as it goes — one consumer of the table.

    The streaming materialiser, and since #200 the *only* thing ``_instantiate`` is: the
    pruning and the bond spaces are ``_edge_table``'s, the per-site block table is
    [EdgeTable.edge_blocks][tenet.network.EdgeTable.edge_blocks]'s, and this function asks
    ``EdgeTable.site`` for one site at a time. The ``cutoff=None`` branch that used to
    live here is gone, because an MPO that keeps its finite-state machine now keeps the
    description rather than a materialised copy of it.

    Materialisation runs **inside** the backward compressing sweep, which is the only
    consumer of a site here. Site ``N-1`` is placed and its left bond truncated; site
    ``n`` is placed **against the carry**, which ``_place`` folds into the scatter so that
    the site is born on the already-truncated right leg, and is truncated in turn. The
    forward sweep in ``from_terms`` then runs on tensors that are already at the operator
    Schmidt rank. Right to left first, exactly as ``MPS.compress_`` canonizes before it
    truncates: a forward sweep alone measures the rank of the *left* part against the raw
    FSM bond, which overshoots wherever the redundancy is only visible from the right. The
    widest object that ever exists is ``D_FSM x d**2 x chi_Schmidt``, so the full-width
    MPO never exists as a whole.

    **The SVD acts on the open row slab only** (#204): the two corner channels are taken
    out first, the rest is rotated and truncated, and the carry handed to site ``n-1`` is
    the block-diagonal ``1_IdL (+) (u.s) (+) 1_IdR`` -- which ``_place`` folds exactly as
    it folds the free carry, its right end being one rank-2 map either way. The corner
    rows are removed by subtracting their own one-row slabs rather than by restricting to
    the open group, because a whole-group embedding of the *uncompressed* bond is a
    ``D_FSM x D_FSM`` object (7.9 GiB at K=26, #202) and a one-row selector is not.

    Returns the sites and, per cut, ``(IdL live, the open block's space or None, IdR
    live)`` -- the description of the partition the sweep just preserved, which
    ``_compress_forward`` carries on and ``_compressed_table`` turns into a block table.
    """
    sym = tab.phys.provider
    unit = GradedSpace.new(sym, {sym.unit: 1})
    n_sites = len(tab.edges)
    cuts: list = [None] * (n_sites + 1)
    for i in (0, n_sites):
        cuts[i] = (tab.groups[i]["idl"] is not None, None, tab.groups[i]["idr"] is not None)
    out, carry = [], None
    for n in reversed(range(n_sites)):
        w = tab.site(n, carry)
        if n:
            bond, dual_b = tab.bonds[n], tab.dual
            live = [g for g in ("idl", "idr") if tab.groups[n][g] is not None]
            slots = _corner_slots(bond, sym) if live else {}
            corners, rest = {}, w
            for g in live:
                take = _corner_map(
                    bond, slots[g], (Leg(unit, IN, dual_b), Leg(bond, OUT, dual_b)), column=False
                )
                put = _corner_map(
                    bond, slots[g], (Leg(bond, IN, dual_b), Leg(unit, OUT, dual_b)), column=True
                )
                corners[g] = tenet.einsum("xpqb,vx->vpqb", w, take)
                rest = tenet.subtract(rest, tenet.einsum("vpqb,xv->xpqb", corners[g], put))
            u, s, vh = tenet.linalg.svd_truncated(rest, ((0,), (1, 2, 3)), cutoff=cutoff)
            # ``u`` came back on the map partition, its ``wl`` leg bent; spell the bend
            # with ``repartition`` so the join below is a composition, not an implicit cap.
            open_w, open_c = _as_w(vh), tenet.einsum("xy,yz->xz", u, s)
            ref = open_c.legs[1]
            rows, cols = [], []
            for g in ("idl", "open", "idr"):
                if g == "open":
                    rows.append(open_w)
                    cols.append(open_c)
                elif g in live:
                    rows.append(_aligned(corners[g], open_w.legs[0].dual, 0))
                    cols.append(
                        _corner_map(
                            bond,
                            slots[g],
                            (open_c.legs[0], Leg(unit, ref.side, ref.dual)),
                            column=True,
                        )
                    )
            w = _joined(rows, 0)
            carry = tenet.repartition(_joined(cols, 1), (), (0, 1))
            cuts[n] = ("idl" in live, open_w.legs[0].space, "idr" in live)
        out.append(w)
    out.reverse()
    return out, cuts


def _compress_forward(
    sites: list[SymmetricTensor], cuts: list, cutoff: float
) -> list[SymmetricTensor]:
    """The left-to-right half of the compressing sweep, in place over ``sites``.

    ``_instantiate`` ran the right-to-left half while it placed the sites; this is the
    return leg, and it is the same shape for every builder that takes a float ``cutoff``,
    so it lives here rather than once per builder. It is ``_instantiate``'s pinning
    mirrored onto the open **column** slab, and it narrows ``cuts`` in place as it goes.
    """
    sym = sites[0].legs[1].space.provider
    unit = GradedSpace.new(sym, {sym.unit: 1})
    for n in range(len(sites) - 1):
        has_l, _open, has_r = cuts[n + 1]
        bond, dual_b = sites[n].legs[3].space, sites[n].legs[3].dual
        live = [g for g, on in (("idl", has_l), ("idr", has_r)) if on]
        slots = _corner_slots(bond, sym) if live else {}
        corners, rest = {}, sites[n]
        for g in live:
            take = _corner_map(
                bond, slots[g], (Leg(bond, IN, dual_b), Leg(unit, OUT, dual_b)), column=True
            )
            put = _corner_map(
                bond, slots[g], (Leg(unit, IN, dual_b), Leg(bond, OUT, dual_b)), column=False
            )
            corners[g] = tenet.einsum("xv,apqx->apqv", take, sites[n])
            rest = tenet.subtract(rest, tenet.einsum("vx,apqv->apqx", put, corners[g]))
        u, s, vh = tenet.linalg.svd_truncated(rest, ((0, 1, 2), (3,)), cutoff=cutoff)
        open_w, open_c = _as_w(u), tenet.einsum("xy,yz->xz", s, vh)
        ref = open_c.legs[0]
        rows, cols = [], []
        for g in ("idl", "open", "idr"):
            if g == "open":
                rows.append(open_w)
                cols.append(open_c)
            elif g in live:
                rows.append(_aligned(corners[g], open_w.legs[3].dual, 3))
                cols.append(
                    _corner_map(
                        bond,
                        slots[g],
                        (Leg(unit, ref.side, ref.dual), open_c.legs[1]),
                        column=False,
                    )
                )
        sites[n] = _joined(rows, 3)
        # ``vh``'s ``wr`` leg came back bent; spell the bend, as in ``_instantiate``'s sweep.
        carry = tenet.repartition(_joined(cols, 0), (0, 1), ())
        sites[n + 1] = _as_w(tenet.einsum("ypqr,xy->xpqr", sites[n + 1], carry))
        cuts[n + 1] = (has_l, open_w.legs[3].space, has_r)
    return sites


def _compressed_table(sites: list[SymmetricTensor], cuts: list, phys: GradedSpace) -> EdgeTable:
    """The compressed operator as a description: the sites plus each cut's three slabs.

    The carrier #204 asks for. A float-``cutoff`` builder no longer hands ``MPO`` bare
    site tensors; it hands this, so ``MPO.edges`` stays the one dispatch source for
    [Env.heff2][tenet.network.Env.heff2] whether the MPO was compressed or not. What the
    prepared machinery consumes of a bond is the direct-sum decomposition ``IdL (+) open
    (+) IdR`` and the four blocks placed against it -- not the FSM's edges, which no
    consumer of [EdgeBlocks][tenet.network.EdgeBlocks] reads -- and after the pinned
    sweeps a compressed bond still has that decomposition, with one *open state per cut*
    where the FSM had one per open string.

    So the per-edge dicts are empty and every field the matvec does read is sliced out of
    the compressed ``W'`` by [EdgeTable.edge_blocks][tenet.network.EdgeTable.edge_blocks].
    Simplification: the open state's label is the cut index, which is enough because
    nothing outside this module reads a state label's value.
    """
    sym = phys.provider
    unit = GradedSpace.new(sym, {sym.unit: 1})
    # The compressed MPO carries one dual convention on its interior bonds (the two ends
    # are ``D=1`` and non-dual, as ``site``'s caps and ``from_w``'s ends are), so
    # ``_embed``'s ``dual and 0 < i < N`` rule describes it unchanged.
    dual = sites[-1].legs[0].dual
    states: dict = {_IDL: unit, _IDR: unit}
    ordered, bonds, starts, groups = [], [], [], []
    for i, (has_l, open_space, has_r) in enumerate(cuts):
        key = _IDR + 1 + i  # one open state per cut, labelled by the cut
        if open_space is not None:
            states[key] = open_space
        live = (("idl", _IDL, has_l), ("open", key, open_space is not None), ("idr", _IDR, has_r))
        cut = [k for _n, k, on in live if on]
        bond, per_state = _merge(sym, cut, states)
        ordered.append(cut)
        bonds.append(bond)
        starts.append(per_state)
        groups.append({n: (*_merge(sym, [k], states), [k]) if on else None for n, k, on in live})
    tab = EdgeTable(
        [{} for _ in range(len(sites))], ordered, bonds, starts, groups, states, phys, dual
    )
    tab.compressed = sites
    return tab


class MPO:
    """A finite MPO: one rank-4 ``SymmetricTensor`` per site, ``(wl IN, p OUT, p IN, wr OUT)``.

    Parameters
    ----------
    sites : Iterable of SymmetricTensor or None, optional
        The rank-4 site tensors, left to right. Exactly one of ``sites`` and
        ``edges`` is given.
    edges : EdgeTable or None, optional
        The edge description [from_terms][tenet.network.MPO.from_terms] and
        [from_arrays][tenet.network.MPO.from_arrays] keep -- the finite-state
        machine at ``cutoff=None``, the compressed sites and their per-cut
        slabs at a float cutoff -- from which sites and block tables come on
        request; ``None`` for every other MPO. Keyword-only.

    Raises
    ------
    ValueError
        If neither or both of ``sites`` and ``edges`` are given.

    Notes
    -----
    Invariance reads ``q(p_out) + q(wr) = q(wl) + q(p_in)``. The first and last sites
    carry a ``D=1`` boundary MPO bond, which is what makes *every* ``W_n`` rank 4 and
    removes the boundary-vector special case.

    **A separate class from [MPS][tenet.network.MPS], with no shape flag** -- the comparison
    with YASTN's and TenPy's choices is in ``docs/design.md`` "Milestone 11".

    **Two internal representations, and ``edges`` is the description** (#200). Given an
    ``EdgeTable`` the container may hold no tensor at all: ``self[n]``
    materialises site ``n`` on request and caches it, and
    [edge_blocks][tenet.network.MPO.edge_blocks] does the same for the site's block table,
    so an MPO whose only consumer is the prepared two-site matvec never allocates a
    full-width rank-4 ``W``. A *compressed* description (#204) already holds its sites --
    the two truncating sweeps built them -- and answers both accessors off those. Given
    site tensors and no description the container holds exactly those and
    ``edge_blocks`` is ``None`` throughout. #141's standing trade -- two representations
    in one class, because ``from_w``'s numeric path and ``to_dense`` still need the sites
    -- is accepted here one level deeper, with its deletion condition unchanged.

    ``edges`` and
    [edge_blocks][tenet.network.MPO.edge_blocks] are the two read-only accessors beyond
    the container protocol, and they exist so that [Env][tenet.network.Env] can reach the
    symbolic structure without touching a private name -- #138 refused public exposure of
    the symbolic layer "if a caller ever needs to inspect it, that is a separate issue
    with an argument attached", and the prepared two-site matvec is that caller and that
    argument (#141).
    """

    edges: EdgeTable | None

    def __init__(
        self,
        sites: Iterable[SymmetricTensor] | None = None,
        *,
        edges: EdgeTable | None = None,
    ) -> None:
        if (sites is None) == (edges is None):
            raise ValueError("MPO takes either its site tensors or an edge description, not both")
        self.edges = edges
        self._sites: dict[int, SymmetricTensor] = {} if sites is None else dict(enumerate(sites))
        self._n = len(edges) if edges is not None else len(self._sites)

    @property
    def sites(self) -> list[SymmetricTensor]:
        """The site tensors, materialising every one that is not built yet."""
        return [self[n] for n in range(self._n)]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, n: int) -> SymmetricTensor:
        if n < 0:
            n += self._n
        got = self._sites.get(n)
        if got is None:
            if self.edges is None or not 0 <= n < self._n:
                raise IndexError(f"MPO site index {n} is outside range({self._n})")
            got = self._sites[n] = self.edges.site(n)
        return got

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
        Every ``from_terms`` / ``from_arrays`` operator carries a table, at either
        cutoff. The float cutoff used to give one up, because the compressing sweep's SVD
        gauge mixed the FSM states and left zero identity edges on every model (#141);
        since #204 both sweeps pin the two corner channels, so a compressed bond still
        decomposes as ``IdL (+) open (+) IdR`` and still has a table -- one open state per
        cut rather than one per open string. [from_w][tenet.network.MPO.from_w] never had
        a description and returns ``None``, which routes
        [Env.heff2][tenet.network.Env.heff2] onto its compatibility entry, which is what
        that branch is for.

        Since #200 the table is *built* here rather than stored here: the call goes
        through to ``EdgeTable.edge_blocks``, which places one
        site's blocks against the group slot maps and caches them. No full-width site
        tensor is involved, which is what lets a Hamiltonian be assembled and swept
        without one ever existing.
        """
        return None if self.edges is None else self.edges.edge_blocks(n)

    @classmethod
    def identity(cls, n_sites: int, phys: GradedSpace) -> "MPO":
        """The identity operator as an MPO: ``D=1`` bonds, ``eye`` on every site.

        Parameters
        ----------
        n_sites : int
            Chain length.
        phys : GradedSpace
            The physical space of every site.

        Returns
        -------
        MPO
            An ``n_sites``-site MPO with unit-sector ``D=1`` bonds throughout and no
            edge description, so every consumer takes the full-contraction path.

        Examples
        --------
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPO, MPS, Env
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1)]).canonize_()
        >>> phi = MPS.product(phys, [U1Sector(1), U1Sector(-1)]).canonize_()
        >>> round(Env(psi, MPO.identity(2, phys), bra=phi).measure(), 12)
        1.0

        Notes
        -----
        It exists because an **overlap is an environment**: ``<phi|psi>`` and its
        per-bond projection vectors are what a two-state [Env][tenet.network.Env] over
        this operator produces, which is exactly how block2 builds the ``ext_mes`` its
        excited-state projection reads -- ``impo = self.get_identity_mpo()``
        (``pyblock2/driver/core.py``:4817-4830). One operator spelled once beats a second
        environment class whose contractions would be this one's with a leg deleted.

        Deliberately carries **no** ``EdgeTable``: the prepared matvec's one-sided terms
        are a gauge statement about a state against itself, and this operator's only
        caller is the two-state path where that statement is false. The site tensor is
        [tenet.identity][] on ``(unit, phys)`` transposed into the MPO's
        ``(wl IN, p OUT, p IN, wr OUT)`` axis order -- no ``einsum``, so no composition
        rule to state.
        """
        sym = phys.provider
        unit = GradedSpace.new(sym, {sym.unit: 1})
        eye = tenet.identity((Leg(unit, OUT), Leg(phys, OUT)))  # (u OUT, p OUT, u IN, p IN)
        return cls([tenet.transpose(eye, (2, 1, 3, 0))] * n_sites)

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
    def from_entries(cls, entries: Iterable[Mapping[tuple[int, int], Any]]) -> "MPO":
        """The non-zero ``(i, j)`` entries of each site's ``W``, as a graded MPO with symbols.

        Parameters
        ----------
        entries : Iterable of Mapping
            One mapping per site, left to right, from the ``(row, column)`` index
            pair of that site's ``W`` to what sits there. ``0`` is the ``IdL``
            channel of a bond and ``-1`` its ``IdR`` channel, at every bond
            (see Notes); the open channels are ``1, 2, ...`` and need not be
            contiguous. Chain length is ``len(entries)``, so a uniform bulk is
            ``[w] * n_sites``. An entry is

            * ``None`` -- the identity, which on ``(i, i)`` is a spectator ride;
            * a number ``c`` -- ``c`` times the identity;
            * a rank-3 charge-leg operator from
              [local_op][tenet.network.local_op];
            * the pair ``(c, op)`` -- ``c`` times that operator.

        Returns
        -------
        MPO
            The assembled operator, carrying the per-site
            [edge_blocks][tenet.network.MPO.edge_blocks] table -- so it takes
            [Env.heff2][tenet.network.Env.heff2]'s one prepared engine path.

        Raises
        ------
        ValueError
            If ``entries`` is empty, or every entry is an identity (the physical
            space is read off an operator); if an entry is none of the four
            forms above, or holds ``local_op``'s invariant rank-2*k* operator,
            which spans *k* sites and has nowhere to put them in one site's
            ``W``; if the operators disagree about the physical space; if a key
            is not a pair of ``int``\\ s, or names a bond index below ``-1``, or
            enters ``IdL`` (``(i, 0)`` with ``i != 0``) or leaves ``IdR``
            (``(-1, j)`` with ``j != -1``); if ``(0, 0)`` or ``(-1, -1)`` holds
            anything but the identity; if two entries reach one bond state with
            different charges; if a term closing into ``IdR`` has not brought
            the bond charge back to the unit sector; or if an interior bond
            state is dead -- unreachable from ``IdL`` or unable to reach
            ``IdR``.

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
        >>> w = {  # the 3-site XY chain's bulk W, written as the textbook prints it
        ...     (0, 0): None,
        ...     (0, 1): (0.5, opp),
        ...     (1, -1): opm,
        ...     (0, 2): (0.5, opm),
        ...     (2, -1): opp,
        ...     (-1, -1): None,
        ... }
        >>> h = MPO.from_entries([w] * 3)
        >>> len(h), h.to_dense().shape, h.edges is not None
        (3, (8, 8), True)

        Notes
        -----
        **The hand-build entry that keeps its symbols.** [from_w][tenet.network.MPO.from_w]
        takes a fully-formed rank-4 ``W``, so its caller writes every zero of the
        finite-state machine out, gets four legs and a ``dual`` convention right, and
        hands over a grading by hand -- and what comes back is numeric, with no edge
        description, so it routes onto [Env.heff2][tenet.network.Env.heff2]'s
        compatibility entry. This builder takes the *same* ``W``, named entry by entry,
        and produces an [EdgeTable][tenet.network.EdgeTable]: the very object
        [from_terms][tenet.network.MPO.from_terms] produces, indistinguishable to
        [Env][tenet.network.Env], on the one engine path. **The bond spaces are derived,
        never declared** -- the charge is on the operator's third leg, each state's
        [GradedSpace][tenet.GradedSpace] is the running fused charge, and the bond at a
        cut is the direct sum over its states -- so there is no grading argument and no
        ``dual`` convention to state. ``from_w`` is unchanged and is not deprecated: it is
        the entry for a ``W`` that arrives as a *dense array* (a paper, another library),
        where the entries are numbers and no charge can be recovered from them (#141).

        **``IdL`` is index ``0`` and ``IdR`` is index ``-1``, by convention, at every
        bond.** MPSKit fixes the same two by position -- ``V[1] = V[end] = _rightunit``
        in ``mpohamiltonian.jl``, the ``(1 C D; . A B; . . 1)`` partition
        ``EdgeBlocks`` implements -- and tenet's own bond layout already assumes it:
        ``_merge`` direct-sums a cut in ``[_IDL, *open, _IDR]`` order and the pinned
        sweeps read the two corners off the first and last slot of the unit sector.
        TenPy carries ``IdL``/``IdR`` *explicitly* alongside its ``W`` list because its
        ``MPOGraph`` keys are arbitrary hashables with no order to lean on; here an
        explicit pair would be a second source of truth that ``_merge`` could contradict.
        Python's ``-1`` is what makes the convention cost nothing: **no bond width is
        ever declared or inferred**, because the last index needs no width to name. The
        convention is then made self-enforcing rather than assumed -- an entry into
        ``IdL`` or out of ``IdR``, or a non-identity on either corner, is refused by
        name, which is the same four zeros the corner-exactness property asserts.

        **The two boundary bonds are ``D=1``**, so bond ``0`` keeps only its ``IdL``
        channel and bond ``len(entries)`` only its ``IdR`` one. Everything else at those
        two bonds is dropped silently -- that is exactly ``from_w``'s ``start`` row and
        ``end`` column, and it is what lets one bulk ``W`` be handed over for every site
        including the first and the last. Only *interior* dead states raise.

        **Every operator is rank 3**, for [from_arrays][tenet.network.MPO.from_arrays]'s
        reason: one ``W`` entry sits on one site, and ``local_op``'s invariant *k*-site
        form spans ``k`` sites through an SVD, so it is refused with a pointer to
        ``from_terms``. YASTN is the third reference and its contribution is a negative
        one: between a fully formed tensor (``A[n] = t``) and a term list (``Hterm``,
        ``generate_mpo``) it offers nothing at all, which is the gap this builder fills.

        No compressing sweep runs and there is no ``cutoff``: the caller wrote the bond,
        so there is nothing combinatorial to cut down. ``from_terms``' sweeps exist
        because *its* finite-state machine is built from a term list and can be
        numerically low-rank -- a power law, an integral file -- which is a property of
        the term list, not of a ``W`` somebody sat down and wrote. An operator that wants
        the sweeps wants ``from_terms``, which is where its ``cutoff`` lives. Outside
        ``jit``/``grad`` like the rest of this
        module, because the assembly decides [GradedSpace][tenet.GradedSpace]\\ s.
        """
        rows = [dict(e) for e in entries]
        n_sites = len(rows)
        if not n_sites:
            raise ValueError("from_entries: no sites; the chain length is len(entries)")
        phys = None
        parsed: list[dict] = []
        for n, row in enumerate(rows):
            out = {}
            for key, value in row.items():
                if not (isinstance(key, tuple) and len(key) == 2):
                    raise ValueError(
                        f"from_entries: site {n} has the key {key!r}; a W entry is keyed by "
                        "the pair (row, column) of bond indices"
                    )
                i, j = key
                if not (isinstance(i, int) and isinstance(j, int)):
                    raise ValueError(
                        f"from_entries: entry {key} of site {n} is not a pair of ints; a bond "
                        "index is 0 for IdL, -1 for IdR and 1, 2, ... for the open channels"
                    )
                if i < -1 or j < -1:
                    raise ValueError(
                        f"from_entries: entry {key} of site {n} names a bond index below -1; "
                        "only -1 names the IdR channel"
                    )
                coeff, op = _w_entry(value, key, n)
                if i != 0 and j == 0:
                    raise ValueError(
                        f"from_entries: entry {key} of site {n} enters the IdL channel, which "
                        "is index 0 and means 'only identities to the left'; nothing can "
                        "arrive there"
                    )
                if i == -1 and j != -1:
                    raise ValueError(
                        f"from_entries: entry {key} of site {n} leaves the IdR channel, which "
                        "is index -1 and means 'the term is already finished'; nothing can "
                        "leave it"
                    )
                if i == j and i in (0, -1) and (op is not None or coeff != 1):
                    raise ValueError(
                        f"from_entries: entry {key} of site {n} is not the identity; the IdL "
                        "and IdR channels are identities by definition, so both corners hold "
                        "None (or are simply omitted)"
                    )
                if op is not None:
                    if op.ndim != 3:
                        raise ValueError(
                            f"from_entries: entry {key} of site {n} holds a rank-{op.ndim} "
                            "operator; one W entry sits on one site, so it is local_op's "
                            "rank-3 charge-leg form. An invariant k-site operator spans k "
                            "sites through an SVD and has nowhere to put them here -- hand "
                            "the whole term to MPO.from_terms instead"
                        )
                    phys = _check_op(op, phys)
                out[key] = (coeff, op)
            parsed.append(out)
        if phys is None:
            raise ValueError(
                "from_entries: every entry is an identity; the physical space is read off an "
                "operator and there is none"
            )
        # No k-site split runs here, so the MPO stays on the non-dual convention, exactly
        # as ``from_arrays`` does.
        walk = _Walk(n_sites, phys, False)
        # ``(bond, index) -> state label``. The two corners are the *same* label at every
        # bond -- that is what makes them the corners -- and an open index gets a fresh
        # label per bond, except where a spectator ride carries one across (below).
        label: dict[tuple[int, int], int] = {}
        for c in range(n_sites + 1):
            label[(c, 0)], label[(c, -1)] = _IDL, _IDR
        charge: dict[int, int] = {_IDL: 0, _IDR: 0}
        for n, row in enumerate(parsed):
            # A spectator ride reuses its own label across the cut, which is what
            # ``spectators`` means: one state whose space runs through the site untouched.
            for (i, j), (coeff, op) in row.items():
                if i == j and i not in (0, -1) and op is None and coeff == 1:
                    carried = label.get((n, i))
                    if carried is not None and carried in charge:
                        label[(n + 1, i)] = carried
            for (i, j), (coeff, op) in sorted(row.items()):
                left = label.get((n, i))
                if left is None or left not in charge:
                    continue  # unreachable from IdL; ``_edge_table`` prunes it either way
                if i == j and i in (0, -1):
                    continue  # a corner identity, already implicit in ``walk.spectators``
                if op is None:
                    w = _identity_w(Leg(walk.states[left], IN, False), phys)
                    emitted, space = charge[left], walk.states[left]
                else:
                    slot = walk.slot(id(op), op)
                    key = (slot, charge[left])
                    edge = walk.trans.get(key) or walk.transition(slot, charge[left], n)
                    w, emitted, space = edge
                if coeff != 1:
                    w = tenet.multiply(w, coeff)
                if j == -1:
                    if emitted:
                        raise ValueError(
                            f"from_entries: entry {(i, j)} of site {n} closes into IdR at "
                            f"bond charge {space}; both MPO boundaries are the trivial D=1 "
                            "leg, so a channel's operator charges must sum to the unit "
                            "sector before it closes"
                        )
                    stops = walk.stops[n]
                    stops[left] = stops[left] + w if left in stops else w
                    continue
                right = label.get((n + 1, j))
                if right is None:
                    right = label[(n + 1, j)] = len(walk.order) + 2
                    walk.order.append(right)
                if charge.get(right, emitted) != emitted:
                    raise ValueError(
                        f"from_entries: state {j} of bond {n + 1} is reached with two "
                        f"different charges; a bond state carries one GradedSpace, so the "
                        f"entries writing into it must agree (entry {(i, j)} of site {n} "
                        f"brings {space})"
                    )
                charge[right], walk.states[right] = emitted, space
                if right == left:
                    walk.spectators[n].add(right)
                else:
                    walk.moves[n][(left, right)] = w
        tab = _edge_table(
            n_sites,
            phys,
            False,
            walk.states,
            walk.order,
            walk.moves,
            walk.stops,
            walk.spectators,
        )
        live = set().union(*tab.ordered)
        dead = sorted(
            (c, i)
            for (c, i), k in label.items()
            if k not in (_IDL, _IDR) and 0 < c < n_sites and k not in live
        )
        if dead:
            c, i = dead[0]
            raise ValueError(
                f"from_entries: state {i} of bond {c} is dead -- it cannot be reached from "
                f"IdL at bond 0, or cannot reach IdR at bond {n_sites}, so nothing that runs "
                "through it is part of the operator"
            )
        return cls(edges=tab)

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
            The assembled operator, carrying the per-site
            [edge_blocks][tenet.network.MPO.edge_blocks] table at either cutoff.

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

        **``cutoff`` is the regime knob, and it is a build-time choice rather than a
        hidden runtime one.** Both settings now keep the block table
        [edge_blocks][tenet.network.MPO.edge_blocks] exposes and both run through
        [Env.heff2][tenet.network.Env.heff2]'s **one** engine path, because the sweeps pin
        the ``IdL``/``IdR`` channels through their SVDs (#204) instead of rotating them
        away (#141). What the choice decides is the operator the engine runs on:

        * ``cutoff=None`` for a **finite-range lattice model**. The compressing sweeps
          reduce its bond by exactly nothing, and the finite-state machine keeps its
          identity channels separable, so every spectator site rides a rank-2 map with no
          ``W`` contraction. Measured on N=20 U(1) Heisenberg at ``chi=64``: **1.96 s**
          against **3.53 s** at ``1e-13``.
        * a float ``cutoff`` for **power-law couplings and ab initio integrals**, where
          the sweep is the difference between a bond of 31,441 and one of 736 and the
          operator does not fit otherwise. The rotation mixes the open states, so no
          spectator separates any more and every open state is operator-carrying; that
          is a real constant factor and it is the same uniform mechanism block2 uses,
          which carries its identity as an ordinary entry in its operator map.

        The default stays ``1e-13``. The measurements are in ``docs/design.md``
        "Milestone 16" and "Milestone 39".

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

        # The term list is canonicalized into the same integer rows ``from_arrays``
        # hands over -- slots and sites in site order, the Koszul sign already on the
        # coefficient -- so both builders run the one walk (#197).
        walk = _Walk(n_sites, phys, dual)
        rows, sites, coeffs = [], [], []
        for coeff, ops in strings:
            row, where, c = _canonical_term(walk, n_sites, coeff, ops, split)
            rows.append(row)
            sites.append(where)
            coeffs.append(c)
        _term_edges(walk, rows, sites, coeffs)
        walk.close()
        states, order = walk.states, walk.order
        moves, stops, spectators = walk.moves, walk.stops, walk.spectators
        # The edge description is the finite-state machine, and at ``cutoff=None`` it *is*
        # the MPO: nothing is instantiated until a consumer asks (#200). Given a float,
        # ``_instantiate`` streams -- one site at a time in the backward sweep's order,
        # truncating as it goes -- so what comes back is already at the operator Schmidt
        # rank and the full-width MPO never exists.
        tab = _edge_table(n_sites, phys, dual, states, order, moves, stops, spectators)
        if cutoff is None:
            return cls(edges=tab)
        sites, cuts = _instantiate(tab, cutoff)
        return cls(edges=_compressed_table(_compress_forward(sites, cuts, cutoff), cuts, phys))

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
            The assembled operator, carrying the per-site
            [edge_blocks][tenet.network.MPO.edge_blocks] table at either cutoff.

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
        # No k-site split runs here, so the MPO stays on the non-dual convention.
        walk = _Walk(n_sites, phys, False)
        slots = np.array([walk.slot(id(op), op) for op in table], dtype=np.intp)
        seen = 0
        for labels, sites, coeffs in merged:
            # ``tolist`` once per group: the walk reads Python ints out of lists rather
            # than boxing a numpy scalar per operator.
            _term_edges(walk, slots[labels].tolist(), sites.tolist(), coeffs.tolist())
            seen += len(coeffs)
        if not seen:
            raise ValueError(
                "from_arrays: no term survived the merge and screen "
                f"(screen={screen}); an MPO is read off its terms and there are none"
            )
        walk.close()
        tab = _edge_table(
            n_sites,
            phys,
            False,
            walk.states,
            walk.order,
            walk.moves,
            walk.stops,
            walk.spectators,
        )
        if cutoff is None:
            return cls(edges=tab)
        sites, cuts = _instantiate(tab, cutoff)
        return cls(edges=_compressed_table(_compress_forward(sites, cuts, cutoff), cuts, phys))

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

    def apply(self, psi: MPS) -> MPS:
        """``H|psi>`` as a new [MPS][tenet.network.MPS], **untruncated**; ``psi`` is untouched.

        Parameters
        ----------
        psi : MPS
            The state; any gauge, any norm. Not modified -- the product is built from its
            frozen tensors into a new container.

        Returns
        -------
        MPS
            The product state, on the site convention ``MPS.__setitem__`` enforces. Its
            bond at every cut is the **fusion** of this operator's bond with ``psi``'s, so
            the bond dimension is the product of the two and the result is exact.

        Raises
        ------
        ValueError
            If the operator and the state have different lengths.

        Examples
        --------
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPO, MPS, overlap
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1), U1Sector(1)])
        >>> ident = MPO.identity(3, phys)
        >>> round(overlap(psi, ident.apply(psi)), 12)
        1.0

        Notes
        -----
        **Truncation is not hidden here and is not a keyword: it is
        [MPS.compress_][tenet.network.MPS.compress_], by name.**

            phi = h.apply(psi)
            discarded = phi.compress_(chi=64, cutoff=1e-12)

        That call already takes the ``chi``/``cutoff`` pair [Sweep][tenet.network.Sweep] and
        the sweep take and already returns the **total** discarded weight
        ``sqrt(sum_bond dw)``, which is the convention this question wants and which
        [sweep_][tenet.network.sweep_]'s per-bond *maximum* deliberately is not. Giving
        ``apply`` its own ``chi=`` would put a second name on that number and be the one
        place the two conventions could blur. Simplification: the untruncated product costs
        ``D_w`` times ``psi``'s bond, so for a wide operator compress promptly rather than
        holding the product; the zip-up apply that truncates *during* the sweep (YASTN's
        ``zipper``, TenPy's ``apply_zipup``) is the named upgrade and is a change with a
        measurement attached.

        **A deferred operator is materialised, site by site, through ``MPO.__getitem__``** --
        so an ``MPO`` built at
        ``cutoff=None``, which carries an edge description and no tensors, pays one full
        ``W`` per site here. That is stated rather than avoided: this is a whole-state
        product, not a sweep step, and there is no bond at which a symbolic operator could
        be kept symbolic. The sweep's own path (#200, #204) is untouched.

        **The virtual leg is turned around once, and that is the whole graded content.**
        The operator's bond and the state's bond cross a site in *opposite* directions --
        the fact ``docs/design.md`` "Milestone 11" spends a section on -- so they cannot be
        fused until one of them is turned. Turning the operator's left virtual leg is a
        duality relabel, [tenet.flip_dual][], which charges ``chi * theta`` per fusion tree:
        ``+1`` on every bosonic sector and ``-1`` on an odd fermionic one, which is exactly
        the sign that is missing if the fusion is written without it. The direction is fixed
        by the leg rather than by the flag: ``inv=not dual`` charges the same categorical
        map whether the leg was written dual or plain, and it has to be, because
        [from_terms][tenet.network.MPO.from_terms]'s two representations write that flag
        differently -- compressed bonds come back ``dual``, a deferred table's do not.
        Charging by the flag instead would make ``H|psi>`` depend on which representation
        built ``H``, silently and only for fermions. Both are tested against the dense
        oracle, under fermionic parity and under SU(2).
        """
        if len(self) != len(psi):
            raise ValueError(
                f"MPO.apply needs an operator and a state of the same length, got "
                f"{len(self)} and {len(psi)}"
            )
        sites, last = [], len(psi) - 1
        for n in range(len(psi)):
            w = self[n]
            w = tenet.flip_dual(w, (0,), inv=not w.legs[0].dual)
            w = tenet.repartition(w, (0, 1), (2, 3))  # (x OUT, P OUT | p IN, m IN)
            t = tenet.einsum("xPpm,apr->axPrm", w, psi[n])
            if n == 0:  # the operator's own D=1 boundary, capped rather than fused
                cap = ones((Leg(t.legs[1].space, IN, t.legs[1].dual),))
                t = tenet.einsum("x,axPrm->aPrm", cap, t)
            else:
                t = tenet.fuse(t, (0, 1))
            if n == last:
                cap = ones((Leg(t.legs[3].space, OUT, t.legs[3].dual),))
                t = tenet.einsum("aPrm,m->aPr", t, cap)
            else:
                t = tenet.fuse(t, (2, 3))
            sites.append(t)
        return MPS(sites)

    def variance(self, psi: MPS) -> float:
        """``<psi|H^2|psi> / <psi|psi> - E**2`` -- the convergence check that is not a change test.

        Parameters
        ----------
        psi : MPS
            The state, normally a converged [DMRG_out][tenet.network.DMRG_out]'s ``psi``;
            any gauge, any norm, and not modified.

        Returns
        -------
        float
            The energy variance. Zero for an exact eigenstate, and it falls as ``chi``
            grows for a state that is converging on one.

        Examples
        --------
        >>> from tenet import GradedSpace
        >>> from tenet.network import MPO, MPS
        >>> from tenet.symmetry import U1, U1Sector
        >>> phys = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
        >>> psi = MPS.product(phys, [U1Sector(1), U1Sector(-1)])
        >>> round(MPO.identity(2, phys).variance(psi), 12)  # every state is an eigenstate of 1
        0.0

        Notes
        -----
        TenPy names the same quantity ``MPO.variance(psi, exp_val=None)``. ``dmrg_``'s own
        convergence test (``network/dmrg.py``) is a **change** test -- the energy stopped
        moving and the Schmidt values stopped moving -- and both references say plainly that
        a change test can be satisfied by a run stuck on a wrong bond structure. This is the
        check that is not a change test, and ``docs/tutorials/dmrg.md`` shows it beside the
        convergence discussion.

        **One line over [apply][tenet.network.MPO.apply] and
        [overlap][tenet.network.overlap], and no ``MPO @ MPO``.** With ``|Hpsi> = H|psi>``
        exact, ``<psi|H^2|psi>`` is ``<Hpsi|Hpsi>`` and ``<psi|H|psi>`` is
        ``<psi|Hpsi>`` -- three overlaps and one product. Expanding ``H**2`` as a term list
        would be quadratic in the term count and would ask the caller to multiply every
        operator pair by hand, which is why #214 is about the apply and not about an
        operator algebra.

        The product is **untruncated**, so this is the variance of ``psi`` under the exact
        ``H`` and not of a compressed approximation to it; the cost is one state of bond
        ``D_w`` times ``psi``'s. ``<psi|H|psi>`` read this way agrees with
        [Env.measure][tenet.network.Env.measure] to solver precision, which is tested and is
        the statement that the apply is the operator it claims to be.
        """
        hpsi = self.apply(psi)
        norm2 = overlap(psi, psi)
        energy = overlap(psi, hpsi) / norm2
        return overlap(hpsi, hpsi) / norm2 - energy * energy
