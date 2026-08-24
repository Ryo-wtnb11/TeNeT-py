"""Square-lattice geometry and the site-indexed container every 2D object is built on.

A transcription of YASTN's ``yastn/tn/fpeps/_geometry.py`` (b0187c4), which M79/#277
adopts wholesale rather than reinventing: [Site][tenet.network.Site],
[Bond][tenet.network.Bond], [SquareLattice][tenet.network.SquareLattice] with its three
boundaries, the two pattern subclasses, and [Lattice][tenet.network.Lattice] -- a
geometry plus one object per *unique* site, which the state, the environment and the
projectors all instantiate.

**Coordinates are matrix indices.** ``Site(nx, ny)`` is ``(row, column)`` with
``Site(0, 0)`` at the top-left::

    +----- y (columns) ----->
    |
    x (rows)   (0,0) (0,1) ...
    |          (1,0) (1,1) ...
    v

**The fermionic order is column-major: left before right, top before bottom.**
[SquareLattice.f_ordered][tenet.network.SquareLattice.f_ordered] is the predicate, and
[SquareLattice.bonds][tenet.network.SquareLattice.bonds] emits every bond already
oriented that way (``Bond(s, nn_site(s, 'r'))`` and ``Bond(s, nn_site(s, 'b'))``), so no
caller has to re-derive which end of a bond comes first. It is the *same* statement the
composition rule makes about operand order in
[peps][tenet.network.peps]: a fermionic contraction is not symmetric in its two ends, so
the order is data and it is written down once, here.

**Deliberately not transcribed** (M79a scope): YASTN's ``_patch`` shadow dictionary,
``to_dict``/``from_dict`` serialization, and ``TriangularLattice``. The patch mechanism
serves evolution's provisional per-site updates (M79d), serialization has no consumer
yet, and the triangular lattice is out of scope for every stage of #277.
"""

from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

__all__ = [
    "Bond",
    "CheckerboardLattice",
    "Lattice",
    "RectangularUnitcell",
    "Site",
    "SquareLattice",
]


class Site(NamedTuple):
    """A lattice site, ``(row, column)``.

    Attributes
    ----------
    nx : int
        Row index; increases downwards.
    ny : int
        Column index; increases rightwards.
    """

    nx: int = 0
    ny: int = 0

    def __str__(self) -> str:
        return f"Site({self.nx}, {self.ny})"


class Bond(NamedTuple):
    """A nearest-neighbour bond, ``site0`` before ``site1`` in the fermionic order.

    Attributes
    ----------
    site0 : Site
        The left (horizontal bond) or top (vertical bond) end.
    site1 : Site
        The right or bottom end.
    """

    site0: Site
    site1: Site

    def __str__(self) -> str:
        return f"Bond({self.site0}, {self.site1})"


#: Per boundary, the wrap rule for each axis: ``i`` infinite, ``o`` open, ``p`` periodic.
_PERIODIC = {"infinite": "ii", "obc": "oo", "cylinder": "po"}

#: The nine direction labels, as ``(dx, dy)`` shifts.
_DIR = {
    "tl": (-1, -1),
    "t": (-1, 0),
    "tr": (-1, 1),
    "l": (0, -1),
    "r": (0, 1),
    "bl": (1, -1),
    "b": (1, 0),
    "br": (1, 1),
}


class SquareLattice:
    """Geometry of a 2D square lattice: which sites exist, and who neighbours whom.

    Parameters
    ----------
    dims : tuple[int, int]
        Unit-cell size as ``(rows, columns)``.
    boundary : str
        ``'infinite'`` (the default), ``'obc'`` for a finite patch, or ``'cylinder'``
        for a finite cylinder periodic along the rows.

    Raises
    ------
    ValueError
        If ``boundary`` is not one of the three, or ``dims`` is not two positive ints.

    Examples
    --------
    >>> from tenet.network import SquareLattice
    >>> lat = SquareLattice(dims=(2, 3))
    >>> lat.nn_site((0, 2), "r")            # infinite: wraps by the unit cell
    Site(nx=0, ny=3)
    >>> lat.site2index((0, 3))
    (0, 0)
    >>> SquareLattice(dims=(2, 3), boundary="obc").nn_site((0, 2), "r") is None
    True

    Notes
    -----
    An infinite lattice never returns ``None`` from
    [nn_site][tenet.network.SquareLattice.nn_site]: it shifts, and
    [site2index][tenet.network.SquareLattice.site2index] folds back into the unit cell
    later. That split -- *which* site versus *which tensor* -- is what lets an algorithm
    walk the plane in absolute coordinates and only fold when it reads a tensor.
    """

    def __init__(self, dims: tuple[int, int] = (2, 2), boundary: str = "infinite") -> None:
        if boundary not in _PERIODIC:
            raise ValueError(f"boundary={boundary!r} is not one of 'infinite', 'obc' or 'cylinder'")
        if len(dims) != 2 or any(int(d) != d or d < 1 for d in dims):
            raise ValueError(f"dims={dims!r} should be two positive integers (rows, columns)")
        self.boundary = boundary
        self._periodic = _PERIODIC[boundary]
        self._dims = (int(dims[0]), int(dims[1]))
        self._sites: tuple[Site, ...] = tuple(
            Site(nx, ny) for ny in range(self.Ny) for nx in range(self.Nx)
        )
        self._rebuild_bonds()

    def _rebuild_bonds(self) -> None:
        """Unique bonds, each oriented along the fermionic order (see the module docstring)."""
        horizontal, vertical = [], []
        for s in self._sites:
            right = self.nn_site(s, "r")  # left before right
            if right is not None:
                horizontal.append(Bond(s, right))
            below = self.nn_site(s, "b")  # top before bottom
            if below is not None:
                vertical.append(Bond(s, below))
        self._bonds_h = tuple(horizontal)
        self._bonds_v = tuple(vertical)

    @property
    def Nx(self) -> int:
        """Rows in the unit cell."""
        return self._dims[0]

    @property
    def Ny(self) -> int:
        """Columns in the unit cell."""
        return self._dims[1]

    @property
    def dims(self) -> tuple[int, int]:
        """Unit-cell size, ``(rows, columns)``."""
        return self._dims

    def sites(self, reverse: bool = False) -> tuple[Site, ...]:
        """The unique sites, column-major (the fermionic order)."""
        return self._sites[::-1] if reverse else self._sites

    def bonds(self, dirn: str | None = None, reverse: bool = False) -> tuple[Bond, ...]:
        """The unique nearest-neighbour bonds.

        Parameters
        ----------
        dirn : str or None
            ``'h'`` for horizontal, ``'v'`` for vertical, ``None`` for horizontal
            followed by vertical.
        reverse : bool
            Reverse the sequence (and, for ``None``, the two groups as well).

        Returns
        -------
        tuple[Bond, ...]
            Each bond with ``site0`` before ``site1`` in the fermionic order.
        """
        if dirn == "h":
            return self._bonds_h[::-1] if reverse else self._bonds_h
        if dirn == "v":
            return self._bonds_v[::-1] if reverse else self._bonds_v
        if dirn is not None:
            raise ValueError(f"dirn={dirn!r} should be 'h', 'v' or None")
        if reverse:
            return self._bonds_v[::-1] + self._bonds_h[::-1]
        return self._bonds_h + self._bonds_v

    def nn_site(self, site: Site | tuple[int, int] | None, d: str | tuple[int, int]) -> Site | None:
        """The site reached from ``site`` by the shift ``d``, or ``None`` if there is none.

        Parameters
        ----------
        site : Site or tuple[int, int] or None
            The starting site; ``None`` propagates.
        d : str or tuple[int, int]
            One of ``'t'``, ``'b'``, ``'l'``, ``'r'``, ``'tl'``, ``'tr'``, ``'bl'``,
            ``'br'``, or an explicit ``(dx, dy)``.

        Returns
        -------
        Site or None
            ``None`` when the shift leaves an open boundary. On an infinite axis the
            site is returned unfolded; on a periodic axis it wraps.
        """
        if site is None:
            return None
        dx, dy = _DIR[d] if isinstance(d, str) else d
        x, y = site[0] + dx, site[1] + dy
        if self._periodic[0] == "o" and not 0 <= x < self._dims[0]:
            return None
        if self._periodic[1] == "o" and not 0 <= y < self._dims[1]:
            return None
        if self._periodic[0] == "p":
            x = x % self._dims[0]
        return Site(x, y)

    def nn_bond_dirn(self, s0: Any, s1: Any = None) -> str:
        """``'lr'``, ``'tb'``, ``'rl'`` or ``'bt'`` for a nearest-neighbour pair.

        Raises
        ------
        ValueError
            If the two sites are not nearest neighbours.
        """
        if s1 is None:
            s0, s1 = s0
        pairs = (("r", "l", "lr"), ("b", "t", "tb"), ("l", "r", "rl"), ("t", "b", "bt"))
        for d, back, name in pairs:
            if self.nn_site(s0, d) == tuple(s1) and self.nn_site(s1, back) == tuple(s0):
                return name
        raise ValueError(f"{Site(*s0)}, {Site(*s1)} are not nearest-neighbour sites")

    def f_ordered(self, s0: Any, s1: Any) -> bool:
        """Are ``s0`` and ``s1`` in the fermionic order (column-major), or identical?

        Left before right, and within a column top before bottom -- the order
        [bonds][tenet.network.SquareLattice.bonds] already orients every bond by.
        """
        return bool(s0[1] < s1[1] or (s0[1] == s1[1] and s0[0] <= s1[0]))

    def site2index(self, site: Any) -> Any:
        """Fold any site of the plane onto the key of the unique tensor that sits there."""
        if site is None:
            return None
        x = site[0] % self._dims[0] if self._periodic[0] in "ip" else site[0]
        y = site[1] % self._dims[1] if self._periodic[1] == "i" else site[1]
        return (x, y)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        twin: SquareLattice = other
        return (
            self._periodic == twin._periodic
            and self._dims == twin._dims
            and self._sites == twin._sites
            and all(
                self.site2index((nx, ny)) == twin.site2index((nx, ny))
                for ny in range(self.Ny)
                for nx in range(self.Nx)
            )
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._periodic, self._dims, self._sites))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dims={self._dims}, boundary={self.boundary!r})"


class CheckerboardLattice(SquareLattice):
    """The infinite bipartite lattice: a 2x2 cell holding two unique tensors.

    Examples
    --------
    >>> from tenet.network import CheckerboardLattice
    >>> lat = CheckerboardLattice()
    >>> lat.site2index((0, 0)), lat.site2index((0, 1)), lat.site2index((1, 1))
    (0, 1, 0)
    >>> len(lat.sites())
    2
    """

    def __init__(self) -> None:
        super().__init__(dims=(2, 2), boundary="infinite")
        self._sites = (Site(0, 0), Site(0, 1))
        self._bonds_h = (Bond(Site(0, 0), Site(0, 1)), Bond(Site(0, 1), Site(0, 2)))
        self._bonds_v = (Bond(Site(0, 0), Site(1, 0)), Bond(Site(0, 1), Site(1, 1)))

    def site2index(self, site: Any) -> Any:
        """``(nx + ny) % 2`` -- the two sublattices."""
        return None if site is None else (site[0] + site[1]) % 2

    def __repr__(self) -> str:
        return "CheckerboardLattice()"


class RectangularUnitcell(SquareLattice):
    """An infinite lattice tiled by a rectangular pattern of unique-tensor labels.

    Parameters
    ----------
    pattern : Sequence[Sequence] or dict[tuple[int, int], object]
        Labels of the unique tensors at each site of the cell. A dict must cover the
        rectangle from ``(0, 0)`` to ``(Nx - 1, Ny - 1)``.

    Raises
    ------
    ValueError
        If the pattern is not a rectangle, if its labels are unhashable, or if two
        sites carrying the *same* label do not see the same four neighbours.

    Examples
    --------
    >>> from tenet.network import RectangularUnitcell
    >>> lat = RectangularUnitcell([[0, 1], [1, 0]])   # the checkerboard, spelled out
    >>> lat.sites()
    (Site(nx=0, ny=0), Site(nx=0, ny=1))
    >>> RectangularUnitcell([[0, 1], [1, 1]])
    Traceback (most recent call last):
        ...
    ValueError: RectangularUnitcell: each unique label must have the same neighbours

    Notes
    -----
    The neighbourhood check is the class's whole reason to exist. A pattern such as
    ``[[0, 1], [1, 1]]`` assigns one tensor to sites whose environments differ, so a
    single environment per label cannot describe it -- YASTN states that as a warning
    (``_geometry.py``:255) and raises; here it is the same refusal.

    Only patterns of a single momentum ``Q`` survive the check, which is exactly the
    family this parameterization can represent (after B. Ponsioen's ``ad-peps``).
    """

    def __init__(self, pattern: Sequence[Sequence[Any]] | dict[tuple[int, int], Any]) -> None:
        rows = _pattern_rows(pattern)
        super().__init__(dims=(len(rows), len(rows[0])), boundary="infinite")
        self._pattern = {
            (nx, ny): label for nx, row in enumerate(rows) for ny, label in enumerate(row)
        }
        label_sites: dict[Any, list[Site]] = {}
        neighbourhoods: dict[Any, set[Any]] = {}
        try:
            for nx in range(self.Nx):
                for ny in range(self.Ny):
                    label = self._pattern[nx, ny]
                    around = tuple(
                        self.site2index(s)
                        for s in ((nx - 1, ny), (nx, ny - 1), (nx + 1, ny), (nx, ny + 1))
                    )
                    label_sites.setdefault(label, []).append(Site(nx, ny))
                    neighbourhoods.setdefault(label, set()).add(around)
        except TypeError as exc:
            raise ValueError("RectangularUnitcell: pattern labels must be hashable") from exc
        if any(len(seen) > 1 for seen in neighbourhoods.values()):
            raise ValueError("RectangularUnitcell: each unique label must have the same neighbours")
        self._sites = tuple(sorted(min(group) for group in label_sites.values()))
        self._bonds_h = tuple(Bond(s, self.nn_site(s, "r")) for s in self._sites)
        self._bonds_v = tuple(Bond(s, self.nn_site(s, "b")) for s in self._sites)

    def site2index(self, site: Any) -> Any:
        """The pattern's label at ``site``, folded into the unit cell."""
        if site is None:
            return None
        return self._pattern[site[0] % self.Nx, site[1] % self.Ny]

    def __repr__(self) -> str:
        return f"RectangularUnitcell(pattern={self._pattern})"


def _pattern_rows(pattern: Sequence[Sequence[Any]] | dict[tuple[int, int], Any]) -> list[list[Any]]:
    """Normalize either accepted pattern spelling into a rectangle of rows."""
    if type(pattern) is dict:
        if not pattern:
            raise ValueError("RectangularUnitcell: pattern is empty")
        max_row = max(k[0] for k in pattern)
        max_col = max(k[1] for k in pattern)
        if min(k[0] for k in pattern) != 0 or min(k[1] for k in pattern) != 0:
            raise ValueError(
                "RectangularUnitcell: pattern keys should cover (0, 0) to (Nx - 1, Ny - 1)"
            )
        try:
            return [[pattern[r, c] for c in range(max_col + 1)] for r in range(max_row + 1)]
        except KeyError as exc:
            raise ValueError(
                "RectangularUnitcell: pattern keys should cover (0, 0) to (Nx - 1, Ny - 1)"
            ) from exc
    rows = [list(row) for row in pattern]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("RectangularUnitcell: pattern should be a rectangular matrix of labels")
    return rows


class Lattice:
    """A geometry plus one object per *unique* site: the container everything 2D subclasses.

    Parameters
    ----------
    geometry : SquareLattice
        The geometry, or anything carrying one as ``.geometry``.
    objects : optional
        One object, a nested sequence, or a ``{site: object}`` mapping. A single object
        is spread over every unique site.

    Raises
    ------
    ValueError
        If an assignment lands outside the geometry, if two different objects reach one
        unique site, or if some unique site is left unassigned.

    Examples
    --------
    >>> from tenet.network import CheckerboardLattice, Lattice
    >>> lat = Lattice(CheckerboardLattice(), {(0, 0): "A", (0, 1): "B"})
    >>> lat[0, 0], lat[1, 1], lat[0, 3]
    ('A', 'A', 'B')

    Notes
    -----
    Reads and writes go through
    [site2index][tenet.network.SquareLattice.site2index], so ``lat[5, 7]`` is the
    tensor of the unique site ``(5, 7)`` folds onto -- the container stores one entry
    per unique site and nothing per lattice site. It is a **mutable container of
    immutable tensors**, the arrangement ``MPS`` and ``Env`` already use.
    """

    def __init__(self, geometry: Any, objects: Any = None) -> None:
        self.geometry: SquareLattice = getattr(geometry, "geometry", geometry)
        self._data: dict[Any, Any] = {self.site2index(site): None for site in self.sites()}
        if objects is None:
            return
        # ``type(...) is`` rather than ``isinstance``: a SymmetricTensor has ``items()``
        # too, so duck typing would read a single spread-me tensor as a mapping, and the
        # package's hygiene fence forbids ``isinstance`` on anything but ``int``/``str``.
        kind = type(objects)
        if kind is dict:
            items = list(objects.items())
        elif kind is list or kind is tuple:
            items = [
                ((nx, ny), obj) for nx, row in enumerate(objects) for ny, obj in enumerate(row)
            ]
        else:
            items = [(site, objects) for site in self.sites()]
        for site, obj in items:
            key = self.site2index(site)
            if key not in self._data:
                raise ValueError(f"{type(self).__name__}: {Site(*site)} is outside the geometry")
            if self._data[key] is None:
                self._data[key] = obj
            elif self._data[key] is not obj:
                raise ValueError(
                    f"{type(self).__name__}: two different objects assigned to unique "
                    f"site {self.site2index(site)}"
                )
        missing = [key for key, obj in self._data.items() if obj is None]
        if missing:
            raise ValueError(f"{type(self).__name__}: unique sites {missing} were not assigned")

    # -- the geometry, forwarded so a Lattice reads like the lattice it lives on -----

    @property
    def Nx(self) -> int:
        """Rows in the unit cell."""
        return self.geometry.Nx

    @property
    def Ny(self) -> int:
        """Columns in the unit cell."""
        return self.geometry.Ny

    @property
    def dims(self) -> tuple[int, int]:
        """Unit-cell size, ``(rows, columns)``."""
        return self.geometry.dims

    @property
    def boundary(self) -> str:
        """The geometry's boundary condition."""
        return self.geometry.boundary

    def sites(self, reverse: bool = False) -> tuple[Site, ...]:
        """The unique sites."""
        return self.geometry.sites(reverse)

    def bonds(self, dirn: str | None = None, reverse: bool = False) -> tuple[Bond, ...]:
        """The unique bonds."""
        return self.geometry.bonds(dirn, reverse)

    def nn_site(self, site: Any, d: str | tuple[int, int]) -> Site | None:
        """The neighbour of ``site`` in direction ``d``."""
        return self.geometry.nn_site(site, d)

    def nn_bond_dirn(self, s0: Any, s1: Any = None) -> str:
        """The orientation of a nearest-neighbour bond."""
        return self.geometry.nn_bond_dirn(s0, s1)

    def f_ordered(self, s0: Any, s1: Any) -> bool:
        """Whether two sites are in the fermionic order."""
        return self.geometry.f_ordered(s0, s1)

    def site2index(self, site: Any) -> Any:
        """The unique-tensor key of ``site``."""
        return self.geometry.site2index(site)

    # -- the container --------------------------------------------------------------

    def __getitem__(self, site: Any) -> Any:
        return self._data[self.site2index(site)]

    def __setitem__(self, site: Any, obj: Any) -> None:
        self._data[self.site2index(site)] = obj

    def items(self) -> Iterator[tuple[Site, Any]]:
        """``(site, object)`` for every unique site."""
        return ((site, self[site]) for site in self.sites())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(geometry={self.geometry!r}, objects={self._data!r})"
