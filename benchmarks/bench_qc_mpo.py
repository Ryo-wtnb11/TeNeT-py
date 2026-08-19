"""Where ``MPO.from_terms`` breaks on ab initio quantum-chemistry integrals (#184 Part 1).

Feeds real and synthetic ab initio Hamiltonians into
[MPO.from_terms][tenet.network.MPO.from_terms] on spin-orbital ``fZ2`` sites and records,
per orbital count ``K``:

* the bond width per cut at ``cutoff=None`` (the finite-state machine's own bond) and at
  the default ``1e-13`` (after the two compressing SVD sweeps);
* peak resident memory and wall time split by phase -- term construction, ``_term_edges``,
  ``_instantiate``, sweep 1 (right to left) and sweep 2 (left to right) -- taken from the
  real ``from_terms`` call by wrapping its own module globals, not from a re-implementation;
* the term count, and the count of terms ``mps.py``'s same-site rule refuses;
* the **minimum vertex cover** of the same cut, computed independently and combinatorially
  (Kuhn maximum matching plus Koenig, no ``scipy``), reported beside the FSM width and the
  post-SVD width.

Not a test, not part of the package, on no CI path. Run from the repo root:

    uv run python benchmarks/bench_qc_mpo.py            # the shipped input set
    uv run python benchmarks/bench_qc_mpo.py --synthetic-only   # no network at all
    uv run python benchmarks/bench_qc_mpo.py --list     # what would run
    uv run python benchmarks/bench_qc_mpo.py --dmrg --only N2.CAS.6-31G   # the #202 trade

``--dmrg`` is the second instrument (#202): full DMRG runs against the ``cutoff=None``
operator, one per cache byte budget, reporting peak resident memory split across the
sweep's five per-bond caches, the wall time, and how many block tables and merged cores
each budget made the run build. The construction measurement above and this one answer
different questions and share the inputs, the caps and the JSON trail.

Each (input, cutoff) measurement runs in its own subprocess under a hard wall-clock cap
and a hard address-space cap (``--wall``, ``--mem``), so an input that will not finish
reports the phase it stopped in and the resource it exhausted instead of swapping the
machine to death. Nothing here is asserted in the test suite.

Licence decision, and its reversal criterion
--------------------------------------------
The real FCIDUMPs live in **block2-preview**, which is **GPL-3.0**; tenet is
**Apache-2.0**. Copying a GPL-licensed repository's data files into an Apache-2.0 tree is
a licensing question this benchmark is not the right place to answer, so the files are
**fetched, never vendored**: each is downloaded by URL with a pinned sha256 into a
git-ignored cache (``benchmarks/.cache/fcidump/``) and verified before use, with the
provenance block below committed in this script. The FCIDUMP *reader* and the *synthetic*
generator are in-repo and carry no licence question at all. CI never runs this script, so
the network access ``REPOSITORY_RULES.md`` forbids in tests is not involved.

**Reversal criterion, and it was checked rather than assumed:** if a permissively licensed
source for a small ab initio FCIDUMP at ``K >= 4`` is found, the two smallest are vendored
under ``tests/fixtures/`` in the racah pattern (``tests/fixtures/su2_f.txt``:1-9) and this
fetch path is deleted. Searched at implementation time across every public FCIDUMP a code
search reaches; what turned up:

* ``block2-preview`` and ``pyblock3-preview`` -- **GPL-3.0**;
* ``theochem/ModelHamiltonian`` -- **LGPL-3.0**;
* ``hande-qmc/hande`` -- **LGPL-2.1**;
* ``qiskit-community/qiskit-nature``, ``docs/migration/aux_files/h2.fcidump`` --
  **Apache-2.0**, and it is the only permissive ab initio FCIDUMP found. Its header reads
  ``NORB=2, NELEC=2``: four spin-orbital sites, which measures nothing about a bond that
  grows as ``K**3``. ``pyscf`` (Apache-2.0) ships an FCIDUMP *writer* and no data file.

So a permissive source exists at ``K = 2`` and at no larger ``K``, the criterion does not
fire for the input set this measurement needs, and the fetch path stands. It is worth
re-checking: one Apache-2.0 file at ``K >= 8`` would delete this whole section.
"""

import argparse
import hashlib
import json
import math
import pathlib
import platform
import random
import resource
import subprocess
import sys
import time
import urllib.request

import numpy as np

import tenet
from tenet import GradedSpace
from tenet.network import MPO, MPS, local_op, mps
from tenet.symmetry import FZ2Sector, fZ2

# --- provenance ---------------------------------------------------------------------
#
# source     block2-preview, ``data/`` at branch ``master``
# licence    GPL-3.0 -- fetched, not vendored; see the module docstring
# retrieved  2026-08-19 (UTC)
# generator  block2-preview's own repository; NORB and NELEC below are read from each
#            file's own ``&FCI`` header at run time and checked against these values
BASE_URL = "https://raw.githubusercontent.com/block-hczhai/block2-preview/master/data/"
REAL = {  # name: (sha256, NORB, NELEC, bytes)
    "H4.STO6G.R1.8": (
        "45aa698a55d955dc12f4652a620683c069bc22c34649f21ef7f75b49fb38bf72",
        4,
        4,
        1480,
    ),
    "H8.STO6G.R1.8": (
        "6d44c9ec5121be93a644b425ab793be3adfad49917e2676bda5cf61fe9677fb7",
        8,
        8,
        13661,
    ),
    "N2.STO3G": (
        "affea0210605a25fb3d25b41804a1adf0a302a210392dac91ab31e6d3cda3f17",
        10,
        14,
        11770,
    ),
    "H10.STO6G.R1.8": (
        "2d3aaec047cf4961d0a1437264ee02c533a10091990f5323c2a579d2e11fed07",
        10,
        10,
        30463,
    ),
    "N2.CAS.6-31G": (
        "b2f85d818b60f97ee6bc0d49d20ed6fa84b71cdfd94084c3dd1b7b7ea872a885",
        16,
        10,
        54747,
    ),
    "C2.CAS.PVDZ": (
        "68259d8ad9a75d2346051b3312052aff13933504d41b77ce22da86085ec62588",
        26,
        8,
        288422,
    ),
}
CACHE = pathlib.Path(__file__).resolve().parent / ".cache" / "fcidump"

# Integrals below this magnitude are dropped. Every FCIDUMP written by a program with
# point-group symmetry carries symmetry-forbidden entries at ~1e-15 (N2.STO3G has seven);
# keeping them would inflate the term count with terms that are numerically zero.
SCREEN = 1e-12
CHI_PRACTICAL = 2000  # the "practical chi" the FSM bond is asked to stay under
MEM_CEILING_GIB = 8.0
WALL_CAP_S = 600.0
# Kuhn is O(V*E); above this the cover is reported as capped rather than run for hours.
MVC_WORK_CAP = 4e8


# --- FCIDUMP ------------------------------------------------------------------------


def read_fcidump(path):
    """``(norb, nelec, [(value, i, j, k, l), ...])``; ``NORB``/``NELEC`` from the header.

    The whole format tenet needs: a Fortran namelist header ending in ``&END`` (or a bare
    ``/``), then ``value i j k l`` records with 1-based orbital indices, ``k == l == 0``
    for the one-body part and ``i == j == k == l == 0`` for the core energy.
    """
    text = pathlib.Path(path).read_text()
    head, sep, body = text.partition("&END")
    if not sep:
        head, _, body = text.partition("/\n")
    flat = head.upper().replace(" ", "")
    norb = int(flat.split("NORB=", 1)[1].split(",")[0])
    nelec = int(flat.split("NELEC=", 1)[1].split(",")[0])
    recs = []
    for line in body.splitlines():
        f = line.split()
        if len(f) == 5:
            recs.append((float(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4])))
    return norb, nelec, recs


def fetch(name):
    """The cached FCIDUMP for ``name``, downloaded once and verified against its sha256."""
    sha, norb, nelec, size = REAL[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.FCIDUMP"
    if not path.exists():
        with urllib.request.urlopen(f"{BASE_URL}{name}.FCIDUMP", timeout=60) as r:
            path.write_bytes(r.read())
    blob = path.read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    if got != sha or len(blob) != size:
        raise SystemExit(f"{path}: sha256 {got} ({len(blob)} B) != pinned {sha} ({size} B)")
    got_norb, got_nelec, recs = read_fcidump(path)
    if (got_norb, got_nelec) != (norb, nelec):
        raise SystemExit(
            f"{path}: header says NORB={got_norb} NELEC={got_nelec}, pinned {norb}/{nelec}"
        )
    return got_norb, got_nelec, recs


def synthetic(k, seed=0, screen=1e-6):
    """A deterministic ab initio-shaped integral set on ``k`` localized orbitals.

    **The decay model, stated so the numbers can be read.** The orbitals sit on a 1-D
    lattice at integer positions. Two properties of real localized-basis integrals are
    reproduced and nothing else is:

    * the *overlap distribution* ``phi_i phi_j`` is a product of two localized functions,
      so it decays as ``exp(-(i - j)**2 / w**2)`` -- Gaussian in the orbital separation;
    * the Coulomb interaction between two such distributions decays as the inverse
      distance between their centroids, ``1 / (1 + |(i + j)/2 - (k + l)/2|)``.

    So ``V_ijkl = exp(-((i-j)**2 + (k-l)**2) / w**2) / (1 + d)``, times a bounded
    deterministic jitter in ``[0.8, 1.2]`` keyed on the sorted index quadruple so the
    8-fold permutational symmetry of ``(ij|kl)`` holds *exactly*. The tensor is therefore
    **dense but structured** -- every entry is nonzero before screening, which is the
    property the whole measurement turns on -- and a one-body part ``h`` is a banded
    hopping matrix with a site-dependent diagonal.

    Screened at ``screen`` (default ``1e-6``), as every quantum-chemistry code screens; the
    threshold is reported with the term count. Deterministic under ``seed``, no download,
    no licence question.
    """
    w = 1.5
    rng = random.Random(seed)
    jitter = {}

    def jit(q):
        key = min((q, q[::-1], (q[1], q[0], q[3], q[2]), (q[2], q[3], q[0], q[1])))
        if key not in jitter:
            jitter[key] = 0.8 + 0.4 * rng.random()
        return jitter[key]

    recs = [(0.0, 0, 0, 0, 0)]
    for i in range(k):
        recs.append((-0.5 - 0.05 * i, i + 1, i + 1, 0, 0))
        if i + 1 < k:
            recs.append((-1.0, i + 2, i + 1, 0, 0))
    for i in range(k):
        for j in range(i + 1):
            for a in range(k):
                for b in range(a + 1):
                    if (i, j) < (a, b):
                        continue
                    d = abs((i + j) / 2 - (a + b) / 2)
                    v = math.exp(-((i - j) ** 2 + (a - b) ** 2) / w**2) / (1.0 + d)
                    v *= jit((i, j, a, b))
                    if abs(v) > screen:
                        recs.append((v, i + 1, j + 1, a + 1, b + 1))
    return k, k, recs


# --- the term list ------------------------------------------------------------------
#
# ``H = sum_pq h_pq sum_s a+_ps a_qs + 1/2 sum_pqrs (pq|rs) sum_st a+_ps a+_rt a_st a_qs``
# on spin-orbital sites ``P = 2p + s``. The 8-fold permutational symmetry of ``(ij|rs)``
# is expanded explicitly, so a record contributes every distinct quadruple it stands for.

EIGHTFOLD = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
             (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0))  # fmt: skip


def spin_orbital_terms(recs, screen=SCREEN):
    """``[(coeff, ((kind, site), ...)), ...]`` -- one operator per index occurrence."""
    h, v = {}, {}
    for val, i, j, k, m in recs:
        if abs(val) <= screen:
            continue
        if k == 0 and m == 0:
            if i == 0:
                continue  # the core energy is a constant shift, not a term
            h[(i - 1, j - 1)] = h[(j - 1, i - 1)] = val
        else:
            q = (i - 1, j - 1, k - 1, m - 1)
            for perm in EIGHTFOLD:
                v[tuple(q[x] for x in perm)] = val
    terms = []
    for (p, q), c in h.items():
        for s in (0, 1):
            terms.append((c, (("cd", 2 * p + s), ("c", 2 * q + s))))
    for (p, q, r, s), c in v.items():
        for sa in (0, 1):
            for sb in (0, 1):
                ops = (
                    ("cd", 2 * p + sa),
                    ("cd", 2 * r + sb),
                    ("c", 2 * s + sb),
                    ("c", 2 * q + sa),
                )
                terms.append((0.5 * c, ops))
    return terms


PHYS = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
C = np.array([[0.0, 1.0], [0.0, 0.0]])  # |0> even, |1> odd; c|1> = |0>
MAT = {"c": C, "cd": C.T}
_OPS: dict = {}


_LABELS: dict = {}


def _label(matrix, odd):
    """A small hashable id per distinct on-site matrix -- the combinatorial stand-in for
    ``id(op)``."""
    key = (matrix.tobytes(), odd)
    if key not in _LABELS:
        _LABELS[key] = (len(_LABELS), matrix, odd)
    return _LABELS[key][0]


def _op(label):
    """One ``local_op`` object per distinct on-site matrix.

    Object identity is load-bearing: ``_term_edges`` keys an FSM state on ``id(op)``
    (``mps.py``:817), so a fresh ``local_op`` per term would give every term its own state
    and the measurement would be of the cache, not of the machine.
    """
    if label not in _OPS:
        _, matrix, odd = next(v for v in _LABELS.values() if v[0] == label)
        _OPS[label] = local_op(matrix, phys=PHYS, charge=FZ2Sector(1 if odd else 0))
    return _OPS[label]


def fold_terms(terms, premultiply=True):
    """``(folded terms, refused count)``, still symbolic: ``[(coeff, ((label, site), ...))]``.

    ``_term_edges`` refuses two operators of one term on one site (``mps.py``:815), and
    every quantum-chemistry term with a repeated spin-orbital index is one. Pre-multiplying
    the coincident operators into a single on-site operator is what a caller must do today,
    and this is that caller: the operator list is sorted into site order, the Koszul sign of
    the reorder is paid on the coefficient exactly as ``_term_edges`` pays it (strict ``>``,
    so two operators on one site never contribute a sign), and the surviving per-site
    products are taken in their original relative order. A product that vanishes
    (``a+_P a+_P``) drops the term.

    The ``label`` is the *matrix*, canonicalized to a small integer -- deliberately not the
    operator's spelling, because ``_term_edges`` keys an FSM state on ``id(op)``
    (``mps.py``:817) and this benchmark hands it one object per distinct matrix. Two
    spellings that multiply to the same matrix must therefore share a state, and they do.
    """
    out, refused = [], 0
    for coeff, ops in terms:
        sites = [s for _k, s in ops]
        if len(set(sites)) != len(sites):
            refused += 1
            if not premultiply:
                continue
        inversions = sum(
            1 for a in range(len(sites)) for b in range(a + 1, len(sites)) if sites[a] > sites[b]
        )
        groups: dict = {}
        for kind, site in ops:
            groups.setdefault(site, []).append(kind)
        entry, dead = [], False
        for site in sorted(groups):
            kinds = groups[site]
            m = MAT[kinds[0]]
            for kind in kinds[1:]:
                m = m @ MAT[kind]
            if not m.any():
                dead = True
                break
            entry.append((_label(m, len(kinds) % 2 == 1), site))
        if not dead:
            out.append((coeff * (-1.0) ** inversions, tuple(entry)))
    return out, refused


def to_tenet_terms(folded):
    """The folded symbolic terms as ``from_terms`` input: one ``local_op`` object per label."""
    return [(c, [(_op(label), site) for label, site in ops]) for c, ops in folded]


# --- the combinatorial cross-check --------------------------------------------------


def _walks(terms):
    """``(first site, last site, ((kind, site), ...) in site order)`` per accepted term."""
    for _c, ops in terms:
        sites = [s for _k, s in ops]
        if len(set(sites)) != len(sites):
            continue
        ordered = tuple(sorted(ops, key=lambda o: o[1]))
        yield ordered[0][1], ordered[-1][1], ordered


def fsm_profile(n_sites, terms):
    """The FSM bond per cut under tenet's exact state rule (``mps.py``:874).

    A cut's bond is the open left-partial strings plus the two identity channels where
    they are live: ``IdL`` while some term has not started, ``IdR`` once some term has
    finished. That is exactly the bond ``_instantiate`` leaves after pruning, and the
    ``check`` column of the report is that claim measured against the real MPO.
    """
    opened = [set() for _ in range(n_sites + 1)]
    idl = [0] * (n_sites + 1)
    idr = [0] * (n_sites + 1)
    for first, last, ordered in _walks(terms):
        for cut in range(first + 1):
            idl[cut] = 1
        for cut in range(last + 1, n_sites + 1):
            idr[cut] = 1
        prefix: tuple = ()
        for m in range(len(ordered) - 1):
            prefix = (*prefix, ordered[m])
            for cut in range(ordered[m][1] + 1, ordered[m + 1][1] + 1):
                opened[cut].add(prefix)
    return [len(o) + a + b for o, a, b in zip(opened, idl, idr, strict=True)]


def mvc_profile(n_sites, terms, work_cap=MVC_WORK_CAP):
    """The minimum vertex cover per cut, plus the same two identity channels.

    At each cut the terms form a bipartite graph of left partial strings against right
    partial strings; a bond basis that can carry the Hamiltonian must cover every edge, and
    the minimum such basis is a minimum vertex cover. Covering a right vertex *is* emitting
    one complementary operator that sums a row of coefficients -- block2's ``Bipartite``,
    which is its default for arbitrary term lists.
    """
    walks = list(_walks(terms))
    prof = []
    for cut in range(n_sites + 1):
        edges, idl, idr = set(), 0, 0
        for first, last, ordered in walks:
            if first >= cut:
                idl = 1
            if last < cut:
                idr = 1
            left = tuple(o for o in ordered if o[1] < cut)
            right = tuple(o for o in ordered if o[1] >= cut)
            if left and right:
                edges.add((left, right))
        cover = min_vertex_cover(edges, work_cap)
        prof.append(None if cover is None else cover + idl + idr)
    return prof


def min_vertex_cover(edges, work_cap=MVC_WORK_CAP):
    """|MVC| of a bipartite graph: Kuhn maximum matching, then Koenig. ``None`` if capped.

    Koenig's theorem makes the cover the matching's size, and the *construction* names it:
    let ``Z`` be everything reachable by alternating paths from the unmatched left
    vertices; the cover is ``(L \\ Z) union (R and Z)``. No ``scipy`` -- it is deliberately
    not a dependency of this repository.
    """
    adj: dict = {}
    for u, w in edges:
        adj.setdefault(u, []).append(w)
    if len(adj) * max(len(edges), 1) > work_cap:
        return None
    match_r, match_l = {}, {}

    def augment(u, seen):
        for w in adj[u]:
            if w in seen:
                continue
            seen.add(w)
            if w not in match_r or augment(match_r[w], seen):
                match_r[w], match_l[u] = u, w
                return True
        return False

    for u in adj:
        augment(u, set())
    reach_l = {u for u in adj if u not in match_l}
    reach_r: set = set()
    stack = list(reach_l)
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in reach_r and match_l.get(u) != w:
                reach_r.add(w)
                nxt = match_r.get(w)
                if nxt is not None and nxt not in reach_l:
                    reach_l.add(nxt)
                    stack.append(nxt)
    return sum(1 for u in adj if u not in reach_l) + len(reach_r)


# --- the instrument -----------------------------------------------------------------


def rss_gib():
    """Peak resident set size so far, in GiB (``ru_maxrss`` is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


class Phases:
    """Wall time and peak RSS per phase, taken from ``from_terms``' own module globals.

    Not a re-implementation of ``from_terms``: the real classmethod runs, and
    ``_term_edges``, ``_instantiate``, ``_braids_with_signs`` and ``svd_truncated`` are
    wrapped in place for the duration. The two compressing sweeps are separated by the
    partition their SVD asks for -- ``((0,), (1, 2, 3))`` is the right-to-left sweep,
    ``((0, 1, 2), (3,))`` the left-to-right one, and ``_split``'s internal SVD asks for
    neither -- so each sweep is bracketed by its own first SVD's timestamp. Since #191
    ``_instantiate`` *contains* the backward sweep at a float cutoff -- it places one
    site at a time in that sweep's order -- so ``t_instantiate`` is instantiation and
    sweep 1 together and ``t_sweep1`` is the span inside it, not a phase after it.

    ``_braids_with_signs`` is counted and timed because #184 predicted it to be an
    unprofiled hot spot worth a one-line ``functools.cache``. **Measured, it is not, and
    the cache does not land.** It is 0.1--1.9 % of ``from_terms``' wall time across every
    input here and its share *falls* as K rises (H8: 0.10 s of 35.5 s over 30 775 calls).
    Wrapping it in ``functools.cache`` at H8 takes the real call count from 30 775 to 2
    and the wall time from 29.90 s to 30.77 s -- inside run-to-run noise, because the
    other 99.7 % is ``_instantiate``. A cache that removes 99.99 % of the calls to a
    function and moves the total by nothing is a measurement that the function was never
    the cost.
    """

    def __init__(self):
        self.t = dict.fromkeys(("term_edges", "edge_table", "instantiate", "braids"), 0.0)
        self.braid_calls = 0
        self.t_instantiate_end = None
        self.t_sweep1_start = None
        self.t_sweep2_start = None
        self._saved = {}

    def __enter__(self):
        self._saved = {
            "_term_edges": mps._term_edges,
            "_edge_table": mps._edge_table,
            "_instantiate": mps._instantiate,
            "_braids_with_signs": mps._braids_with_signs,
            "svd": tenet.linalg.svd_truncated,
        }
        outer = self

        def timed(name, fn, after=None):
            def wrapper(*a, **kw):
                t0 = time.perf_counter()
                try:
                    return fn(*a, **kw)
                finally:
                    outer.t[name] += time.perf_counter() - t0
                    if after:
                        after()

            return wrapper

        def mark_instantiate():
            outer.t_instantiate_end = time.perf_counter()
            note(event="phase", phase="instantiate", status="done")

        def count_braids(*a, **kw):
            outer.braid_calls += 1
            t0 = time.perf_counter()
            try:
                return self._saved["_braids_with_signs"](*a, **kw)
            finally:
                outer.t["braids"] += time.perf_counter() - t0

        def svd(t, axes, **kw):
            if axes == ((0,), (1, 2, 3)) and outer.t_sweep1_start is None:
                outer.t_sweep1_start = time.perf_counter()
                note(event="phase", phase="sweep 1", status="start")
            if axes == ((0, 1, 2), (3,)) and outer.t_sweep2_start is None:
                outer.t_sweep2_start = time.perf_counter()
                note(event="phase", phase="sweep 2", status="start")
            return self._saved["svd"](t, axes, **kw)

        def instantiate(*a, **kw):
            note(event="phase", phase="instantiate", status="start")
            return timed("instantiate", self._saved["_instantiate"], mark_instantiate)(*a, **kw)

        mps._term_edges = timed("term_edges", self._saved["_term_edges"])
        mps._edge_table = timed("edge_table", self._saved["_edge_table"])
        mps._instantiate = instantiate
        mps._braids_with_signs = count_braids
        tenet.linalg.svd_truncated = svd
        return self

    def __exit__(self, *exc):
        mps._term_edges = self._saved["_term_edges"]
        mps._edge_table = self._saved["_edge_table"]
        mps._instantiate = self._saved["_instantiate"]
        mps._braids_with_signs = self._saved["_braids_with_signs"]
        tenet.linalg.svd_truncated = self._saved["svd"]
        return False


def bond_widths(h):
    """The MPO's bond dimension at every cut, boundaries included.

    Off the edge description where there is one (#200): asking ``h[n]`` for its leg would
    materialise the very site tensor the deferred path exists not to build, which would
    make this reporting line the thing that decides the measurement. Both spellings give
    the same numbers; the fallback is what runs against the base commit.
    """
    edges = getattr(h, "edges", None)
    if edges is not None:
        return [space.dim for space in edges.bonds]
    return [h[n].legs[0].space.dim for n in range(len(h))] + [h[len(h) - 1].legs[3].space.dim]


def note(**kw):
    """One machine-readable JSON line; a killed child still leaves its trail behind."""
    print(json.dumps(kw), flush=True)


def measure(name, cutoff):
    """One (input, cutoff) measurement, in this process, under this process's caps."""
    t0 = time.perf_counter()
    norb, nelec, recs = synthetic(int(name[4:])) if name.startswith("syn-") else fetch(name)
    screen = 1e-6 if name.startswith("syn-") else SCREEN
    raw = spin_orbital_terms(recs, screen=screen)
    folded, refused = fold_terms(raw)
    terms = to_tenet_terms(folded)
    n_sites = 2 * norb
    t_terms = time.perf_counter() - t0
    note(
        event="input", name=name, k=norb, nelec=nelec, sites=n_sites, raw_terms=len(raw),
        terms=len(terms), refused=refused, screen=screen, t_terms=t_terms, rss=rss_gib(),
    )  # fmt: skip

    note(event="phase", name=name, cutoff=cutoff, phase="from_terms", status="start")
    with Phases() as ph:
        t1 = time.perf_counter()
        h = MPO.from_terms(n_sites, terms, cutoff=cutoff)
        t2 = time.perf_counter()
    note(
        event="phase", name=name, cutoff=cutoff, phase="from_terms", status="done",
        t=t2 - t1, rss=rss_gib(),
    )  # fmt: skip
    extra = {}
    if cutoff is None:
        # The consumer of the description, and what makes the ``cutoff=None`` row a
        # like-for-like comparison across #200: before it, ``from_terms`` built every
        # site *and* every block table; after it, it builds neither and this is where the
        # block tables the sweep actually needs get built. On the base commit the loop
        # hands back the tables ``_instantiate`` already made and costs nothing.
        note(event="phase", name=name, cutoff=cutoff, phase="edge_blocks", status="start")
        t3 = time.perf_counter()
        for site in range(n_sites):
            h.edge_blocks(site)
        extra = {"t_edge_blocks": time.perf_counter() - t3}
    sweeps = {}
    if cutoff is not None and ph.t_sweep1_start and ph.t_sweep2_start:
        sweeps = {
            "t_sweep1": ph.t_sweep2_start - ph.t_sweep1_start,
            "t_sweep2": t2 - ph.t_sweep2_start,
        }
    note(
        event="measure", name=name, k=norb, cutoff=cutoff, sites=n_sites, terms=len(terms),
        refused=refused, t_terms=t_terms, t_total=t2 - t1, t_term_edges=ph.t["term_edges"],
        t_instantiate=ph.t["instantiate"], t_edge_table=ph.t["edge_table"],
        t_braids=ph.t["braids"], braid_calls=ph.braid_calls,
        rss=rss_gib(), widths=bond_widths(h), **sweeps, **extra,
    )  # fmt: skip


def payload(obj, seen):
    """Bytes of distinct array buffers reachable from ``obj``; a shared buffer counts once.

    Reads ``SymmetricTensor.blocks`` on purpose. That is forbidden *inside*
    ``src/tenet/network/`` (``tests/network/test_hygiene.py``) and this is a benchmark,
    not the package: what a cache costs is the size of the arrays it keeps alive, and
    nothing above the block layer can say that. ``seen`` is threaded across buckets so a
    tensor a downstream cache merely *references* is charged to the bucket that walked it
    first -- which is the whole question #202 asks, since ``Env._cores`` holds several of
    ``EdgeBlocks``' own tensors by reference rather than by copy.
    """
    if isinstance(obj, tenet.SymmetricTensor):
        total = 0
        for block in obj.blocks:
            if id(block) not in seen:
                seen.add(id(block))
                total += block.nbytes
        return total
    if isinstance(obj, dict):
        return sum(payload(v, seen) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(payload(v, seen) for v in obj)
    return 0


def cache_bytes(h, env):
    """The buckets #202 splits ``18.9 GiB`` into, in GiB, exclusive then own.

    ``exclusive`` walks the buckets in the order the data flows -- the block tables, then
    the group embeddings they contain, then ``Env``'s merged cores, prepared operators and
    environments -- charging each buffer to the first bucket that reaches it, so the
    numbers sum to the resident total. ``own`` re-walks each bucket alone, so the gap
    between the two is exactly the sharing: the cores' ``own`` counts tensors the block
    tables already hold.
    """
    tab, seen = h.edges, set()
    buckets = {
        "edge_blocks": tab._table,
        "embeddings": tab._embeds,
        "identities": tab._identities,
        "cores": env._cores,
        "prepared": env._prepared,
        "environments": env.F,
        "mpo_sites": h._sites,
    }
    out = {f"gib_{k}": payload(v, seen) / 2**30 for k, v in buckets.items()}
    out.update({f"own_{k}": payload(v, set()) / 2**30 for k, v in buckets.items()})
    out["gib_cached"] = sum(v for k, v in out.items() if k.startswith("gib_"))
    out["entries"] = {k: len(v) for k, v in buckets.items()}
    return out


def _counting(fn, tally, key):
    """``fn``, counting its calls into ``tally[key]`` -- what an eviction actually costs."""

    def wrapper(*a, **kw):
        tally[key] += 1
        return fn(*a, **kw)

    return wrapper


def dmrg_run(name, chi, sweeps, budget):
    """A **full** DMRG run at ``cutoff=None`` -- the consumer #202 is about.

    ``sweep_`` is driven here rather than through ``dmrg_`` for one reason: ``dmrg_``
    builds its own [Env][tenet.network.Env] and never hands it back, and the caches this
    measures live on that object. The loop is ``dmrg_``'s own, minus the convergence test,
    so the sweep count is fixed and the two sides of a before/after comparison do the same
    arithmetic. ``budget`` overrides ``common.CACHE_BUDGET`` for the process, which is the
    whole policy (#202) and therefore the only knob.

    Rebuilds are counted rather than inferred: what a budget costs is how many block
    tables and merged cores the sweep has to build a second time, and that count is what
    the wall-clock difference is made of.
    """
    from tenet.network import Env, common, sweep_
    from tenet.network import env as env_module

    common.CACHE_BUDGET = budget
    builds = {"tables": 0, "cores": 0}
    mps.EdgeTable._build_table = _counting(mps.EdgeTable._build_table, builds, "tables")
    env_module._cores2 = _counting(env_module._cores2, builds, "cores")
    t0 = time.perf_counter()
    norb, nelec, recs = synthetic(int(name[4:])) if name.startswith("syn-") else fetch(name)
    screen = 1e-6 if name.startswith("syn-") else SCREEN
    terms = to_tenet_terms(fold_terms(spin_orbital_terms(recs, screen=screen))[0])
    n_sites = 2 * norb
    h = MPO.from_terms(n_sites, terms, cutoff=None)
    note(
        event="dmrg_build", name=name, k=norb, sites=n_sites, chi=chi, budget=budget,
        t=time.perf_counter() - t0, rss=rss_gib(), widths=bond_widths(h),
    )  # fmt: skip

    phys = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
    triv = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
    mid = GradedSpace.new(fZ2, {FZ2Sector(0): chi - chi // 2, FZ2Sector(1): chi // 2})
    psi = MPS.random(phys, [triv] + [mid] * (n_sites - 1) + [triv], seed=0)
    t1 = time.perf_counter()
    psi.canonize_(0)
    env = Env(psi, h).setup_(0)
    note(event="dmrg_setup", name=name, budget=budget, t=time.perf_counter() - t1, rss=rss_gib(),
         **cache_bytes(h, env))  # fmt: skip

    schmidt, energies = {}, []
    for it in range(sweeps):
        t2 = time.perf_counter()
        energy, _ = sweep_(psi, h, env, schmidt, chi=chi, cutoff=1e-10)
        energies.append(energy)
        note(event="dmrg_sweep", name=name, budget=budget, sweep=it, energy=energy,
             t=time.perf_counter() - t2, rss=rss_gib(), **cache_bytes(h, env))  # fmt: skip
    note(
        event="dmrg", name=name, k=norb, sites=n_sites, chi=chi, sweeps=sweeps, budget=budget,
        energy=energies[-1], energies=energies, t_total=time.perf_counter() - t1, **builds,
        rss=rss_gib(), **cache_bytes(h, env),
    )  # fmt: skip


def combinatorial(name):
    """The FSM/MVC cross-check for one input; no tenet assembly, so it reaches large K."""
    t0 = time.perf_counter()
    norb, nelec, recs = synthetic(int(name[4:])) if name.startswith("syn-") else fetch(name)
    screen = 1e-6 if name.startswith("syn-") else SCREEN
    raw = spin_orbital_terms(recs, screen=screen)
    n_sites = 2 * norb
    dropped, refused = fold_terms(raw, premultiply=False)
    kept, _ = fold_terms(raw, premultiply=True)
    out = {"event": "combinatorial", "name": name, "k": norb, "nelec": nelec, "sites": n_sites}
    for label, tl in (("premul", kept), ("drop", dropped)):
        out[f"{label}_terms"] = len(tl)
        out[f"{label}_fsm"] = fsm_profile(n_sites, tl)
        out[f"{label}_mvc"] = mvc_profile(n_sites, tl)
    out["refused"] = refused
    out["raw_terms"] = len(raw)
    out["t"] = time.perf_counter() - t0
    note(**out)


# --- the harness --------------------------------------------------------------------


def child(argv, mem):
    """``--child KIND NAME [CUTOFF]``: one measurement under this process's own caps."""
    kind, name = argv[0], argv[1]
    try:
        resource.setrlimit(resource.RLIMIT_AS, (int(mem * 2**30),) * 2)
    except (ValueError, OSError) as exc:  # not lowerable on every platform; say so
        note(event="warn", detail=f"RLIMIT_AS not enforced: {exc}")
    sys.setrecursionlimit(100000)
    if kind == "comb":
        combinatorial(name)
    elif kind == "dmrg":
        chi, sweeps, budget = int(argv[2]), int(argv[3]), int(argv[4])
        dmrg_run(name, chi, sweeps, budget)
    else:
        measure(name, None if argv[2] == "none" else float(argv[2]))


def run_capped(args, wall, mem):
    """Spawn one child; return its JSON lines and how it ended."""
    here = str(pathlib.Path(__file__).resolve())
    cmd = [sys.executable, here, "--mem", str(mem), "--child", *args]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=wall, check=False)
        out, status = p.stdout, ("ok" if p.returncode == 0 else f"exit {p.returncode}")
        if p.returncode != 0:
            tail = (p.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
            status = f"{status}: {tail[0][:120]}"
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        status = f"wall cap {wall:.0f}s exceeded"
    rows = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    return rows, status, time.perf_counter() - t0


def fmt(xs, width=5):
    return " ".join("--" if x is None else f"{x:>{width}}" for x in xs)


def _dmrg_table(names, a):
    """``--dmrg``: one full run per (input, budget), and the trade printed as a table.

    The budget is #202's whole policy, so it is the only column that varies: the same
    Hamiltonian, the same ``chi``, the same sweep count and the same seed on every row,
    with a budget of a terabyte standing for the unbounded cache the issue is about.
    ``blocks``/``embed``/``cores``/``prep`` are what is *still held* at the end of the run
    and ``builds``/``rebuilt`` are how many block tables and merged cores the run had to
    make -- the cost side of the trade, counted rather than inferred from the clock.
    """
    print(f"# machine   {platform.platform()} / {platform.processor() or platform.machine()}")
    print(f"# dmrg      chi={a.chi}, {a.sweeps} sweeps, cutoff=None operator, fZ2 spin orbitals")
    print(f"# caps      wall {a.wall:.0f} s and address space {a.mem:.1f} GiB per run")
    header = (
        f"{'input':<14} {'GiB cap':>8} {'peak GiB':>9} {'cached':>7} {'blocks':>7} "
        f"{'embed':>7} {'cores':>7} {'prep':>6} {'wall s':>8} {'builds':>7} "
        f"{'rebuilt':>8} {'energy':>16}"
    )
    for name in names:
        print(f"\n== {name}")
        print(header)
        for gib in a.budgets:
            args = ["dmrg", name, str(a.chi), str(a.sweeps), str(int(gib * 2**30))]
            rows, status, wall = run_capped(args, a.wall, a.mem)
            row = next((r for r in rows if r.get("event") == "dmrg"), None)
            if row is None:
                last = [r for r in rows if r.get("event", "").startswith("dmrg_")]
                where = last[-1]["event"] if last else "term construction"
                print(
                    f"{name:<14} {gib:>8.2f} DID NOT FINISH after {where}: {status} ({wall:.1f}s)"
                )
                continue
            print(
                f"{name:<14} {gib:>8.2f} {row['rss']:>9.2f} {row['gib_cached']:>7.2f} "
                f"{row['gib_edge_blocks']:>7.2f} {row['gib_embeddings']:>7.2f} "
                f"{row['gib_cores']:>7.2f} {row['gib_prepared']:>6.2f} "
                f"{row['t_total']:>8.1f} {row['tables']:>7} {row['cores']:>8} "
                f"{row['energy']:>16.9f}"
            )
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("--wall", type=float, default=WALL_CAP_S, help="per-measurement cap, seconds")
    ap.add_argument("--mem", type=float, default=MEM_CEILING_GIB, help="per-measurement cap, GiB")
    ap.add_argument("--synthetic-only", action="store_true", help="no network access at all")
    ap.add_argument("--only", nargs="*", help="run just these inputs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-check", action="store_true", help="the model-vs-assembler check")
    ap.add_argument("--dmrg", action="store_true", help="full DMRG runs, the #202 cache trade")
    ap.add_argument("--chi", type=int, default=16, help="--dmrg bond dimension")
    ap.add_argument("--sweeps", type=int, default=3, help="--dmrg sweeps per run")
    ap.add_argument(
        "--budgets", type=float, nargs="*", default=[1024.0, 4.0, 1.0, 0.25],
        help="--dmrg cache byte budgets in GiB; a huge one is the unbounded cache",
    )  # fmt: skip
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # a run this long has to be watchable live

    if a.child:
        return child(a.child, a.mem)
    if a.self_check:
        _self_check()
        print("self-check ok: the combinatorial FSM profile is the assembler's own bond")
        return None

    names = [] if a.synthetic_only else list(REAL)
    names += [f"syn-{k}" for k in (8, 16, 24, 32)]
    if a.only:
        names = [n for n in names if n in a.only]
    if a.list:
        print("\n".join(names))
        return None

    if a.dmrg:
        return _dmrg_table(names, a)

    print(f"# machine   {platform.platform()} / {platform.processor() or platform.machine()}")
    print(
        f"# python    {platform.python_version()}, numpy {np.__version__}, "
        f"tenet {tenet.__version__}"
    )
    print(f"# caps      wall {a.wall:.0f} s and address space {a.mem:.1f} GiB per measurement")
    print(f"# criteria  practical chi {CHI_PRACTICAL}, memory ceiling {a.mem:.1f} GiB")
    print(f"# screening |integral| > {SCREEN:g} (real), 1e-06 (synthetic)\n")

    records, summary = [], []
    for name in names:
        rows, status, wall = run_capped(["comb", name], a.wall, a.mem)
        comb = next((r for r in rows if r.get("event") == "combinatorial"), None)
        print(f"== {name}")
        if comb is None:
            print(f"   combinatorial: {status} ({wall:.1f} s)")
            continue
        share = 100.0 * comb["refused"] / max(comb["raw_terms"], 1)
        print(
            f"   K={comb['k']} NELEC={comb['nelec']} sites={comb['sites']} "
            f"terms={comb['premul_terms']} (raw {comb['raw_terms']}, same-site refused "
            f"{comb['refused']} = {share:.0f}%)   combinatorial pass {comb['t']:.1f} s"
        )
        records.append(comb)

        runs, fail = {}, {}
        for cutoff in ("none", "1e-13"):
            rows, status, wall = run_capped(["run", name, cutoff], a.wall, a.mem)
            m = next((r for r in rows if r.get("event") == "measure"), None)
            if m is None:
                phase = [r for r in rows if r.get("event") == "phase"]
                where = phase[-1]["phase"] if phase else "term construction"
                if phase and phase[-1].get("status") == "done":
                    where = f"after {where}"
                fail[cutoff] = where
                print(f"   cutoff={cutoff:<5} DID NOT FINISH in {where}: {status} ({wall:.1f} s)")
                records.append({"event": "measure", "name": name, "k": comb["k"],
                                "cutoff": cutoff, "status": status, "stopped_in": where,
                                "wall": wall})  # fmt: skip
                continue
            runs[cutoff] = m
            sw = ""
            if "t_sweep1" in m:
                sw = f" sweep1 {m['t_sweep1']:.1f} sweep2 {m['t_sweep2']:.1f}"
            if "t_edge_blocks" in m:
                sw = f" edge_blocks {m['t_edge_blocks']:.1f}"
            print(
                f"   cutoff={cutoff:<5} widest {max(m['widths']):<6} {m['t_total']:>7.1f} s "
                f"(terms {m['t_terms']:.1f} _term_edges {m['t_term_edges']:.1f} "
                f"_edge_table {m.get('t_edge_table', 0.0):.1f} "
                f"_instantiate {m['t_instantiate']:.1f}{sw})   peak RSS {m['rss']:.2f} GiB   "
                f"_braids_with_signs {m['t_braids']:.2f} s / {m['braid_calls']} calls "
                f"({100 * m['t_braids'] / max(m['t_total'], 1e-9):.1f}%)"
            )
            records.append(m)

        # The three widths per cut, side by side: the FSM's own bond, the independently
        # computed minimum vertex cover, and what the two compressing SVD sweeps left.
        fsm = runs["none"]["widths"] if "none" in runs else comb["premul_fsm"]
        mvc = comb["premul_mvc"]
        svd = runs["1e-13"]["widths"] if "1e-13" in runs else [None] * len(mvc)
        print(f"   FSM  {fmt(fsm)}")
        print(f"   MVC  {fmt(mvc)}")
        print(f"   SVD  {fmt(svd)}")
        pairs = [(f, m) for f, m in zip(fsm, mvc, strict=True) if m]
        wf = max(fsm)
        wm = max(m for _, m in pairs) if pairs else None
        ws = max((x for x in svd if x is not None), default=None)
        line = f"   widest  FSM {wf}"
        if wm:
            line += f"   MVC {wm} ({wf / wm:.2f}x)"
        if ws:
            line += f"   post-SVD {ws}" + (f" ({ws / wm:.2f}x MVC)" if wm else "")
        print(line)
        # The same numbers under the model the issue was drafted with: same-site terms
        # dropped rather than pre-multiplied, so the two tables can be compared directly.
        dfsm, dmvc = comb["drop_fsm"], comb["drop_mvc"]
        dm = max((x for x in dmvc if x), default=None)
        print(
            f"   drop-refused model: terms {comb['drop_terms']}  widest FSM {max(dfsm)}"
            + (f"  widest MVC {dm}  ratio {max(dfsm) / dm:.2f}x" if dm else "")
        )
        summary.append((name, comb, runs, fail, wf, wm, ws))
        print()

    print("# --- the three K values ---------------------------------------------------")
    first = {}
    for name, comb, runs, fail, wf, _wm, _ws in sorted(summary, key=lambda r: r[1]["k"]):
        k = comb["k"]
        if wf > CHI_PRACTICAL and "chi" not in first:
            first["chi"] = f"K={k} ({name}): FSM bond {wf} > practical chi {CHI_PRACTICAL}"
        peak = max((m["rss"] for m in runs.values()), default=0.0)
        if peak > a.mem and "mem" not in first:
            first["mem"] = f"K={k} ({name}): peak RSS {peak:.1f} GiB > ceiling {a.mem:.1f} GiB"
        # A run that died before the sweeps ever started did not fail *in* the sweeps,
        # and saying so would put the wall in the wrong phase.
        if "sweep" in fail.get("1e-13", "") and "sweep" not in first:
            first["sweep"] = f"K={k} ({name}): stopped in {fail['1e-13']}"
        if fail.get("1e-13") and "before" not in first:
            first["before"] = f"K={k} ({name}): cutoff=1e-13 stopped in {fail['1e-13']}"
    for key, label in (
        ("chi", f"FSM bond first exceeds chi={CHI_PRACTICAL}"),
        ("mem", f"peak memory first exceeds {a.mem:.1f} GiB"),
        ("sweep", f"compressing sweeps first fail to finish in {a.wall:.0f} s"),
        ("before", "from_terms first fails to finish at all, and where"),
    ):
        print(f"  {label:<52} {first.get(key, 'not reached in this input set')}")
    print()

    out = CACHE.parent / "bench_qc_mpo.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"# machine-readable rows -> {out}")
    return None


def _self_check():
    """The smallest thing that fails if the combinatorial model drifts from ``from_terms``.

    Four spinless-fermion terms on six sites: the FSM profile predicted combinatorially
    must equal the bond the real assembler leaves at ``cutoff=None``, and the cover must
    never exceed it.
    """
    pairs = [(i, j) for i in range(6) for j in range(6) if i != j]
    cd, c = _label(MAT["cd"], True), _label(MAT["c"], True)
    symbolic = [(1.0, ((cd, i), (c, j))) for i, j in pairs]
    h = MPO.from_terms(6, to_tenet_terms(symbolic), cutoff=None)
    predicted = fsm_profile(6, symbolic)
    assert bond_widths(h) == predicted, (bond_widths(h), predicted)
    cover = mvc_profile(6, symbolic)
    assert all(m <= f for m, f in zip(cover, predicted, strict=True)), (cover, predicted)


if __name__ == "__main__":
    main()
