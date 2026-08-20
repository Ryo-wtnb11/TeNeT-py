"""Gate 1 for #222: the **pre-placement** bond width against #218's post-placement one.

#218 (M39) chooses each cut's bond basis *after* ``_place`` has scattered the finite-state
machine into a ``D_FSM x d**2 x chi`` buffer. block2 chooses it before anything numeric
exists (``general_mpo.hpp`` Part 2, :666-780), so no wide object is ever formed. Whether
the two choices agree on the *width* is not established -- today's SVD sees the numerical
rank of placed tensors, a pre-placement SVD sees only the scalar structure of the cut's
coefficients -- and #222 does not proceed past the measurement.

**This script is that measurement.** It carries a self-contained pre-placement assembler
(``stream``) that never builds a rank-4 tensor to choose a basis:

* the coefficient stream is carried forward on the *kept* basis, so the rows of every
  decomposed matrix are ``(kept basis) x (site fan-out)`` and never ``D_FSM``;
* both sides are interned per cut -- left rows by ``(kept index, site slot)``, right
  columns by the remaining operator string -- and the SVD runs dense on the resulting
  ``szl x szr`` block, per quantum number;
* the singular values are folded into the *next* cut's coefficient stream rather than
  into a tensor, so ``U`` is the site's factors and ``S.Vh`` is the new stream;
* ``IdL`` and ``IdR`` are excluded from the choice and given their own slots, which is
  #218's corner pinning applied one stage earlier.

That is block2's shape, described from ``general_mpo.hpp``:540-541, :666-780, :723-730,
:763, :764-805, :828-833, :1210-1214 and :1381+ and written independently; block2 is
GPL-3.0 and no line of it is reproduced here.

Gate: ``max(pre / post)`` at or below 1.10 on every fixture, reported for the inner cuts
and for the two boundary cuts separately. #218's gate failed on its all-cuts reading for a
structural reason -- the free sweep's boundary width sat *below* the minimum vertex cover,
which no partitioned operator can reach -- so every reading is printed and the verdict
names the one it rests on.

**Three widths of one mechanism**, because the basis choice and the truncation policy are
two separate things and only the first is what #222 is about:

* ``pre`` -- the basis chosen *and truncated* inside the pre-placement SVD. This is the
  gate's literal reading.
* ``exact`` -- the same choice at a rank-revealing threshold: the width ``_place`` would
  be handed, and the number the ``D_FSM`` claim is about. Printed beside ``D_FSM``.
* ``swept`` -- ``exact`` after #218's own two pinned truncating sweeps, which on the
  pre-placement bond decompose a ``chi x d**2 x chi`` tensor rather than a
  ``D_FSM x d**2 x chi`` one.

``pre`` and ``swept`` differ for one reason and it is not the basis: ``rsum2`` weighs a
discarded direction against the norm of the matrix it decomposes, and the *coefficient*
matrix's norm is not the placed operator's. The scalar stream carries each slot's
Frobenius norm so the two metrics agree wherever the string basis is orthogonal, and a
dense synthetic integral set is exactly where it is not.

The width is worthless if the operator is wrong, so every fixture small enough to expand
is checked with ``to_dense`` against ``from_terms`` at ``cutoff=None`` and at ``1e-13``
before any width is believed.

Restrictions, stated rather than hidden: the prototype handles rank-3 term operators on an
abelian grading (``qdim == 1``), which is every fixture below. A k-site operator's split
pieces and a non-abelian ``qdim`` weight are not covered, and the first of those is a
design question rather than an implementation gap -- see ``docs/design.md`` "M55".

Not a test, not part of the package, on no CI path. It reuses ``bench_qc_mpo.py``'s
FCIDUMP fetch, its synthetic generator and its term folding unchanged, so the licence
decision recorded in that module's docstring covers this one too. Run from the repo root:

    uv run python benchmarks/bench_preplace_mpo.py
    uv run python benchmarks/bench_preplace_mpo.py --only H4.STO6G.R1.8
    uv run python benchmarks/bench_preplace_mpo.py --synthetic-only
"""

import argparse
import json
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np

import tenet
from tenet import IN, OUT, GradedSpace, Leg
from tenet.network import MPO
from tenet.network.mps import (
    _aligned,
    _as_w,
    _canonical_term,
    _compress_forward,
    _corner_map,
    _corner_slots,
    _identity_w,
    _joined,
    _Walk,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bench_pinned_mpo as pin  # noqa: E402
import bench_qc_mpo as qc  # noqa: E402

# --- the pre-placement basis choice --------------------------------------------------


def _rows(n_sites, terms):
    """``(walk, [(slots, sites, coeff), ...])`` -- ``from_terms``' own canonicalization.

    The same ``_canonical_term`` the shipped builder runs, so the pre-placement stream
    starts from bit-identical integer rows: the sort into site order and the Koszul sign
    it costs are already paid, and a width difference can only come from the basis choice.
    """
    phys = None
    for _c, ops in terms:
        for op, _s in ops:
            if op.ndim != 3:
                raise ValueError("the pre-placement prototype takes rank-3 term operators only")
            phys = op.legs[1].space
    walk = _Walk(n_sites, phys, False)

    def split(op):
        raise ValueError("no k-site split in this prototype")

    out = []
    for coeff, ops in terms:
        row, where, c = _canonical_term(walk, n_sites, coeff, list(ops), split)
        out.append((tuple(row), tuple(where), c))
    return walk, out


def _keep(blocks, cutoff, mode):
    """How many singular values each block keeps, in one of ``svd_truncated``'s modes.

    ``blocks`` maps a quantum number to its ``(u, s, vh)``. ``"rsum2"`` pools the spectrum
    across quantum numbers and drops the largest tail whose summed ``sigma**2`` stays
    under ``cutoff * sum(sigma**2)``, which is ``ops/linalg.py``'s ``_admissible`` at
    ``qdim == 1``; ``"rel"`` keeps ``sigma > cutoff * sigma_max``, which at a tight cutoff
    is a rank reveal rather than a truncation.
    """
    spectrum = sorted(
        ((float(sig), q, i) for q, (_u, s, _v) in blocks.items() for i, sig in enumerate(s)),
        key=lambda e: -e[0],
    )
    if mode == "rel":
        kept = sum(1 for sig, _q, _i in spectrum if sig > cutoff * spectrum[0][0])
    else:
        weights = [sig**2 for sig, _q, _i in spectrum]
        threshold = cutoff * sum(weights)
        kept, dropped = len(spectrum), 0.0
        for w in reversed(weights):
            if dropped + w >= threshold:
                break
            dropped += w
            kept -= 1
    counts: dict = {}
    for _sig, q, _i in spectrum[:kept]:
        counts[q] = counts.get(q, 0) + 1
    return counts


class _Cut:
    """One cut's chosen basis: its quantum numbers, its corners and the site factors.

    ``q`` is the quantum number of every kept basis element in the cut's own order
    (``IdL`` first, the open block next, ``IdR`` last -- ``_merge``'s order, so the
    compressed description could reuse it). ``fac[o]`` is the ``(kept left, kept right)``
    coefficient matrix of the site's slot ``o``, which is ``U`` for the open columns,
    the delayed identity for the ``IdL`` column and the closing coefficients for ``IdR``.
    """

    __slots__ = ("fac", "idl", "idr", "q", "slots")

    def __init__(self, q, idl, idr, slots, fac):
        self.q = q
        self.idl = idl
        self.idr = idr
        self.slots = slots
        self.fac = fac


def stream(n_sites, walk, rows, cutoff, *, mode="rsum2", factors=False):
    """Choose every cut's bond basis from the coefficient structure, left to right.

    Parameters
    ----------
    n_sites : int
        Chain length.
    walk : _Walk
        The interning walk ``_rows`` filled; ``walk.ops`` is the slot table.
    rows : list
        ``(slots, sites, coeff)`` per term, in site order.
    cutoff : float
        ``svd_truncated``'s ``rsum2`` cutoff, applied to the pooled spectrum.
    factors : bool, optional
        Keep the per-site coefficient matrices so the operator can be built and
        checked. Default ``False`` -- the width measurement does not need them and
        they are the one object that grows as ``chi**2``.

    Returns
    -------
    list of _Cut
        One entry per cut, ``n_sites + 1`` of them.

    Notes
    -----
    The stream at a cut is ``V[x, R]``: the coefficient the kept basis element ``x``
    carries for the remaining operator string ``R``. Site ``n`` splits every ``R`` into
    its site-``n`` slot and the rest, which turns ``V`` into ``M[(x, o), R']`` -- rows
    ``(kept basis) x (site fan-out)``, columns the remaining strings interned at the next
    cut. ``M`` is decomposed per quantum number with the ``IdL`` row and the ``IdR``
    column taken out first, ``U`` becomes site ``n``'s factors and ``S.Vh`` becomes the
    next cut's stream. No rank-4 tensor and no ``D_FSM``-wide object is built anywhere in
    here.
    """
    sym = walk.phys.provider
    unit = sym.unit
    root_d = walk.phys.dim**0.5
    slot_q = [op.legs[2].space.sectors[0][0] for op, _p in walk.ops]
    # The metric the truncation is taken in. ``rsum2`` weighs a discarded direction
    # against the norm of the matrix it decomposes, so the scalar matrix has to carry the
    # operators' own norms or the cutoff measures the wrong thing: a scalar row of
    # coefficients says nothing about how much operator weight it stands for. Each slot's
    # weight is its Frobenius norm relative to the identity's, so the identity is 1 and
    # the whole-chain factor ``d**(N/2)`` -- common to every column -- cancels out.
    slot_w = [float(np.linalg.norm(np.asarray(op.to_dense()))) / root_d for op, _p in walk.ops]

    intern: dict = {}
    strings: list = []
    str_q: list = []
    str_w: list = []
    acc: dict = {}
    for slots, sites, coeff in rows:
        s = tuple(zip(sites, slots, strict=True))
        j = intern.get(s)
        if j is None:
            q, w = unit, 1.0
            for slot in slots:
                q = sym.fusion(q, slot_q[slot])[0]
                w *= slot_w[slot]
            j = intern[s] = len(strings)
            strings.append(s)
            str_q.append(q)
            str_w.append(w)
        acc[j] = acc.get(j, 0.0) + coeff
    v = np.zeros((1, len(strings)))
    for j, c in acc.items():
        v[0, j] = c
    x_q = [unit]
    idl = 0
    cuts = [_Cut([unit], 0, None, [], {})]

    for n in range(n_sites):
        # --- split every remaining string on this site: heads become rows, tails columns
        intern2: dict = {}
        strings2: list = []
        str_q2: list = []
        w2: list = []
        head = np.empty(len(strings), dtype=np.intp)
        kid = np.empty(len(strings), dtype=np.intp)
        for j, s in enumerate(strings):
            if s and s[0][0] == n:
                o = s[0][1]
                rest = s[1:]
                q = sym.fusion(sym.dual(slot_q[o]), str_q[j])[0]
                w = str_w[j] / slot_w[o]
            else:
                o, rest, q, w = -1, s, str_q[j], str_w[j]
            k = intern2.get(rest)
            if k is None:
                k = intern2[rest] = len(strings2)
                strings2.append(rest)
                str_q2.append(q)
                w2.append(w)
            head[j] = o
            kid[j] = k
        slots = sorted(set(head.tolist()))
        col_idr = intern2.get(())
        cw = np.array(w2)
        rw = {o: (1.0 if o < 0 else slot_w[o]) for o in slots}

        # --- the rows and columns of each quantum-number block, interned per cut
        rows_of: dict = {}
        runs: dict = {}
        for o in slots:
            qo = unit if o < 0 else slot_q[o]
            for x, qx in enumerate(x_q):
                q = sym.fusion(qx, qo)[0]
                runs.setdefault((q, o), []).append(x)
                rows_of.setdefault(q, []).append((x, o))
        cols_of: dict = {}
        for k, q in enumerate(str_q2):
            cols_of.setdefault(sym.dual(q), []).append(k)
        row_at = {q: {p: i for i, p in enumerate(pairs)} for q, pairs in rows_of.items()}
        col_at = {q: np.full(len(strings2), -1, dtype=np.intp) for q in cols_of}
        for q, ks in cols_of.items():
            col_at[q][np.array(ks, dtype=np.intp)] = np.arange(len(ks))

        # --- M, block by block; the only dense object here is szl x szr scalars
        mats = {q: np.zeros((len(rows_of[q]), len(cols_of[q]))) for q in rows_of if q in cols_of}
        col_blk = [sym.dual(q) for q in str_q2]
        for o in slots:
            src = np.flatnonzero(head == o)
            if not src.size:
                continue
            by_block: dict = {}
            for j in src.tolist():
                by_block.setdefault(col_blk[int(kid[j])], []).append(j)
            for q, js in by_block.items():
                m = mats.get(q)
                if m is None:
                    continue
                xs = np.array(runs[(q, o)], dtype=np.intp)
                start = row_at[q][(int(xs[0]), o)]
                cols = np.array(js, dtype=np.intp)
                ck = col_at[q][kid[cols]]
                m[start : start + xs.size, ck] = v[np.ix_(xs, cols)] * (rw[o] * cw[kid[cols]])

        # --- pin the two corners out of the choice, block2 :764-805's position
        idl_row = None if idl is None else row_at.get(unit, {}).get((idl, -1))
        idr_col = None if col_idr is None else int(col_at[unit][col_idr])
        idl_vec = np.zeros(len(strings2))
        idr_vec: dict = {}
        if idl_row is not None:
            m = mats[unit]
            take = col_at[unit] >= 0
            idl_vec[take] = m[idl_row, col_at[unit][take]]
            if idr_col is not None:
                idl_vec[col_idr] = 0.0
            m[idl_row, :] = 0.0
        if idr_col is not None:
            m = mats[unit]
            idr_vec = {p: m[i, idr_col] for p, i in row_at[unit].items() if m[i, idr_col]}
            m[:, idr_col] = 0.0

        # --- the choice itself: one dense SVD per quantum number
        blocks = {}
        for q, m in mats.items():
            if m.size and np.any(m):
                blocks[q] = np.linalg.svd(m, full_matrices=False)
        counts = _keep(blocks, cutoff, mode) if blocks else {}

        # --- the kept basis, in _merge's order, and the stream rebuilt on it, unweighted
        has_l = idl_row is not None and bool(np.any(idl_vec))
        has_r = idr_col is not None
        chi = int(has_l) + sum(counts.values()) + int(has_r)
        v2 = np.zeros((chi, len(strings2)))
        x_q2: list = []
        fac: dict = {}
        at = 0
        if has_l:
            v2[0] = idl_vec / cw
            x_q2.append(unit)
            if factors:
                fac.setdefault(-1, np.zeros((len(x_q), chi)))[idl, 0] = 1.0
            at = 1
        for q in sorted(counts, key=str):
            u, s, vh = blocks[q]
            k = counts[q]
            cols = np.array(cols_of[q], dtype=np.intp)
            v2[at : at + k, cols] = (s[:k, None] * vh[:k]) / cw[cols]
            x_q2.extend([q] * k)
            if factors:
                for (x, o), i in row_at[q].items():
                    if np.any(u[i, :k]):
                        got = fac.setdefault(o, np.zeros((len(x_q), chi)))
                        got[x, at : at + k] = u[i, :k] / rw[o]
            at += k
        if has_r:
            v2[at, col_idr] = 1.0
            x_q2.append(unit)
            if factors:
                scale = cw[col_idr]
                for (x, o), val in idr_vec.items():
                    fac.setdefault(o, np.zeros((len(x_q), chi)))[x, at] = val / (rw[o] * scale)
        cuts[-1].slots = slots
        cuts[-1].fac = fac
        cuts.append(_Cut(x_q2, 0 if has_l else None, (chi - 1) if has_r else None, [], {}))
        strings, str_q, str_w, v, x_q = strings2, str_q2, w2, v2, x_q2
        idl = 0 if has_l else None
    return cuts


def widths(cuts):
    """Each cut's dense bond width -- ``qdim == 1``, so the kept count is the dimension."""
    return [len(c.q) for c in cuts]


# --- turning the chosen basis into the operator, for the correctness half ------------


def build(n_sites, walk, cuts):
    """The site tensors on the chosen bases; the buffer is ``chi x d**2 x chi``.

    One ``from_dense`` per site against the *chosen* basis, exactly as ``_place`` scatters
    against the finite-state machine's -- so this is where the ``D_FSM`` factor would have
    been and is not.

    Returns
    -------
    tuple
        ``(sites, cuts)`` where ``cuts`` is ``_instantiate``'s own
        ``(IdL live, the open block's space or None, IdR live)`` per cut, so the shipped
        pinned sweeps can run on the result.
    """
    sym, phys, d = walk.phys.provider, walk.phys, walk.phys.dim

    def cidx(c):
        if c not in walk.charges:
            walk.charges[c] = len(walk.sectors)
            walk.sectors.append(c)
        return walk.charges[c]

    atom: dict = {}

    def block(o, qx):
        got = atom.get((o, qx))
        if got is None:
            if o < 0:
                one = GradedSpace.new(sym, {qx: 1})
                got = np.asarray(_identity_w(Leg(one, IN), phys).to_dense())[0, :, :, 0]
            else:
                got = np.asarray(walk.transition(o, cidx(qx), 0)[0].to_dense())[0, :, :, 0]
            atom[o, qx] = got
        return got

    spaces, pos, shapes, by_q = [], [], [], []
    for cut in cuts:
        counts: dict = {}
        opens: dict = {}
        seen: dict = {}
        where, groups = [], {}
        for q in cut.q:
            counts[q] = counts.get(q, 0) + 1
        space = GradedSpace.new(sym, counts)
        for i, q in enumerate(cut.q):
            where.append(space.sector_offset(q) + seen.get(q, 0))
            seen[q] = seen.get(q, 0) + 1
            groups.setdefault(q, []).append(i)
            if i not in (cut.idl, cut.idr):
                opens[q] = opens.get(q, 0) + 1
        spaces.append(space)
        pos.append(np.array(where, dtype=np.intp))
        by_q.append({q: np.array(xs, dtype=np.intp) for q, xs in groups.items()})
        shapes.append(
            (
                cut.idl is not None,
                GradedSpace.new(sym, opens) if opens else None,
                cut.idr is not None,
            )
        )

    sites = []
    axes = (np.arange(d), np.arange(d))
    for n in range(n_sites):
        left, right = cuts[n], cuts[n + 1]
        acc = np.zeros((len(left.q), d, d, len(right.q)))
        for o, mat in left.fac.items():
            for qx, rows in by_q[n].items():
                acc[rows] += np.einsum("xz,pq->xpqz", mat[rows], block(o, qx))
        buf = np.zeros((spaces[n].dim, d, d, spaces[n + 1].dim))
        buf[np.ix_(pos[n], *axes, pos[n + 1])] = acc
        legs = (Leg(spaces[n], IN), Leg(phys, OUT), Leg(phys, IN), Leg(spaces[n + 1], OUT))
        sites.append(tenet.SymmetricTensor.from_dense(buf, legs))
    return sites, shapes


def sweep_back(sites, cuts, cutoff):
    """``_instantiate``'s pinned backward truncation, on tensors that are already narrow.

    The mirror of the shipped ``_compress_forward``, and the half of #218's compression
    that ``_instantiate`` fuses into placement. On the pre-placement bond there is nothing
    to place, so it runs on plain site tensors: the two corner rows come out, the open row
    slab is rotated and truncated, and the block-diagonal carry goes into site ``n - 1``.
    """
    sym = sites[0].legs[1].space.provider
    unit = GradedSpace.new(sym, {sym.unit: 1})
    for n in reversed(range(1, len(sites))):
        has_l, _open, has_r = cuts[n]
        bond = sites[n].legs[0].space
        live = [g for g, on in (("idl", has_l), ("idr", has_r)) if on]
        slots = _corner_slots(bond, sym) if live else {}
        corners, rest = {}, sites[n]
        for g in live:
            take = _corner_map(bond, slots[g], (Leg(unit, IN), Leg(bond, OUT)), column=False)
            put = _corner_map(bond, slots[g], (Leg(bond, IN), Leg(unit, OUT)), column=True)
            corners[g] = tenet.einsum("xpqb,vx->vpqb", sites[n], take)
            rest = tenet.subtract(rest, tenet.einsum("vpqb,xv->xpqb", corners[g], put))
        u, s, vh = tenet.linalg.svd_truncated(rest, ((0,), (1, 2, 3)), cutoff=cutoff)
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
        sites[n] = _joined(rows, 0)
        carry = tenet.repartition(_joined(cols, 1), (), (0, 1))
        # ``_place`` folds this carry into its scatter; with the site already placed there
        # is nothing to scatter, so the fold is spelled the way ``_place`` spells it -- a
        # dense contraction on the right end, the new leg being the carry's second leg
        # moved to the codomain (same space, ``OUT``, ``dual`` flipped).
        left = sites[n - 1]
        dense = np.asarray(left.to_dense()) @ np.asarray(carry.to_dense())
        legs = (
            *left.legs[:3],
            Leg(carry.legs[1].space, OUT, not carry.legs[1].dual),
        )
        sites[n - 1] = tenet.SymmetricTensor.from_dense(dense, legs)
        cuts[n] = (has_l, open_w.legs[0].space, has_r)
    return sites


# --- the measurement ------------------------------------------------------------------


def rss_gib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20


def _buffer(left, right, d):
    """The widest ``_place`` buffer a sweep over these two bond profiles allocates, in GiB.

    ``_place`` scatters into ``(left bond) x d x d x (right bond)`` at ``float64``, one site
    at a time (M33's carry fold), so the transient the ``D_FSM`` claim is about is the
    largest of those products. Passing the finite-state machine's bonds and the compressed
    ones gives what #218 allocates; passing the pre-placement bonds twice gives what a
    pre-placement builder would.
    """
    widest = max(left[n] * d * d * right[n + 1] for n in range(len(right) - 1))
    return widest * 8 / 2**30


def measure(name, n_sites, terms, cutoff=1e-13, rank=1e-14, dense_check=False):
    """One fixture, three width readings of the same mechanism against #218's.

    * ``pre`` -- the basis chosen *and truncated* inside the pre-placement SVD, which is
      the literal reading of the gate;
    * ``exact`` -- the same choice at a rank-revealing threshold, which is the width
      ``_place`` would be handed and the number the ``D_FSM`` claim is about;
    * ``swept`` -- ``exact`` after #218's own two pinned truncating sweeps, run on the
      pre-placement bond where they are ``chi x d**2 x chi`` rather than ``D_FSM``-wide.

    The three separate the basis choice from the truncation policy, which is the only
    thing that can differ between them: a relative cutoff inside the pre-placement SVD
    measures a discarded direction against the *coefficient* matrix's norm, and the
    shipped one measures it against the placed operator's.
    """
    base = rss_gib()
    walk, rows = _rows(n_sites, terms)
    t0 = time.perf_counter()
    cuts = stream(n_sites, walk, rows, cutoff)
    t_pre = time.perf_counter() - t0
    rss_pre = rss_gib()
    row = {
        "name": name,
        "n_sites": n_sites,
        "pre": widths(cuts),
        "wall_pre": round(t_pre, 2),
        "rss_pre": round(rss_pre - base, 3),
        "rss_pre_peak": round(rss_pre, 3),
    }
    t0 = time.perf_counter()
    exact = stream(n_sites, walk, rows, rank, mode="rel", factors=True)
    row["exact"] = widths(exact)
    sites, shapes = build(n_sites, walk, exact)
    sweep_back(sites, shapes, cutoff)
    _compress_forward(sites, shapes, cutoff)
    row["swept"] = pin.widths(sites)
    row["wall_swept"] = round(time.perf_counter() - t0, 2)
    row["rss_swept_peak"] = round(rss_gib(), 3)
    t0 = time.perf_counter()
    post = MPO.from_terms(n_sites, terms, cutoff=cutoff)
    row["wall_post"] = round(time.perf_counter() - t0, 2)
    row["post"] = pin.widths(post.sites)
    row["fsm"] = [b.dim for b in MPO.from_terms(n_sites, terms, cutoff=None).edges.bonds]
    row["rss_post_peak"] = round(rss_gib(), 3)
    row["place_fsm"] = round(_buffer(row["fsm"], row["post"], walk.phys.dim), 4)
    row["place_pre"] = round(_buffer(row["exact"], row["exact"], walk.phys.dim), 4)
    if dense_check:
        ref = np.asarray(MPO.from_terms(n_sites, terms, cutoff=None).to_dense())
        got = np.asarray(MPO(sites).to_dense())
        row["err_pre"] = float(np.abs(got - ref).max())
        row["err_post"] = float(np.abs(np.asarray(post.to_dense()) - ref).max())
        row["err_vs_post"] = float(np.abs(got - np.asarray(post.to_dense())).max())
    return row


def ratios(row, key):
    """``(max over every cut, max away from the two boundary-adjacent cuts)``."""
    n = len(row["post"])
    pairs = [(p / f, i) for i, (f, p) in enumerate(zip(row["post"], row[key], strict=True)) if f]
    inner = [r for r, i in pairs if 1 < i < n - 2]
    return max(r for r, _ in pairs), (max(inner) if inner else 1.0)


def child(name, cutoff):
    n_sites, terms = pin.qc_terms(name)
    qc.note(kind="preplace-width", **measure(name, n_sites, terms, cutoff=cutoff))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--cutoff", type=float, default=1e-13)
    ap.add_argument("--child", nargs=2, help=argparse.SUPPRESS)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if a.child:
        return child(a.child[0], float(a.child[1]))

    print("# correctness: pre-placement vs post-placement vs uncompressed to_dense\n")
    for label, (n, terms) in pin.SMALL.items():
        row = measure(label, n, terms, cutoff=a.cutoff, dense_check=True)
        print(
            f"{label:<18} err(pre) {row['err_pre']:.2e}  err(post) {row['err_post']:.2e}  "
            f"pre-vs-post {row['err_vs_post']:.2e}"
        )
        print(f"{'':<18} post  {row['post']}")
        print(f"{'':<18} swept {row['swept']}")
        print(f"{'':<18} pre   {row['pre']}")

    names = pin.FIXTURES
    if a.synthetic_only:
        names = [x for x in names if x.startswith("syn-")]
    if a.only:
        names = [x for x in names if x in a.only]
    print("\n# gate 1: pre-placement vs post-placement bond width per cut\n")
    head = (
        "fixture",
        "N",
        "D_FSM",
        "post",
        "pre",
        "exact",
        "swept",
        "pre/post",
        "inner",
        "swept/post",
        "inner",
        "place#218",
        "place pre",
        "pre wall",
        "pre RSS",
    )
    fmt = (
        "{:<16} {:>4} {:>7} {:>6} {:>6} {:>6} {:>6} {:>9} {:>7} {:>10} {:>7} "
        "{:>10} {:>10} {:>9} {:>8}"
    )
    print(fmt.format(*head))
    inners, alls = [], []
    sw_inners, sw_alls = [], []
    for label in names:
        out = subprocess.run(
            [sys.executable, __file__, "--child", label, str(a.cutoff)],
            capture_output=True,
            text=True,
            check=False,
        )
        got = None
        for line in out.stdout.splitlines():
            if line.startswith("{"):
                got = json.loads(line)
        if got is None:
            print(fmt.format(label, *(["-"] * 13), "fail"))
            print("   " + (out.stderr.strip().splitlines() or ["no output"])[-1])
            continue
        hi, inner = ratios(got, "pre")
        sw_hi, sw_inner = ratios(got, "swept")
        alls.append(hi)
        inners.append(inner)
        sw_alls.append(sw_hi)
        sw_inners.append(sw_inner)
        print(
            fmt.format(
                label,
                got["n_sites"],
                max(got["fsm"]),
                max(got["post"]),
                max(got["pre"]),
                max(got["exact"]),
                max(got["swept"]),
                f"{hi:.3f}",
                f"{inner:.3f}",
                f"{sw_hi:.3f}",
                f"{sw_inner:.3f}",
                f"{got['place_fsm']:.3f} G",
                f"{got['place_pre']:.3f} G",
                f"{got['wall_pre']:.1f} s",
                f"{got['rss_pre']:.2f} G",
            )
        )
        qc.note(**got)
    if alls:
        print("\n# gate 1, truncating inside the pre-placement SVD")  # fmt: skip
        print(f"#   all cuts   max ratio {max(alls):.3f} "
              f"{'PASS' if max(alls) <= 1.10 else 'FAIL'}")  # fmt: skip
        print(f"#   inner cuts max ratio {max(inners):.3f} "
              f"{'PASS' if max(inners) <= 1.10 else 'FAIL'}")  # fmt: skip
        print("\n# gate 1, pre-placement basis + #218's two pinned sweeps on it")
        print(f"#   all cuts   max ratio {max(sw_alls):.3f} "
              f"{'PASS' if max(sw_alls) <= 1.10 else 'FAIL'}")  # fmt: skip
        print(f"#   inner cuts max ratio {max(sw_inners):.3f} "
              f"{'PASS' if max(sw_inners) <= 1.10 else 'FAIL'}")  # fmt: skip


if __name__ == "__main__":
    main()
