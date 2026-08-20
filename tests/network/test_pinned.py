"""M39 (#204): the compressing sweeps keep the ``IdL (+) open (+) IdR`` partition.

Both truncating sweeps pin the two corner channels, so a float-``cutoff`` MPO is
compressed *and* partitioned where it used to be one or the other. Three things have to
hold and each is a test here:

* the pinned operator is the **same operator** the free sweep produced and the same one
  ``cutoff=None`` builds -- checked by ``to_dense`` on a spin chain, a spinless fermion
  chain and the spinful ``d=4`` Hubbard chain, at tight tolerance;
* every compressed cut's two corner channels are **exact unit slabs** -- the identity on
  the physical space, with nothing entering ``IdL`` and nothing leaving ``IdR``, read off
  the compressed ``W'`` itself;
* the description the compressed MPO carries is a real block table, with the corner
  channels in ``idmap`` and the four blocks summing back to ``W'``.

The free sweep, which #204 replaced, is kept in ``_free_sites`` for the first of those:
it is the only thing that can fail if pinning quietly changed the operator rather than
only its gauge. ``benchmarks/bench_pinned_mpo.py`` is the same comparison at ab initio
scale, for width rather than for values.
"""

import numpy as np
import pytest

import tenet
from tenet import GradedSpace
from tenet.network import MPO, local_op
from tenet.network.mps import _IDL, _IDR, _as_w, _corner_slots
from tenet.symmetry import U1, FZ2Sector, U1Sector, fZ2

_A = np.array([[0.0, 1.0], [0.0, 0.0]])
U1_PHYS = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
FZ2_PHYS = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
PHYS4 = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})
_C_UP = np.zeros((4, 4))
_C_UP[0, 2] = _C_UP[3, 1] = 1.0
_C_DN = np.zeros((4, 4))
_C_DN[0, 3] = 1.0
_C_DN[2, 1] = -1.0


def _spin_terms(n):
    """U(1) Heisenberg plus a power-law ``SzSz`` tail -- something for the sweep to cut."""
    sp = local_op(_A.T, phys=U1_PHYS, charge=U1Sector(-2))
    sm = local_op(_A, phys=U1_PHYS, charge=U1Sector(2))
    sz = local_op(np.diag([-0.5, 0.5]), phys=U1_PHYS, charge=U1Sector(0))
    terms = []
    for i in range(n - 1):
        terms += [(0.5, [(sp, i), (sm, i + 1)]), (0.5, [(sm, i), (sp, i + 1)])]
    for i in range(n):
        for j in range(i + 1, n):
            terms.append((1.0 / (j - i) ** 3, [(sz, i), (sz, j)]))
    return terms


def _spinless_terms(n):
    """fZ2 hopping plus a power-law density tail."""
    cd = local_op(_A.T, phys=FZ2_PHYS, charge=FZ2Sector(1))
    c = local_op(_A, phys=FZ2_PHYS, charge=FZ2Sector(1))
    nop = local_op(_A.T @ _A, phys=FZ2_PHYS, charge=FZ2Sector(0))
    terms = []
    for i in range(n - 1):
        terms += [(-1.0, [(cd, i), (c, i + 1)]), (-1.0, [(cd, i + 1), (c, i)])]
    for i in range(n):
        for j in range(i + 1, n):
            terms.append((0.8 / (j - i) ** 3, [(nop, i), (nop, j)]))
    return terms


def _hubbard_terms(n):
    """The spinful ``d=4`` chain of ``test_hubbard.py``, on its documented on-site matrices."""
    ops = {}
    for label, m in (("u", _C_UP), ("d", _C_DN)):
        ops[label] = local_op(m, phys=PHYS4, charge=FZ2Sector(1))
        ops[label + "+"] = local_op(m.T, phys=PHYS4, charge=FZ2Sector(1))
    nn = local_op((_C_UP.T @ _C_UP) @ (_C_DN.T @ _C_DN), phys=PHYS4, charge=FZ2Sector(0))
    terms = [(4.0, [(nn, i)]) for i in range(n)]
    for i in range(n - 1):
        for f in ("u", "d"):
            terms += [
                (-1.0, [(ops[f + "+"], i), (ops[f], i + 1)]),
                (-1.0, [(ops[f + "+"], i + 1), (ops[f], i)]),
            ]
    return terms


MODELS = {
    "spin U(1)": (6, _spin_terms(6)),
    "spinless fZ2": (6, _spinless_terms(6)),
    "Hubbard fZ2 d=4": (4, _hubbard_terms(4)),
}


def _free_sites(tab, cutoff):
    """The compressing sweeps as they stood before #204: the SVD rotates the whole bond."""
    out, carry = [], None
    for n in reversed(range(len(tab.edges))):
        w = tab.site(n, carry)
        if n:
            u, s, vh = tenet.linalg.svd_truncated(w, ((0,), (1, 2, 3)), cutoff=cutoff)
            w = _as_w(vh)
            carry = tenet.repartition(tenet.einsum("xy,yz->xz", u, s), (), (0, 1))
        out.append(w)
    out.reverse()
    for n in range(len(out) - 1):
        u, s, vh = tenet.linalg.svd_truncated(out[n], ((0, 1, 2), (3,)), cutoff=cutoff)
        out[n] = _as_w(u)
        carry = tenet.repartition(tenet.einsum("xy,yz->xz", s, vh), (0, 1), ())
        out[n + 1] = _as_w(tenet.einsum("ypqr,xy->xpqr", out[n + 1], carry))
    return out


@pytest.mark.parametrize("model", sorted(MODELS), ids=lambda k: k)
def test_the_pinned_sweeps_build_the_operator_the_free_sweeps_did(model):
    """``to_dense`` against the free sweep and against ``cutoff=None``, element-wise.

    Pinning is a restriction of the two SVDs' gauge freedom, so it may move the bond
    dimension and may not move a single matrix element. On the fermionic models this is
    also the check that the block-diagonal carry's corner block is the *bent* identity
    the free carry's corner would have been -- a bare delta in the dense buffer is not
    that, and the difference shows up as a sign on the odd sector.
    """
    n, terms = MODELS[model]
    ref = np.asarray(MPO.from_terms(n, terms, cutoff=None).to_dense())
    free = np.asarray(
        MPO(_free_sites(MPO.from_terms(n, terms, cutoff=None).edges, 1e-13)).to_dense()
    )
    pinned = np.asarray(MPO.from_terms(n, terms, cutoff=1e-13).to_dense())
    scale = np.abs(ref).max()
    assert np.abs(free - ref).max() < 1e-12 * scale
    assert np.abs(pinned - ref).max() < 1e-12 * scale
    assert np.abs(pinned - free).max() < 1e-12 * scale


@pytest.mark.parametrize("model", sorted(MODELS), ids=lambda k: k)
def test_every_compressed_cut_keeps_its_corner_channels_exact(model):
    """The partition survives the sweep: both corners are unit slabs of the compressed ``W'``.

    The mechanism is a dense read of the site tensor at the two rows the corner channels
    occupy -- the first and last degeneracy slot of the bond's unit sector, which is where
    ``_merge``'s ``[_IDL, *open, _IDR]`` order and ``direct_sum``'s prefix placement put
    them. Three things are asserted per site: ``W'[IdL, :, :, IdL]`` and
    ``W'[IdR, :, :, IdR]`` are the identity on the physical space, nothing else in the
    ``IdL`` *column* is nonzero (nothing enters the not-yet-started channel), and nothing
    else in the ``IdR`` *row* is (nothing leaves the already-finished one). Those are
    exactly the four zeros of MPSKit's ``(1 C D; . A B; . . 1)``.
    """
    n, terms = MODELS[model]
    h = MPO.from_terms(n, terms, cutoff=1e-13)
    tab = h.edges
    assert tab is not None
    sym = tab.phys.provider
    d = tab.phys.dim
    for site in range(n):
        w = np.asarray(h[site].to_dense())
        lo = _corner_slots(tab.bonds[site], sym)
        hi = _corner_slots(tab.bonds[site + 1], sym)
        if _IDL in tab.ordered[site] and _IDL in tab.ordered[site + 1]:
            assert np.abs(w[lo["idl"], :, :, hi["idl"]] - np.eye(d)).max() < 1e-13
            column = np.delete(w[:, :, :, hi["idl"]], lo["idl"], axis=0)
            assert column.size == 0 or np.abs(column).max() < 1e-13
        if _IDR in tab.ordered[site] and _IDR in tab.ordered[site + 1]:
            assert np.abs(w[lo["idr"], :, :, hi["idr"]] - np.eye(d)).max() < 1e-13
            row = np.delete(w[lo["idr"]], hi["idr"], axis=2)
            assert row.size == 0 or np.abs(row).max() < 1e-13


@pytest.mark.parametrize("model", sorted(MODELS), ids=lambda k: k)
def test_the_compressed_block_table_reassembles_the_compressed_site(model):
    """``a``/``b``/``c``/``d`` are all of ``W'`` outside the two corner channels.

    The block table is what ``Env.heff2``'s prepared path reads, and it is now sliced out
    of ``W'`` rather than scattered from edges -- so the check that no term fell out of
    the slicing is that the six embeddings put the four blocks back where they came from
    and the only remainder is the two corner identities, which the test above pins and
    this one blanks rather than rebuilding (an identity channel of a sign-braiding bond
    is not the dense ``idmap (x) eye`` a naive reassembly would write).
    """
    n, terms = MODELS[model]
    h = MPO.from_terms(n, terms, cutoff=1e-13)
    tab = h.edges
    for site in range(n):
        blocks = h.edge_blocks(site)
        assert blocks is not None
        assert blocks.spec_op is None and blocks.a_real_op is blocks.a_op
        assert not blocks.a and not blocks.b and not blocks.c and not blocks.d
        w = np.asarray(h[site].to_dense())
        left, right = tab._embed(site), tab._embed(site + 1)
        rebuilt = np.zeros_like(w)
        for name, (gl, gr) in (
            ("a", ("open", "open")),
            ("b", ("open", "idr")),
            ("c", ("idl", "open")),
            ("d", ("idl", "idr")),
        ):
            op = getattr(blocks, f"{name}_op")
            if op is None:
                continue
            # both einsums are compositions: operand 1 supplies IN on the shared wire,
            # which for these 0/1 embeddings is what makes the round trip sign-exact
            inner = tenet.einsum("wy,vpqw->vpqy", right[gr][1], op)
            back = tenet.einsum("vpqy,xv->xpqy", inner, left[gl][0])
            rebuilt += np.asarray(back.to_dense())
        sym = tab.phys.provider
        lo, hi = _corner_slots(tab.bonds[site], sym), _corner_slots(tab.bonds[site + 1], sym)
        rest = w.copy()
        for corner, key in ((_IDL, "idl"), (_IDR, "idr")):
            if corner in tab.ordered[site] and corner in tab.ordered[site + 1]:
                rest[lo[key], :, :, hi[key]] = 0.0
        assert np.abs(rebuilt - rest).max() < 1e-12 * max(np.abs(w).max(), 1.0)
