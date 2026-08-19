"""M37 (#200): the edge description is the MPO, and the matvec never materialises a site.

Stage 1 of #184's staging: ``MPO.from_terms(cutoff=None)`` keeps its finite-state machine
as an [EdgeTable][tenet.network.EdgeTable] rather than as a list of rank-4 site tensors,
``Env``'s ``_cores2`` builds its cores from that description, and the group embedding is
part of the per-site core builder instead of a step ``_instantiate`` ran for the whole
MPO. One test per acceptance criterion, and each one fails if its criterion stops holding
-- the allocation claim is instrumented on ``_place``'s buffer, not asserted in prose.
"""

import ast
import pathlib

import numpy as np
import pytest

import tenet
from tenet import GradedSpace
from tenet.network import MPO, MPS, Env, local_op
from tenet.network import mps as mps_module
from tenet.symmetry import U1, FZ2Sector, U1Sector, fZ2

MPS_PY = pathlib.Path(mps_module.__file__)

# --- the three models -----------------------------------------------------------------

SPIN = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
SPINLESS = GradedSpace.new(fZ2, {FZ2Sector(0): 1, FZ2Sector(1): 1})
HUBBARD = GradedSpace.new(fZ2, {FZ2Sector(0): 2, FZ2Sector(1): 2})


def _spin_terms(n):
    """U(1) Heisenberg with a next-nearest-neighbour rung, so every block is populated."""
    sp = np.array([[0.0, 0.0], [1.0, 0.0]])
    op_sp = local_op(sp, phys=SPIN, charge=U1Sector(-2))
    op_sm = local_op(sp.T, phys=SPIN, charge=U1Sector(2))
    op_sz = local_op(np.diag([-0.5, 0.5]), phys=SPIN, charge=U1Sector(0))
    terms = []
    for i, j in [(m, m + 1) for m in range(n - 1)] + [(m, m + 2) for m in range(n - 2)]:
        c = 1.0 if j == i + 1 else 0.4
        terms += [
            (c, [(op_sz, i), (op_sz, j)]),
            (0.5 * c, [(op_sp, i), (op_sm, j)]),
            (0.5 * c, [(op_sm, i), (op_sp, j)]),
        ]
    return terms


def _spinless_terms(n):
    """Spinless fermions on fZ2: hops, a non-adjacent hop over a spectator, a density."""
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    cd = local_op(a.T, phys=SPINLESS, charge=FZ2Sector(1))
    c = local_op(a, phys=SPINLESS, charge=FZ2Sector(1))
    op_n = local_op(np.diag([0.0, 1.0]), phys=SPINLESS, charge=FZ2Sector(0))
    terms = [(0.8, [(op_n, m)]) for m in range(n)]
    for i, j in [(m, m + 1) for m in range(n - 1)] + [(0, n - 1)]:
        terms += [(1.0, [(cd, i), (c, j)]), (1.0, [(cd, j), (c, i)])]
    return terms


def _hubbard_terms(n, u=4.0):
    """The spinful Hubbard chain of ``tests/network/test_hubbard.py``, same conventions."""
    c_up = np.zeros((4, 4))
    c_up[0, 2] = 1.0
    c_up[3, 1] = 1.0
    c_dn = np.zeros((4, 4))
    c_dn[0, 3] = 1.0
    c_dn[2, 1] = -1.0
    pairs = [
        (
            local_op(m.T, phys=HUBBARD, charge=FZ2Sector(1)),
            local_op(m, phys=HUBBARD, charge=FZ2Sector(1)),
        )
        for m in (c_up, c_dn)
    ]
    nn = local_op((c_up.T @ c_up) @ (c_dn.T @ c_dn), phys=HUBBARD, charge=FZ2Sector(0))
    terms = [(u, [(nn, m)]) for m in range(n)]
    for m in range(n - 1):
        for cd, c in pairs:
            terms += [(-1.0, [(cd, m), (c, m + 1)]), (-1.0, [(cd, m + 1), (c, m)])]
    return terms


MODELS = {
    "spin": (SPIN, _spin_terms),
    "spinless": (SPINLESS, _spinless_terms),
    "hubbard": (HUBBARD, _hubbard_terms),
}


def _bonds(phys, n, chi):
    """A bond space per cut, ``D=1`` at the two ends, ``chi`` per sector in between.

    U(1) needs the *reachable* charges -- the sum of ``i`` physical ``+-1`` charges, which
    must still reach 0 by site ``n`` -- because a bond graded with unreachable sectors has
    no allowed block at all. The two fZ2 gradings are closed under fusion, so every cut
    carries both sectors.
    """
    sym = phys.provider
    unit = GradedSpace.new(sym, {sym.unit: 1})
    if sym is U1:
        mids = [
            GradedSpace.new(U1, {U1Sector(q): chi for q in range(-w, w + 1, 2)})
            for w in (min(i, n - i) for i in range(1, n))
        ]
        return [unit, *mids, unit]
    mid = GradedSpace.new(sym, {a: chi for a, _ in phys.sectors})
    return [unit] + [mid] * (n - 1) + [unit]


def _walk_to(hs, phys, n, bond, chi, seed=1):
    """One state and one ``Env`` per Hamiltonian, walked into the sweep's own gauge.

    ``heff2``'s prepared path assumes the two-site sweep's mixed-canonical gauge -- sites
    left of the bond left-orthonormal -- so the harness *walks* it with exact SVDs rather
    than asserting it, which is ``tests/integration/test_dmrg_prepared.py``'s method.
    """
    psi = MPS.random(phys, _bonds(phys, n, chi), seed=seed).canonize_()
    envs = [Env(psi, h).setup_() for h in hs]
    for b in range(bond):
        aa = tenet.einsum("apx,xqr->apqr", psi[b], psi[b + 1])
        u, s, vh = tenet.linalg.svd(aa, ((0, 1), (2, 3)))
        psi[b] = u
        psi[b + 1] = tenet.einsum("xy,yqr->xqr", s, vh)
        for env in envs:
            env.clear_(b, b + 1)
            env.update_(b, to="last")
    return psi, envs


# --- the allocation claim ---------------------------------------------------------------


def _recording_place(monkeypatch):
    """Record ``(left dim, right dim)`` of every buffer ``_place`` allocates."""
    seen = []
    original = mps_module._place

    def spy(items, space_l, space_r, phys, dual_l, dual_r, carry=None):
        seen.append((space_l.dim, space_r.dim))
        return original(items, space_l, space_r, phys, dual_l, dual_r, carry)

    monkeypatch.setattr(mps_module, "_place", spy)
    return seen


def test_the_prepared_matvec_allocates_no_full_width_site_tensor(monkeypatch):
    """The memory claim of #184 candidate (a), instrumented on the allocation itself.

    Every rank-4 buffer the MPO layer ever allocates goes through ``_place``, so the pair
    of bond dimensions it is given *is* the width of the tensor about to exist. A
    full-width dense-blocked site at cut ``n`` is ``(bonds[n].dim, bonds[n+1].dim)``;
    the group-restricted blocks the prepared path needs are strictly narrower at both
    ends, because ``IdL`` and ``IdR`` are one dimension each and the open group is the
    rest. Building the operator, every environment and the two-site matvec must never
    produce that pair -- and the positive control below shows the instrument catches it
    the moment a site *is* materialised.
    """
    n, bond = 8, 3
    seen = _recording_place(monkeypatch)
    h = MPO.from_terms(n, _spin_terms(n), cutoff=None)
    assert seen == []  # the description is symbolic: assembly allocates nothing at all
    full = [(h.edges.bonds[m].dim, h.edges.bonds[m + 1].dim) for m in range(n)]
    assert min(min(pair) for pair in full[1:-1]) >= 4  # narrower is a real statement here
    psi, (env,) = _walk_to([h], SPIN, n, bond, chi=8)
    aa = tenet.einsum("apx,xqr->apqr", psi[bond], psi[bond + 1])
    env.heff2(bond, aa)
    assert seen, "the instrument never fired: the prepared path built no block at all"
    assert not set(seen) & set(full)

    h[bond]  # the positive control: one materialised site, and the instrument sees it
    assert full[bond] in seen


def test_a_deferred_mpo_holds_no_site_tensor_until_one_is_asked_for():
    """``_instantiate`` is not on the ``cutoff=None`` path at all, so nothing is placed.

    The container-level statement behind the allocation test: the MPO *is* the edge
    description until a consumer that needs a dense site turns up, and ``to_dense`` is
    such a consumer while ``Env`` is not.
    """
    h = MPO.from_terms(6, _spin_terms(6), cutoff=None)
    assert h.edges is not None
    assert h._sites == {}
    assert len(h) == 6
    h.to_dense()
    assert len(h._sites) == 6
    assert MPO(h.sites).edges is None  # the site-tensor container carries no description


def test_group_embedding_happens_in_the_core_builder_not_in_assembly(monkeypatch):
    """The restructuring #184 named as the real cost of (a), counted.

    ``_group_embedding`` used to run for every cut of the MPO inside ``_instantiate``.
    It now runs when a site's block table is asked for, which is the core builder, and
    a cut's embeddings are built once and shared by the two sites that meet at it.
    """
    calls = []
    original = mps_module._group_embedding
    monkeypatch.setattr(
        mps_module,
        "_group_embedding",
        lambda *a, **kw: calls.append(1) or original(*a, **kw),
    )
    h = MPO.from_terms(8, _spin_terms(8), cutoff=None)
    assert calls == []  # assembly embeds nothing
    h.to_dense()
    assert calls == []  # neither does full numeric instantiation
    h.edge_blocks(3)  # the core builder, and the first thing that embeds anything
    assert len(calls) == 2 * 3 * 2  # three groups, two orientations, cuts 3 and 4
    h.edge_blocks(4)
    assert len(calls) == 2 * 3 * 3  # cut 4 was already built: the two sites share it


def test_instantiate_is_one_consumer_of_the_description_not_its_producer():
    """The call structure, read off the source rather than asserted about.

    ``_instantiate`` takes the table and returns sites: it neither prunes, nor derives a
    bond space, nor embeds a group. The producer is ``_edge_table`` and both builders
    call it directly.
    """
    tree = ast.parse(MPS_PY.read_text())
    funcs = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    inst = funcs["_instantiate"]
    first = inst.args.args[0]
    assert first.arg == "tab" and ast.unparse(first.annotation) == "EdgeTable"
    called = {ast.unparse(node.func) for node in ast.walk(inst) if isinstance(node, ast.Call)}
    assert not called & {"_edge_table", "_group_embedding", "_merge", "_blocks"}
    for builder in ("from_terms", "from_arrays"):
        body = funcs[builder]
        assert "_edge_table" in {
            ast.unparse(node.func) for node in ast.walk(body) if isinstance(node, ast.Call)
        }


# --- the two paths are the same operator ------------------------------------------------


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("chi", [4, 12])
def test_the_deferred_and_numeric_heff2_paths_agree(model, chi):
    """``heff2`` off the description against ``heff2`` off the materialised ``W`` pair.

    The same MPO twice: once as the edge description, once as the plain site tensors
    ``MPO(h.sites)`` carries, which routes ``Env`` onto the dense path. Two bond
    dimensions, three models -- a spin chain, a spinless fermionic chain and the spinful
    Hubbard chain -- and the two fermionic ones are what make this a statement about the
    Jordan-Wigner string rather than only about bookkeeping.
    """
    phys, terms = MODELS[model]
    n, bond = 6, 2
    h = MPO.from_terms(n, terms(n), cutoff=None)
    psi, (env, ref) = _walk_to([h, MPO(h.sites)], phys, n, bond, chi)
    aa = tenet.einsum("apx,xqr->apqr", psi[bond], psi[bond + 1])
    yp, yd = env.heff2(bond, aa), ref.heff2(bond, aa)
    gap = float(tenet.norm(tenet.subtract(yp, yd)))
    assert gap < 1e-11 * float(tenet.norm(yd))
