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
from tenet.network import MPO, MPS, Env, common, dmrg_, local_op, sweep_
from tenet.network import env as env_module
from tenet.network import mps as mps_module
from tenet.symmetry import U1, FZ2Sector, U1Sector, fZ2

MPS_PY = pathlib.Path(mps_module.__file__)
ENV_PY = pathlib.Path(env_module.__file__)

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
    ``MPO(h.sites)`` carries, which routes ``Env`` onto the site-tensor path. Two bond
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


# --- the caches are bounded (M38, #202) -------------------------------------------------


def _sweep_caches(h, env):
    """Every per-bond cache the sweep fills that holds tensors, by its owner's name."""
    return {
        "EdgeTable._table": h.edges._table,
        "EdgeTable._embeds": h.edges._embeds,
        "Env._cores": env._cores,
        "Env._prepared": env._prepared,
    }


def _swept(n, chi, model="spinless", seed=1):
    """One full sweep of a ``cutoff=None`` MPO, and the ``(h, env)`` whose caches it filled."""
    phys, terms = MODELS[model]
    h = MPO.from_terms(n, terms(n), cutoff=None)
    psi = MPS.random(phys, _bonds(phys, n, chi), seed=seed)
    psi.canonize_(0)
    env = Env(psi, h).setup_(0)
    sweep_(psi, h, env, {}, chi=chi, cutoff=1e-12)
    return h, env


def test_the_sweep_caches_never_grow_past_the_budget(monkeypatch):
    """#202's memory criterion: a full sweep visits every bond and keeps only the last few.

    The unbounded caches this replaced held one entry per bond after a half sweep, and at
    quantum-chemistry scale that total *is* the operator -- the object the instantiation
    boundary above exists not to build. A model small enough for a test suite sits under
    the shipped budget by design, so the budget is driven to both ends here instead: at
    zero every cache falls to the two-entry floor a two-site bond needs, and the same
    sweep with the budget effectively infinite holds one entry per bond, which is what
    makes the first half a real bound rather than a statement about a small model.

    **``Env._compiled`` left the budget in #227 and is asserted about separately below.**
    Four caches hold tensors and are bounded by bytes; the fifth holds a tuple of legs and
    a callable, weighs nothing, and is bounded by the bond count. Weighing it was not
    conservative but systematically wrong: its entry used to carry the very ``_Prepared``
    ``Env._prepared`` holds, so those bytes were charged twice and evicted a callable
    whose whole purpose is to outlive the visit.
    """
    n, held = 12, {}
    for tag, budget in (("evicting", 0), ("unbounded", 1 << 40)):
        monkeypatch.setattr(common, "CACHE_BUDGET", budget)
        h, env = _swept(n, 8)
        caches = _sweep_caches(h, env)
        assert set(caches) == {  # a cache added later must make this list or explain itself
            "EdgeTable._table",
            "EdgeTable._embeds",
            "Env._cores",
            "Env._prepared",
        }
        held[tag] = {name: len(cache) for name, cache in caches.items()}
        for name, cache in caches.items():
            assert isinstance(cache, common.Recent), name
        # The one cache outside the budget: a plain dict, one slot per bond at both ends
        # of the budget, and no backend array reachable from any entry.
        assert not isinstance(env._compiled, common.Recent)
        assert len(env._compiled) == n - 1
        assert common.payload(dict(env._compiled)) == 0
    assert set(held["evicting"].values()) == {2}
    assert min(held["unbounded"].values()) > 2


def test_a_model_under_the_budget_keeps_every_entry():
    """The other half of the policy, and the reason it is bytes and not a count of entries.

    A fixed count evicts on a two-megabyte MPO exactly as eagerly as on a twenty-gibibyte
    one, and measured that cost 22-26 % of the wall time of a small ``dmrg_`` run for no
    memory saved at all. Under the shipped budget nothing here evicts, so the common case
    pays nothing -- which is what this asserts, at the same sweep the test above bounds.
    """
    n = 12
    h, env = _swept(n, 8)
    assert common.payload(dict(h.edges._table)) < common.CACHE_BUDGET
    assert len(h.edges._table) == n  # every site's block table still cached
    assert len(env._cores) == n - 1  # and every bond's merged cores


def test_the_budget_evicts_by_use_and_keeps_a_two_site_bond_addressable(monkeypatch):
    """The eviction order, on the class rather than through a sweep.

    Oldest-*used* first, not oldest-written: a bond revisited on the return leg of a sweep
    is one step away, which is the access pattern the whole policy is chosen for. And the
    floor of two entries is what keeps a two-site bond -- which asks for site ``n`` and
    site ``n + 1`` in one breath -- from thrashing however small the budget is set.
    """
    monkeypatch.setattr(common, "CACHE_BUDGET", 3 * np.zeros(64).nbytes)
    cache = common.Recent()
    for i in range(3):
        cache[i] = np.zeros(64)
    cache[0]  # a read is a use, so 0 is now the youngest and 1 the oldest
    cache[3] = np.zeros(64)
    assert 0 in cache and 1 not in cache
    monkeypatch.setattr(common, "CACHE_BUDGET", 0)
    tiny = common.Recent()
    tiny["a"], tiny["b"], tiny["c"] = np.zeros(64), np.zeros(64), np.zeros(64)
    assert len(tiny) == 2  # never below two, whatever the budget says


def test_the_budget_changes_no_energy(monkeypatch):
    """#202's correctness criterion: a cache policy that changes a number is a bug.

    The same ``dmrg_`` run twice, once with the caches evicting on almost every insert and
    once with them effectively unbounded, and the histories must agree **bitwise** -- not
    to a tolerance. Eviction only ever forces a recomputation of a pure function of the
    edge description, so a difference here would mean the rebuilt object was not the
    evicted one.
    """
    n, terms = 8, MODELS["hubbard"][1]
    out = {}
    for tag, budget in (("evicting", 8 * 1024), ("unbounded", 1 << 40)):
        monkeypatch.setattr(common, "CACHE_BUDGET", budget)
        h = MPO.from_terms(n, terms(n), cutoff=None)
        psi = MPS.random(HUBBARD, _bonds(HUBBARD, n, 8), seed=3)
        out[tag] = dmrg_(psi, h, chi=8, cutoff=1e-12, max_sweeps=3).history
    assert out["evicting"] == out["unbounded"]


def test_the_cache_policy_is_one_decision_in_one_place():
    """#202's structural criterion, read off the source rather than asserted about.

    ``CACHE_BUDGET`` is *set* in ``common.py`` and nowhere else in the package, and every
    cache is constructed as a bare ``Recent()`` -- so there is no per-cache budget to set,
    no flag to thread through ``Env`` and ``EdgeTable`` separately, and no way for two of
    them to disagree about the policy. Other modules may name it in a comment; what would
    break the criterion is a second module deciding it.
    """
    package = MPS_PY.parent
    sets = set()
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                else []
            )
            if any(getattr(t, "id", None) == "CACHE_BUDGET" for t in targets):
                sets.add(path.name)
    assert sets == {"common.py"}
    for path in (MPS_PY, ENV_PY):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "Recent":
                assert not node.args and not node.keywords, f"{path.name}:{node.lineno}"
