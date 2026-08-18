"""M16 (#141): the edge table survives instantiation and the matvec consumes it.

Everything here asserts *structure* -- edge counts, ``None``-ness, populated fields,
dispatch counts, cache sizes -- never wall clock. The 1e-12 numerical agreement between
the prepared and dense paths lives in ``tests/integration/test_dmrg_prepared.py``,
where the full sweep-gauge walk has the time budget it needs.
"""

import ast
import pathlib
import sys

import heisenberg_walkthrough as example  # noqa: E402  (see conftest.py)
import pytest

import tenet
from tenet.network import MPO, MPS, Env, lanczos, local_op, sweep_
from tenet.network import env as env_module
from tenet.symmetry import U1Sector


def _ops():
    _, sz, sp, sm = example._spin_half()
    return (
        local_op(sz, phys=example.PHYS, charge=U1Sector(0)),
        local_op(sp, phys=example.PHYS, charge=U1Sector(-2)),
        local_op(sm, phys=example.PHYS, charge=U1Sector(2)),
    )


def _pair_terms(pairs):
    op_sz, op_sp, op_sm = _ops()
    terms = []
    for i, j in pairs:
        terms += [
            (1.0, [(op_sz, i), (op_sz, j)]),
            (0.5, [(op_sp, i), (op_sm, j)]),
            (0.5, [(op_sm, i), (op_sp, j)]),
        ]
    return terms


def _cylinder_pairs(n, ly):
    """Heisenberg pairs on a width-``ly`` cylinder over sites ``0..n-1``, column-major."""
    pairs = []
    for x in range(-(-n // ly)):
        for y in range(ly):
            i = x * ly + y
            pairs.append((i, x * ly + (y + 1) % ly))
            pairs.append((i, i + ly))
    return sorted({(min(i, j), max(i, j)) for i, j in pairs if i != j and max(i, j) < n})


def _heis(n):
    return _pair_terms([(i, i + 1) for i in range(n - 1)])


@pytest.fixture(scope="module")
def ly10():
    """The width-10 cylinder at N=24, ``cutoff=None`` -- #141's measurement-2 subject."""
    return MPO.from_terms(24, _pair_terms(_cylinder_pairs(24, 10)), cutoff=None)


def _mid_env(h, seed=1):
    """An ``Env`` with every environment the middle bond needs, in the sweep's own order."""
    n = len(h)
    psi = MPS.random(example.PHYS, example.bond_spaces(n), seed=seed).canonize_()
    env = Env(psi, h).setup_()
    for m in range(n // 2):
        env.update_(m, to="last")
    return env, psi


# --- the table is present, and absent where it should be absent ----------------------


def test_only_cutoff_none_from_terms_carries_a_table():
    """``from_terms(cutoff=None)`` exposes a block table; the default and ``from_w`` expose
    ``None`` -- measurement 2's reason: the compressing sweep leaves zero identity edges
    on every model, so there is nothing to recover and no table to hand out."""
    kept = MPO.from_terms(6, _heis(6), cutoff=None)
    assert all(kept.jordan(n) is not None for n in range(6))
    assert all(MPO.from_terms(6, _heis(6)).jordan(n) is None for n in range(6))
    from_w = example.mpo(6)
    assert all(from_w.jordan(n) is None for n in range(6))
    assert MPO(kept.sites).jordan(0) is None  # rebuilding the container drops the table


def test_the_ly10_cylinder_middle_site_has_38_edges_of_which_29_identities(ly10):
    """Measurement 2's row, asserted on the table itself: 38 live edges against
    ``D_w^2 = 1024``, 29 of them identities. The two corner identities are implicit
    (``jordanmpotensor.jl``:64-65), so the four dicts hold 36 edges and 27 ``None``s."""
    assert ly10[12].legs[0].space.dim == 32  # D_w, so 32**2 = 1024 dense (l, r) pairs
    table = ly10.jordan(12)
    stored = [v for name in "abcd" for v in getattr(table, name).values()]
    assert len(stored) + 2 == 38
    assert sum(v is None for v in stored) + 2 == 29


def test_identity_edges_are_stored_as_none_never_materialised(ly10):
    """The ``tensors``/``scalars`` split with the scalar always 1: every spectator edge in
    the ``a`` block is ``None``, and every stored tensor is a genuine operator."""
    table = ly10.jordan(12)
    assert table.a and all(v is None for v in table.a.values())
    for name in "bcd":
        for v in getattr(table, name).values():
            assert v is None or v.ndim == 4


def test_the_su3_state_space_survives_in_the_table_with_its_multiplicity():
    """``tests/network/test_mpo.py``'s SU(3) probe, one layer deeper: the state carrying
    ``degeneracy(EIGHT) == 2`` appears in the block table with that space intact."""
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "symmetry"))
    import numpy as np  # noqa: PLC0415
    from _su3_fixture import EIGHT, SU3  # noqa: PLC0415

    from tenet import GradedSpace  # noqa: PLC0415

    d = SU3.irrep_dim(EIGHT)
    swap = np.einsum("ad,bc->abcd", np.eye(d), np.eye(d))
    op = local_op(swap, phys=GradedSpace.new(SU3, {EIGHT: 1}))
    h = MPO.from_terms(2, [(1.0, [(op, (0, 1))])], cutoff=None)
    assert h[0].legs[3].space.degeneracy(EIGHT) == 2
    ((_, edge),) = h.jordan(1).b.items()
    assert edge.legs[0].space.degeneracy(EIGHT) == 2


# --- the prepared operator: fields, sparsity, dispatch --------------------------------


def test_the_prepared_operator_populates_at_most_six_fields_for_nn_heisenberg():
    """The empty fields are the structural zeros, counted after preparation: nearest
    neighbour has no onsite and no continuing block, so ``ID``, ``DE``, ``CA``, ``AB``
    and ``AA`` are all absent at the middle bond."""
    h = MPO.from_terms(8, _heis(8), cutoff=None)
    env, _ = _mid_env(h)
    env._prepare2(4)
    fields = env._prepared[4][3]
    assert set(fields) == {"II", "IC", "CB", "BE", "EE"}
    assert len(fields) <= 6


def test_the_prepared_operator_populates_at_most_nine_fields_on_the_ly10_cylinder():
    """The cylinder adds the continuing block (``CA``, ``AB``, ``AA``) but still has no
    onsite term, so ``ID`` and ``DE`` stay structurally absent. Two columns of the
    width-10 cylinder are enough: the middle bond already carries every field pattern
    the N=24 table does, and the dispatch inequality on the full N=24 object lives in
    ``tests/integration/test_dmrg_prepared.py`` with the budget it needs."""
    h = MPO.from_terms(12, _pair_terms(_cylinder_pairs(12, 10)), cutoff=None)
    psi = MPS.random(example.PHYS, example.bond_spaces(12), seed=1).canonize_()
    env = Env(psi, h).setup_()
    for m in range(2):
        env.update_(m, to="last")
    env._prepare2(2)  # the first bond whose two sites carry every populated pattern
    fields = env._prepared[2][3]
    assert set(fields) == {"II", "IC", "CB", "CA", "AB", "AA", "BE", "EE"}
    assert len(fields) <= 9


# --- caching discipline ---------------------------------------------------------------


def test_one_lanczos_solve_builds_the_prepared_operator_exactly_once(monkeypatch):
    """``ncv=3`` means three matvecs per bond; the fold over the environments must be
    paid once, which is the entire amortisation argument of #141 decision 2."""
    h = MPO.from_terms(6, _heis(6), cutoff=None)
    env, psi = _mid_env(h)
    builds = []
    original = env_module._build2
    monkeypatch.setattr(env_module, "_build2", lambda *a: builds.append(1) or original(*a))
    aa = tenet.einsum("apx,xqr->apqr", psi[3], psi[4])
    lanczos(lambda v: env.heff2(3, v), aa, ncv=3)
    assert builds == [1]


def test_the_compiled_cache_keeps_one_entry_per_bond_across_two_chis():
    """Two sweeps at different ``chi``: the bond spaces move, the structure keys change,
    and every changed key *replaces* its entry -- the cache never accumulates."""
    h = MPO.from_terms(6, _heis(6), cutoff=None)
    psi = MPS.random(example.PHYS, example.bond_spaces(6), seed=2).canonize_()
    env = Env(psi, h).setup_()
    sweep_(psi, h, env, {}, chi=4, cutoff=1e-14)
    first = {n: entry[0] for n, entry in env._compiled.items()}
    sweep_(psi, h, env, {}, chi=16, cutoff=1e-14)
    assert set(env._compiled) <= set(range(5))  # one slot per bond index, nothing else
    assert any(env._compiled[n][0] != first[n] for n in first)  # replaced, not appended
    assert all(len(entry) == 3 for entry in env._compiled.values())


def test_compile_is_called_once_per_structure_key_and_its_result_is_used():
    """The identity-recording stub: no accelerator needed. One ``lanczos`` solve compiles
    once and runs the compiled callable for all three matvecs; re-solving at the same
    bond with unchanged environments recompiles nothing."""
    h = MPO.from_terms(6, _heis(6), cutoff=None)
    compiled, runs = [], []

    def stub(fn):
        compiled.append(fn)

        def wrapped(*args):
            runs.append(1)
            return fn(*args)

        return wrapped

    n = len(h)
    psi = MPS.random(example.PHYS, example.bond_spaces(n), seed=1).canonize_()
    env = Env(psi, h, compile=stub).setup_()
    for m in range(3):
        env.update_(m, to="last")
    aa = tenet.einsum("apx,xqr->apqr", psi[3], psi[4])
    lanczos(lambda v: env.heff2(3, v), aa, ncv=3)
    assert len(compiled) == 1 and compiled[0] is env_module._apply2
    assert len(runs) == 3
    lanczos(lambda v: env.heff2(3, v), aa, ncv=3)
    assert len(compiled) == 1  # same structure key: the stored callable was reused
    assert len(runs) == 6


# --- hygiene: the accessor is the only door ------------------------------------------


def test_env_reads_no_private_attribute_of_any_other_object():
    """The ``ImportFrom`` check in ``test_hygiene.py`` cannot see an attribute read, so
    this pins the rest: every ``_``-prefixed attribute access in ``env.py`` hangs off
    ``self``. The block table is reached through ``MPO.jordan`` alone."""
    path = pathlib.Path(__file__).parents[2] / "src" / "tenet" / "network" / "env.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            assert isinstance(node.value, ast.Name) and node.value.id == "self", (
                f"env.py reads {ast.dump(node)} at line {node.lineno}"
            )
