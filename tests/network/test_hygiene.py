"""The layer's hygiene invariants, enforced rather than stated (#112).

``src/tenet/network/`` may use **public ``tenet`` API only**. Concretely, and each one is
a test below:

* no ``import jax`` / ``torch`` / ``scipy`` / ``quimb`` / ``opt_einsum``, extending
  ``tests/test_import.py`` and REPOSITORY_RULES:31-33 ("core never imports them") to the
  new package. ``numpy`` and ``autoray`` are core dependencies and are allowed;
* no reach into a ``_``-prefixed name of another ``tenet`` module. The package's own
  private helpers (``mps._as_site``, ``dmrg._schmidt_change``, ``ctmrg._spectrum_change``) are its
  own business; importing someone else's is not;
* no **numerical** use of ``t.blocks``. Reduced blocks are the library's storage, and a
  driver that reads them is doing arithmetic below the public API.

**The one named exception**, spelled out because it is a line and not a loophole:
reading ``t.provider`` and the provider's ``qdim``, ``unit``, ``fusion``, ``dual`` and
``permute_tree`` is allowed, and each read has an owner: ``common.spectrum`` needs
``sqrt(qdim)`` for its Schmidt weights, ``ctmrg.py`` needs ``provider.unit`` for the
trivial sector, ``mps.py`` accumulates charges with ``fusion`` (#133), derives
``MPS.product``'s bonds backwards with ``fusion`` and ``dual`` (M14), and probes the
braiding with ``permute_tree`` for spectator classification and the fermionic
term dressing (M13b's refusal probe, repurposed by #160/#147). That is
*symmetry-generic metadata*: ``provider.qdim(c)`` is fine, ``isinstance(provider,
SU2Provider)`` is not, and the second is what the branch test below forbids. Since M14
the metadata test also catches reads through a local binding (``sym = space.provider``),
so the list enforced and the list ``network/__init__.py`` claims are the same list.

**The duplicated-``scalar`` finding #112 recorded, corrected and closed (#114, #126).** It
was recorded as *three* files writing the same five-line ``sum(qdim(c) * trace(m))``; it
was **two**. ``examples/vmc_mps.py``:158-160 is a *different* function --
``t.blocks[0][0, 0]``, the trivial-leg shortcut for a network with no non-unit sector on
the open bond, not qdim-weighted at all -- and it is deliberately untouched. The two real
copies were ``examples/ctmrg.py`` and ``tenet/network/mps.py::scalar``; #114 moved them
into ``network/common.py`` with their bodies unchanged, and #126 finished the job by
promoting the qdim-weighted trace to ``tenet.full_trace`` (and ``inner`` to
``tenet.inner``) in ``tenet.ops``, next to ``trace``. There is now a public spelling, so
``scalar`` is deleted rather than allowed, and ``spectrum`` alone owns the ``qdim``
allowance above.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).parents[2] / "src" / "tenet" / "network"
MODULES = sorted(PACKAGE.glob("*.py"))
FORBIDDEN = {"jax", "torch", "scipy", "quimb", "opt_einsum"}


def trees():
    for path in MODULES:
        yield path, ast.parse(path.read_text())


def test_the_package_has_the_modules_it_claims():
    assert {p.name for p in MODULES} == {
        "__init__.py",
        "common.py",
        "mps.py",
        "env.py",
        "dmrg.py",
        "ctmrg.py",
    }


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_imports_a_forbidden_dependency(path):
    tree = ast.parse(path.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN, sorted(roots & FORBIDDEN)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_reaches_into_another_modules_private_names(path):
    tree = ast.parse(path.read_text())
    own = f"tenet.network.{path.stem}"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module != own:
            private = [a.name for a in node.names if a.name.startswith("_")]
            assert not private, f"{path.name} imports {private} from {node.module}"


def test_direct_sum_left_the_mpo_builder():
    """M15 (#138): ``from_terms`` assembles a finite-state machine, edge by edge.

    ``tenet.direct_sum``'s block-diagonal placement is the wrong primitive for a graph
    whose edges are deliberately off-diagonal, so it does not appear in ``mps.py`` at all.
    """
    assert "direct_sum" not in (PACKAGE / "mps.py").read_text()


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_uses_reduced_blocks_numerically(path):
    """``.blocks`` is storage. ``apply_blocks`` is the public spelling and is not this."""
    tree = ast.parse(path.read_text())
    reads = [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "blocks"
    ]
    assert not reads, f"{path.name} reads .blocks at line(s) {[n.lineno for n in reads]}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_branches_on_which_provider_it_has(path):
    """Metadata reads yes, provider dispatch no -- the line the module docstring draws.

    No provider class is so much as *named* in the package, which is the cheapest way to
    say ``isinstance(provider, SU2Provider)`` never happens: the only ``isinstance`` here
    is the write barrier's index check, and it tests ``int``.
    """
    source = path.read_text()
    assert "Provider" not in source, f"{path.name} names a provider class"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "isinstance":
            assert getattr(node.args[1], "id", None) in {"int", "str"}, node.lineno


def test_the_allowed_metadata_reads_stay_on_the_named_list():
    """Whatever this package reads off a provider must stay on the named list.

    Two AST shapes are covered: the direct chain ``x.provider.ATTR``, and reads through a
    local binding -- ``sym = space.provider`` then ``sym.ATTR`` -- which #133's
    ``sym.fusion`` introduced and the original test could not see. The binding scan is
    per-module and by name, so a name bound to a provider anywhere in a module counts
    everywhere in it; that over-approximates, which is the right direction for a limit.
    """
    allowed = {"qdim", "unit", "provider", "fusion", "dual", "permute_tree"}
    seen = set()
    for _path, tree in trees():
        bound = set()  # names locally bound to a provider, e.g. ``sym = space.provider``
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            target, value = node.targets[0], node.value
            pairs = (
                list(zip(target.elts, value.elts, strict=True))
                if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)
                else [(target, value)]
            )
            for t, v in pairs:
                if (
                    isinstance(t, ast.Name)
                    and isinstance(v, ast.Attribute)
                    and v.attr == "provider"
                ):
                    bound.add(t.id)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if isinstance(node.value, ast.Attribute) and node.value.attr == "provider":
                seen.add(node.attr)
            elif isinstance(node.value, ast.Name) and node.value.id in bound:
                seen.add(node.attr)
    assert seen <= allowed, sorted(seen - allowed)


def test_every_two_operand_einsum_is_a_composition(monkeypatch):
    """The composition rule (#160), pinned: operand 1 supplies IN on every shared wire.

    ``network/__init__.py`` states the rule once; this test is what makes it a
    convention rather than a one-time cleanup. ``tenet.einsum`` is wrapped for one
    smoke over the MPS/MPO/DMRG modules -- ``from_terms`` at both cutoffs with a term
    list that populates every Jordan channel (a non-adjacent term over a spectator, a
    single-site term, an operator-carrying string, a k-site split), ``to_dense``,
    ``dmrg_`` sweeps on both ``heff2`` paths, ``Env.measure``, ``MPS.norm``, the two
    expectation values, ``compress_`` -- and every two-operand call must have operand
    1 supplying the IN end of every shared wire. Zero exemptions: a wire that bends is
    bent explicitly with ``tenet.repartition`` *before* the einsum
    (``env.py::_composed``), so what reaches ``tenet.einsum`` is always a composition.

    The coverage claim is asserted, not assumed: every ``tenet.einsum`` call site the
    AST finds in ``mps.py``, ``env.py`` and ``dmrg.py`` must be reached by the smoke
    (``_composed`` call sites route through the one einsum inside the helper, which is
    itself on the list). ``ctmrg.py`` is deliberately outside: a 2D network has loops,
    where operand order is necessary but not sufficient (``ops/contraction.py``
    :575-587), and no fermionic PEPS caller exists.
    """
    import traceback

    import numpy as np

    import tenet
    from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    from tenet.network import MPO, MPS, Env, dmrg_, expectation_1site, expectation_2site, local_op
    from tenet.symmetry import U1, U1Sector

    reached, violations = set(), []
    real = tenet.einsum

    def wrapped(equation, *operands, **kwargs):
        if len(operands) == 2:
            stack = traceback.extract_stack()
            frame = stack[-2]
            where = f"{pathlib.Path(frame.filename).name}:{frame.lineno}"
            if pathlib.Path(frame.filename).parent == PACKAGE:
                reached.add((pathlib.Path(frame.filename).name, frame.lineno))
                if frame.name == "_composed":  # credit the site that stated the bend
                    caller = stack[-3]
                    if pathlib.Path(caller.filename).parent == PACKAGE:
                        reached.add((pathlib.Path(caller.filename).name, caller.lineno))
                terms = equation.split("->")[0].split(",")
                for label in terms[0]:
                    if label not in terms[1]:
                        continue
                    end1 = operands[0].legs[terms[0].index(label)].side
                    end2 = operands[1].legs[terms[1].index(label)].side
                    if not (end1 is IN and end2 is not IN):
                        violations.append(
                            f"{where}: {equation}: wire {label!r} has operand 1 "
                            f"supplying {end1} and operand 2 supplying {end2}"
                        )
        return real(equation, *operands, **kwargs)

    monkeypatch.setattr(tenet, "einsum", wrapped)

    phys = GradedSpace.new(U1, {U1Sector(1): 1, U1Sector(-1): 1})
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    sz = np.diag([0.5, -0.5])
    op_sp = local_op(sp, phys=phys, charge=U1Sector(2))
    op_sm = local_op(sp.T, phys=phys, charge=U1Sector(-2))
    op_sz = local_op(sz, phys=phys, charge=U1Sector(0))
    zz = local_op(np.kron(sz, sz), phys=phys)
    n = 4
    terms = []
    for m in range(n - 1):
        terms += [
            (0.5, [(op_sp, m), (op_sm, m + 1)]),
            (0.5, [(op_sm, m), (op_sp, m + 1)]),
            (1.0, [(op_sz, m), (op_sz, m + 1)]),
        ]
    terms += [
        (0.25, [(zz, (0, 2))]),  # a k-site split whose derived bond runs over a spectator
        (0.1, [(op_sz, 2)]),  # a single-site term: the d channel
        (0.3, [(op_sp, 0), (op_sz, 1), (op_sz, 2), (op_sm, 3)]),  # operator-carrying string
        (0.2, [(op_sp, 0), (op_sm, 3)]),  # open-group spectators: the AA-remainder chains
    ]
    unit = GradedSpace.new(U1, {U1Sector(0): 1})
    bonds = [unit]
    for i in range(1, n):
        qs = (-1, 1) if i % 2 else (-2, 0, 2)
        bonds.append(GradedSpace.new(U1, {U1Sector(q): 2 for q in qs}))
    bonds.append(unit)
    for cutoff in (None, 1e-13):
        h = MPO.from_terms(n, terms, cutoff=cutoff)
        h.to_dense()
        psi = MPS.random(phys, bonds, seed=3)
        psi.norm()
        psi.to_dense()
        sz2 = SymmetricTensor.from_dense(sz, (Leg(phys, OUT), Leg(phys, IN)))
        zz4 = SymmetricTensor.from_dense(
            np.reshape(np.kron(sz, sz), (2, 2, 2, 2)),
            (Leg(phys, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(phys, IN)),
        )
        expectation_1site(psi, sz2, 1)
        expectation_2site(psi, zz4, 1)
        Env(psi.copy(), h).measure()
        dmrg_(psi, h, chi=8, cutoff=1e-12, max_sweeps=2)
    psi.compress_(chi=4)

    assert not violations, "\n".join(violations)

    wanted = set()
    for path in (PACKAGE / "mps.py", PACKAGE / "env.py", PACKAGE / "dmrg.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            direct = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "einsum"
                and len(node.args) == 3
            )
            bent = isinstance(node.func, ast.Name) and node.func.id == "_composed"
            if direct or bent:
                wanted.add((path.name, node.lineno))
    missing = sorted(wanted - reached)
    assert not missing, f"the smoke never reached these einsum call sites: {missing}"
