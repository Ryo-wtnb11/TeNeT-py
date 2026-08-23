"""The composition rule at every step of every ``einsum_chain`` in the driver layer.

``test_hygiene.py::test_every_two_operand_einsum_is_a_composition`` pins the rule (#160)
for the two-operand ``tenet.einsum`` calls and for ``_composed``. A chain step never
reaches either: it contracts through ``ops.contraction`` directly, which is the whole
point — no tensor is written between two steps. So the rule is pinned here for the chain
sites, in the same shape: operand 1 supplies the ``IN`` end of every shared wire, zero
exemptions, checked on the operands the step actually contracts (after the wires named in
the step's ``bend`` have been moved), and every chain site the AST finds in ``env.py`` has
to be reached by the smoke.
"""

import ast
import pathlib
import traceback

import numpy as np

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import MPO, MPS, Env, dmrg_, local_op
from tenet.ops import contraction as ct
from tenet.symmetry import U1, U1Sector

PACKAGE = pathlib.Path(tenet.network.__file__).parent


def _smoke():
    """Both ``heff2`` routes, both environment folds, on a chain with every channel."""
    phys = GradedSpace.new(U1, {U1Sector(1): 1, U1Sector(-1): 1})
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    sz = np.diag([0.5, -0.5])
    op_sp = local_op(sp, phys=phys, charge=U1Sector(2))
    op_sm = local_op(sp.T, phys=phys, charge=U1Sector(-2))
    op_sz = local_op(sz, phys=phys, charge=U1Sector(0))
    n = 4
    terms = []
    for m in range(n - 1):
        terms += [
            (0.5, [(op_sp, m), (op_sm, m + 1)]),
            (0.5, [(op_sm, m), (op_sp, m + 1)]),
            (1.0, [(op_sz, m), (op_sz, m + 1)]),
        ]
    zz = local_op(np.kron(sz, sz), phys=phys)
    terms += [
        (0.25, [(zz, (0, 2))]),  # the k-site split whose remainder is the sr2 family
        (0.1, [(op_sz, 2)]),
        (0.3, [(op_sp, 0), (op_sz, 1), (op_sz, 2), (op_sm, 3)]),
        (0.2, [(op_sp, 0), (op_sm, 3)]),
    ]
    unit = GradedSpace.new(U1, {U1Sector(0): 1})
    bonds = [unit]
    for i in range(1, n):
        qs = (-1, 1) if i % 2 else (-2, 0, 2)
        bonds.append(GradedSpace.new(U1, {U1Sector(q): 2 for q in qs}))
    bonds.append(unit)
    built = [MPO.from_terms(n, terms, cutoff=c, symbolic=True) for c in (None, 1e-13)]
    built.append(MPO(built[1].sites))  # no edge table: the four full contractions
    for h in built:
        psi = MPS.random(phys, bonds, seed=3)
        Env(psi.copy(), h).measure()
        dmrg_(psi, h, chi=8, cutoff=1e-12, max_sweeps=2)


def test_every_chain_step_is_a_composition(monkeypatch):
    """Operand 1 supplies IN on every wire a chain step contracts, at every site."""
    reached, violations = set(), []
    real_chain, real_contract = tenet.einsum_chain, ct._contracted

    def chain(steps):
        frame = traceback.extract_stack()[-2]
        if pathlib.Path(frame.filename).parent == PACKAGE:
            reached.add((pathlib.Path(frame.filename).name, frame.lineno))
        return real_chain(steps)

    def contracted(a, b, axes, after):
        for i, j in zip(*axes, strict=True):
            if not (a.legs[i].side is IN and b.legs[j].side is not IN):
                violations.append(
                    f"a wire of a chain step has operand 1 supplying {a.legs[i].side} "
                    f"and operand 2 supplying {b.legs[j].side}"
                )
        return real_contract(a, b, axes, after)

    monkeypatch.setattr(tenet, "einsum_chain", chain)
    monkeypatch.setattr(tenet.network.env.tenet, "einsum_chain", chain)
    monkeypatch.setattr(ct, "_contracted", contracted)
    _smoke()
    assert not violations, "\n".join(sorted(set(violations)))

    wanted = set()
    for path in (PACKAGE / "mps.py", PACKAGE / "env.py", PACKAGE / "dmrg.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "einsum_chain"
            ):
                wanted.add((path.name, node.lineno))
    assert wanted, "no chain site left to cover; delete this test with the last one"
    assert not sorted(wanted - reached), f"the smoke never reached {sorted(wanted - reached)}"


def test_a_chain_step_that_breaks_the_rule_is_caught():
    """The check above is not vacuous: a reversed wire is a violation, not a shrug."""
    v = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    a = SymmetricTensor.random((Leg(v, OUT), Leg(v, OUT)), seed=0)
    b = SymmetricTensor.random((Leg(v, IN), Leg(v, IN)), seed=1)
    seen = []
    real = ct._contracted

    def contracted(x, y, axes, after):
        seen.extend((x.legs[i].side, y.legs[j].side) for i, j in zip(*axes, strict=True))
        return real(x, y, axes, after)

    ct._contracted = contracted
    try:
        tenet.einsum_chain([("ab,bc->ac", a, b, "b")])
    finally:
        ct._contracted = real
    assert seen and all(x is IN and y is not IN for x, y in seen)
