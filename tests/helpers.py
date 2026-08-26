"""Test helpers shared between test modules. Not part of the library.

``supersign`` is the dense-side Koszul sign of an axis permutation, written for
issue #39 and promoted here unchanged when #51 needed the same oracle for
``tensordot``: the fermionic correctness of a contraction is *inherited* from
``transpose``, so the two suites must weigh it on the same scale.
"""

import dataclasses
import math
import os
import pathlib
import re

import numpy as np

from tenet.space import GradedSpace

__all__ = [
    "NoBendProvider",
    "check_example_page",
    "dense_compose",
    "dense_repartition",
    "dense_step",
    "parity_vector",
    "sector_parity",
    "supersign",
]

DOCS_EXAMPLES = pathlib.Path(__file__).parents[1] / "docs" / "examples"

_FENCE = re.compile(r"```text\n(.*?)```", re.DOTALL)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def check_example_page(page_path: str, captured_stdout: str) -> None:
    """The docs example page's output fence against the run CI just performed (#164).

    Non-numeric text is compared exactly; every number to a relative tolerance of 1e-6
    (with a 1e-12 absolute floor, so float-noise diagnostics like ``max|<S^z>|`` survive
    a BLAS change while any physical digit does not). ``TENET_UPDATE_EXAMPLE_PAGES=1``
    rewrites the fence in place instead of asserting — that is the regeneration path.
    """
    page = DOCS_EXAMPLES / page_path
    text = page.read_text()
    match = _FENCE.search(text)
    assert match is not None, f"{page} has no ```text output fence"
    if os.environ.get("TENET_UPDATE_EXAMPLE_PAGES") == "1":
        page.write_text(text[: match.start(1)] + captured_stdout + text[match.end(1) :])
        return
    fence = match.group(1)
    assert _NUMBER.sub("#", fence) == _NUMBER.sub("#", captured_stdout), (
        f"{page}: the output fence's text no longer matches the run — regenerate with "
        f"TENET_UPDATE_EXAMPLE_PAGES=1.\n--- page ---\n{fence}--- run ---\n{captured_stdout}"
    )
    for committed, fresh in zip(
        _NUMBER.findall(fence), _NUMBER.findall(captured_stdout), strict=True
    ):
        assert math.isclose(float(committed), float(fresh), rel_tol=1e-6, abs_tol=1e-12), (
            f"{page}: committed {committed} vs computed {fresh} — regenerate with "
            f"TENET_UPDATE_EXAMPLE_PAGES=1"
        )


def sector_parity(sector) -> int:
    """Fermionic parity of a sector, summed over the factors of a product sector.

    ``ProductSector`` carries no ``parity`` of its own; a product of a bosonic and
    a fermionic factor is graded by the fermionic one (#52 needs this to give the
    product provider the same oracle as fZ2).
    """
    if hasattr(sector, "parity"):
        return sector.parity
    return sum(sector_parity(c) for c in getattr(sector, "components", ())) % 2


def parity_vector(space: GradedSpace) -> np.ndarray:
    """Parity of each dense index of ``space``, in canonical sector order."""
    return np.concatenate([np.full(m, sector_parity(a)) for a, m in space.sectors])


def supersign(legs, p: tuple[int, ...], *, per_side: bool) -> np.ndarray:
    """Dense-side Koszul sign array, shaped like ``np.transpose(dense, p)``.

    ``per_side=False`` counts every inversion of ``p`` (correct when every leg
    lives on one side); ``per_side=True`` counts only inversions between two axes
    of the same side, which is TeNeT-py's stated convention.
    """
    pars = [parity_vector(legs[ax].space) for ax in p]
    sides = [legs[ax].side for ax in p]
    sign = np.ones(tuple(len(v) for v in pars))
    n = len(p)
    for j in range(n):
        for k in range(j + 1, n):
            if p[j] <= p[k] or (per_side and sides[j] is not sides[k]):
                continue
            shape_j = [1] * n
            shape_j[j] = len(pars[j])
            shape_k = [1] * n
            shape_k[k] = len(pars[k])
            product = pars[j].reshape(shape_j) * pars[k].reshape(shape_k)
            sign = sign * (-1.0) ** product
    return sign


# --- the graded dense oracle for a two-operand composition (M79a, #277) -------------
#
# ``supersign`` above is the sign of a *transpose*. A PEPS primitive is a composition
# with an explicit bend, so the oracle needs two more things: what a ``repartition``
# does to a dense array, and what a bent ``einsum_chain`` step does. Both fall out of
# one observation, measured rather than assumed and pinned by
# ``tests/network/test_peps.py::test_the_oracle_reproduces_repartition_and_einsum``:
#
#   a tensor's fermionic **line order** is its OUT axes in public order followed by its
#   IN axes in *reversed* public order -- the domain is built by bending the last
#   codomain line down, so the domain reads back to front --
#
# and the rule that moving between two (public order, line order) pairs costs the
# Koszul sign of the permutation relating the two line sequences. Every provider shares
# the code; for a non-fermionic one every parity is zero and the sign is 1.


def _line_order(legs) -> tuple[int, ...]:
    """OUT axes in public order, then IN axes reversed -- the fermionic line sequence."""
    from tenet import IN, OUT

    outs = tuple(i for i, leg in enumerate(legs) if leg.side is OUT)
    ins = tuple(i for i, leg in enumerate(legs) if leg.side is IN)
    return outs + ins[::-1]


def _koszul_sign(pars, q) -> np.ndarray:
    """``(-1)`` per inversion of ``q``, weighted by the parities at the two positions."""
    n = len(q)
    sign = np.ones(tuple(len(v) for v in pars))
    for j in range(n):
        for k in range(j + 1, n):
            if q[j] <= q[k]:
                continue
            shape_j, shape_k = [1] * n, [1] * n
            shape_j[j], shape_k[k] = len(pars[j]), len(pars[k])
            sign = sign * (-1.0) ** (pars[j].reshape(shape_j) * pars[k].reshape(shape_k))
    return sign


def _parities(space, n: int) -> np.ndarray:
    """Fermionic parity of each **dense** index of ``space``, in canonical sector order.

    ``parity_vector`` gives one entry per *multiplet*, which is the reduced dimension.
    A dense axis is longer wherever a sector has ``irrep_dim > 1``, so the two disagree
    on any non-Abelian space and this repeats each sector's parity across its dense
    indices. Parity is a property of the sector, not of the magnetic index inside it, so
    the within-sector layout does not matter here.

    It used to pad with zeros on a length mismatch, which read as "no fermionic sign on
    a non-Abelian space". That is right for a bosonic provider, where every parity is
    zero anyway, and **silently wrong** for a fermionic factor sitting beside a
    non-Abelian one in a Deligne product -- ``fZ2 x SU2``'s odd ``j = 1/2`` sector has
    parity 1 and dense dimension 2, so every Koszul sign on it was being deleted. The
    ``assert`` is what turns the next such mismatch into a failure instead of a silent
    zero.
    """
    provider = space.provider
    irrep_dim = getattr(provider, "irrep_dim", None)
    parities = np.concatenate(
        [
            np.full(m * (irrep_dim(a) if irrep_dim else 1), sector_parity(a), dtype=int)
            for a, m in space.sectors
        ]
    )
    assert len(parities) == n, f"dense axis is {n} long, parities cover {len(parities)}"
    return parities


def _regauge(arr, spaces, pub_old, seq_old, pub_new, seq_new):
    """Move a dense array between two (public order, line order) pairs."""
    dims = {x: arr.shape[pub_old.index(x)] for x in pub_old}
    arr = np.transpose(arr, [pub_old.index(x) for x in seq_old])
    q = tuple(seq_old.index(x) for x in seq_new)
    pars = [_parities(spaces[x], dims[x]) for x in seq_new]
    arr = _koszul_sign(pars, q) * np.transpose(arr, q)
    return np.transpose(arr, [seq_new.index(x) for x in pub_new])


def dense_repartition(arr, legs, outs, ins):
    """``tenet.repartition(t, outs, ins).to_dense()`` from ``t.to_dense()`` alone."""
    import dataclasses

    from tenet import IN, OUT

    outs, ins = tuple(outs), tuple(ins)
    spaces = [leg.space for leg in legs]
    new_legs = tuple(dataclasses.replace(legs[i], side=OUT) for i in outs) + tuple(
        dataclasses.replace(legs[i], side=IN) for i in ins
    )
    arr = _regauge(
        arr, spaces, tuple(range(len(legs))), _line_order(legs), outs + ins, outs + ins[::-1]
    )
    return arr, new_legs


def dense_compose(equation, arr_a, legs_a, arr_b, legs_b):
    """``tenet.einsum(equation, a, b).to_dense()`` for a *composition*.

    Refuses -- with an assertion naming the wire -- anything that is not one: operand 1
    must supply the ``IN`` end of every shared wire. That refusal is half the point of
    the oracle.
    """
    from tenet import IN, OUT

    lhs, out = equation.split("->")
    ta, tb = lhs.split(",")
    shared = [c for c in ta if c in tb]
    fa = [c for c in ta if c not in shared]
    fb = [c for c in tb if c not in shared]
    for c in shared:
        assert legs_a[ta.index(c)].side is IN, f"wire {c!r}: operand 1 must supply IN"
        assert legs_b[tb.index(c)].side is not IN, f"wire {c!r}: operand 2 must supply OUT"
    a2, _ = dense_repartition(
        arr_a, legs_a, [ta.index(c) for c in fa], [ta.index(c) for c in shared]
    )
    b2, _ = dense_repartition(
        arr_b, legs_b, [tb.index(c) for c in shared], [tb.index(c) for c in fb]
    )
    k = len(shared)
    prod = np.tensordot(a2, b2, axes=(list(range(len(fa), len(fa) + k)), list(range(k))))
    labels = fa + fb
    legs_res = tuple(legs_a[ta.index(c)] for c in fa) + tuple(legs_b[tb.index(c)] for c in fb)
    n = len(labels)
    seq_old = tuple(range(len(fa))) + tuple(range(len(fa), n))[::-1]
    pub_new = tuple(labels.index(c) for c in out)
    outs = tuple(i for i in pub_new if legs_res[i].side is OUT)
    ins = tuple(i for i in pub_new if legs_res[i].side is IN)
    arr = _regauge(
        prod, [leg.space for leg in legs_res], tuple(range(n)), seq_old, pub_new, outs + ins[::-1]
    )
    return arr, tuple(legs_res[i] for i in pub_new)


def dense_step(equation, arr_a, legs_a, arr_b, legs_b, bend=""):
    """One ``tenet.einsum_chain`` step -- the bend, then the composition."""
    from tenet import OUT

    lhs, out = equation.split("->")
    ta, tb = lhs.split(",")
    if bend:
        flip = set(bend)

        def bent(arr, legs, term):
            outs = tuple(i for i, c in enumerate(term) if (legs[i].side is OUT) != (c in flip))
            ins = tuple(i for i in range(len(term)) if i not in outs)
            arr, legs = dense_repartition(arr, legs, outs, ins)
            return arr, legs, "".join(term[i] for i in (*outs, *ins))

        arr_a, legs_a, ta = bent(arr_a, legs_a, ta)
        arr_b, legs_b, tb = bent(arr_b, legs_b, tb)
    return dense_compose(f"{ta},{tb}->{out}", arr_a, legs_a, arr_b, legs_b)


# --- a provider that cannot bend (#312) ---------------------------------------------
#
# `ProductProvider` used to be the repository's only provider without
# `BendingCoefficients`, so eight refusal tests across four modules reached for it when
# they needed one. #312 forwarded bending through products and took that vehicle away.
#
# The refusals themselves are contract and must stay pinned, so the vehicle is replaced
# rather than the tests deleted: `NoBendProvider` delegates every capability to a real
# provider and withholds exactly two, `bend_right`/`bend_left` and `z_matrix`. Written as
# explicit delegation, not `__getattr__`, because the capability lattice is a *structural*
# check -- a catch-all would answer `hasattr` for the very methods this exists to lack.


@dataclasses.dataclass(frozen=True, slots=True)
class NoBendProvider:
    """``base`` with ``BendingCoefficients`` and ``DualBasis`` withheld.

    Sectors, fusion, duals, quantum dimensions, Clebsch-Gordan tensors and tree
    permutation are the base provider's, so spaces and tensors behave normally and a
    test reaches the bend on real data. What it cannot do is move a line between the two
    trees, or expand a dual leg.
    """

    base: object = None
    name: str = "NoBend"

    def __post_init__(self) -> None:
        if self.base is None:
            from tenet.symmetry import U1

            object.__setattr__(self, "base", U1)

    @property
    def unit(self):
        return self.base.unit

    def dual(self, a):
        return self.base.dual(a)

    def fusion(self, a, b):
        return self.base.fusion(a, b)

    def n_symbol(self, a, b, c):
        return self.base.n_symbol(a, b, c)

    def qdim(self, a):
        return self.base.qdim(a)

    def irrep_dim(self, a):
        return self.base.irrep_dim(a)

    def cgc(self, a, b, c):
        return self.base.cgc(a, b, c)

    def frobenius_schur(self, a):
        return self.base.frobenius_schur(a)

    def twist(self, a):
        return self.base.twist(a)

    def permute_tree(self, tree, perm):
        return self.base.permute_tree(tree, perm)
