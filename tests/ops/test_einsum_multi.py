"""Tests for multi-operand ``tenet.einsum`` — one test per acceptance criterion of #67.

The scheduler owns no mathematics either: ``opt_einsum.contract_path`` orders the
pairwise contractions of #52 and every step is one of those calls. So what is
checked here is, again, *layering*, plus the two things the path brings with it —
that the planner is fed **physical** shapes, and that the answer does not depend
on which path it came back with.

The dense oracle is :func:`dense_fold`: the pairwise oracle of
``tests/ops/test_einsum.py`` applied left to right. For a non-graded provider it
is asserted below to be plain ``np.einsum`` on the expansions; for fZ2 and the
product provider it carries the Koszul signs of the same reorders, weighed on
``helpers.supersign`` like every other fermionic test in the repository.

Two layout facts shape the networks:

* the product provider has no ``BendingCoefficients``, so its network is the
  bend-free matrix chain (free legs on the two ends only) that keeps every step
  composition-shaped — contracted legs of the left operand IN, of the right
  operand OUT;
* the other providers get a chain whose middle tensors carry free legs on both
  sides and whose bonds alternate side and ``dual``, so most steps *must* bend.
"""

import itertools
import subprocess
import sys

import numpy as np
import pytest
from helpers import NoBendProvider, supersign

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.ops import contraction
from tenet.symmetry import (
    SU2,
    U1,
    CapabilityError,
    FZ2Sector,
    ProductProvider,
    ProductSector,
    Sector,
    SU2Sector,
    Trivial,
    TrivialSector,
    U1Sector,
    fZ2,
)

# --- spaces -----------------------------------------------------------------------

ZERO, HALF, ONE = SU2Sector(0), SU2Sector(1), SU2Sector(2)
V = GradedSpace.new(SU2, {ZERO: 2, HALF: 2, ONE: 1})
W = GradedSpace.new(SU2, {ZERO: 1, HALF: 2})
U = GradedSpace.new(SU2, {HALF: 1, ONE: 2})
X = GradedSpace.new(SU2, {ZERO: 1, HALF: 1})

Q = GradedSpace.new(U1, {U1Sector(-1): 2, U1Sector(0): 3, U1Sector(1): 1})
P = GradedSpace.new(U1, {U1Sector(0): 2, U1Sector(1): 2})

T3 = GradedSpace.new(Trivial, {TrivialSector(): 3})
T2 = GradedSpace.new(Trivial, {TrivialSector(): 2})

EVEN, ODD = FZ2Sector(0), FZ2Sector(1)
FP = GradedSpace.new(fZ2, {EVEN: 2, ODD: 3})
FQ = GradedSpace.new(fZ2, {EVEN: 1, ODD: 2})
FR = GradedSpace.new(fZ2, {EVEN: 2, ODD: 1})

UF = ProductProvider((U1, fZ2))


def uf(charge: int, parity: int) -> ProductSector:
    return ProductSector((U1Sector(charge), FZ2Sector(parity)))


S1 = GradedSpace.new(UF, {uf(0, 0): 2, uf(1, 1): 1, uf(-1, 1): 1})
S2 = GradedSpace.new(UF, {uf(0, 0): 1, uf(1, 1): 2})

# A provider with the product's sectors and *no* bending: #312 forwarded
# ``BendingCoefficients`` through products, so ``UF`` no longer refuses a bend and the
# refusal cases below need a vehicle that still does. ``helpers.NoBendProvider`` withholds
# exactly that capability and delegates the rest, so the layout, the fZ2 signs and the
# sector pattern are unchanged -- only the bend is out of contract.
NB = NoBendProvider(UF)
NB_S1 = GradedSpace.new(NB, {uf(0, 0): 2, uf(1, 1): 1, uf(-1, 1): 1})
NB_S2 = GradedSpace.new(NB, {uf(0, 0): 1, uf(1, 1): 2})

BONDS, FREES = "cegik", "dfhjl"

# (spaces, graded)
PROVIDERS = {
    "trivial": ((T3, T2, T3, T2), False),
    "u1": ((Q, P, Q, P), False),
    "su2": ((V, W, U, X), False),
    "fz2": ((FP, FQ, FR, FQ), True),
}


# --- networks ------------------------------------------------------------------------


def chain(spaces, n):
    """``(equation, legs per operand)`` for an ``n``-operand chain that bends.

    ``T0[a,b,c] T1[c,d,e] T2[e,f,g] ... T_{n-1}[.,.]``: bond ``k`` alternates
    space, side and ``dual`` between its two ends, so a step whose operand has a
    free leg on the "wrong" side must bend it back — which is the whole point of
    running these against SU(2) and fZ2.
    """
    x, y, z, w = spaces
    bond = [(z, True) if k % 2 == 0 else (w, False) for k in range(n - 1)]
    free = [(y, OUT) if k % 2 else (x, IN) for k in range(n)]

    terms = ["ab" + BONDS[0]]
    legs = [[Leg(x, OUT), Leg(y, IN), Leg(bond[0][0], OUT, dual=bond[0][1])]]
    for k in range(1, n - 1):
        terms.append(BONDS[k - 1] + FREES[k - 1] + BONDS[k])
        legs.append(
            [
                Leg(bond[k - 1][0], IN, dual=bond[k - 1][1]),
                Leg(*free[k]),
                Leg(bond[k][0], OUT, dual=bond[k][1]),
            ]
        )
    terms.append(BONDS[n - 2] + FREES[n - 2])
    legs.append([Leg(bond[n - 2][0], IN, dual=bond[n - 2][1]), Leg(*free[n - 1])])
    out = "ab" + FREES[: n - 1]
    return ",".join(terms) + "->" + out, [tuple(each) for each in legs]


def matrix_chain(spaces, n, duals=False):
    """``(equation, legs)`` for a **bend-free** ``n``-operand chain.

    Free legs live on the two ends only and every operand is already
    composition-shaped for a left-to-right path: no leg ever changes side. Two
    things need that. A provider without ``BendingCoefficients`` (the product
    one) cannot run anything else, and — the reason the graded providers use this
    layout too — ``helpers.supersign`` is the dense oracle of a *composition*,
    not of a bend, so :func:`dense_fold` is only an oracle here. Bending is
    covered for fZ2 by the chain-equivalence test instead.
    """
    x, y, z, w = spaces
    bond = [z if k % 2 == 0 else w for k in range(n - 1)]
    dual = [duals and k % 2 == 0 for k in range(n - 1)]
    terms = ["ab" + BONDS[0]]
    legs = [[Leg(x, OUT), Leg(y, OUT), Leg(bond[0], IN, dual=dual[0])]]
    for k in range(1, n - 1):
        terms.append(BONDS[k - 1] + BONDS[k])
        legs.append([Leg(bond[k - 1], OUT, dual=dual[k - 1]), Leg(bond[k], IN, dual=dual[k])])
    terms.append(BONDS[n - 2] + FREES[0])
    legs.append([Leg(bond[n - 2], OUT, dual=dual[n - 2]), Leg(x, IN)])
    return ",".join(terms) + "->ab" + FREES[0], [tuple(each) for each in legs]


def chain_path(n):
    """The left-to-right path, in ``opt_einsum``'s numbering (pop both, append result)."""
    path = [(0, 1)]
    path += [(0, m - 1) for m in range(n - 1, 1, -1)]
    return path


def build(equation, legs, seed=0):
    return tuple(SymmetricTensor.random(each, seed=seed + k) for k, each in enumerate(legs))


def split(equation):
    lhs, _, rhs = equation.partition("->")
    return lhs.split(","), rhs


# --- the dense oracle ------------------------------------------------------------------


def dense_fold(tensors, equation, graded):
    """``np.einsum`` on the expansions, folded left to right with the Koszul signs.

    One iteration is exactly ``test_einsum.dense_oracle``: graded-transpose each
    operand into ``(free..., contracted...)`` / ``(contracted..., free...)``,
    ``np.tensordot``, and carry the resulting legs forward. For a non-graded
    provider every sign is ``+1``, which the test below asserts against
    ``np.einsum``.
    """
    terms, out = split(equation)
    arrays = [t.to_dense() for t in tensors]
    legs = [t.legs for t in tensors]
    while len(terms) > 1:
        ta, tb = terms[0], terms[1]
        shared = [x for x in ta if x in tb]
        fa = [x for x in ta if x not in shared]
        fb = [x for x in tb if x not in shared]
        pa = tuple(ta.index(x) for x in fa + shared)
        pb = tuple(tb.index(x) for x in shared + fb)
        da, db = np.transpose(arrays[0], pa), np.transpose(arrays[1], pb)
        if graded:
            da = supersign(legs[0], pa, per_side=True) * da
            db = supersign(legs[1], pb, per_side=True) * db
        merged_legs = tuple(legs[0][ta.index(x)] for x in fa)
        merged_legs += tuple(legs[1][tb.index(x)] for x in fb)
        arrays = [np.tensordot(da, db, axes=len(shared)), *arrays[2:]]
        legs = [merged_legs, *legs[2:]]
        terms = ["".join(fa + fb), *terms[2:]]
    perm = tuple(terms[0].index(x) for x in out)
    got = np.transpose(arrays[0], perm)
    return supersign(legs[0], perm, per_side=True) * got if graded else got


def use_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import tenet.pytree  # noqa: F401  registration is the import's side effect

    return jax


def to_numpy(t: SymmetricTensor) -> SymmetricTensor:
    return SymmetricTensor(t.structure, tuple(np.asarray(x) for x in t.blocks))


NS = [3, 4, 5]


# --- dense oracles, every provider × 3, 4 and 5 operands --------------------------------


@pytest.mark.parametrize("provider_id", [p for p in PROVIDERS if not PROVIDERS[p][1]])
@pytest.mark.parametrize("n", NS)
def test_dense_oracle_is_np_einsum_with_duals_and_bends(provider_id, n):
    equation, legs = chain(PROVIDERS[provider_id][0], n)
    tensors = build(equation, legs, seed=3)
    np.testing.assert_allclose(
        tenet.einsum(equation, *tensors).to_dense(),
        np.einsum(equation, *(t.to_dense() for t in tensors)),
        atol=1e-10,
    )


@pytest.mark.parametrize("n", NS)
def test_the_bending_chain_really_bends(n):
    """Teeth for the layout above: the same network is refused for a provider with
    no ``BendingCoefficients``, which is only possible if legs must cross.

    The vehicle is ``NB``, not ``UF``: #312 forwarded bending through products, so the
    product itself no longer refuses anything. ``NB`` carries the same sectors and the
    same fZ2 signs and withholds only the bend, which is what this case is weighing.
    """
    equation, legs = matrix_chain((NB_S1, NB_S2, NB_S1, NB_S2), n)  # that layout
    tensors = build(equation, legs, seed=3)
    tenet.einsum(equation, *tensors, optimize=chain_path(n))  # bend-free: fine

    equation, legs = chain((NB_S1, NB_S2, NB_S1, NB_S2), n)
    tensors = build(equation, legs, seed=3)
    with pytest.raises(CapabilityError, match="BendingCoefficients"):
        tenet.einsum(equation, *tensors)


def graded_network(provider_id, n, braided):
    """The bend-free chain for a graded provider, output legs optionally braided.

    ``braided`` swaps the two free legs of the first operand, which are both OUT:
    a within-side reorder, i.e. the one place in this layout where the Koszul sign
    is not identically ``+1``.
    """
    spaces = PROVIDERS["fz2"][0] if provider_id == "fz2" else (S1, S2, S1, S2)
    equation, legs = matrix_chain(spaces, n, duals=provider_id == "fz2")
    if braided:
        equation = equation.replace("->ab", "->ba")
    # the product provider has no BendingCoefficients, so its path is pinned to the
    # left-to-right one, which is the only one that keeps every step a composition
    optimize = "auto" if provider_id == "fz2" else chain_path(n)
    return equation, build(equation, legs, seed=3), optimize


@pytest.mark.parametrize("braided", [False, True])
@pytest.mark.parametrize("provider_id", ["fz2", "product"])
@pytest.mark.parametrize("n", NS)
def test_graded_dense_oracle(provider_id, n, braided):
    """The graded oracle: ``np.einsum`` on the expansions times the Koszul signs of
    each fold, on the bend-free layout where that is what a fold *is*."""
    equation, tensors, optimize = graded_network(provider_id, n, braided)
    np.testing.assert_allclose(
        tenet.einsum(equation, *tensors, optimize=optimize).to_dense(),
        dense_fold(tensors, equation, graded=True),
        atol=1e-10,
    )


@pytest.mark.parametrize("provider_id", ["fz2", "product"])
@pytest.mark.parametrize("n", NS)
def test_the_graded_oracle_is_not_plain_np_einsum(provider_id, n):
    """The graded criterion with teeth: with the output legs braided the signs are
    not identically ``+1``, so a scheduler that lost them fails the test above."""
    equation, tensors, _ = graded_network(provider_id, n, braided=True)
    plain = np.einsum(equation, *(t.to_dense() for t in tensors))
    assert not np.allclose(dense_fold(tensors, equation, graded=True), plain, atol=1e-8)


@pytest.mark.parametrize("provider_id", [p for p in PROVIDERS if not PROVIDERS[p][1]])
@pytest.mark.parametrize("n", NS)
def test_the_fold_oracle_is_plain_np_einsum_when_nothing_is_graded(provider_id, n):
    equation, legs = chain(PROVIDERS[provider_id][0], n)
    tensors = build(equation, legs, seed=3)
    np.testing.assert_allclose(
        dense_fold(tensors, equation, graded=False),
        np.einsum(equation, *(t.to_dense() for t in tensors)),
        atol=1e-12,
    )


# --- output legs -----------------------------------------------------------------------


@pytest.mark.parametrize("provider_id", list(PROVIDERS))
@pytest.mark.parametrize("n", NS)
def test_output_legs_are_the_input_legs_unchanged(provider_id, n):
    equation, legs = chain(PROVIDERS[provider_id][0], n)
    tensors = build(equation, legs, seed=3)
    terms, out = split(equation)
    want = []
    for label in out:
        k = next(i for i, term in enumerate(terms) if label in term)
        want.append(tensors[k].legs[terms[k].index(label)])
    assert tenet.einsum(equation, *tensors).legs == tuple(want)


@pytest.mark.parametrize("out", ["abdf", "bafd", "dfab", "fdba"])
def test_the_output_subscript_orders_the_legs(out):
    """One transpose at the end, and it is categorical: the fZ2 result is *not* the
    naive ``np.transpose`` of the plain one."""
    equation, legs = chain(PROVIDERS["fz2"][0], 3)
    tensors = build(equation, legs, seed=3)
    got = tenet.einsum(equation.partition("->")[0] + "->" + out, *tensors)
    plain = tenet.einsum(equation, *tensors)
    perm = tuple("abdf".index(label) for label in out)
    assert got.legs == tuple(plain.legs[p] for p in perm)
    assert tenet.allclose(got, tenet.transpose(plain, perm), atol=1e-12)


# --- the scheduler is the pairwise call, in the caller's operand order --------------------


@pytest.mark.parametrize("provider_id", list(PROVIDERS))
def test_multi_operand_is_the_hand_written_chain_of_pairwise_calls(provider_id):
    equation, legs = chain(PROVIDERS[provider_id][0], 4)
    a, b, c, d = build(equation, legs, seed=3)
    ab = tenet.einsum("abc,cde->abde", a, b)
    abc = tenet.einsum("abde,efg->abdfg", ab, c)
    want = tenet.einsum("abdfg,gh->abdfh", abc, d)
    assert tenet.allclose(tenet.einsum(equation, a, b, c, d), want, atol=1e-12)


@pytest.mark.parametrize("provider_id", list(PROVIDERS))
def test_every_grouping_of_the_chain_agrees(provider_id):
    """Associativity of the pairwise call, which is what makes the path free to
    choose: ``((AB)C)D``, ``(AB)(CD)``, ``A(B(CD))`` and ``A((BC)D)``."""
    equation, legs = chain(PROVIDERS[provider_id][0], 4)
    a, b, c, d = build(equation, legs, seed=3)
    e = tenet.einsum
    groupings = [
        e("abdfg,gh->abdfh", e("abde,efg->abdfg", e("abc,cde->abde", a, b), c), d),
        e("abde,efh->abdfh", e("abc,cde->abde", a, b), e("efg,gh->efh", c, d)),
        e("abc,cdfh->abdfh", a, e("cde,efh->cdfh", b, e("efg,gh->efh", c, d))),
        e("abc,cdfh->abdfh", a, e("cdfg,gh->cdfh", e("cde,efg->cdfg", b, c), d)),
    ]
    for other in groupings[1:]:
        assert tenet.allclose(groupings[0], other, atol=1e-12)


def test_the_path_does_not_change_the_answer():
    """Every path ``opt_einsum`` could return for the chain, run explicitly."""
    equation, legs = chain(PROVIDERS["su2"][0], 4)
    tensors = build(equation, legs, seed=3)
    want = tenet.einsum(equation, *tensors)
    paths = itertools.product([(0, 1), (1, 2), (2, 3)], [(0, 1), (0, 2), (1, 2)], [(0, 1)])
    checked = 0
    for path in paths:
        got = tenet.einsum(equation, *tensors, optimize=list(path))
        assert tenet.allclose(got, want, atol=1e-10), path
        checked += 1
    assert checked == 9


# --- opt_einsum is not imported for one or two operands ------------------------------------


def test_no_opt_einsum_import_for_one_or_two_operands(monkeypatch):
    """A monkeypatched ``__import__``, because ``sys.modules`` is polluted by JAX,
    which depends on ``opt_einsum`` itself."""
    import builtins

    real = builtins.__import__

    def refusing(name, *args, **kwargs):
        assert name != "opt_einsum", "the pairwise path must not import opt_einsum"
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refusing)
    equation, legs = chain(PROVIDERS["su2"][0], 3)
    a, b, _ = build(equation, legs, seed=3)
    tenet.einsum("abc->cab", a)
    tenet.einsum("abc,cde->abde", a, b)


def test_no_cotengra_and_no_top_level_opt_einsum_anywhere_in_src():
    import pathlib
    import re

    root = pathlib.Path(tenet.__file__).parent
    sources = {path: path.read_text() for path in root.rglob("*.py")}
    assert sources, "the source tree was not found"
    imports_cotengra = re.compile(r"^\s*(import cotengra|from cotengra)", re.M)
    assert not [path for path, text in sources.items() if imports_cotengra.search(text)]
    importers = [path for path, text in sources.items() if "import opt_einsum" in text]
    assert [path.name for path in importers] == ["contraction.py"]
    # ...and there it is inside a function, i.e. indented
    assert "\n    import opt_einsum as oe\n" in sources[importers[0]]


# --- the path itself -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cold_path_cache():
    """``einsum`` caches the path per ``(equation, shapes, strategy)`` (#317).

    Every test that counts searches, or fakes one, needs a cold cache to see its own
    call — and must not leave a fake path behind for the next test to hit.
    """
    contraction._path.cache_clear()
    yield
    contraction._path.cache_clear()


def recorded_paths(monkeypatch):
    """``(paths, shapes)`` recorded from every ``contract_path`` call."""
    import opt_einsum as oe

    paths, shapes = [], []
    real = oe.contract_path

    def recording(subscripts, *operands, **kwargs):
        shapes.append(tuple(operands))
        path, info = real(subscripts, *operands, **kwargs)
        paths.append(path)
        return path, info

    monkeypatch.setattr(oe, "contract_path", recording)
    return paths, shapes


@pytest.mark.parametrize("optimize", ["auto", "greedy", "optimal"])
def test_the_path_is_deterministic_in_one_process(monkeypatch, optimize):
    paths, _ = recorded_paths(monkeypatch)
    equation, legs = chain(PROVIDERS["su2"][0], 5)
    tensors = build(equation, legs, seed=3)
    # cleared per call: the question is whether the *search* is deterministic, and the
    # path cache would otherwise answer the second and third calls without searching
    results = []
    for _ in range(3):
        contraction._path.cache_clear()
        results.append(tenet.einsum(equation, *tensors, optimize=optimize))
    assert len(paths) == 3 and paths[0] == paths[1] == paths[2]
    assert all(tenet.allclose(results[0], other, atol=1e-12) for other in results[1:])


PATH_SCRIPT = """
import opt_einsum as oe
shapes = {shapes!r}
print([oe.contract_path({equation!r}, *shapes, shapes=True, optimize=o)[0]
       for o in ("auto", "greedy", "optimal")])
"""


def test_the_path_is_the_same_in_another_process(monkeypatch):
    paths, _ = recorded_paths(monkeypatch)
    equation, legs = chain(PROVIDERS["su2"][0], 5)
    tensors = build(equation, legs, seed=3)
    for optimize in ("auto", "greedy", "optimal"):
        tenet.einsum(equation, *tensors, optimize=optimize)
    script = PATH_SCRIPT.format(shapes=[t.shape for t in tensors], equation=equation)
    other = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert other.stdout.strip() == repr(paths)


def test_an_explicit_path_is_used_verbatim(monkeypatch):
    paths, _ = recorded_paths(monkeypatch)
    equation, legs = chain(PROVIDERS["su2"][0], 4)
    tensors = build(equation, legs, seed=3)
    explicit = [(1, 2), (0, 2), (0, 1)]
    got = tenet.einsum(equation, *tensors, optimize=explicit)
    assert paths == [explicit]
    # an explicit path is already a path: it never reaches the cache (#317)
    assert contraction._path.cache_info().misses == 0
    assert tenet.allclose(got, tenet.einsum(equation, *tensors), atol=1e-10)


def test_the_planner_sees_physical_shapes_not_reduced_ones(monkeypatch):
    """#19's cost model: ``Σ_a m_a d_a``, not the degeneracy count. Asserted on the
    shapes ``contract_path`` actually received."""
    _, shapes = recorded_paths(monkeypatch)
    equation, legs = chain(PROVIDERS["su2"][0], 4)
    tensors = build(equation, legs, seed=3)
    tenet.einsum(equation, *tensors)
    assert shapes == [tuple(t.shape for t in tensors)]
    assert any(t.shape != t.reduced_shape for t in tensors), (
        "a network whose physical and reduced shapes agree would prove nothing"
    )


def test_a_path_step_that_is_not_a_pair_is_refused(monkeypatch):
    import opt_einsum as oe

    equation, legs = chain(PROVIDERS["su2"][0], 3)
    tensors = build(equation, legs, seed=3)
    monkeypatch.setattr(oe, "contract_path", lambda *a, **k: ([(0,), (0, 1)], None))
    with pytest.raises(ValueError) as excinfo:
        tenet.einsum(equation, *tensors)
    assert "every step must be a pair" in str(excinfo.value)
    assert str(excinfo.value).startswith("einsum: ")


def test_a_cotengra_optimizer_is_accepted_as_it_is():
    """Duck typing, no adapter: cotengra's optimizers *are* ``PathOptimizer``s."""
    cotengra = pytest.importorskip("cotengra")
    import opt_einsum as oe

    assert issubclass(cotengra.HyperOptimizer, oe.paths.PathOptimizer)
    equation, legs = chain(PROVIDERS["su2"][0], 4)
    tensors = build(equation, legs, seed=3)
    got = tenet.einsum(
        equation, *tensors, optimize=cotengra.HyperOptimizer(max_repeats=4, parallel=False)
    )
    np.testing.assert_allclose(
        got.to_dense(), dense_fold(tensors, equation, graded=False), atol=1e-10
    )


# --- the path cache (#317) ------------------------------------------------------------


@pytest.mark.parametrize("provider_id", [*PROVIDERS, "product"])
@pytest.mark.parametrize("n", [3, 4, 6])
def test_the_cached_path_gives_bit_identical_results(provider_id, n):
    """A cached path is the path the search returned, so the second call is not merely
    close to the first: it is the same blocks. Run graded for fZ2 and the product
    provider, where the contraction order carries Koszul signs."""
    if provider_id in ("fz2", "product"):
        equation, tensors, optimize = graded_network(provider_id, n, braided=True)
    else:
        equation, legs = chain(PROVIDERS[provider_id][0], n)
        tensors, optimize = build(equation, legs, seed=3), "auto"
    cold = tenet.einsum(equation, *tensors, optimize=optimize)  # the fixture cleared it
    warm = tenet.einsum(equation, *tensors, optimize=optimize)
    assert cold.structure == warm.structure
    for one, other in zip(cold.blocks, warm.blocks, strict=True):
        np.testing.assert_array_equal(one, other)


def test_the_path_is_searched_once_for_repeated_calls(monkeypatch):
    """The defect of #317, stated structurally: the search must not run per call.

    No wall clock — the assertion is on the number of searches, which is what the
    timing measured.
    """
    paths, _ = recorded_paths(monkeypatch)
    equation, legs = chain(PROVIDERS["su2"][0], 5)
    tensors = build(equation, legs, seed=3)
    for _ in range(5):
        tenet.einsum(equation, *tensors)
    assert len(paths) == 1
    assert contraction._path.cache_info() == (4, 1, None, 1)  # hits, misses, maxsize, size


def test_the_cache_is_keyed_on_the_shapes_too():
    """Same equation, different operands: two searches, not one shared path."""
    equation, legs = chain(PROVIDERS["su2"][0], 4)
    tenet.einsum(equation, *build(equation, legs, seed=3))
    other, other_legs = chain(PROVIDERS["u1"][0], 4)
    assert other == equation
    tenet.einsum(equation, *build(equation, other_legs, seed=3))
    assert contraction._path.cache_info().misses == 2


def test_a_path_optimizer_object_is_consulted_every_call():
    """A ``PathOptimizer`` may be stateful and deliberately non-deterministic
    (cotengra's are), so it is never cached — the docstring promises it reaches
    ``opt_einsum`` unchanged."""
    import opt_einsum as oe

    class Counting(oe.paths.PathOptimizer):
        def __init__(self):
            self.calls = 0

        def __call__(self, inputs, output, size_dict, memory_limit=None):
            self.calls += 1
            return oe.paths.greedy(inputs, output, size_dict, memory_limit)

    equation, legs = chain(PROVIDERS["su2"][0], 4)
    tensors = build(equation, legs, seed=3)
    optimizer = Counting()
    first = tenet.einsum(equation, *tensors, optimize=optimizer)
    second = tenet.einsum(equation, *tensors, optimize=optimizer)
    assert optimizer.calls == 2
    assert contraction._path.cache_info().misses == 0
    assert tenet.allclose(first, second, atol=1e-12)


# --- a provider without ClebschGordanData ---------------------------------------------------------


class Bare:
    """U(1) fusion without ``irrep_dim``: no physical ``shape``, so no ``ClebschGordanData``.

    Mirrors ``tests/test_space.py``'s ``_NoCGCProvider``; the point here is that
    the path finder falls back to ``reduced_shape`` instead of raising.
    """

    name = "Bare"

    def __eq__(self, other):
        return isinstance(other, Bare)

    def __hash__(self):
        return hash(Bare)

    @property
    def unit(self) -> U1Sector:
        return U1Sector(0)

    def dual(self, a: U1Sector) -> U1Sector:
        return U1Sector(-a.charge)

    def fusion(self, a: U1Sector, b: U1Sector) -> tuple[Sector, ...]:
        return (U1Sector(a.charge + b.charge),)

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        return 1


@pytest.mark.parametrize("n", NS)
def test_a_provider_without_clebsch_gordan_still_plans(monkeypatch, n):
    _, shapes = recorded_paths(monkeypatch)
    bare = Bare()
    b1 = GradedSpace.new(bare, {U1Sector(0): 2, U1Sector(1): 3})
    b2 = GradedSpace.new(bare, {U1Sector(0): 1, U1Sector(-1): 2})
    equation, legs = matrix_chain((b1, b2, b1, b2), n)
    tensors = build(equation, legs, seed=3)
    with pytest.raises(CapabilityError, match="ClebschGordanData"):
        _ = tensors[0].shape
    got = tenet.einsum(equation, *tensors, optimize=chain_path(n))
    assert shapes == [tuple(t.reduced_shape for t in tensors)]
    assert got.ndim == 3


# --- refusals still fire with three or more operands ---------------------------------------------


REFUSALS = [
    pytest.param("abc,cde,ef", 2, ["3 comma-separated term", "2 operand"], id="operand-count"),
    pytest.param("ab,cde,ef->abdf", 3, ["term 'ab'", "operand 0 is 3-"], id="term-length"),
    pytest.param("aac,cde,ef->cdf", 3, ["tenet.trace", "repeated inside"], id="trace"),
    pytest.param("aac,cde,ef->acdf", 3, ["diagonal is not defined"], id="diagonal"),
    pytest.param("abc,cda,ea->bd", 3, ["occurs 3 times"], id="label-thrice"),
    pytest.param("abc,cde,ef->abdfz", 3, ["appears in no input term"], id="unknown-output"),
    pytest.param("abc,cde,ef->abdd", 3, ["repeated in the output"], id="repeated-output"),
    pytest.param("abc,cde,ef->abd", 3, ["missing from the output"], id="summed"),
    pytest.param("a...,cde,ef->abdf", 3, ["ellipsis"], id="ellipsis"),
    pytest.param("ab1,cde,ef->ab1df", 3, ["not an ASCII letter"], id="non-alpha"),
]


@pytest.mark.parametrize(("equation", "n", "fragments"), REFUSALS)
def test_loud_refusals(equation, n, fragments):
    _, legs = chain(PROVIDERS["su2"][0], 3)
    tensors = build("", legs, seed=3)
    with pytest.raises(ValueError) as excinfo:
        tenet.einsum(equation, *tensors[:n])
    message = str(excinfo.value)
    assert message.startswith("einsum: ")
    for fragment in fragments:
        assert fragment in message, message


def test_three_operands_is_no_longer_refused():
    equation, legs = chain(PROVIDERS["su2"][0], 3)
    tensors = build(equation, legs, seed=3)
    assert tenet.einsum(equation, *tensors).ndim == 4


def test_a_contraction_leaving_no_free_leg_is_still_refused():
    """Inherited from #51 unchanged: a rank-0 SymmetricTensor does not exist."""
    a = SymmetricTensor.random((Leg(V, OUT), Leg(W, IN)), seed=0)
    b = SymmetricTensor.random((Leg(W, OUT), Leg(U, IN)), seed=1)
    c = SymmetricTensor.random((Leg(U, OUT), Leg(V, IN)), seed=2)
    with pytest.raises(ValueError, match="leaves no free leg"):
        tenet.einsum("ab,bc,ca->", a, b, c)


def test_the_implicit_output_rule_still_applies():
    equation, legs = chain(PROVIDERS["su2"][0], 4)
    tensors = build(equation, legs, seed=3)
    lhs = equation.partition("->")[0]
    assert tenet.allclose(tenet.einsum(lhs, *tensors), tenet.einsum(equation, *tensors), atol=1e-12)


def test_whitespace_is_ignored():
    equation, legs = chain(PROVIDERS["su2"][0], 3)
    tensors = build(equation, legs, seed=3)
    spaced = " abc , cde , ef -> abdf "
    assert tenet.allclose(tenet.einsum(spaced, *tensors), tenet.einsum(equation, *tensors))


def test_ar_do_einsum_takes_three_operands():
    import autoray as ar

    equation, legs = chain(PROVIDERS["su2"][0], 3)
    tensors = build(equation, legs, seed=3)
    assert ar.do("einsum", equation, *tensors) == tenet.einsum(equation, *tensors)


# --- jit and grad ---------------------------------------------------------------------------------


def test_jit_of_four_operands_matches_the_numpy_backend():
    jax = use_jax()
    equation, legs = chain(PROVIDERS["su2"][0], 4)
    tensors = build(equation, legs, seed=3)
    want = tenet.einsum(equation, *tensors)

    traced = []

    def f(*args):
        traced.append(None)
        return tenet.einsum(equation, *args)

    jitted = jax.jit(f)
    moved = [t.to_backend("jax") for t in tensors]
    got = jitted(*moved)
    again = jitted(*moved)
    assert len(traced) == 1, "the path is structural, so one structure set traces once"
    assert got.backend == "jax"
    assert tenet.allclose(to_numpy(got), want, atol=1e-10)
    assert tenet.allclose(to_numpy(again), want, atol=1e-10)


def test_grad_of_four_operands_matches_central_differences():
    jax = use_jax()
    equation, legs = chain(PROVIDERS["su2"][0], 4)
    a, *rest = build(equation, legs, seed=3)
    others = [t.to_backend("jax") for t in rest]

    def loss(t):
        return tenet.norm(tenet.einsum(equation, t, *others))

    grad = jax.grad(loss)(a.to_backend("jax"))
    h = 1e-6
    checked = 0
    for i, block in enumerate(a.blocks):
        if not block.size:
            continue
        for k in (0, block.size - 1):
            shifted = []
            for sign in (+1, -1):
                # a block is a live view into its coupled-sector matrix, so this needs a
                # copy that is both real (writing into it must not reach ``a``) and
                # C-contiguous (``reshape(-1)`` on a non-contiguous copy hands back
                # another copy, and the write below vanishes into it)
                blocks = [np.array(x, order="C", copy=True) for x in a.blocks]
                blocks[i].reshape(-1)[k] += sign * h
                shifted.append(float(loss(SymmetricTensor(a.structure, tuple(blocks)))))
            fd = (shifted[0] - shifted[1]) / (2 * h)
            assert fd == pytest.approx(float(np.asarray(grad.blocks[i]).reshape(-1)[k]), abs=1e-6)
            checked += 1
    assert checked
