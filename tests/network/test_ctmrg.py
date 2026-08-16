"""#114 — ``tenet.network.ctmrg``: the structural contract, then the trace boundary.

Two halves, deliberately split by what they need. The **structural** half -- the
:class:`~tenet.network.Absorb` contract for both absorbers, :func:`~tenet.network.move`'s
``ndim // 2`` partition, :func:`~tenet.network.init_env`'s leg pattern,
:func:`~tenet.network.ctmrg` reaching ``tol``, :func:`~tenet.network.ring`'s adjoint
pairing -- needs **no JAX at all**, because the package imports none (hygiene forbids it),
and it runs on Z2, U(1) and SU(2). The **trace** half sits below a module-level
``pytest.importorskip("jax")``, per ``tests/backends/test_ad.py``:29-30, and pins the
``svd_truncated``-outside / ``svd(bond=)``-inside pairing (#77) that this module is the
first library API to carry in its signatures.

The physics is borrowed, not copied: ``ising_bulk`` comes from ``examples/ctmrg.py``, as
``conftest.py`` explains. Runtime budget **15 s**: a Z2 Ising CTMRG at ``chi=4``-``8``
converges in tens of milliseconds per sweep and the two jit tests compile one small move
each. Anything slower than that is an integration test in the wrong directory.
"""

import ctmrg as example  # the example: the Boltzmann tensor and the C4v ansatz constraint
import pytest

import tenet
from tenet import IN, OUT, Leg, SymmetricTensor
from tenet.network import (
    Absorb,
    CTMEnv,
    ctmrg,
    ctmrg_unrolled,
    double_layer,
    double_layer_ctm,
    init_env,
    layers,
    move,
    ring,
    single_layer,
    single_layer_ctm,
)

BETA = 0.4
CHI = 4


def ipeps(provider: str = "u1", seed: int = 1) -> SymmetricTensor:
    """A C4v-symmetric single-site iPEPS on the example's spaces, on the numpy backend.

    ``example.build_ipeps`` hops to ``jax``; the structural half of this module must run
    without it, so only the *spaces* are borrowed.
    """
    phys, virt = example.SPACES[provider]
    legs = (Leg(phys, OUT), Leg(virt, OUT), Leg(virt, OUT), Leg(virt, IN), Leg(virt, IN))
    return example.c4v(SymmetricTensor.random(legs, seed=seed))


# --- structural: no JAX anywhere below this line until the importorskip -------------


def test_single_layer_meets_the_absorb_contract():
    """``corner`` is rank ``2n`` and ``edge`` keeps ``e``'s rank -- the two documented
    callable contracts, on the model-free single-layer absorber."""
    absorb, c, e = single_layer_ctm(example.ising_bulk(BETA))
    assert isinstance(absorb, Absorb)
    big_c = absorb.corner(c, e)
    assert big_c.ndim == 4  # rank 2n with n = 2
    n = big_c.ndim // 2
    p = tenet.linalg.svd_truncated(big_c, (tuple(range(n)), tuple(range(n, 2 * n))), max_bond=CHI)[
        0
    ]
    assert absorb.edge(e, p).ndim == e.ndim


def test_double_layer_meets_the_same_contract_at_rank_six():
    absorb, c, e = double_layer_ctm(ipeps())
    big_c = absorb.corner(c, e)
    assert big_c.ndim == 6  # rank 2n with n = 3: the partition is ndim // 2, not a branch
    n = big_c.ndim // 2
    p = tenet.linalg.svd_truncated(big_c, (tuple(range(n)), tuple(range(n, 2 * n))), max_bond=CHI)[
        0
    ]
    assert absorb.edge(e, p).ndim == e.ndim == 4


def test_the_su2_double_layer_absorbs_and_moves():
    """REPOSITORY_RULES:61 -- one structural test on a non-Abelian provider.

    SU(2) is where the ``ndim // 2`` partition is worth asserting: the enlarged corner's
    two index triples are mirror images whose *multiplet* content is not, so the projector
    is a genuine graded truncation rather than a relabelling.
    """
    ket = ipeps("su2", seed=3)
    absorb, c, e = double_layer_ctm(ket)
    assert absorb.corner(c, e).ndim == 6
    env = move(c, e, absorb, chi=6)
    assert isinstance(env, CTMEnv)
    assert env.e.ndim == 4
    assert env.bond.provider is ket.provider


def test_init_env_leg_pattern_for_one_and_for_two_bonds():
    """The corner is a rank-2 map on the unit sector; the edge carries the bulk bonds."""
    bulk = example.ising_bulk(BETA)
    virt = bulk.legs[0].space
    c, e = init_env(bulk, Leg(virt, IN))
    assert (c.ndim, e.ndim) == (2, 3)
    assert c.legs[0].space.dim == 1 and e.legs[0].space.dim == 1
    assert e.legs[2].space == virt

    ket = ipeps()
    v = ket.legs[1].space
    c, e = init_env(ket, Leg(v, IN), Leg(v, IN, dual=True))
    assert (c.ndim, e.ndim) == (2, 4)
    assert not e.legs[2].dual and e.legs[3].dual  # the bra bond is the bent one


def test_init_env_reads_no_reduced_block_and_keeps_the_dtype():
    """``tenet.identity(legs, dtype=..., like=...)``, not ``like=site.blocks[0]`` (#114).

    The hygiene test forbids ``.blocks`` in the package; the seed is bit-identical either
    way, and that is what this asserts rather than states.
    """
    bulk = example.ising_bulk(BETA)
    c, e = init_env(bulk, Leg(bulk.legs[0].space, IN))
    assert c.dtype == bulk.dtype and e.dtype == bulk.dtype
    assert c.backend == bulk.backend
    assert all(float(v) == 1.0 for b in c.blocks for v in b.ravel())
    assert all(float(v) == 1.0 for b in e.blocks for v in b.ravel())


@pytest.mark.parametrize("provider", ["u1", "su2"])
def test_ring_is_the_four_adjoint_paired_tensors(provider):
    absorb, c, e = double_layer_ctm(ipeps(provider))
    env = move(c, e, absorb, chi=CHI)
    cc, ca, ec, ea = ring(env.c, env.e)
    assert cc is env.c and ec is env.e
    # the far side of the ring is the near side seen from the other side: every leg flips
    assert all(x.side != y.side for x, y in zip(ca.legs, cc.legs, strict=True))
    assert all(x.side != y.side for x, y in zip(ea.legs, ec.legs, strict=True))
    assert float(tenet.norm(ca)) == pytest.approx(float(tenet.norm(cc)))
    assert float(tenet.norm(ea)) == pytest.approx(float(tenet.norm(ec)))


def test_converge_reaches_tol_and_freezes_the_projector_bond():
    env, history = ctmrg(*single_layer_ctm(example.ising_bulk(BETA)), chi=CHI, tol=1e-10)
    assert isinstance(env, CTMEnv)
    assert history[-1] < 1e-10
    assert len(history) < 100  # it stopped on the tolerance, not on max_sweeps

    # the frozen bond *is* the projector's space: one more move on it changes no structure
    again = move(env.c, env.e, single_layer(example.ising_bulk(BETA)), bond=env.bond)
    assert again.bond == env.bond
    assert again.c.structure == env.c.structure and again.e.structure == env.e.structure


def test_converge_stops_at_max_sweeps_when_the_tolerance_is_unreachable():
    _env, history = ctmrg(
        *single_layer_ctm(example.ising_bulk(BETA)), chi=CHI, tol=0.0, max_sweeps=3
    )
    assert len(history) == 3


def test_layers_bends_the_bra_and_flips_dual():
    ket = ipeps()
    same, bra = layers(ket)
    assert same is ket
    assert bra.ndim == 5
    assert bra.legs[0].dual and bra.legs[1].dual  # the bend is what flips ``dual``
    assert isinstance(double_layer(ket, bra), Absorb)


# --- the trace boundary: JAX from here down ----------------------------------------

jax = pytest.importorskip("jax")

import tenet.pytree  # noqa: E402, F401  # registration is the import's side effect


def test_move_on_a_frozen_bond_traces_once_and_retraces_on_a_different_one():
    """The #77 pairing, at unit scope: ``bond=B`` is shape-static, and ``B`` is the key."""
    from functools import partial

    absorb, c, e = single_layer_ctm(example.ising_bulk(BETA))
    env, _ = ctmrg(absorb, c, e, chi=CHI)
    smaller, _ = ctmrg(*single_layer_ctm(example.ising_bulk(BETA)), chi=2)
    assert smaller.bond != env.bond

    count = 0

    @partial(jax.jit, static_argnums=(2, 3))
    def one(c, bulk, bond, k):
        nonlocal count
        count += 1  # a Python side effect: trace time only
        # the absorber is built *inside*, from the traced bulk, so the closure captures a
        # traced value -- the behaviour a Protocol would neither help nor hinder and a
        # stateful class would tempt someone to break
        return ctmrg_unrolled(c, env.e, single_layer(bulk), bond, k=k)[0]

    out = one(env.c, example.ising_bulk(BETA), env.bond, 1)
    assert out.structure == env.c.structure
    one(env.c, example.ising_bulk(BETA + 0.01), env.bond, 1)
    assert count == 1  # different block values, same frozen bond: one trace
    one(env.c, example.ising_bulk(BETA), smaller.bond, 1)
    assert count == 2  # a different GradedSpace is a different cache key


def test_move_with_chi_is_refused_under_jit():
    """``chi=`` decides a structure from singular *values*: that is not traceable (#64)."""
    from functools import partial

    absorb, c, e = single_layer_ctm(example.ising_bulk(BETA))
    with pytest.raises(tenet.StructureChangingError, match="tenet.linalg.svd"):
        jax.jit(partial(move, chi=CHI), static_argnums=(2,))(c, e, absorb)


def test_converge_is_refused_under_jit_before_it_reaches_an_svd():
    """Its loop exit reads a corner spectrum, which is a *data-dependent* exit -- the thing
    the outside/inside split exists to keep outside, and it raises earlier than the SVD."""
    from functools import partial

    absorb, c, e = single_layer_ctm(example.ising_bulk(BETA))
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(partial(ctmrg, chi=CHI, max_sweeps=1), static_argnums=(0,))(absorb, c, e)
