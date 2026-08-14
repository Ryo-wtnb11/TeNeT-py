"""Issue #69 — the M8 composability test: pytree + einsum + svd in one differentiated program.

This is the first test that composes ``tenet.pytree`` (#41), pairwise contraction
(#51/#52) and fixed-structure ``svd`` (#57) into a single ``jax.grad``-ed objective,
which is what README's "VMC and neural quantum states" diagram claims the library can
already do. It adds nothing to ``src/tenet``: it imports ``examples/vmc_mps.py`` and
runs it, so the example cannot rot.

x64 is enabled process-globally in ``tests/conftest.py``; the finite-difference
tolerances here depend on it, and follow ``tests/backends/test_pytree.py``.
"""

import pathlib
import sys

import numpy as np
import pytest

import tenet
from tenet import SymmetricTensor

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import tenet.pytree  # noqa: E402, F401  # registration is the import's side effect

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples"))
import vmc_mps  # noqa: E402

PROVIDERS = ["u1", "su2"]
N_SITES = 4


@pytest.fixture(params=PROVIDERS)
def provider(request):
    return request.param


@pytest.fixture
def model(provider):
    return vmc_mps.build_mps(N_SITES, provider=provider, seed=5), vmc_mps.build_h(provider)


# --- 1. pytree round-trip --------------------------------------------------------


def test_leaves_are_the_blocks_in_block_order_and_round_trip(model):
    mps, _ = model
    leaves, treedef = jax.tree_util.tree_flatten(mps)
    want = [b for t in mps for b in t.blocks]
    assert len(leaves) == len(want)
    assert all(a is b for a, b in zip(leaves, want, strict=True))
    for t in mps:
        assert len(t.blocks) == t.structure.num_blocks  # blocks *are* block_order

    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert all(a == b and a.structure is b.structure for a, b in zip(back, mps, strict=True))


# --- 2. batched amplitudes -------------------------------------------------------


def test_vmap_over_a_stacked_batch_equals_the_per_sample_loop(provider):
    """The batching recipe from ``tenet/pytree.py``, applied to a list-of-tensors ansatz.

    Same sector pattern for every sample, so one treedef; see the example's docstring
    on why a sample-dependent projector could not be batched this way.
    """
    h = vmc_mps.build_h(provider)
    batch = [vmc_mps.build_mps(N_SITES, provider=provider, seed=s) for s in (5, 15, 25)]
    stacked = jax.tree.map(lambda *bs: jnp.stack(bs), *batch)

    got = jax.vmap(lambda m: vmc_mps.energy(m, h))(stacked)
    assert got.shape == (3,)
    np.testing.assert_allclose(
        np.asarray(got), [float(vmc_mps.energy(m, h)) for m in batch], atol=1e-12
    )


# --- 3. gradient vs central differences ------------------------------------------


def test_grad_matches_central_differences_on_one_block(model):
    mps, h = model
    grads = jax.grad(vmc_mps.energy)(mps, h)
    site, blk, delta = 1, 0, 1e-5

    def shifted(idx, d):
        blocks = list(mps[site].blocks)
        blocks[blk] = blocks[blk].at[idx].add(d)
        perturbed = list(mps)
        perturbed[site] = SymmetricTensor(mps[site].structure, tuple(blocks))
        return float(vmc_mps.energy(perturbed, h))

    block = mps[site].blocks[blk]
    want = np.zeros(block.shape)
    for idx in np.ndindex(block.shape):
        want[idx] = (shifted(idx, delta) - shifted(idx, -delta)) / (2 * delta)
    np.testing.assert_allclose(np.asarray(grads[site].blocks[blk]), want, rtol=1e-5, atol=1e-8)


# --- 4. the svd / svd_truncated boundary -----------------------------------------


def test_svd_runs_inside_the_jitted_and_differentiated_path(model):
    """``energy`` calls ``canonicalize``, i.e. ``tenet.linalg.svd``, on every evaluation."""
    mps, h = model
    eager = jax.grad(vmc_mps.energy)(mps, h)
    jitted = jax.jit(jax.grad(vmc_mps.energy))(mps, h)
    for a, b in zip(jitted, eager, strict=True):
        for x, y in zip(a.blocks, b.blocks, strict=True):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-10)

    # the gauge transformation is exact: canonicalizing does not move the energy
    assert float(vmc_mps.energy(vmc_mps.canonicalize(mps), h)) == pytest.approx(
        float(vmc_mps.energy(mps, h)), rel=1e-10
    )


def test_jitted_step_traces_once_and_matches_the_eager_step(model):
    mps, h = model
    count = 0

    @jax.jit
    def jstep(m, h):
        nonlocal count
        count += 1  # a Python side effect: trace time only
        return vmc_mps.step(m, h, 0.01)

    a, ea = jstep(mps, h)
    b, eb = jstep(a, h)
    assert count == 1  # the structure is preserved by the step, so one trace covers both

    want_a, want_ea = vmc_mps.step(mps, h, 0.01)
    want_b, want_eb = vmc_mps.step(want_a, h, 0.01)
    assert float(ea) == pytest.approx(float(want_ea), rel=1e-10)
    assert float(eb) == pytest.approx(float(want_eb), rel=1e-10)
    for got, want in zip(b, want_b, strict=True):
        for x, y in zip(got.blocks, want.blocks, strict=True):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-10)


def test_svd_truncated_works_outside_and_is_refused_under_jit(model):
    """Pinned, not merely commented: the structure-changing factorization cannot trace."""
    mps, _ = model
    u, s, vh = vmc_mps.compress(mps[0], max_bond=2)
    assert u.legs[-1].space.dim <= 2
    assert float(tenet.norm(u @ s @ vh)) <= float(tenet.norm(mps[0])) + 1e-10

    with pytest.raises(tenet.StructureChangingError, match="tenet.linalg.svd"):
        jax.jit(vmc_mps.compress)(mps[0])


# --- 5. optimization -------------------------------------------------------------


def test_main_energy_decreases_monotonically(provider):
    trace = vmc_mps.main(n_sites=N_SITES, steps=3, provider=provider)
    assert len(trace) == 3
    assert all(b < a for a, b in zip(trace, trace[1:], strict=False)), trace


def test_final_tensors_still_construct_with_the_original_structure(model):
    mps, h = model
    trained = mps
    for _ in range(3):
        trained, _ = vmc_mps.step(trained, h, 0.01)
    for got, ref in zip(trained, mps, strict=True):
        assert got.structure is ref.structure
        SymmetricTensor(got.structure, got.blocks)  # the trust boundary, re-run
