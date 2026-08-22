"""M67 (#251): the lattice lane runs on the site-tensor path, and it says so in the code.

The routing was never in doubt -- ``Env.heff2`` branches on ``self.h.edges is not None``
and on nothing else -- so what these tests hold down is the *spelling*: that a caller who
follows ``docs/guide/models-and-sites.md`` (a ``tenet.models`` site, a builder,
``dmrg_``) lands on the site-tensor path without ever meeting an ``EdgeTable``, and that
the same builder without ``materialize()`` still lands on the prepared one, which is what
the quantum-chemistry lane depends on. Which branch ran is *instrumented*, the way
``test_deferred.py`` instruments its allocation claim, rather than inferred from
``h.edges``.
"""

import numpy as np
import pytest

from tenet.models import spin_half, spinful_fermion
from tenet.network import MPO, MPS, Env, dmrg_
from tenet.network import env as env_module
from tenet.symmetry import FZ2Sector, U1Sector

# --- the two models, built exactly as the guide builds them ---------------------------


def guide_heisenberg(n=8):
    """The guide's "A spin model, end to end", verbatim but for its ``n``."""
    site = spin_half()
    bond = [(i, i + 1) for i in range(n - 1)]
    h = MPO.from_arrays(
        n,
        site.ops,
        [
            ("Sz Sz", bond, [1.0] * (n - 1)),
            ("S+ S-", bond, [0.5] * (n - 1)),
            ("S- S+", bond, [0.5] * (n - 1)),
        ],
    )
    psi = MPS.product(site.phys, [U1Sector(1 if i % 2 else -1) for i in range(n)])
    return h, psi


def guide_hubbard(n=6, u=4.0):
    """The guide's "A fermionic model, end to end", verbatim but for its ``n``."""
    site = spinful_fermion()
    fwd = [(m, m + 1) for m in range(n - 1)]
    bwd = [(m + 1, m) for m in range(n - 1)]
    blocks = []
    for flavour in ("up", "dn"):
        expr = f"c+_{flavour} c_{flavour}"
        blocks += [(expr, fwd, [-1.0] * (n - 1)), (expr, bwd, [-1.0] * (n - 1))]
    blocks.append(("n_up n_dn", [(m, m) for m in range(n)], [u] * n))
    h = MPO.from_arrays(n, site.ops, blocks)
    psi = MPS.product(site.phys, [FZ2Sector(i % 2) for i in range(n)])
    return h, psi


MODELS = {"heisenberg": guide_heisenberg, "hubbard": guide_hubbard}


@pytest.fixture
def branches(monkeypatch):
    """``{"prepared": count, "sites": count}`` over whatever runs while this is live.

    The two module globals ``Env.heff2`` looks up are the branch, so counting calls to
    them is the routing claim itself and not a proxy for it.
    """
    seen = {"prepared": 0, "sites": 0}
    apply2, full = env_module._apply2, env_module._heff2_full

    def spy_prepared(*args, **kwargs):
        seen["prepared"] += 1
        return apply2(*args, **kwargs)

    def spy_sites(*args, **kwargs):
        seen["sites"] += 1
        return full(*args, **kwargs)

    monkeypatch.setattr(env_module, "_apply2", spy_prepared)
    monkeypatch.setattr(env_module, "_heff2_full", spy_sites)
    return seen


# --- the acceptance criterion ---------------------------------------------------------


@pytest.mark.parametrize("model", sorted(MODELS))
def test_the_guide_lands_a_lattice_user_on_the_site_tensor_path(model, branches):
    """``tenet.models`` site -> builder -> ``materialize()`` -> ``dmrg_``, and no symbols.

    Nothing in the chain names ``EdgeTable``, ``edges`` or ``edge_blocks``; the only
    thing the caller writes is the ``materialize()`` the guide teaches.
    """
    h, psi = MODELS[model]()
    out = dmrg_(psi, h.materialize(), chi=16, sweeps=2)
    assert branches["sites"] > 0
    assert branches["prepared"] == 0
    assert np.isfinite(out.energy)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_the_builder_without_materialize_still_takes_the_prepared_path(model, branches):
    """The quantum-chemistry lane's default, unchanged: symbols kept, prepared engine.

    ``from_arrays`` is the ab initio front end and this is the routing #218 decided for
    it. The N2 ``K=16`` and C2 ``K=26`` runs behind that decision need an FCIDUMP off the
    network and live in ``benchmarks/``; the routing they depend on is asserted here.
    """
    h, psi = MODELS[model]()
    dmrg_(psi, h, chi=16, sweeps=2)
    assert branches["prepared"] > 0
    assert branches["sites"] == 0


@pytest.mark.parametrize("model", sorted(MODELS))
def test_materialize_changes_the_path_and_not_the_operator(model):
    """The same operator on both paths -- dense, and to solver precision through DMRG."""
    h, psi = MODELS[model]()
    sites = h.materialize()
    assert h.edges is not None and sites.edges is None
    assert np.allclose(h.to_dense(), sites.to_dense(), atol=1e-12)
    a = dmrg_(psi, h, chi=16, sweeps=3)
    b = dmrg_(psi, sites, chi=16, sweeps=3)
    assert abs(a.energy - b.energy) < 1e-9


def test_materialize_on_an_operator_that_already_carries_no_symbols():
    """Idempotent, and it copies the container rather than aliasing it."""
    h, _ = guide_heisenberg(6)
    once = h.materialize()
    twice = once.materialize()
    assert twice.edges is None and twice is not once
    assert [t is u for t, u in zip(once.sites, twice.sites, strict=True)] == [True] * 6


def test_the_families_read_falls_back_to_one_vector_on_the_site_tensor_path():
    """What the lattice path gives up, asserted rather than described (M61 Stage C)."""
    import tenet

    h, psi = guide_heisenberg(6)
    psi.canonize_()
    aa = tenet.einsum("apx,xqr->apqr", psi[0], psi[1])
    assert len(Env(psi, h).setup_().heff2_families(0, aa)) > 1
    assert len(Env(psi, h.materialize()).setup_().heff2_families(0, aa)) == 1
