"""``tenet.network.MPO``: the leg structure, the refusal, and the operator it builds."""

import dmrg as example  # noqa: E402  (see conftest.py)
import numpy as np
import pytest

from tenet import IN, OUT, GradedSpace
from tenet.network import MPO
from tenet.symmetry import U1, U1Sector


def test_from_w_gives_every_site_the_same_rank_and_leg_pattern():
    """``(wl IN, p OUT, p IN, wr OUT)`` at every site, with ``D=1`` MPO bonds at the ends.

    The boundary vectors are spelled as ``D=1`` legs rather than as rank-3 end tensors,
    which is what removes the boundary special case from every contraction downstream.
    """
    h = MPO.from_w(
        example.mpo_array(),
        5,
        phys=example.PHYS,
        bond=example.MPO_BOND,
        boundary=example.BOUNDARY,
        start=example._START,
        end=example._END,
    )
    assert len(h) == 5
    for w in h:
        assert w.ndim == 4
        assert tuple(leg.side for leg in w.legs) == (IN, OUT, IN, OUT)
        assert w.legs[1].space == example.PHYS and w.legs[2].space == example.PHYS
    assert h[0].legs[0].space == example.BOUNDARY
    assert h[4].legs[3].space == example.BOUNDARY
    assert h[1].legs[0].space == example.MPO_BOND
    assert h[1].structure == h[2].structure  # one bulk tensor, shared


def test_from_w_refuses_a_perturbed_grading():
    """The refusal is the proof the grading is right; a passing ``allclose`` would not be.

    ``from_dense`` runs at its **default** relative ``atol`` (``ops/dense.py``:301), so
    moving the ``+2`` sector of the MPO bond to ``-2`` -- same total dimension -- makes
    construction *raise* rather than projecting onto a different Hamiltonian. Asserted
    here directly against ``MPO.from_w``, and end to end in
    ``tests/integration/test_dmrg.py``:146-157.
    """
    perturbed = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(-2): 2})
    assert perturbed.dim == example.MPO_BOND.dim
    with pytest.raises(ValueError, match="not symmetric"):
        MPO.from_w(
            example.mpo_array(),
            6,
            phys=example.PHYS,
            bond=perturbed,
            boundary=example.BOUNDARY,
            start=example._START,
            end=example._END,
        )


def test_the_mpo_is_the_heisenberg_hamiltonian_at_n4():
    """Contract to the dense 16 x 16 operator and compare against explicit ``kron``.

    ``tests/integration/test_dmrg.py``:116-143 does this at N=8; at N=4 it is a unit test
    of ``from_w``'s row/column slicing rather than of the physics.
    """
    n_sites = 4
    tensors = [np.asarray(w.to_dense()) for w in example.mpo(n_sites)]
    acc = tensors[0]
    for w in tensors[1:]:
        acc = np.einsum("a...b,bcde->a...cde", acc, w)
    acc = acc[0, ..., 0]
    order = list(range(0, 2 * n_sites, 2)) + list(range(1, 2 * n_sites, 2))
    from_mpo = acc.transpose(order).reshape(2**n_sites, 2**n_sites)

    eye, sz, sp, sm = example._spin_half()

    def at(op, site):
        out = np.array([[1.0]])
        for k in range(n_sites):
            out = np.kron(out, op if k == site else eye)
        return out

    from_kron = sum(
        at(sz, i) @ at(sz, i + 1) + 0.5 * (at(sp, i) @ at(sm, i + 1) + at(sm, i) @ at(sp, i + 1))
        for i in range(n_sites - 1)
    )
    assert np.abs(from_mpo - from_kron).max() < 1e-12


def test_mpo_is_not_an_mps():
    """Two classes, no shape flag: an :class:`MPO` has no ``center`` and no write barrier.

    The claim is that nothing in ``tenet.network`` branches on ``ndim`` or an
    ``nr_phys``-style attribute to decide what it is operating on (YASTN
    ``_mps_obc.py``:284, :291, :438, :443, rejected).
    """
    h = example.mpo(4)
    assert not hasattr(h, "center")
    assert not hasattr(h, "canonize_")
    assert not hasattr(MPO, "__setitem__")
