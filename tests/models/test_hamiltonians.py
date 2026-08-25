"""The named Hamiltonians against oracles that never touch ``tenet.models``.

Two lanes. The **operator** lane builds each model's dense matrix with numpy ``kron``
alone -- the same site matrices spelled out here, the Jordan-Wigner string written by
hand for the fermionic ones -- and asserts it is what ``MPO.to_dense`` returns: a
Hamiltonian is judged as an operator before any energy is quoted. The **energy** lane
runs DMRG against numbers pinned elsewhere in the suite: the open-chain N=12 Heisenberg
ED value, the MPSKit Hubbard fixture, and Sutherland's Bethe-ansatz bracket for the
SU(3) fundamental chain.
"""

import json
import pathlib

import numpy as np
import pytest

from tenet import GradedSpace
from tenet.models import (
    heisenberg,
    hubbard,
    spin_half,
    spinful_fermion,
    spinless_tv,
    sun_exchange,
    sun_heisenberg,
    transverse_field_ising,
    xxz,
)
from tenet.network import MPS, dmrg_
from tenet.symmetry import SU2, FZ2Sector, SU2Sector, SUNProvider, SUNSector, Trivial, U1Sector, fZ2

# The open-boundary N=12 Heisenberg ground energy, ED, as ``tests/integration/test_dmrg.py``
# and ``tests/test_examples.py`` both pin it.
E_OBC_12 = -5.142090632840532

# Sutherland, Phys. Rev. B 12, 3795 (1975): 1 + (2/N)(gamma + psi(1/N)) per site for the
# infinite SU(N) fundamental chain with H = sum_i P_{i,i+1}, at N = 3.
SUTHERLAND_3 = 1.0 - np.log(3.0) - np.pi / (3.0 * np.sqrt(3.0))

HUBBARD_FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "mpskit_hubbard.json"

# The site matrices, written out here so the oracle shares nothing with the library.
SZ, SP = np.diag([-0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 0.0]])
SIGMA_X, SIGMA_Z = np.diag([1.0, -1.0]), np.array([[0.0, 1.0], [1.0, 0.0]])
A = np.array([[0.0, 1.0], [0.0, 0.0]])
C_UP, C_DN = np.zeros((4, 4)), np.zeros((4, 4))
C_UP[0, 2], C_UP[3, 1] = 1.0, 1.0
C_DN[0, 3], C_DN[2, 1] = 1.0, -1.0
N_UP, N_DN = C_UP.T @ C_UP, C_DN.T @ C_DN


def embed(op, i, n, string=None):
    """``op`` on site ``i`` of ``n``, with ``string`` on every site to its left."""
    d = op.shape[0]
    left = [np.eye(d) if string is None else string] * i
    full = np.array([[1.0]])
    for f in left + [op] + [np.eye(d)] * (n - i - 1):
        full = np.kron(full, f)
    return full


def spin_chain(n, jz, jxy):
    """``sum_i jz S^z S^z + jxy (S^+ S^- + S^- S^+)``, dense, over (|down>, |up>)."""
    h = np.zeros((2**n, 2**n))
    for i in range(n - 1):
        h += jz * embed(SZ, i, n) @ embed(SZ, i + 1, n)
        h += jxy * embed(SP, i, n) @ embed(SP.T, i + 1, n)
        h += jxy * embed(SP.T, i, n) @ embed(SP, i + 1, n)
    return h


def swap_matrix(colors):
    """The permutation of two SU(N) fundamentals as an ``(N, N, N, N)`` array."""
    return np.eye(colors**2).reshape((colors,) * 4).transpose(0, 1, 3, 2)


# --------------------------------------------------------------------------------------
# The operator lane: dense form against numpy kron.
# --------------------------------------------------------------------------------------


def test_heisenberg_u1_is_the_dot_product_chain():
    for n in (2, 4):
        for j in (1.0, -2.0):
            assert np.allclose(heisenberg(n, J=j).to_dense(), spin_chain(n, j, 0.5 * j))


def test_heisenberg_su2_has_the_u1_spectrum():
    """SU(2) resolves no ``S^z``, so the operators are compared where they agree: the
    spectrum, which is basis-free and carries the whole multiplet structure."""
    for n in (2, 4):
        su2 = np.linalg.eigvalsh(heisenberg(n, SU2).to_dense())
        u1 = np.linalg.eigvalsh(heisenberg(n).to_dense())
        assert np.abs(np.sort(su2) - np.sort(u1)).max() < 1e-12


def test_xxz_scales_the_two_halves_independently():
    for n in (2, 4):
        assert np.allclose(xxz(n, Delta=0.5, J=2.0).to_dense(), spin_chain(n, 1.0, 1.0))
    # Delta = 1 is the Heisenberg chain, term for term.
    assert np.array_equal(xxz(4).to_dense(), heisenberg(4).to_dense())


def test_transverse_field_ising_is_the_pauli_chain():
    """In the ``sigma^x`` eigenbasis the field is the diagonal operator and the bond the
    off-diagonal one, which is what the Z2 grading names even and odd."""
    n, g = 4, 0.5
    h = -sum(embed(SIGMA_Z, i, n) @ embed(SIGMA_Z, i + 1, n) for i in range(n - 1))
    h -= g * sum(embed(SIGMA_X, i, n) for i in range(n))
    assert np.allclose(transverse_field_ising(n, g=g).to_dense(), h)


def test_spinless_tv_is_the_jordan_wigner_chain():
    n, v, string = 4, 2.0, np.diag([1.0, -1.0])
    h = np.zeros((2**n, 2**n))
    for i in range(n - 1):
        hop = embed(A.T, i, n, string) @ embed(A, i + 1, n, string)
        h += -(hop + hop.T)
        h += v * embed(A.T @ A, i, n) @ embed(A.T @ A, i + 1, n)
    assert np.allclose(spinless_tv(n, V=v).to_dense(), h)


def test_hubbard_is_the_jordan_wigner_chain():
    n, u, string = 3, 4.0, np.diag([1.0, 1.0, -1.0, -1.0])
    h = np.zeros((4**n, 4**n))
    for i in range(n - 1):
        for c in (C_UP, C_DN):
            hop = embed(c.T, i, n, string) @ embed(c, i + 1, n, string)
            h += -(hop + hop.T)
    h += u * sum(embed(N_UP @ N_DN, i, n) for i in range(n))
    assert np.allclose(hubbard(n, U=u).to_dense(), h)


@pytest.mark.parametrize("colors", (2, 3, 4))
def test_sun_exchange_is_the_permutation_and_its_chain_is_the_sum(colors):
    """Two numbers in two blocks are the whole permutation matrix, for every N."""
    swap = swap_matrix(colors)
    assert np.abs(sun_exchange(colors).to_dense() - swap).max() < 1e-13
    n = 3
    flat = swap.reshape(colors**2, colors**2)
    # The two-site block spans sites (i, i+1), so it is embedded by hand rather than
    # through ``embed``, which places one matrix per site.
    h = sum(
        np.kron(np.kron(np.eye(colors**i), flat), np.eye(colors ** (n - i - 2)))
        for i in range(n - 1)
    )
    assert np.abs(sun_heisenberg(n, colors).to_dense() - h).max() < 1e-12


def test_sun_heisenberg_at_two_colors_is_the_spin_half_chain():
    """``P = 2 S.S + 1/2`` per bond, and the two matrices meet in the same basis."""
    for n in (3, 4):
        shifted = 2.0 * spin_chain(n, 1.0, 0.5) + 0.5 * (n - 1) * np.eye(2**n)
        assert np.abs(sun_heisenberg(n, 2).to_dense() - shifted).max() < 1e-12


def test_the_refusals_name_what_is_shipped():
    with pytest.raises(ValueError, match="spin 1/2"):
        heisenberg(4, spin=1.0)
    with pytest.raises(ValueError, match="U1 and SU2"):
        heisenberg(4, Trivial)
    with pytest.raises(ValueError, match="N >= 2"):
        sun_heisenberg(4, 1)


def test_symbolic_is_passed_through():
    """The keyword reaches the builder: the same operator, the description kept."""
    h = heisenberg(4, symbolic=True)
    assert np.allclose(h.materialize().to_dense(), heisenberg(4).to_dense())


# --------------------------------------------------------------------------------------
# The energy lane: DMRG against numbers pinned elsewhere.
# --------------------------------------------------------------------------------------


def test_heisenberg_dmrg_reaches_the_pinned_n12_energy():
    n = 12
    psi = MPS.product(spin_half().phys, [U1Sector(1 if i % 2 else -1) for i in range(n)])
    out = dmrg_(psi, heisenberg(n), chi=32)
    assert out.energy == pytest.approx(E_OBC_12, abs=1e-9)


def test_heisenberg_su2_dmrg_reaches_the_same_energy():
    n = 12
    end = GradedSpace.new(SU2, {SU2Sector(0): 1})
    mid = GradedSpace.new(SU2, {SU2Sector(0): 4, SU2Sector(1): 4, SU2Sector(2): 2})
    psi = MPS.random(spin_half(SU2).phys, [end] + [mid] * (n - 1) + [end], seed=0)
    out = dmrg_(psi, heisenberg(n, SU2), chi=32)
    assert out.energy == pytest.approx(E_OBC_12, abs=1e-9)


def test_hubbard_dmrg_meets_the_mpskit_fixture():
    fix = json.loads(HUBBARD_FIXTURE.read_text())["N4"]["U4"]
    n = 4
    unit = GradedSpace.new(fZ2, {FZ2Sector(0): 1})
    mid = GradedSpace.new(fZ2, {FZ2Sector(0): 8, FZ2Sector(1): 8})
    psi = MPS.random(spinful_fermion().phys, [unit] + [mid] * (n - 1) + [unit], seed=2)
    out = dmrg_(psi, hubbard(n, U=4.0), chi=16, cutoff=1e-14, max_sweeps=100)
    assert out.energy == pytest.approx(fix["energy"], abs=2e-10)


def test_sun_heisenberg_at_three_colors_brackets_sutherland():
    """An open chain of ``n`` sites has ``n - 1`` bonds, so ``E/n`` sits above the
    infinite-chain energy per site and ``E/(n-1)`` below it."""
    su3 = SUNProvider(3)
    one, three = SUNSector((0, 0)), SUNSector((1, 0))
    n = 12
    end = GradedSpace.new(su3, {one: 1})
    mid = GradedSpace.new(
        su3,
        {one: 2, three: 2, SUNSector((0, 1)): 2, SUNSector((1, 1)): 2, SUNSector((2, 0)): 1},
    )
    psi = MPS.random(GradedSpace.new(su3, {three: 1}), [end] + [mid] * (n - 1) + [end], seed=0)
    out = dmrg_(psi, sun_heisenberg(n, 3), chi=64)
    assert out.energy / (n - 1) < SUTHERLAND_3 < out.energy / n
