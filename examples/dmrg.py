"""Finite-chain two-site DMRG over ``tenet.network``: U(1) Heisenberg against ED.

Run it standalone::

    uv run python examples/dmrg.py

**What is left here, and what moved (#112).** The machinery this file used to carry --
the MPS container and its canonical form, the MPO builder, the directed-bond environment
cache with its invalidation, the Krylov step, the two-site sweep and the convergence loop
-- is now :mod:`tenet.network` (``MPS``, ``MPO``, ``Env``, ``lanczos``, ``sweep_``,
``dmrg_``), because every one of those is identical for every finite-MPS algorithm anyone
will write here. What stays is **physics**: the 5x5 Heisenberg ``W`` and its channel
constants, the ``2 S^z in {-1, +1}`` grading, the reachable-charge bond spaces, and the
thermodynamic limit ``main()`` reports against. The line is one sentence long: *the
library takes bond spaces; this file computes which spaces are reachable*. The promotion
changed no arithmetic and no number -- ``tests/integration/test_dmrg.py`` compares the
same floats it compared before.

What it demonstrates (no ``scipy``, no ``quimb``, no ``jax``):

* a **U(1) MPS whose target sector is fixed by the boundary legs alone**. The physical
  charge is ``t = 2 S^z in {-1, +1}`` (``vmc_mps.SPACES["u1"]``'s spin doublet) and both
  boundary legs are :data:`BOUNDARY`, the unit sector with degeneracy 1. Invariance of
  every site tensor then forces ``Sum_i 2 S^z_i = 0``, i.e. **``S^z_tot = 0``, the sector
  the ground state of an even chain lives in, enforced structurally and for free** -- no
  penalty term, no projector, no ``project=`` argument. The general recipe is a ``D=1``
  boundary leg carrying ``U1Sector(q)``, which targets ``S^z_tot = q/2``; YASTN puts
  exactly that leg on the *first* virtual bond (``yastn/tn/mps/_initialize.py``:194).
  This file spells the recipe and uses the trivial one;
* the Heisenberg MPO built by ``tenet.SymmetricTensor.from_dense`` (#82) at the
  **default** relative ``atol`` from the 5x5 ``W`` matrix written out in the carrier
  basis, on the graded MPO bond :data:`MPO_BOND`. A wrong grading makes ``from_dense``
  *raise*, and that refusal -- asserted in ``tests/integration/test_dmrg.py`` and again
  in ``tests/network/test_mpo.py`` -- is the proof the grading is right. A passing
  ``allclose`` would not be;
* ``tenet.linalg.svd_truncated`` (#77) deciding a bond :class:`~tenet.GradedSpace` at
  **every bond of every sweep**, with the discarded weight reported by Pythagoras --
  the mirror image of CTMRG's frozen bond;
* an **iterative Krylov eigensolver written over ``SymmetricTensor`` as a vector**:
  ``tenet.network.lanczos`` needs only ``tenet.add``/``subtract``, scalar
  multiply/divide, ``tenet.norm`` and an inner product. No ``scipy.sparse.linalg``, no
  dense reshaping of the local problem.

**Why there is no ``jit`` and no ``grad`` here, and why that is a decision.** DMRG is a
fixed-point solver whose control flow is data-dependent at every level: the truncation
re-decides the bond space each sweep (``tenet.StructureChangingError`` under a trace, by
design), ``lanczos``'s happy breakdown tests a norm against ``tol``, and ``dmrg_``'s loop
exits on a measured energy change. Every one of those is precisely what tenet refuses to
trace, and correctly. So this module runs on the eager NumPy backend and makes no
differentiability claim of any kind -- and neither does ``tenet.network``, which is
outside ``jit``/``grad`` by construction (docs/design.md, M11). The #77 pairing --
``svd_truncated`` *outside* the trace, ``svd(bond=)`` *inside* it -- has **two**
legitimate halves, and this file uses one: ``ctmrg.py`` needs the inside half because it
differentiates through its sweeps; DMRG needs only the outside half because it does not.
There is also, for the same reason, no XLA compile floor here.

**Leg conventions** are stated where they are now enforced -- ``MPS``, ``MPO`` and
``Env``'s docstrings in :mod:`tenet.network`. The one this file still has to know: with
``W_n`` on ``(wl IN, p OUT, p IN, wr OUT)`` invariance reads
``q(p_out) + q(wr) = q(wl) + q(p_in)``, so an ``S^-`` emitted from the start channel sends
the MPO bond to ``+2`` and an ``S^+`` sends it to ``-2``. That is what
:data:`MPO_BOND` grades.

Deliberate simplifications, each with its ceiling:

ponytail: **two-site DMRG only; single-site plus subspace expansion is deferred, and
``tenet.linalg.left_null`` (#88) therefore gets no demo here.** Two-site is what makes
``svd_truncated`` the bond-deciding step, which is the tenet feature this example exists
to exercise, and it grows a bond by a factor of ``d`` per site with no extra concept.
Strictly-single-site DMRG at fixed bond *cannot grow a bond at all*, so it is only honest
with subspace expansion (Hubig-McCulloch-Schollwoeck-Wall, PRB 91, 155115 (2015)) -- which
needs ``left_null``, a mixing factor ``alpha``, its own schedule and a second ``heff1``
contraction chain. Named upgrade path, and a good one: ``left_null`` is the only #88
export with no end-to-end user.

ponytail: **hand-written pairwise contraction orders, not ``optimize=`` on a five-operand
einsum** -- now in ``tenet.network.env``, unchanged. Same reason as ``ctmrg._halves``:
``opt_einsum`` costs a graded network from *physical* leg sizes, and a U(1) MPS bond whose
sectors are unevenly filled is exactly where that estimate is wrong. The orders are
YASTN's own (``yastn/tn/mps/_env.py``:496-518), which its ``_dmrg.py``:102-108 documents
as ``O(D^3 M d + D^2 M^2 d^2)`` per matvec -- optimal for *one* matvec, which is all a
Krylov step ever wants.

ponytail: **the MPO is written out, not generated.** YASTN builds MPOs from ``Hterm``
NamedTuples through ``generate_mpo`` and a series of compressing SVDs
(``_generate_mpo.py``:30-51, :73-112). That is the right API for a library that must accept
arbitrary Hamiltonians; it is 300 lines and the wrong thing for a file whose Hamiltonian is
one line of physics. ``tenet.network.MPO.from_w`` takes the array.
"""

import numpy as np

from tenet import GradedSpace, network
from tenet.symmetry import U1, U1Sector

# Physical space: charge t = 2 S^z, so the spin doublet is {-1, +1} -- exactly
# ``vmc_mps.SPACES["u1"]``'s physical leg. BOUNDARY is the unit sector with degeneracy 1,
# used for *both* ends of the MPS (fixing S^z_tot = 0) and for both ends of the MPO.
PHYS = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
BOUNDARY = GradedSpace.new(U1, {U1Sector(0): 1})

# The MPO bond: three charge-0 channels (start, S^z, end) and the two S^± channels at
# +-2, because S^± shifts 2 S^z by +-2. The sign convention is **not derived by hand**:
# the dense W below is handed to ``from_dense`` at the default relative atol, and a wrong
# grading makes it raise. See ``test_dmrg.py::test_mpo_refuses_a_perturbed_grading``.
MPO_BOND = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(2): 1, U1Sector(-2): 1})

# Dense positions of the five MPO channels. ``GradedSpace`` sorts its sectors ascending
# (space.py:68), so the -2 channel is index 0, the three charge-0 channels are 1..3 and
# the +2 channel is index 4. The names, not the numbers, are what the W matrix is written
# in below: ``_SM_CHANNEL`` is the channel *entered* by emitting an S^-, which raises the
# MPO bond charge by +2, and ``_SP_CHANNEL`` the one entered by emitting an S^+.
_SP_CHANNEL, _END, _SZ_CHANNEL, _START, _SM_CHANNEL = 0, 1, 2, 3, 4

# The thermodynamic limit, 1/4 - ln 2 (Bethe 1931; Hulthen 1938), for main()'s report.
E_INF = -0.4431471805599453


def _spin_half() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(I, Sz, S+, S-)`` as 2x2 dense arrays in this module's physical basis.

    Index 0 is charge ``-1`` (spin down) and index 1 is charge ``+1`` (spin up), because
    :data:`PHYS` sorts its sectors ascending. The matrices are ``O[out, in]``.
    """
    eye = np.eye(2)
    sz = np.diag([-0.5, 0.5])
    sp = np.array([[0.0, 0.0], [1.0, 0.0]])  # |down> -> |up>
    return eye, sz, sp, sp.T


def mpo_array() -> np.ndarray:
    """The 5x5 Heisenberg ``W``, dense, indexed ``[wl, p_out, p_in, wr]``.

    ``H = Sum_i (S^z_i S^z_{i+1} + (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) / 2)``, J = 1,
    **open** boundaries. In the channel names of :data:`_START` and friends the standard
    lower-triangular matrix reads

    * ``W[START, START] = I``   -- nothing emitted yet;
    * ``W[START, SM] = S^-/2``, ``W[SM, END] = S^+``   -- the ``S^- S^+/2`` term;
    * ``W[START, SP] = S^+/2``, ``W[SP, END] = S^-``   -- the ``S^+ S^-/2`` term;
    * ``W[START, SZ] = S^z``,  ``W[SZ, END] = S^z``    -- the ``S^z S^z`` term;
    * ``W[END, END] = I``      -- the term is finished.

    The left boundary vector is ``e_START`` and the right one ``e_END``; both are spelled
    as ``D=1`` MPO bond legs by ``MPO.from_w``, which is what makes every ``W_n`` rank 4.
    """
    eye, sz, sp, sm = _spin_half()
    w = np.zeros((5, 2, 2, 5))
    w[_START, :, :, _START] = eye
    w[_END, :, :, _END] = eye
    w[_START, :, :, _SM_CHANNEL] = 0.5 * sm
    w[_SM_CHANNEL, :, :, _END] = sp
    w[_START, :, :, _SP_CHANNEL] = 0.5 * sp
    w[_SP_CHANNEL, :, :, _END] = sm
    w[_START, :, :, _SZ_CHANNEL] = sz
    w[_SZ_CHANNEL, :, :, _END] = sz
    return w


def mpo(n_sites: int, bond: GradedSpace = MPO_BOND) -> network.MPO:
    """The Heisenberg MPO on ``bond``. ``bond`` is a parameter so a test can perturb it."""
    return network.MPO.from_w(
        mpo_array(),
        n_sites,
        phys=PHYS,
        bond=bond,
        boundary=BOUNDARY,
        start=_START,
        end=_END,
    )


def bond_spaces(n_sites: int) -> list[GradedSpace]:
    """The ``n_sites + 1`` virtual spaces, degeneracy 1 in every reachable sector.

    The charge on bond ``i`` is the sum of ``i`` physical charges (each ``+-1``) and must
    still be able to reach 0 by site ``n_sites``, so it runs over
    ``-w, -w+2, ..., w`` with ``w = min(i, n_sites - i)``; bonds 0 and ``n_sites`` are
    :data:`BOUNDARY`, which is the whole ``S^z_tot = 0`` statement. This function is the
    reason the library takes bond *spaces* and not a chi: which charges are reachable is
    physics, and it is the only thing about this chain the container cannot know.

    ponytail: degeneracy 1 per sector, not a full-rank seed. Two-site DMRG multiplies a
    bond by ``d = 2`` per sweep, so a chi=64 bond is three sweeps away and the first
    sweeps are correspondingly cheap. A larger seed is one ``min(..., cap)`` here if a
    workload ever wants the bond full on sweep one.
    """
    spaces = []
    for i in range(n_sites + 1):
        w = min(i, n_sites - i)
        spaces.append(GradedSpace.new(U1, {U1Sector(q): 1 for q in range(-w, w + 1, 2)}))
    return spaces


def dmrg(n_sites: int, chi: int = 64, *, seed: int = 0, **kwargs) -> network.DMRG_out:
    """Seed a random U(1) MPS in the ``S^z_tot = 0`` sector and hand it to the driver."""
    psi = network.MPS.random(PHYS, bond_spaces(n_sites), seed=seed)
    return network.dmrg_(psi, mpo(n_sites), chi=chi, **kwargs)


def main(n_sites: int = 12, chi: int = 64, big_sites: int = 32, big_chi: int = 32):
    """N=12 at chi=64 against the exact ground state, then N=32 at chi=32 against ``e_inf``.

    The N=12 reference printed here is the **open**-boundary energy
    ``-5.142090632840532``; the periodic chain's ``-5.387390917445203`` is a different
    number for a different model and an OBC MPS cannot reproduce it.
    ``tests/integration/test_dmrg.py`` computes the OBC value rather than trusting it.
    """
    small = dmrg(n_sites, chi)
    print(f"N={n_sites} chi={chi}  E={small.energy:+.12f}  exact=-5.142090632840532")
    for i, (e, de, ds, dw) in enumerate(small.history, start=1):
        print(f"  sweep {i:2d}  E={e:+.12f}  dE={de:.3e}  dS={ds:.3e}  dw={dw:.3e}")

    big = dmrg(big_sites, big_chi)
    print(
        f"N={big_sites} chi={big_chi}  E={big.energy:+.12f}  "
        f"E/N={big.energy / big_sites:+.12f}  e_inf={E_INF:+.12f}  "
        f"sweeps={big.sweeps}  max_dw={big.max_discarded_weight:.3e}"
    )
    return small, big


if __name__ == "__main__":
    main()
