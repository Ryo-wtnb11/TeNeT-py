"""The U(1) Heisenberg chain through ``tenet.network``, with the symmetry input spelled out.

Run it standalone::

    uv run python examples/heisenberg_walkthrough.py

**Where this sits.** ``examples/heisenberg.py`` is the short usage example: a term list,
``MPO.from_terms``, a Neel ``MPS.product`` seed, ``dmrg_``, done -- and it never mentions a
``W`` matrix or a bond space. ``examples/toy_codes/dmrg.py`` is the other end: the whole
algorithm written out on ``SymmetricTensor``, importing nothing from ``tenet.network``.
This file is the middle one, and the reason it exists is that the two things a symmetric
DMRG really needs from its user -- **which MPO bond grading**, and **which virtual charges
are reachable** -- are invisible in the first file and buried in the second. It calls the
library for everything (``MPO.from_w``, ``MPO.from_terms``, ``MPS.random``, ``dmrg_``) and
computes only the symmetry input. It carries the physics commentary #110 wrote and #183
moved here when the teaching lane took its algorithm back.

**What moved and what stayed (#112, #183).** The MPS container and its canonical form, the
MPO builders, the directed-bond environment cache with its invalidation, the Krylov step,
the two-site sweep and the convergence loop are :mod:`tenet.network` (``MPS``, ``MPO``,
``Env``, ``lanczos``, ``sweep_``, ``dmrg_``), because every one of those is identical for
every finite-MPS algorithm anyone will write here. What stays in a *user's* file is
physics: the 5x5 Heisenberg ``W`` and its channel constants, the ``2 S^z in {-1, +1}``
grading, and the reachable-charge bond spaces. The line is one sentence long: *the library
takes bond spaces; this file computes which spaces are reachable.*

What it demonstrates:

* a **U(1) MPS whose target sector is fixed by the boundary legs alone**. The physical
  charge is ``t = 2 S^z in {-1, +1}`` (``toy_codes/vmc_mps.SPACES["u1"]``'s spin doublet)
  and both boundary legs are :data:`BOUNDARY`, the unit sector with degeneracy 1.
  Invariance of every site tensor then forces ``Sum_i 2 S^z_i = 0``, i.e. **``S^z_tot =
  0``, the sector the ground state of an even chain lives in, enforced structurally and
  for free** -- no penalty term, no projector, no ``project=`` argument. The general
  recipe is a ``D=1`` boundary leg carrying ``U1Sector(q)``, which targets ``S^z_tot =
  q/2``; YASTN puts exactly that leg on the *first* virtual bond
  (``yastn/tn/mps/_initialize.py``:194, ``ll = Leg(config, s=-1, t=(n0,), D=(1,))``, with
  the last virtual leg required to be zero at :177-179) because it must support any target
  charge. This file spells the recipe and uses the trivial one, because the physics
  question here is the ground state;
* the Heisenberg MPO built from the 5x5 ``W`` matrix written out in the carrier basis, on
  the hand-graded MPO bond :data:`MPO_BOND`. ``MPO.from_w`` hands the array to
  ``SymmetricTensor.from_dense`` (#82) at the **default** relative ``atol``, so a wrong
  grading makes it *raise* -- and that refusal, asserted in
  ``tests/integration/test_dmrg.py`` and again in ``tests/network/test_mpo.py``, is the
  proof the grading is right. A passing ``allclose`` would not be;
* ``tenet.linalg.svd_truncated`` (#77) deciding a bond :class:`~tenet.GradedSpace` at
  **every bond of every sweep** inside ``dmrg_``, with the discarded weight reported by
  Pythagoras -- the mirror image of CTMRG's frozen bond.

**Three routes to the same MPO, and what each is for (#133, #217).**

* :func:`mpo` writes the 5x5 ``W`` out, grades the bond by hand as :data:`MPO_BOND` and
  hands the dense array to ``MPO.from_w``. It stays **first** because the ``W`` matrix and
  its channel table are what teach what an MPO *is*, and a reader who has only ever seen a
  term list cannot debug one. It is also the entry to use when the ``W`` arrives as an
  *array* -- out of a paper, out of another library -- because then the entries are
  numbers and no charge can be recovered from them.
* :func:`mpo_entries` names the same ``W``'s eight non-zero entries and hands them to
  ``MPO.from_entries``. **This is the one to reach for when writing an MPO by hand**: same
  finite-state machine, none of the zeros, and no grading, no boundary vectors and no
  ``dual`` convention to declare -- the charge is already on the operator.
* :func:`mpo_from_terms` lists the Hamiltonian's terms and hands them to
  ``MPO.from_terms``, which derives the same bond spaces from the operators' own charges
  and never mentions a ``W`` at all.

The payoff of keeping all three is a cross-check no single route can produce: a
hand-derived grading and two derived ones must agree as *operators*, and :func:`main` runs
exactly that comparison (``tests/network/test_mpo.py`` asserts it too, including that
``from_terms`` recovers :data:`MPO_BOND` sector for sector, and
``tests/network/test_from_entries.py`` is ``from_entries``' own oracle). The last two also
carry an **edge description** and so run on ``Env.heff2``'s prepared engine path, which
:func:`main` prints; the ``from_w`` operator carries none and takes the compatibility
entry.

**Why there is no ``jit`` and no ``grad`` here, and why that is a decision.** DMRG is a
fixed-point solver whose control flow is data-dependent at every level: the truncation
re-decides the bond space each sweep (``tenet.StructureChangingError`` under a trace, by
design), ``lanczos``'s happy breakdown tests a norm against ``tol``, and ``dmrg_``'s loop
exits on a measured energy change. Every one of those is precisely what tenet refuses to
trace, and correctly. So this module runs on the eager NumPy backend and makes no
differentiability claim of any kind -- and neither does ``tenet.network``, which is outside
``jit``/``grad`` by construction (docs/design.md, M11). There is also, for the same reason,
no XLA compile floor here.

**Leg conventions** are stated where they are enforced -- ``MPS``, ``MPO`` and ``Env``'s
docstrings in :mod:`tenet.network`. The one this file has to know: with ``W_n`` on
``(wl IN, p OUT, p IN, wr OUT)`` invariance reads ``q(p_out) + q(wr) = q(wl) + q(p_in)``,
so an ``S^-`` emitted from the start channel sends the MPO bond to ``+2`` and an ``S^+``
sends it to ``-2``. That is what :data:`MPO_BOND` grades.
"""

import numpy as np

from tenet import GradedSpace, network
from tenet.symmetry import U1, U1Sector

# Physical space: charge t = 2 S^z, so the spin doublet is {-1, +1} -- exactly
# ``toy_codes/vmc_mps.SPACES["u1"]``'s physical leg. BOUNDARY is the unit sector with
# degeneracy 1, used for *both* ends of the MPS (fixing S^z_tot = 0) and of the MPO.
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

# The open-boundary N=12 ground-state energy, computed by exact diagonalization in
# ``tests/integration/test_dmrg.py``; printed by main() as the target, never trusted there.
E_OBC_12 = -5.142090632840532


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


def mpo_from_terms(n_sites: int) -> network.MPO:
    """The same Heisenberg MPO as :func:`mpo`, listed as terms instead of written as ``W``.

    The charges are the only symmetry input, one per operator, and the MPO bond spaces
    fall out of the compressing SVD sweep -- :data:`MPO_BOND` is never mentioned.
    """
    _, sz, sp, sm = _spin_half()
    op = {
        q: network.local_op(o, phys=PHYS, charge=U1Sector(q))
        for q, o in ((0, sz), (-2, sp), (2, sm))
    }
    terms = []
    for i in range(n_sites - 1):
        terms.append((1.0, [(op[0], i), (op[0], i + 1)]))
        terms.append((0.5, [(op[-2], i), (op[2], i + 1)]))
        terms.append((0.5, [(op[2], i), (op[-2], i + 1)]))
    return network.MPO.from_terms(n_sites, terms)


def mpo_entries(n_sites: int) -> network.MPO:
    """The same ``W`` as :func:`mpo_array`, named entry by entry instead of written out.

    This is the middle route, and the one to reach for when the ``W`` matrix *is* what you
    have (M52, #217). Compare it with :func:`mpo_array` above: it is the same finite-state
    machine, the same eight non-zero channels, minus every zero of the 5x5 and minus all
    three pieces of symmetry bookkeeping. There is no :data:`MPO_BOND` -- the charge is
    already on ``local_op``'s third leg, so each channel's :class:`~tenet.GradedSpace` is
    derived and the bond at each cut is the direct sum over its channels; there is no
    ``boundary``, ``start`` or ``end``, because ``0`` is the IdL channel and ``-1`` the IdR
    one at every bond and the two ``D=1`` ends follow; and there is no ``dual`` convention
    to get right, because no rank-4 tensor is ever handed over.

    The channel numbering is the textbook's, not the grading's. :data:`_START` and friends
    above are ordered by *charge*, because ``from_w``'s dense rows have to line up with
    ``GradedSpace``'s ascending sector order -- that constraint is gone here, so ``0`` is
    the start channel, ``1``/``2``/``3`` are ``S^-``/``S^+``/``S^z`` and ``-1`` is the end,
    which is how the lower-triangular ``W`` is printed.

    What it buys beyond the writing: the result carries an edge description, so ``Env``
    cannot tell it from :func:`mpo_from_terms`' operator and it runs on the prepared engine
    path. :func:`mpo`'s output carries none and takes the compatibility entry instead.
    """
    _, sz, sp, sm = _spin_half()
    op = {
        q: network.local_op(o, phys=PHYS, charge=U1Sector(q))
        for q, o in ((0, sz), (-2, sp), (2, sm))
    }
    w = {
        (0, 0): None,  # I -- nothing emitted yet
        (0, 1): (0.5, op[2]),  # S^-/2 out of the start channel ...
        (1, -1): op[-2],  # ... closed by S^+
        (0, 2): (0.5, op[-2]),  # S^+/2 ...
        (2, -1): op[2],  # ... closed by S^-
        (0, 3): op[0],  # S^z ...
        (3, -1): op[0],  # ... closed by S^z
        (-1, -1): None,  # I -- the term is finished
    }
    return network.MPO.from_entries([w] * n_sites)


def bond_spaces(n_sites: int) -> list[GradedSpace]:
    """The ``n_sites + 1`` virtual spaces, degeneracy 1 in every reachable sector.

    The charge on bond ``i`` is the sum of ``i`` physical charges (each ``+-1``) and must
    still be able to reach 0 by site ``n_sites``, so it runs over
    ``-w, -w+2, ..., w`` with ``w = min(i, n_sites - i)``; bonds 0 and ``n_sites`` are
    :data:`BOUNDARY`, which is the whole ``S^z_tot = 0`` statement. This function is the
    reason the library takes bond *spaces* and not a chi: which charges are reachable is
    physics, and it is the only thing about this chain the container cannot know.

    Simplification: degeneracy 1 per sector, not a full-rank seed. Two-site DMRG multiplies a
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


def main(n_sites: int = 12, chi: int = 64):
    """N=12 at chi=64 from the hand-graded ``W``, then the same chain from a term list.

    The two routes are separately converged and compared: a hand-derived MPO bond grading
    and one the library derives from the operators' charges must agree as *operators*, so
    their ground-state energies must agree to solver precision. The N=12 reference printed
    here is the **open**-boundary energy ``-5.142090632840532``; the periodic chain's
    ``-5.387390917445203`` is a different number for a different model and an OBC MPS
    cannot reproduce it.
    """
    out = dmrg(n_sites, chi)
    print(f"from_w      N={n_sites} chi={chi}  E={out.energy:+.12f}  exact={E_OBC_12:+.12f}")
    for i, (e, de, ds, dw) in enumerate(out.history, start=1):
        print(f"  sweep {i:2d}  E={e:+.12f}  dE={de:.3e}  dS={ds:.3e}  dw={dw:.3e}")

    seeded = network.MPS.random(PHYS, bond_spaces(n_sites), seed=0)
    terms = network.dmrg_(seeded, mpo_from_terms(n_sites), chi=chi)
    print(f"from_terms  N={n_sites} chi={chi}  E={terms.energy:+.12f}")
    print(f"  |E(from_w) - E(from_terms)| = {abs(out.energy - terms.energy):.3e}")

    seeded = network.MPS.random(PHYS, bond_spaces(n_sites), seed=0)
    entries = network.dmrg_(seeded, mpo_entries(n_sites), chi=chi)
    print(f"from_entries N={n_sites} chi={chi}  E={entries.energy:+.12f}")
    print(f"  |E(from_w) - E(from_entries)| = {abs(out.energy - entries.energy):.3e}")
    carries = {
        "from_w": mpo(n_sites).edges is not None,
        "from_entries": mpo_entries(n_sites).edges is not None,
        "from_terms": mpo_from_terms(n_sites).edges is not None,
    }
    print(f"  carries an edge description: {carries}")

    def grading(space: GradedSpace) -> str:
        return " ".join(f"{sector.charge:+d}:{m}" for sector, m in space.sectors)

    hand = mpo(n_sites)[n_sites // 2].legs[0].space
    derived = mpo_from_terms(n_sites)[n_sites // 2].legs[0].space
    print(f"  MPO bond, hand-graded: {grading(hand)}")
    print(f"  MPO bond, derived:     {grading(derived)}")

    seed_bonds = [s.dim for s in bond_spaces(n_sites)]
    final = [out.psi[0].legs[0].space.dim] + [t.legs[2].space.dim for t in out.psi]
    print(f"  seed bond dims:  {seed_bonds}")
    print(f"  final bond dims: {final}")
    return out, terms


if __name__ == "__main__":
    main()
