"""The U(1) Heisenberg chain, stated once in both the forms an algorithm can consume.

``H = Sum_i (S^z_i S^z_{i+1} + (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) / 2)``, J = 1, open
boundaries. :func:`h_bonds` hands it over as two-site gates, which is what ``tebd.py``
exponentiates; :func:`mpo` hands over the same operator as a matrix product, which is what
``dmrg.py`` builds environments from. Two algorithms, one model -- which is the reason the
model is a file of its own rather than a section of either.

**Physical convention**: charge ``t = 2 S^z``, so the spin doublet is ``{-1, +1}`` and the
dense basis runs ``(down, up)``. :data:`BOUNDARY` is the unit sector with degeneracy 1,
used for both ends of the MPS in ``mps.py`` -- which forces ``S^z_tot = 0`` structurally --
and for both ends of the MPO here.

**MPO leg convention**: site ``W_n`` is ``(wl IN, p OUT, p IN, wr OUT)``. Invariance reads
``q(p_out) + q(wr) = q(wl) + q(p_in)``, so an ``S^-`` emitted from the start channel sends
the MPO bond to ``+2`` and an ``S^+`` sends it to ``-2``. The first and last sites carry a
``D=1`` :data:`BOUNDARY` MPO bond, which is what makes *every* ``W_n`` rank 4 and removes
the boundary-vector special case.

**Two-site gate convention**: ``h`` is ``(P OUT, Q OUT, p IN, q IN)`` -- an operator on the
pair, ready for ``tenet.linalg.expm`` on the partition ``((0, 1), (2, 3))``.

Simplification: **both forms are written out, not generated.** Deriving MPO bonds from a
term list is a library feature (``MPO.from_terms``), demonstrated in
``examples/heisenberg_walkthrough.py``. Here the Hamiltonian is one page of blocks, twice,
and the two pages carry the same six numbers -- which is the point a reader is meant to
check by eye.
"""

import numpy as np

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.symmetry import U1, U1Sector

# Physical space: charge t = 2 S^z, so the spin doublet is {-1, +1} -- exactly
# ``vmc_mps.SPACES["u1"]``'s physical leg. BOUNDARY is the unit sector with degeneracy 1,
# used for *both* ends of the MPS (fixing S^z_tot = 0) and for both ends of the MPO.
PHYS = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
BOUNDARY = GradedSpace.new(U1, {U1Sector(0): 1})

# The thermodynamic limit, 1/4 - ln 2 (Bethe 1931; Hulthen 1938), for the reports.
E_INF = -0.4431471805599453

# The MPO bond: three charge-0 channels (start, S^z, end) and the two S^± channels at
# +-2, because S^± shifts 2 S^z by +-2.
MPO_BOND = GradedSpace.new(U1, {U1Sector(0): 3, U1Sector(2): 1, U1Sector(-2): 1})

# The sectors the blocks below are named in: physical down and up, and the three MPO bond
# charges. ``S^-`` raises the bond charge by 2 and ``S^+`` lowers it by 2.
DOWN, UP = U1Sector(-1), U1Sector(1)
ZERO, PLUS, MINUS = U1Sector(0), U1Sector(2), U1Sector(-2)

# Degeneracy indices inside the charge-0 MPO channel: the "nothing emitted yet" channel,
# the S^z channel and the "term finished" channel. ``GradedSpace`` keeps a sector's
# degeneracies in the order they were given, so these three names are the whole layout.
_END, _SZ, _START = 0, 1, 2


def _blocks(legs, values: dict) -> SymmetricTensor:
    """Build from ``{(sector on each leg, in axis order): block}``.

    ``SymmetricTensor.from_blocks`` is keyed by a ``FusionBlockKey``, which for these legs
    carries the ``OUT`` sectors and the ``IN`` sectors in axis order; naming the sector on
    each leg is the same statement, read left to right off the leg list. Keys not named
    are zero. A sector combination the legs do not allow is not in ``block_order`` at all
    and raises here -- which is why a wrong MPO bond grading is a *refusal* rather than a
    silent projection onto some other operator.
    """
    structure = TensorStructure(tuple(legs))
    outs = [i for i, leg in enumerate(legs) if leg.side is OUT]
    ins = [i for i, leg in enumerate(legs) if leg.side is IN]
    keys = {}
    for key in structure.block_order:
        sectors = dict(zip(outs, key.output_tree.uncoupled, strict=True))
        sectors.update(zip(ins, key.input_tree.uncoupled, strict=True))
        keys[tuple(sectors[i] for i in range(len(legs)))] = key
    return SymmetricTensor.from_blocks(legs, {keys[s]: values[s] for s in values})


# --- the model as two-site gates: what tebd.py exponentiates -----------------------


def sz() -> SymmetricTensor:
    """``S^z`` on one site, legs ``(p OUT, p IN)``. Diagonal, so one 1x1 block per sector."""
    return _blocks(
        (Leg(PHYS, OUT), Leg(PHYS, IN)),
        {(DOWN, DOWN): np.full((1, 1), -0.5), (UP, UP): np.full((1, 1), 0.5)},
    )


def h_bond() -> SymmetricTensor:
    """One bond's ``S^z S^z + (S^+ S^- + S^- S^+) / 2``, legs ``(P OUT, Q OUT, p IN, q IN)``.

    Six allowed blocks, each 1x1x1x1, and they are the six numbers a textbook writes in the
    ``{uu, ud, du, dd}`` basis: ``+1/4`` on the aligned pairs, ``-1/4`` on the antialigned
    ones, and ``1/2`` off-diagonal where the exchange flips an antialigned pair. The
    remaining ten entries of the 4x4 matrix change ``S^z_tot`` and so have no block to live
    in -- the same statement :func:`mpo_blocks` makes about the MPO bond, made about the
    physical legs instead.
    """
    return _blocks(
        (Leg(PHYS, OUT), Leg(PHYS, OUT), Leg(PHYS, IN), Leg(PHYS, IN)),
        {
            (UP, UP, UP, UP): np.full((1, 1, 1, 1), 0.25),
            (DOWN, DOWN, DOWN, DOWN): np.full((1, 1, 1, 1), 0.25),
            (UP, DOWN, UP, DOWN): np.full((1, 1, 1, 1), -0.25),
            (DOWN, UP, DOWN, UP): np.full((1, 1, 1, 1), -0.25),
            (UP, DOWN, DOWN, UP): np.full((1, 1, 1, 1), 0.5),
            (DOWN, UP, UP, DOWN): np.full((1, 1, 1, 1), 0.5),
        },
    )


def h_bonds(n_sites: int) -> list[SymmetricTensor]:
    """The ``n_sites - 1`` two-site terms of the open chain. Translation-invariant, so one
    tensor repeated: the *list* is the interface ``tebd.py`` and ``exact.py`` consume, and a
    non-uniform chain would fill it differently without either of them changing."""
    return [h_bond()] * (n_sites - 1)


# --- the same model as a matrix product operator: what dmrg.py consumes -------------


def mpo_blocks() -> dict:
    """The Heisenberg ``W``, one block per allowed sector tuple ``(wl, p_out, p_in, wr)``.

    ``H = Sum_i (S^z_i S^z_{i+1} + (S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) / 2)``, J = 1, open
    boundaries. As the standard lower-triangular MPO, with ``SM``/``SP`` naming the channel
    entered by emitting an ``S^-``/``S^+``,

    * ``W[START, START] = I``   -- nothing emitted yet;
    * ``W[START, SM] = S^-/2``, ``W[SM, END] = S^+``   -- the ``S^- S^+/2`` term;
    * ``W[START, SP] = S^+/2``, ``W[SP, END] = S^-``   -- the ``S^+ S^-/2`` term;
    * ``W[START, SZ] = S^z``,  ``W[SZ, END] = S^z``    -- the ``S^z S^z`` term;
    * ``W[END, END] = I``      -- the term is finished.

    The symmetry is what splits that matrix into blocks rather than something checked
    afterwards. ``I`` and ``S^z`` keep the physical charge and so live in the two blocks
    on ``wl = wr = 0``, indexed ``[wl channel, 1, 1, wr channel]``; each ``S^±`` moves the
    bond charge by ``-+2`` and so is a block of its own, of extent 1 on that end.
    """
    return {
        # I and S^z: the charge-0 corner of the bond, as a 3x3 channel matrix
        (ZERO, DOWN, DOWN, ZERO): np.array(
            [[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, -0.5, 1.0]]
        ).reshape(3, 1, 1, 3),
        (ZERO, UP, UP, ZERO): np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 1.0]]).reshape(
            3, 1, 1, 3
        ),
        # S^-/2 and S^+/2 leaving the START channel
        (ZERO, DOWN, UP, PLUS): np.array([0.0, 0.0, 0.5]).reshape(3, 1, 1, 1),
        (ZERO, UP, DOWN, MINUS): np.array([0.0, 0.0, 0.5]).reshape(3, 1, 1, 1),
        # S^+ and S^- arriving at the END channel
        (PLUS, UP, DOWN, ZERO): np.array([1.0, 0.0, 0.0]).reshape(1, 1, 1, 3),
        (MINUS, DOWN, UP, ZERO): np.array([1.0, 0.0, 0.0]).reshape(1, 1, 1, 3),
    }


def mpo(n_sites: int, bond: GradedSpace = MPO_BOND) -> list[SymmetricTensor]:
    """The Heisenberg MPO, one rank-4 ``SymmetricTensor`` per site.

    Legs ``(wl IN, p OUT, p IN, wr OUT)``. The bulk tensor is :func:`mpo_blocks` on
    ``bond`` at both ends; the first site is its ``START`` row and the last its ``END``
    column, each on a ``D=1`` :data:`BOUNDARY` MPO leg -- which is what makes every ``W_n``
    rank 4 and removes the boundary-vector special case.

    ``bond`` is a parameter for one reason: so a test can hand it a grading the blocks do
    not fit and assert the refusal.
    """
    blocks = mpo_blocks()

    def legs(left: GradedSpace, right: GradedSpace):
        return (Leg(left, IN), Leg(PHYS, OUT), Leg(PHYS, IN), Leg(right, OUT))

    bulk = _blocks(legs(bond, bond), blocks)
    first = _blocks(
        legs(BOUNDARY, bond),
        {key: blocks[key][_START : _START + 1] for key in blocks if key[0] == ZERO},
    )
    last = _blocks(
        legs(bond, BOUNDARY),
        {key: blocks[key][..., _END : _END + 1] for key in blocks if key[3] == ZERO},
    )
    return [first, *[bulk] * (n_sites - 2), last]
