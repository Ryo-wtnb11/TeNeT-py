"""The Heisenberg MPO, written out block by block on its named channels.

The Hamiltonian half of ``examples/toy_codes/dmrg.py`` (#268 split it out). It takes the
physical and boundary spaces from ``mps.py`` and is consumed by ``dmrg.py``.

**MPO leg convention**: site ``W_n`` is ``(wl IN, p OUT, p IN, wr OUT)``. Invariance reads
``q(p_out) + q(wr) = q(wl) + q(p_in)``, so an ``S^-`` emitted from the start channel sends
the MPO bond to ``+2`` and an ``S^+`` sends it to ``-2``. The first and last sites carry a
``D=1`` ``mps.BOUNDARY`` MPO bond, which is what makes *every* ``W_n`` rank 4 and removes
the boundary-vector special case.

Simplification: **the MPO is written out, not generated.** Deriving MPO bonds from a term
list is a library feature (``MPO.from_terms``), demonstrated in
``examples/heisenberg_walkthrough.py``. Here the Hamiltonian is one page of blocks.
"""

import numpy as np
from mps import BOUNDARY, PHYS

from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor, TensorStructure
from tenet.symmetry import U1, U1Sector

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
