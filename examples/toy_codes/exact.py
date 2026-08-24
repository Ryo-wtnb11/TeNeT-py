"""Dense exact diagonalization of the same chain: the reference the other two print against.

Run it standalone::

    uv run python examples/toy_codes/exact.py

The whole point of a toy code is to be checkable, and this is what checks the other two.
It is deliberately *not* written on the tensor layer: it takes ``model.h_bonds()`` --
the same two-site operator ``tebd.py`` exponentiates -- reads it out dense with
``to_dense``, and builds the many-body matrix in the ``S^z_tot = 0`` sector with numpy and
nothing else. An oracle that shared the machinery it judges would not be one.

The basis is the ``C(n, n/2)`` bitstrings with ``n/2`` ones, bit ``b`` set meaning spin up
on site ``b``, in ascending integer order -- and the ``(down, up)`` order of the dense
two-site block is ``model.PHYS``'s own sector order, so a bit pair indexes the block
directly. At N=12 that is 924 x 924 and ``eigvalsh`` costs milliseconds.

Simplification: **the ``S^z_tot = 0`` sector only, and dense.** The full ``2**N`` space is
16 times larger at N=12 for a ground state that lives entirely in this sector, and sparse
Lanczos is what ``lanczos.py`` already teaches. Ceiling: ``C(20, 10) = 184756`` is past
what a dense ``eigvalsh`` should be asked for; above N=16 the upgrade path is
``scipy.sparse.linalg.eigsh`` on the same matrix.
"""

import model
import numpy as np


def hamiltonian(n_sites: int) -> np.ndarray:
    """The chain in its ``S^z_tot = 0`` sector, dense, from ``model.h_bonds(n_sites)``.

    Each bond's ``4 x 4`` block is read off the model tensor once; the loop then adds
    ``<bra|h|ket>`` for every basis state and every bond. The exchange terms conserve
    ``S^z_tot`` by construction, so every state they reach is in the basis and there is no
    membership test to get wrong.
    """
    # Each bond operator flattened to 4x4 on (site b, site b+1). The dense readout orders
    # the two-site basis as (dd, du, ud, uu), which is PHYS's own sector order squared.
    gates = [np.asarray(h.to_dense()).reshape(4, 4) for h in model.h_bonds(n_sites)]
    # The basis: every bitstring with exactly half the bits set, i.e. S^z_tot = 0. The
    # ground state of an even Heisenberg chain lives here, so nothing else is built.
    states = [s for s in range(1 << n_sites) if bin(s).count("1") == n_sites // 2]
    index = {s: i for i, s in enumerate(states)}
    h = np.zeros((len(states), len(states)))
    for i, s in enumerate(states):
        for b, gate in enumerate(gates):
            # The two bits this bond acts on, packed high-bit-first into 0..3 to match
            # the gate's column order.
            ket = ((s >> b) & 1) * 2 + ((s >> (b + 1)) & 1)
            for bra, value in enumerate(gate[:, ket]):
                if value:
                    # Clear both bits, then write the row's two bits back in their place:
                    # the same basis state with this one bond's pair replaced.
                    flipped = (
                        (s & ~(1 << b) & ~(1 << (b + 1)))
                        | ((bra >> 1) << b)
                        | ((bra & 1) << (b + 1))
                    )
                    # No membership test: S.S conserves S^z_tot, so every state a nonzero
                    # matrix element reaches has the same bit count and is already indexed.
                    h[index[flipped], i] += value
    return h


def ground_energy(n_sites: int) -> float:
    """The lowest eigenvalue of :func:`hamiltonian`, by ``numpy.linalg.eigvalsh``."""
    # eigvalsh, not eig: H is real symmetric, so the spectrum is real and comes sorted.
    return float(np.linalg.eigvalsh(hamiltonian(n_sites))[0])


def main(sizes=(8, 10, 12)):
    """The open-chain ground-state energy at a few sizes, with the sector dimension."""
    energies = {}
    # Several sizes, so E/N can be watched approaching the thermodynamic limit from
    # above: the two open ends cost energy, and their share shrinks like 1/N.
    for n_sites in sizes:
        h = hamiltonian(n_sites)
        energies[n_sites] = float(np.linalg.eigvalsh(h)[0])
        print(
            f"N={n_sites:2d}  dim={len(h):5d}  E={energies[n_sites]:+.12f}  "
            f"E/N={energies[n_sites] / n_sites:+.12f}  e_inf={model.E_INF:+.12f}"
        )
    return energies


if __name__ == "__main__":
    main()
