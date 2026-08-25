"""U(1) Heisenberg ground state through ``tenet.network``, end to end.

Run it standalone::

    uv run python examples/heisenberg.py

What ``examples/toy_codes/dmrg.py`` writes out by hand -- the 5x5 ``W`` matrix, its
channel table, the reachable bond spaces -- this file never mentions: the Hamiltonian is
one call to ``tenet.models.heisenberg``, which writes the term list and lets
``MPO.from_terms`` derive the graded MPO bond, and the Neel product state seeds the
``S^z_tot = 0`` sector by its own charges. ``examples/heisenberg_walkthrough.py``
is the middle of the three: the same library calls, with the ``W`` and the bond spaces
spelled out beside them. The schedule below ramps ``chi``
with noise because writing one is what a user does; on this chain it buys nothing -- the
flat ``chi=64`` run reaches the same energy, since a degeneracy-1 U(1) seed already
ramps for free (see docs/tutorials/dmrg.md, "Schedules and noise", for when it pays).
"""

from tenet.models import heisenberg, spin_half
from tenet.network import MPS, Sweep, dmrg_, expectation_1site, expectation_2site, local_op
from tenet.symmetry import U1Sector

# The standard spin-1/2 site, graded by U(1): charge t = 2 S^z, so the doublet is
# {-1, +1}. Sz/S+/S- arrive as local_op's rank-3 charge-leg operators and S.S as the
# matrix of the invariant two-site term; no spin matrix is written out in this file.
SITE = spin_half()
PHYS = SITE.phys
SZ = SITE.matrices["Sz"]


def main(n_sites: int = 20, chi: int = 64):
    """Ground state at the defaults CI runs; returns the DMRG_out and the bond profile."""
    # Neel: up, down, up, ... Its charges sum to zero, and DMRG never leaves the sector
    # its seed is in, so this one product state is the whole S^z_tot = 0 input.
    psi = MPS.product(PHYS, [U1Sector(1 if n % 2 else -1) for n in range(n_sites)])
    # Staged chi: early sweeps at a small bond are cheap and move the state most, and
    # they hand the later, expensive sweeps a starting point already near the minimum.
    # Noise repopulates bond charges the product seed left empty -- without it a sector
    # that starts at zero weight can never be reached, since the update only rescales
    # what is already there. It decays because by then the missing sectors are found
    # and further noise would only be energy the last, noiseless sweep has to undo.
    schedule = [Sweep(16, noise=1e-4)] * 3 + [Sweep(32, noise=1e-5)] * 3 + [Sweep(chi)]
    # H = sum_i S_i . S_{i+1}, J = 1, open boundaries. Under U(1) that term list is
    # S^z S^z with the transverse half split into S^+ S^- and S^- S^+, since raising and
    # lowering are separate operators here; examples/heisenberg_walkthrough.py writes it
    # out, and this file asks the model function for it.
    out = dmrg_(psi, heisenberg(n_sites), schedule=schedule)
    # legs[0] of site n is its left virtual bond, so this is the cut through the middle
    # of the chain -- the one that carries the most entanglement and the largest bond.
    mid = out.psi[n_sites // 2].legs[0].space
    print(f"N={n_sites}  {out.sweeps} sweeps  E = {out.energy:.15f}  mid bond: {mid.dim} states")

    # S.S as a single two-site matrix, so the bond energy is one expectation value
    # rather than the three-term sum the MPO was built from.
    ss = local_op(SITE.matrices["S.S"], phys=PHYS)
    profile = [expectation_2site(out.psi, ss, n) for n in range(n_sites - 1)]
    print("bond energies:", " ".join(f"{e:+.4f}" for e in profile))
    # H is the sum of those bonds, so the profile summing to out.energy checks the
    # variational energy against a route that never touches the environment caches.
    print(f"sum of bond energies = {sum(profile):.15f}  vs  out.energy = {out.energy:.15f}")

    # <S^z_n> = 0 site by site is not convergence but symmetry: a state living in one
    # U(1) sector has no local magnetisation to round-off, however far from the minimum.
    op_sz = local_op(SZ, phys=PHYS)
    max_sz = max(abs(expectation_1site(out.psi, op_sz, n)) for n in range(n_sites))
    print(f"max_n |<S^z_n>| = {max_sz:.1e}")
    return out, profile


if __name__ == "__main__":
    main()
