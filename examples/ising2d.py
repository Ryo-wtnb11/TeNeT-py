"""2D classical Ising through ``tenet.network.ctmrg``, on a core install -- no JAX.

Run it standalone::

    uv run python examples/ising2d.py

The library owns the sweep; it must not own the bulk tensor, so the Boltzmann tensor,
Baxter's telescoping and the Onsager oracle are borrowed from the teaching lane
(``examples/toy_codes/ctmrg.py``) rather than copied. The ordered-phase corner spectrum
prints exactly two-fold degenerate across the Z2 parity sectors.
"""

import pathlib
import sys

from tenet.network import ctmrg, single_layer_ctm, spectrum

sys.path.insert(0, str(pathlib.Path(__file__).parent / "toy_codes"))
import ctmrg as toy  # noqa: E402  (the teaching lane: ising_bulk, onsager, log_kappa)


def main(chi: int = 24):
    """Free energy against Onsager at three betas; returns {beta: (beta*f, rel_err)}."""
    results = {}
    for beta in (0.3, toy.BETA_C, 0.5):
        out = ctmrg(*single_layer_ctm(toy.ising_bulk(beta)), chi=chi)
        bf = -float(toy.log_kappa(beta, out.env))
        rel = abs(bf / toy.onsager(beta) - 1)
        print(f"beta={beta:.4f}  {out.sweeps:3d} sweeps  beta*f = {bf:+.10f}  rel {rel:.1e}")
        results[beta] = (bf, rel)
    # The exact cross-sector doublet is a sharper question than the free energy, so the
    # ordered environment is swept to the float64 floor for it (as the integration suite
    # does) rather than to the default 1e-10 the loop above uses.
    ordered = ctmrg(*single_layer_ctm(toy.ising_bulk(0.5)), chi=chi, tol=1e-14, max_sweeps=200)
    corner = spectrum(ordered.env.c)
    print("corner spectrum at beta=0.5:", " ".join(f"{v:.4f}" for v in corner[:6]))
    return results, corner


if __name__ == "__main__":
    main()
