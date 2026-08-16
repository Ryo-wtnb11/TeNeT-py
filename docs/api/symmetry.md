# `tenet.symmetry`

The symmetry groups and their sectors: U(1), Z2, fermionic Z2, SU(2) and products.

SU(N) lives in `tenet.symmetry.sun` and is **not** re-exported here, because it
needs the optional `racah-py` wheel: `pip install "tenet-py[sun]"`. Importing
`tenet.symmetry.sun` without it raises an `ImportError` naming the extra. There is
no pure-Python fallback and there will not be one — the SU(N) coefficients *are*
the gauge, so a second implementation would be a second source of truth. The
refusal is categorical, not a degraded mode.

::: tenet.symmetry
