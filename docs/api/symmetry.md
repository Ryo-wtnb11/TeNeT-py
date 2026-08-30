# `tenet.symmetry`

The symmetry providers and their sectors: U(1), Z2, fermionic Z2, SU(2), SU(N) and
products, plus the capability protocols and the coherence validators.

SU(N) lives in `tenet.symmetry.sun` and is not re-exported here: a provider carries its
own `n`, so there is no singleton to export. Its coefficients, and SU(2)'s, come from
`racah-py`, a core dependency, so `import tenet.symmetry.sun` works on a plain
`pip install tenet-sym`.

::: tenet.symmetry
