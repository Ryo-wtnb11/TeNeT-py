# `tenet.symmetry`

The symmetry groups and their sectors: U(1), Z2, fermionic Z2, SU(2) and products.

SU(N) lives in `tenet.symmetry.sun` and is **not** re-exported here, because a
provider carries its own `n` and there is no singleton to export. Its coefficients
— and, since #180, SU(2)'s too — come from `racah-py`, a core dependency, so
`import tenet.symmetry.sun` works on a plain `pip install tenet-py`. There is no
pure-Python fallback and there will not be one: the coefficients *are* the gauge,
so a second implementation would be a second source of truth.

`racah-py` ships abi3-py312 wheels for linux x86_64/aarch64, macOS arm64/x86_64
and windows x64. On any other platform pip builds it from the sdist, which needs
a Rust toolchain.

::: tenet.symmetry
