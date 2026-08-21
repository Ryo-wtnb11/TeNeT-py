"""``tenet.models``: the standard local operator sets, so a Hamiltonian starts as physics.

An optional layer above the core, imported explicitly (``from tenet.models import
spin_half``) and re-exported nowhere: **``tenet.network`` never imports it**, which is
what keeps #112/#133's operator-zoo rule true verbatim -- the driver layer still decides
nothing about what a caller's operators mean. ``tests/network/test_hygiene.py`` enforces
that edge.

What ships is *sites*, not models: [spin_half][tenet.models.spin_half],
[spinless_fermion][tenet.models.spinless_fermion],
[spinful_fermion][tenet.models.spinful_fermion] and
[hard_core_boson][tenet.models.hard_core_boson], each returning a
[Site][tenet.models.Site] -- a physical ``GradedSpace``, a name-to-operator mapping in
the shape [MPO.from_arrays][tenet.network.MPO.from_arrays] calls ``ops``, and the dense
matrices behind them. There is no lattice geometry, no ``heisenberg(L)``, and no
parameter sweep: a site set is finite and a model zoo is not, and the Hamiltonian stays
the caller's term list.

Examples
--------
>>> from tenet.models import spin_half
>>> from tenet.network import MPO
>>> site = spin_half()
>>> n = 4
>>> blocks = [
...     ("Sz Sz", [(i, i + 1) for i in range(n - 1)], [1.0] * (n - 1)),
...     ("S+ S-", [(i, i + 1) for i in range(n - 1)], [0.5] * (n - 1)),
...     ("S- S+", [(i, i + 1) for i in range(n - 1)], [0.5] * (n - 1)),
... ]
>>> h = MPO.from_arrays(n, site.ops, blocks)
>>> h.to_dense().shape
(16, 16)
"""

from tenet.models.sites import (
    Site,
    hard_core_boson,
    spin_half,
    spinful_fermion,
    spinless_fermion,
)

__all__ = [
    "Site",
    "hard_core_boson",
    "spin_half",
    "spinful_fermion",
    "spinless_fermion",
]
