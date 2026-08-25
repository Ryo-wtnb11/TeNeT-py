"""``tenet.models``: the standard sites and the named Hamiltonians on them.

An optional layer above the core, imported explicitly (``from tenet.models import
spin_half``) and re-exported nowhere: **``tenet.network`` never imports it**, which is
what keeps the driver layer from deciding anything about what a caller's operators
mean. ``tests/network/test_hygiene.py`` enforces
that edge.

The *sites* are [spin_half][tenet.models.spin_half],
[spinless_fermion][tenet.models.spinless_fermion],
[spinful_fermion][tenet.models.spinful_fermion] and
[hard_core_boson][tenet.models.hard_core_boson], each returning a
[Site][tenet.models.Site] -- a physical ``GradedSpace``, a name-to-operator mapping in
the shape [MPO.from_arrays][tenet.network.MPO.from_arrays] calls ``ops``, and the dense
matrices behind them.

The *named Hamiltonians* are one function per model, each returning an
[MPO][tenet.network.MPO] over an open chain and taking the symmetry as a parameter:
[heisenberg][tenet.models.heisenberg], [xxz][tenet.models.xxz],
[transverse_field_ising][tenet.models.transverse_field_ising],
[hubbard][tenet.models.hubbard], [spinless_tv][tenet.models.spinless_tv] and
[sun_heisenberg][tenet.models.sun_heisenberg] (whose bond term,
[sun_exchange][tenet.models.sun_exchange], is public too, since measuring it is how a
bulk energy density is read). There is no lattice geometry and no parameter sweep: what
a function does not name -- a next-nearest-neighbour coupling, a ladder, a chemical
potential -- stays the caller's term list, and every one of these models is a short one.

Examples
--------
>>> from tenet.models import heisenberg, spin_half
>>> from tenet.network import MPO
>>> heisenberg(4).to_dense().shape       # the chain in one call
(16, 16)
>>> site = spin_half()                   # or the site, and the terms are yours
>>> n = 4
>>> blocks = [
...     ("Sz Sz", [(i, i + 1) for i in range(n - 1)], [1.0] * (n - 1)),
...     ("S+ S-", [(i, i + 1) for i in range(n - 1)], [0.5] * (n - 1)),
...     ("S- S+", [(i, i + 1) for i in range(n - 1)], [0.5] * (n - 1)),
... ]
>>> h = MPO.from_arrays(n, site.ops, blocks)
>>> h.to_dense().shape
(16, 16)

The build carries no keyword: a builder hands back the **site tensors**, and a
finite-range model's MPO bond is narrow enough that
[Env.heff2][tenet.network.Env.heff2]'s prepared, symbolic path costs more per sweep than
it returns, so the sites are what the sweep should run on. An ab initio Hamiltonian
writes ``symbolic=True`` and keeps the description.
"""

from tenet.models.hamiltonians import (
    heisenberg,
    hubbard,
    spinless_tv,
    sun_exchange,
    sun_heisenberg,
    transverse_field_ising,
    xxz,
)
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
    "heisenberg",
    "hubbard",
    "spin_half",
    "spinful_fermion",
    "spinless_fermion",
    "spinless_tv",
    "sun_exchange",
    "sun_heisenberg",
    "transverse_field_ising",
    "xxz",
]
