"""Finite tensor-network algorithms — DMRG and CTMRG — over the public ``tenet`` API.

Two families, independent of each other:

* **Matrix product states.** [MPS][tenet.network.MPS] and [MPO][tenet.network.MPO] are the
  containers, [MPO.from_terms][tenet.network.MPO.from_terms] builds a Hamiltonian from a
  term list, [Env][tenet.network.Env] caches the ``<psi|H|psi>`` partial contractions, and
  [dmrg_][tenet.network.dmrg_] / [sweep_][tenet.network.sweep_] /
  [lanczos][tenet.network.lanczos] run the ground-state search.
  [MPS.product][tenet.network.MPS.product], [MPS.compress_][tenet.network.MPS.compress_],
  [MPS.save][tenet.network.MPS.save] / [MPS.load][tenet.network.MPS.load] and the two
  [expectation][tenet.network.expectation_1site] values surround it.
* **Corner transfer matrices.** [CTMEnv][tenet.network.CTMEnv],
  [init_env][tenet.network.init_env], [move][tenet.network.move], ``ctmrg`` and
  [ctmrg_unrolled][tenet.network.ctmrg_unrolled] renormalize a C4v environment;
  [single_layer][tenet.network.single_layer], [double_layer][tenet.network.double_layer]
  and [layers][tenet.network.layers] build its absorbers from a bulk tensor or an iPEPS
  ket.

[spectrum][tenet.network.spectrum] and [ones][tenet.network.ones] are shared by both;
[spectrum_sectors][tenet.network.spectrum_sectors] and [entropy][tenet.network.entropy]
sit beside them and are what [MPS.schmidt_values][tenet.network.MPS.schmidt_values],
[MPS.schmidt_sectors][tenet.network.MPS.schmidt_sectors] and
[MPS.entanglement_entropy][tenet.network.MPS.entanglement_entropy] read a bond with.

**Tracing.** Everything here runs **outside** ``jax.jit``/``jax.grad`` by construction and
makes no differentiability claim: ``svd_truncated`` re-decides a bond
[GradedSpace][tenet.GradedSpace] at every bond of every sweep. Two exceptions state
themselves on their own functions — [Env.heff2][tenet.network.Env.heff2]'s prepared matvec
is fixed-structure and traceable through an injected ``compile=``, and in ``ctmrg.py``
[ctmrg_unrolled][tenet.network.ctmrg_unrolled] and ``move(bond=B)`` are shape-static and
differentiable while ``ctmrg`` is not.

Design reasoning, the composition rule every ``einsum`` here obeys, and the
public-``tenet``-API-only hygiene rules: ``docs/design.md`` "Milestone 11", enforced by
``tests/network/test_hygiene.py``.
"""

from tenet.network.common import entropy, ones, spectrum, spectrum_sectors
from tenet.network.ctmrg import (
    Absorb,
    CTMEnv,
    CTMRG_out,
    ctmrg,
    ctmrg_unrolled,
    double_layer,
    double_layer_ctm,
    init_env,
    layers,
    move,
    normalized,
    ring,
    single_layer,
    single_layer_ctm,
)
from tenet.network.dmrg import DMRG_out, Sweep, dmrg_, lanczos, sweep_
from tenet.network.env import Env, correlation_function, measure_mpo
from tenet.network.mps import (
    MPO,
    MPS,
    expectation_1site,
    expectation_2site,
    expectation_profile,
    local_op,
    overlap,
)

__all__ = [
    "MPO",
    "MPS",
    "Absorb",
    "CTMEnv",
    "CTMRG_out",
    "DMRG_out",
    "Env",
    "Sweep",
    "correlation_function",
    "ctmrg",
    "ctmrg_unrolled",
    "dmrg_",
    "double_layer",
    "double_layer_ctm",
    "entropy",
    "expectation_1site",
    "expectation_2site",
    "expectation_profile",
    "init_env",
    "lanczos",
    "layers",
    "local_op",
    "measure_mpo",
    "move",
    "normalized",
    "ones",
    "overlap",
    "ring",
    "single_layer",
    "single_layer_ctm",
    "spectrum",
    "spectrum_sectors",
    "sweep_",
]
