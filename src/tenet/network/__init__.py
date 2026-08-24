"""Finite tensor-network algorithms — DMRG and CTMRG — over the public ``tenet`` API.

Two families, independent of each other:

* **Matrix product states.** [MPS][tenet.network.MPS] and [MPO][tenet.network.MPO] are the
  containers, [MPO.from_terms][tenet.network.MPO.from_terms] builds a Hamiltonian from a
  term list, [Env][tenet.network.Env] caches the ``<psi|H|psi>`` partial contractions, and
  [dmrg_][tenet.network.dmrg_] / [sweep_][tenet.network.sweep_] /
  [lanczos][tenet.network.lanczos] run the variational search -- the ground state, and
  excited states through ``dmrg_``'s ``orthogonal_to=``.
  [MPS.product][tenet.network.MPS.product], [MPS.compress_][tenet.network.MPS.compress_],
  [MPS.save][tenet.network.MPS.save] / [MPS.load][tenet.network.MPS.load] and the two
  [expectation][tenet.network.expectation_1site] values surround it.
* **Two-dimensional states.** [SquareLattice][tenet.network.SquareLattice] and its two
  pattern subclasses carry the geometry, [Lattice][tenet.network.Lattice] the one
  object per unique site, [Peps][tenet.network.Peps] the rank-5 state and
  [Peps2Layers][tenet.network.Peps2Layers] the *view* whose items are lazy
  [DoubleLayer][tenet.network.DoubleLayer] pairs -- the bra-ket product is never formed.
  ``cor_*``, ``edge_*`` and ``append_vec_*`` are the twelve contractions every 2D
  environment is built from, and [EnvCTM][tenet.network.EnvCTM] is the directional
  corner-transfer environment over them: four corners and four edges per site
  ([EnvLocal][tenet.network.EnvLocal]), eight projectors
  ([EnvProjectors][tenet.network.EnvProjectors]) built by
  [corner2x2][tenet.network.corner2x2] and
  [proj_corners][tenet.network.proj_corners], and ``update_``/``iterate_`` reporting a
  [CTM_out][tenet.network.CTM_out]. Its projectors assume nothing about the corner's
  Hermiticity, which is what M63/#243 measured a C4v route cannot have.
  [EnvCTMc4v][tenet.network.EnvCTMc4v] is its C4v specialization: one corner and one edge
  ([EnvLocalC4v][tenet.network.EnvLocalC4v]) for a point-group-symmetric ansatz, whose
  four identical virtual legs tile the plane as a checkerboard of ``A`` and ``flip(A)``
  through [flip][tenet.network.flip] and [PepsFlip][tenet.network.PepsFlip].
* **Time evolution on that layer.** [gate_nn][tenet.network.gate_nn] exponentiates a bond
  Hamiltonian and splits it across the bond, [gates_nn][tenet.network.gates_nn]
  distributes one over a lattice, [apply_gate][tenet.network.apply_gate] puts a
  [Gate][tenet.network.Gate] on its two sites and
  [truncate_][tenet.network.truncate_] reduces the bond it enlarged, in the metric
  ``bond_metric`` supplies -- [EnvCTM.bond_metric][tenet.network.EnvCTM.bond_metric] from
  the six surrounding environment tensors, or [EnvNTU][tenet.network.EnvNTU] from the
  local ``'NN'`` cluster. [evolution_step_][tenet.network.evolution_step_] runs a list of
  gates and reports one [Evolution_out][tenet.network.Evolution_out] per bond -- the
  truncation error and *what the metric was found to be*, which
  [accumulated_truncation_error][tenet.network.accumulated_truncation_error] adds up over
  a trajectory.

[spectrum][tenet.network.spectrum] and [ones][tenet.network.ones] are shared by both;
[spectrum_sectors][tenet.network.spectrum_sectors] and [entropy][tenet.network.entropy]
sit beside them and are what [MPS.schmidt_values][tenet.network.MPS.schmidt_values],
[MPS.schmidt_sectors][tenet.network.MPS.schmidt_sectors] and
[MPS.entanglement_entropy][tenet.network.MPS.entanglement_entropy] read a bond with.

**Tracing.** Everything here runs **outside** ``jax.jit``/``jax.grad`` by construction and
makes no differentiability claim: ``svd_truncated`` re-decides a bond
[GradedSpace][tenet.GradedSpace] at every bond of every sweep. Two exceptions state
themselves on their own functions — [Env.heff2][tenet.network.Env.heff2]'s prepared matvec
is fixed-structure and traceable through an injected ``compile=``, and the fixed-bond
CTM move ``EnvCTMc4v.update_(bond=B)`` is shape-static and differentiable while the
bond-deciding form is not.

**The composition rule** every two-operand ``tenet.einsum`` here obeys: operand 1
supplies the ``IN`` end of every shared wire. Meeting ``IN`` against ``OUT`` is not
enough, because that condition is symmetric and fixes contractibility only, while the
sign a cap pays depends on which operand supplies which end. A wire that genuinely turns
around is bent explicitly with [tenet.repartition][] before the contraction. This module
uses the public ``tenet`` API only, enforced by ``tests/network/test_hygiene.py``.
"""

from tenet.network.common import composed, entropy, ones, spectrum, spectrum_sectors, supplies_in
from tenet.network.dmrg import DMRG_out, Sweep, dmrg_, lanczos, sweep_
from tenet.network.env import Env, correlation_function, measure_mpo
from tenet.network.envctm import (
    CTM_out,
    EnvCTM,
    EnvCTMc4v,
    EnvLocal,
    EnvLocalC4v,
    EnvProjectors,
    PepsFlip,
    corner2x2,
    flip,
    proj_corners,
)
from tenet.network.evolution import (
    EnvNTU,
    Evolution_out,
    Gate,
    accumulated_truncation_error,
    apply_gate,
    evolution_step_,
    gate_nn,
    gates_nn,
    truncate_,
)
from tenet.network.lattice import (
    Bond,
    CheckerboardLattice,
    Lattice,
    RectangularUnitcell,
    Site,
    SquareLattice,
)
from tenet.network.mps import (
    MPO,
    MPS,
    expectation_1site,
    expectation_2site,
    expectation_profile,
    local_op,
    overlap,
)
from tenet.network.peps import (
    DoubleLayer,
    Peps,
    Peps2Layers,
    append_vec_bl,
    append_vec_br,
    append_vec_tl,
    append_vec_tr,
    cor_bl,
    cor_br,
    cor_tl,
    cor_tr,
    edge_b,
    edge_l,
    edge_r,
    edge_t,
)

__all__ = [
    "Bond",
    "CTM_out",
    "CheckerboardLattice",
    "DMRG_out",
    "DoubleLayer",
    "Env",
    "EnvCTM",
    "EnvCTMc4v",
    "EnvLocal",
    "EnvLocalC4v",
    "EnvNTU",
    "Evolution_out",
    "Gate",
    "EnvProjectors",
    "PepsFlip",
    "Lattice",
    "MPO",
    "MPS",
    "Peps",
    "Peps2Layers",
    "RectangularUnitcell",
    "Site",
    "SquareLattice",
    "Sweep",
    "accumulated_truncation_error",
    "append_vec_bl",
    "apply_gate",
    "append_vec_br",
    "append_vec_tl",
    "append_vec_tr",
    "cor_bl",
    "cor_br",
    "cor_tl",
    "cor_tr",
    "composed",
    "corner2x2",
    "correlation_function",
    "dmrg_",
    "edge_b",
    "edge_l",
    "edge_r",
    "edge_t",
    "entropy",
    "evolution_step_",
    "flip",
    "gate_nn",
    "gates_nn",
    "expectation_1site",
    "expectation_2site",
    "expectation_profile",
    "lanczos",
    "local_op",
    "measure_mpo",
    "ones",
    "overlap",
    "proj_corners",
    "spectrum",
    "spectrum_sectors",
    "supplies_in",
    "sweep_",
    "truncate_",
]
