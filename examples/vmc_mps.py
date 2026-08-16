"""VMC-shaped end-to-end example: symmetric MPS -> JAX pytree -> grad -> SGD step.

Run it standalone::

    uv run --extra jax python examples/vmc_mps.py

What this demonstrates, entirely with library code that already exists (no new
``src/tenet`` module, no ``optax``, no ``quimb``):

* a symmetric open-boundary MPS whose parameters are :class:`~tenet.SymmetricTensor`
  blocks, i.e. a JAX pytree the moment ``tenet.pytree`` is imported;
* an objective -- the Rayleigh quotient ``<psi|h|psi> / <psi|psi>`` -- built from a
  left-to-right chain of *pairwise* ``tenet.einsum`` calls (three or more operands
  need a contraction path, which is a separate concern);
* ``jax.grad`` straight through that objective, and a one-line SGD step written with
  ``jax.tree.map``;
* ``tenet.linalg.svd`` (exact, shape-static) *inside* the differentiated and jitted
  path, and ``tenet.linalg.svd_truncated`` (structure-changing) *outside* it -- plus
  the pairing of the two, ``compress`` deciding a bond space out here and ``project``
  running ``svd(t, bond=...)`` in there.

**Trivial boundary legs, not a rank-0 tensor.** ``SymmetricTensor`` has no rank 0, and
it does not need one: the standard MPS convention gives the left and right boundary
legs the unit sector with degeneracy 1, so the fully closed network is a rank-2 tensor
with two trivial legs and a single ``(1, 1)`` block. That block *is* the scalar, and
:func:`scalar` is where the tensor world is explicitly left -- the same move
``tenet.norm`` makes.

**Honest limitation on batching.** ``jax.vmap`` batches samples that share one
``TensorStructure``, because the structure is the treedef. A per-sample
computational-basis projector for a genuine Monte-Carlo amplitude has a
sample-dependent sector pattern, hence a sample-dependent structure, hence a
different treedef -- those cannot be ``vmap``ed together. What ``vmap`` does batch is
a set of ansaetze sharing one sector pattern (equivalently, one total charge), which
is the physically meaningful batching for a symmetric ansatz anyway. Sampling itself
is out of scope here.

Simplification: ``h`` is a *random* symmetric two-site operator, not a Heisenberg term.
Equivariance is automatic from the legs and the pipeline under test is identical;
build the physical operator when a physics result -- not a plumbing result -- is
wanted.
"""

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector

# Physical and bond spaces per provider. U(1) is spin-1/2 in the Sz basis (charges
# +-1, so a total charge of 0 is reachable on an even chain); SU(2) is the same spin
# doublet as one irrep. The unit sector with degeneracy 1 is the boundary space.
SPACES = {
    "u1": (
        GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1}),
        GradedSpace.new(U1, {U1Sector(c): 2 for c in (-2, -1, 0, 1, 2)}),
        GradedSpace.new(U1, {U1Sector(0): 1}),
    ),
    "su2": (
        GradedSpace.new(SU2, {SU2Sector(1): 1}),
        GradedSpace.new(SU2, {SU2Sector(j): 2 for j in (0, 1, 2)}),
        GradedSpace.new(SU2, {SU2Sector(0): 1}),
    ),
}

# Physical-axis labels for the einsum chain; one letter per site.
_PHYS = "bcdefghijklmnopqrstuvw"


def build_mps(n_sites: int, *, provider: str = "u1", seed: int = 0) -> list[SymmetricTensor]:
    """Open-boundary MPS, ``A_i`` with legs ``(left bond OUT, physical OUT, right bond IN)``.

    The first tensor's left leg and the last tensor's right leg live in the trivial
    space (unit sector, degeneracy 1) -- the reason the closed network is rank 2 and
    not rank 0. Charge flows left to right: ``bond_i (x) phys_i -> bond_{i+1}``.
    """
    phys, bond, trivial = SPACES[provider]
    spaces = [trivial, *[bond] * (n_sites - 1), trivial]
    return [
        SymmetricTensor.random(
            (Leg(spaces[i], OUT), Leg(phys, OUT), Leg(spaces[i + 1], IN)), seed=seed + i
        ).to_backend("jax")
        for i in range(n_sites)
    ]


def build_h(provider: str = "u1", seed: int = 100) -> SymmetricTensor:
    """A random two-site operator on legs ``(P OUT, P OUT, P IN, P IN)``."""
    phys = SPACES[provider][0]
    legs = (Leg(phys, OUT), Leg(phys, OUT), Leg(phys, IN), Leg(phys, IN))
    return SymmetricTensor.random(legs, seed=seed).to_backend("jax")


def canonicalize(mps: list[SymmetricTensor]) -> list[SymmetricTensor]:
    """Left-canonicalize the first bond with ``tenet.linalg.svd``.

    ``svd`` is the *exact* compact factorization: its bond space is
    ``min(rows_c, cols_c)`` per coupled sector, which is metadata and never a
    numerical rank, so the output structure is data-independent. That is precisely
    why it is safe here, inside the differentiated and jitted objective.
    ``svd_truncated`` selects sectors from the singular *values* and therefore cannot
    be -- see :func:`compress`.

    ``A_0 = U @ S @ Vh`` exactly, so absorbing ``S @ Vh`` into ``A_1`` leaves the
    energy unchanged; this is a gauge transformation, not an approximation.
    """
    u, s, vh = tenet.linalg.svd(mps[0])
    # (bond OUT, old bond IN) @ (old bond OUT, phys OUT, next IN) -> (bond, phys, next)
    absorbed = tenet.einsum("xy,yzw->xzw", tenet.einsum("xy,yz->xz", s, vh), mps[1])
    return [u, absorbed, *mps[2:]]


def compress(t: SymmetricTensor, max_bond: int = 2):
    """``svd_truncated`` -- **outside** ``jit``/``grad``, and that is the point.

    Which sectors survive depends on the singular values, so the output
    ``TensorStructure`` depends on the data. Under a trace the values are tracers,
    ``float(sigma)`` raises, and ``tenet.StructureChangingError`` is raised with an
    explanation. Structure-changing steps belong between optimization steps (or in a
    setup phase), never inside one.

    The escape hatch is to keep the *decision* out here and take only its result in::

        bond = compress(t0, max_bond=D)[0].legs[-1].space   # decided once, outside
        jax.jit(jax.grad(lambda t: tenet.norm(project(t, bond)[1])))(t)

    -- see :func:`project`. That is the shape a differentiable CTMRG or variational
    iPEPS has: a truncation lives inside the differentiated loop, and the kept
    subspace is frozen across it, because re-deciding it every iteration would mean
    differentiating through a discrete choice, which has no derivative.
    """
    return tenet.linalg.svd_truncated(t, max_bond=max_bond)


def project(t: SymmetricTensor, bond):
    """``svd`` onto a bond space :func:`compress` decided -- **inside** ``jit``/``grad``.

    Fixed shape, so it traces and differentiates like any other ``svd``. The only
    thing that changes is exactness: this is the best approximation at ``bond``'s
    per-sector ranks, not a factorization of ``t`` (see :func:`tenet.linalg.svd`).
    """
    return tenet.linalg.svd(t, bond=bond)


def contract(mps: list[SymmetricTensor]) -> SymmetricTensor:
    """The open ket ``(left boundary, phys_0 ... phys_{n-1}, right boundary)``.

    An explicit left-to-right sequence of *pairwise* ``einsum`` calls. This is also
    what makes the ``jit`` behaviour easy to reason about: one static chain, one
    trace. It collapses to a single call once a multi-operand contraction path lands.
    """
    psi, eq = mps[0], "a" + _PHYS[0] + "z"
    for i, a in enumerate(mps[1:], start=1):
        out = eq[:-1] + _PHYS[i] + "y"
        psi = tenet.einsum(f"{eq},z{_PHYS[i]}y->{out}", psi, a)
        eq = out[:-1] + "z"
    return psi


def scalar(t: SymmetricTensor):
    """Leave the tensor world: a rank-2 tensor on two trivial legs is its 1x1 block."""
    return t.blocks[0][0, 0]


def energy(mps: list[SymmetricTensor], h: SymmetricTensor):
    """``<psi|h|psi> / <psi|psi>`` with ``h`` acting on sites 0 and 1.

    The bra is ``tenet.adjoint(psi)`` (every leg flips side, blocks conjugated), so
    contracting it against the ket over every physical leg *and* the left boundary
    leaves the two right-boundary legs open: rank 2, one ``(1, 1)`` block.
    """
    psi = contract(canonicalize(mps))
    n = psi.ndim - 2
    phys, rest = _PHYS[:2], _PHYS[2:n]
    hpsi = tenet.einsum(f"BC{phys},a{phys}{rest}z->aBC{rest}z", h, psi)
    bra = tenet.adjoint(psi)
    num = tenet.einsum(f"aBC{rest}s,aBC{rest}z->sz", bra, hpsi)
    den = tenet.einsum(f"a{phys}{rest}s,a{phys}{rest}z->sz", bra, psi)
    return scalar(num) / scalar(den)


def step(mps: list[SymmetricTensor], h: SymmetricTensor, lr: float):
    """One plain SGD step. ``optax`` would slot in here; one line does not need it."""
    import jax

    e, grads = jax.value_and_grad(energy)(mps, h)
    return [jax.tree.map(lambda p, g: p - lr * g, t, g) for t, g in zip(mps, grads, strict=True)], e


def main(n_sites: int = 4, steps: int = 20, seed: int = 5, provider: str = "u1", lr: float = 0.01):
    """Run the loop and return the energy trace (one entry per step, pre-update).

    Simplification: ``seed=5`` is not magic, it is a seed whose initial state is far enough
    from the minimum that 20 steps of this ``lr`` are all still visibly downhill for
    both providers. On a converged plateau consecutive energies differ by less than
    float64 resolution and "strictly decreasing" stops meaning anything.
    """
    import tenet.pytree  # noqa: F401  # registration is the import's side effect

    mps, h = build_mps(n_sites, provider=provider, seed=seed), build_h(provider)
    trace = []
    for _ in range(steps):
        mps, e = step(mps, h, lr)
        trace.append(float(e))
    return trace


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)  # tests/conftest.py does this for the suite
    for provider in ("u1", "su2"):
        trace = main(provider=provider)
        print(f"{provider}: " + " ".join(f"{e:+.6f}" for e in trace))
