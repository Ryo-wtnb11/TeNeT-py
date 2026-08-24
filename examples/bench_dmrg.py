"""The two-site matvec instrument: ``heff2`` wall time, dispatch count, field count.

A script and deliberately not a test (#141 decision 6): every number it prints is a
measurement for a PR body or a machine-to-machine comparison, and nothing here is
asserted. The grid is ``(model, N, chi, backend, compile)`` over the U(1) Heisenberg
chain (``D_w = 5``), the SU(2) Heisenberg chain (invariant two-site term), and a
width-``ly`` Heisenberg cylinder (``D_w = 3*ly + 2`` at the widest cut) -- the three
Hamiltonians whose prepared operators exercise every populated field pattern.

``--backend jax --jit`` hands ``jax.jit`` to :class:`tenet.network.Env` as the
``compile=`` callable -- the network layer names no accelerator, so the injection
happens here, at application level, exactly like ``examples/toy_codes/dmrg.py``'s backend choice.
No GPU number is promised anywhere; this is the instrument that would measure one.
"""

import argparse
import pathlib
import sys
import time

import tenet
from tenet import GradedSpace
from tenet.models import spin_half
from tenet.network import MPO, MPS, Env
from tenet.symmetry import SU2, SU2Sector

sys.path.insert(0, str(pathlib.Path(__file__).parent / "toy_codes"))
import dmrg as example  # noqa: E402  (examples/toy_codes, same pattern as the tests)


def pair_terms(pairs):
    op_sz, op_sp, op_sm = (spin_half().ops[name] for name in ("Sz", "S+", "S-"))
    terms = []
    for i, j in pairs:
        # S.S on an arbitrary pair (i, j), not only nearest neighbours: the longer the
        # pair, the more channels the MPO bond has to carry the term across, which is
        # exactly the D_w the benchmark is varying.
        terms += [
            (1.0, [(op_sz, i), (op_sz, j)]),
            (0.5, [(op_sp, i), (op_sm, j)]),
            (0.5, [(op_sm, i), (op_sp, j)]),
        ]
    return terms


def cylinder_pairs(n, ly):
    # Nearest-neighbour bonds of a width-``ly`` cylinder, snaked onto one chain.
    pairs = []
    # Site i = x*ly + y: the chain visits a whole rung before moving one column right.
    for x in range(-(-n // ly)):
        for y in range(ly):
            i = x * ly + y
            # Around the rung, periodic in y -- this is what makes it a cylinder rather
            # than a strip, and these bonds are short in chain distance.
            pairs.append((i, x * ly + (y + 1) % ly))
            # To the next column: distance ly along the chain, so ly of these bonds are
            # open at once at the widest cut, giving D_w = 3*ly + 2.
            pairs.append((i, i + ly))
    # Deduplicate and drop what the last, possibly partial, column runs off the end of.
    return sorted({(min(i, j), max(i, j)) for i, j in pairs if i != j and max(i, j) < n})


def models(n, ly):
    # cutoff=None keeps the compressing SVD from dropping small channels: the benchmark
    # is timing a matvec at a stated D_w, so the operator must not be silently narrowed.
    heis = MPO.from_terms(n, pair_terms([(i, i + 1) for i in range(n - 1)]), cutoff=None)
    yield "u1-heisenberg", heis, example.PHYS, example.bond_spaces(n)

    su2_site = spin_half(SU2)
    su2_phys = su2_site.phys
    ss = su2_site.ops["S.S"]
    su2 = MPO.from_terms(n, [(1.0, [(ss, (i, i + 1))]) for i in range(n - 1)], cutoff=None)
    tri = GradedSpace.new(SU2, {SU2Sector(0): 1})
    mid = GradedSpace.new(SU2, {SU2Sector(0): 2, SU2Sector(1): 2, SU2Sector(2): 1})
    yield "su2-heisenberg", su2, su2_phys, [tri] + [mid] * (n - 1) + [tri]

    cyl = MPO.from_terms(n, pair_terms(cylinder_pairs(n, ly)), cutoff=None)
    yield f"cylinder-ly{ly}", cyl, example.PHYS, example.bond_spaces(n)


def dispatch_count(fn):
    """Top-level ``autoray.do`` entries during ``fn()`` -- the counter of #141's tables."""
    import autoray

    state = {"count": 0, "depth": 0}
    orig = autoray.do

    def counting(*args, **kwargs):
        # Only depth 0 counts: autoray.do calls itself, and what is being measured is
        # how many array operations the matvec issues, not the recursion inside them.
        state["count"] += state["depth"] == 0
        state["depth"] += 1
        try:
            return orig(*args, **kwargs)
        finally:
            state["depth"] -= 1

    autoray.do = counting
    try:
        fn()
    finally:
        autoray.do = orig
    return state["count"]


def wall(fn, reps):
    fn()  # warm the plan and structure caches; the first call is compilation, not matvec
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


def bench(name, h, phys, bonds, *, backend, compile_, reps, sync=None):
    n = len(h)
    # The middle bond: widest MPO bond, widest state bond, so the worst-case matvec.
    mid = n // 2
    # Canonical form is what a sweep would hand heff2, so the timed call sees the same
    # numbers -- an uncanonical seed can have wildly different magnitudes and denormals.
    psi = MPS.random(phys, bonds, seed=1).canonize_()
    if backend != "numpy":
        # The state moves; the MPO and its block table stay NumPy and are promoted by the
        # backend's own coercion inside each contraction, as in ``MPS.save``'s restore note.
        psi = MPS.from_tensors(t.to_backend(backend) for t in psi)
    envs = {}
    # Two environments over the identical state: the same operator with its edge
    # description kept, and the same operator stripped to bare site tensors. The
    # difference between the two rows is what the prepared path is worth.
    for label, ham in (("prepared", h), ("dense", MPO(h.sites))):
        env = Env(psi, ham, compile=compile_ if label == "prepared" else None)
        env.setup_()
        # Walk the left environment up to the middle bond, so the cached block at ``mid``
        # is the real one a sweep would arrive with rather than a boundary stub.
        for m in range(mid):
            env.update_(m, to="last")
        envs[label] = env
    # The two-site block heff2 acts on: a = left bond, p and q the two physical legs,
    # r = right bond, with the shared bond x between the two site tensors summed away.
    aa = tenet.einsum("apx,xqr->apqr", psi[mid], psi[mid + 1])
    d_w = h[mid].legs[0].space.dim
    chi = psi[mid].legs[0].space.dim
    envs["prepared"].heff2(mid, aa)  # build + (maybe) compile, off the clock
    # Which of the ten term-family fields this Hamiltonian actually populates -- the
    # printed count is why the three models are here rather than one.
    fields = envs["prepared"]._prepared[mid][3]
    row = {}
    for label, env in envs.items():

        def call(env=env):
            out = env.heff2(mid, aa)
            return sync(out) if sync is not None else out

        row[label] = (wall(call, reps), dispatch_count(call))
    print(
        f"{name:16s} N={n:3d} chi={chi:4d} D_w={d_w:3d} backend={backend:5s} "
        f"jit={'y' if compile_ else 'n'}  "
        f"prepared {row['prepared'][0]:8.3f} ms / {row['prepared'][1]:5d} disp  "
        f"dense {row['dense'][0]:8.3f} ms / {row['dense'][1]:5d} disp  "
        f"fields {len(fields)}/10 {','.join(fields)}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sites", type=int, nargs="+", default=[16, 24])
    parser.add_argument("--ly", type=int, default=10)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--jit", action="store_true", help="pass jax.jit as Env(compile=)")
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args(argv)
    compile_ = sync = None
    if args.backend == "jax" or args.jit:
        import jax  # application level: the network layer never names an accelerator

        tenet.enable_jax()  # registers SymmetricTensor as a JAX pytree

        sync = jax.block_until_ready  # async dispatch would otherwise flatter the clock
        if args.jit:
            compile_ = jax.jit
    for n in args.sites:
        for name, h, phys, bonds in models(n, args.ly):
            bench(
                name,
                h,
                phys,
                bonds,
                backend=args.backend,
                compile_=compile_,
                reps=args.reps,
                sync=sync,
            )


if __name__ == "__main__":
    sys.exit(main())
