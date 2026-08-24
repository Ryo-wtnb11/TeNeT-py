"""#243 Part 3: the enlarged corner's Hermiticity under `EnvCTMc4v`, and whether it matters.

`EnvCTMc4v`'s projector is the `U` of an SVD of the 2x2 enlarged corner, and the
renormalized corner is `V† U S`: the two index groups leave the factorization as two
factors and the correction between them is kept rather than assumed to be one. So the
question this instrument asks is a *conditional* one -- what happens to the sweep when the
corner is nowhere near Hermitian -- and the answer is that the two have come apart: the row
that is far from Hermitian converges too.

The rows vary the one thing worth varying, how much of the point group the ansatz carries,
over ansaetze this lane accepts at all: four **identical** virtual legs, without which no
rotation acts and the lane refuses the tensor. Hermiticity is a property of that ansatz --
it holds at every move exactly when the state carries the whole C4v point group, the four
rotations as well as the four reflections. `tests/network/test_envctmc4v.py` pins both
directions of that statement as assertions; this file prints the numbers behind them.

Run: `uv run python benchmarks/bench_ctm_corner_signs.py`. On no CI path.
"""

import numpy as np

import tenet
from tenet import OUT, GradedSpace, Leg, SymmetricTensor
from tenet.network import EnvCTMc4v, Peps, SquareLattice, corner2x2
from tenet.symmetry import U1, U1Sector


def _c4v_permutations():
    """The eight elements of C4v as permutations of `(t, l, b, r)`, physical leg last."""
    compose = lambda p, q: tuple(p[i] for i in q)  # noqa: E731
    out, current = [], (0, 1, 2, 3, 4)
    for _ in range(4):
        out.append(current)
        out.append(compose(current, (1, 0, 3, 2, 4)))
        current = compose(current, (1, 2, 3, 0, 4))
    return out


def part_three():
    """`||B - B^H|| / ||B||` per move, and the sweep's verdict, for two ansaetze."""
    virtual = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(0): 1, U1Sector(1): 1})
    physical = GradedSpace.new(U1, {U1Sector(-1): 1, U1Sector(1): 1})
    legs = (Leg(virtual, OUT),) * 4 + (Leg(physical, OUT),)
    raw = SymmetricTensor.random(legs, seed=5)
    averaged = sum((tenet.transpose(raw, p) for p in _c4v_permutations()), start=raw * 0) / 8
    print("\n# #243 Part 3 -- the enlarged corner under `EnvCTMc4v`\n")
    print("| ansatz | `||B - B^H|| / ||B||`, moves 0-5 | converges |")
    print("|---|---|---|")
    for name, a in (("no rotation", raw), ("full C4v (8 elements)", averaged)):
        env = EnvCTMc4v(Peps(SquareLattice(dims=(1, 1)), a))
        row = []
        for _ in range(6):
            env.update_(max_bond=8)
            big = np.asarray(corner2x2(env, "tl", (0, 0)).to_dense())
            side = int(np.prod(big.shape[: big.ndim // 2]))
            m = big.reshape(side, side)
            row.append(np.linalg.norm(m - m.conj().T) / np.linalg.norm(m))
        converged = env.iterate_(max_bond=8, max_sweeps=300, corner_tol=1e-10).converged
        print(f"| {name} | " + "  ".join(f"{x:.2e}" for x in row) + f" | {converged} |")
    print(
        "\nThe first row carries the signature and not the group -- four identical legs, no\n"
        "rotation -- and it converges with the corner far from Hermitian at every move,\n"
        "because the projector never asks."
    )


if __name__ == "__main__":
    part_three()
