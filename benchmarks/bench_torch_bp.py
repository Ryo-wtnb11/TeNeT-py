"""One 6x6 D=32 SU(2) BP iteration on a reproducible random PEPS (#353).

Requires PEPS_VMC's circuit_synthesis package and its dependencies. This isolates
BP contractions; it does not reproduce the issue's complete circuit realization.
Run both library checkouts via PYTHONPATH with the same PEPS_VMC checkout.
"""

import cProfile
import json
import statistics
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from tenet import GradedSpace, SymmetricTensor
from tenet.symmetry import SU2, SU2Sector


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--peps-vmc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [
        str(args.peps_vmc / "projects/circuit_synthesis/src"),
        str(args.peps_vmc / "src"),
    ]
    from circuit_synthesis.tenet_su2 import _site_legs, bp_fixed_point_tenet

    torch.set_num_threads(1)
    physical = GradedSpace.new(SU2, {SU2Sector(1): 1})
    unit = GradedSpace.new(SU2, {SU2Sector(0): 1})
    virtual = GradedSpace.new(SU2, {SU2Sector(0): 5, SU2Sector(1): 6, SU2Sector(2): 5})
    sites = tuple(
        tuple(
            SymmetricTensor.random(
                _site_legs(
                    (
                        physical,
                        virtual if y else unit,
                        virtual if x < 5 else unit,
                        virtual if y < 5 else unit,
                        virtual if x else unit,
                    )
                ),
                dtype=np.complex64,
                seed=6 * y + x,
            ).to_backend("torch")
            for x in range(6)
        )
        for y in range(6)
    )
    sites = tuple(
        tuple(
            SymmetricTensor.from_data(t.structure, tuple(m + 0.3j * m.flip(-1) for m in t.data))
            for t in row
        )
        for row in sites
    )
    leaves = tuple(m for row in sites for t in row for m in t.data)

    def forward():
        messages, _, _ = bp_fixed_point_tenet(sites, maxiter=1, tol=1e-6)
        return tuple(
            m
            for row in messages
            for site in row
            for msg in site
            if msg is not None
            for m in msg.data
        )

    def timed(fn):
        fn()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    plain = timed(forward)
    profile = cProfile.Profile()
    profile.runcall(forward)
    # Enable AD on fresh tensors, before any block views can be cached.
    sites = tuple(
        tuple(
            SymmetricTensor.from_data(
                t.structure, tuple(m.detach().requires_grad_(True) for m in t.data)
            )
            for t in row
        )
        for row in sites
    )
    leaves = tuple(m for row in sites for t in row for m in t.data)
    graph = timed(forward)

    def backward():
        return torch.autograd.grad(sum(m.real.sum() for m in forward()), leaves)

    full = timed(backward)
    args.output.write_text(
        json.dumps(
            dict(
                forward_ms=plain,
                forward_graph_ms=graph,
                forward_backward_ms=full,
                forward_calls=sum(e.callcount for e in profile.getstats()),
            ),
            indent=2,
        )
    )
    np.savez(
        args.output.with_suffix(".npz"),
        values=np.concatenate([m.detach().resolve_conj().numpy().ravel() for m in forward()]),
        gradients=np.concatenate([m.detach().resolve_conj().numpy().ravel() for m in backward()]),
    )
    print(args.output.read_text())


if __name__ == "__main__":
    main()
