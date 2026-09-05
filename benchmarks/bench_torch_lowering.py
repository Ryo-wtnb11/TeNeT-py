"""Torch repartition, including backward and saved storage (issue #353).

Run each checkout in a fresh process, with the same script and thread count:
    PYTHONPATH=/path/to/checkout/src .venv/bin/python benchmarks/bench_torch_lowering.py

JSON lines include cold planning, warm medians, call counts and output digests.
The wide case has three SU(2) sectors, unlike the cheaper rank-8 unit-test fixture.
No GPU, network, or external project is needed.
"""

import cProfile
import hashlib
import json
import platform
import resource
import statistics
import time
from argparse import ArgumentParser

import numpy as np
import torch

import tenet
from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
from tenet.symmetry import SU2, U1, SU2Sector, U1Sector


def tensor(provider, rank, degeneracies):
    sectors = [SU2Sector(i) if provider is SU2 else U1Sector(i - 1) for i in range(3)]
    space = GradedSpace.new(provider, dict(zip(sectors, degeneracies, strict=True)))
    t = SymmetricTensor.random(
        tuple(Leg(space, OUT if i % 2 == 0 else IN) for i in range(rank)),
        seed=rank,
        dtype=np.complex64,
    ).to_backend("torch")
    return SymmetricTensor.from_data(t.structure, tuple(m + 0.3j * m.flip(-1) for m in t.data))


def median_ms(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def measure(name, t, repeats):
    def forward():
        return tenet.repartition(t, tuple(range(1, t.ndim, 2)), tuple(range(0, t.ndim, 2)))

    start = time.perf_counter()
    result = forward()
    cold = (time.perf_counter() - start) * 1000
    digest = hashlib.sha256()
    for m in result.data:
        digest.update(m.detach().numpy().tobytes())
    plain = median_ms(forward, repeats)
    profile = cProfile.Profile()
    profile.runcall(forward)
    calls = sum(entry.callcount for entry in profile.getstats())
    # Fresh wrappers: views cached during no-grad execution must not become AD leaves.
    t = SymmetricTensor.from_data(
        t.structure, tuple(m.detach().requires_grad_(True) for m in t.data)
    )
    graph = median_ms(forward, repeats)

    def backward():
        result = forward()
        return torch.autograd.grad(sum(m.abs().square().sum() for m in result.data), t.data)

    full = median_ms(backward, repeats)
    saved = {}

    def pack(m):
        storage = m.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return m

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda m: m):
        result = forward()
    print(
        json.dumps(
            dict(
                case=name,
                blocks=t.structure.num_blocks,
                cold_ms=cold,
                forward_ms=plain,
                forward_graph_ms=graph,
                forward_backward_ms=full,
                forward_calls=calls,
                saved_storage_mib=sum(saved.values()) / 2**20,
                output_sha256=digest.hexdigest(),
            )
        ),
        flush=True,
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(
        json.dumps(
            dict(
                platform=platform.platform(),
                torch=torch.__version__,
                tenet=tenet.__file__,
                **vars(args),
            )
        ),
        flush=True,
    )
    for name, provider, rank, deg in (
        ("su2-small", SU2, 2, (3, 2, 1)),
        ("u1-ragged", U1, 6, (3, 2, 1)),
        ("su2-ragged", SU2, 6, (3, 2, 1)),
        ("su2-wide", SU2, 8, (1, 1, 1)),
    ):
        measure(name, tensor(provider, rank, deg), args.repeats)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(peak_rss_mib=rss / (2**20 if platform.system() == "Darwin" else 1024))))


if __name__ == "__main__":
    main()
