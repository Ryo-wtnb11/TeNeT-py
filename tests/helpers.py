"""Test helpers shared between test modules. Not part of the library.

``supersign`` is the dense-side Koszul sign of an axis permutation, written for
issue #39 and promoted here unchanged when #51 needed the same oracle for
``tensordot``: the fermionic correctness of a contraction is *inherited* from
``transpose``, so the two suites must weigh it on the same scale.
"""

import math
import os
import pathlib
import re

import numpy as np

from tenet.space import GradedSpace

__all__ = ["check_example_page", "parity_vector", "sector_parity", "supersign"]

DOCS_EXAMPLES = pathlib.Path(__file__).parents[1] / "docs" / "examples"

_FENCE = re.compile(r"```text\n(.*?)```", re.DOTALL)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def check_example_page(page_path: str, captured_stdout: str) -> None:
    """The docs example page's output fence against the run CI just performed (#164).

    Non-numeric text is compared exactly; every number to a relative tolerance of 1e-6
    (with a 1e-12 absolute floor, so float-noise diagnostics like ``max|<S^z>|`` survive
    a BLAS change while any physical digit does not). ``TENET_UPDATE_EXAMPLE_PAGES=1``
    rewrites the fence in place instead of asserting — that is the regeneration path.
    """
    page = DOCS_EXAMPLES / page_path
    text = page.read_text()
    match = _FENCE.search(text)
    assert match is not None, f"{page} has no ```text output fence"
    if os.environ.get("TENET_UPDATE_EXAMPLE_PAGES") == "1":
        page.write_text(text[: match.start(1)] + captured_stdout + text[match.end(1) :])
        return
    fence = match.group(1)
    assert _NUMBER.sub("#", fence) == _NUMBER.sub("#", captured_stdout), (
        f"{page}: the output fence's text no longer matches the run — regenerate with "
        f"TENET_UPDATE_EXAMPLE_PAGES=1.\n--- page ---\n{fence}--- run ---\n{captured_stdout}"
    )
    for committed, fresh in zip(
        _NUMBER.findall(fence), _NUMBER.findall(captured_stdout), strict=True
    ):
        assert math.isclose(float(committed), float(fresh), rel_tol=1e-6, abs_tol=1e-12), (
            f"{page}: committed {committed} vs computed {fresh} — regenerate with "
            f"TENET_UPDATE_EXAMPLE_PAGES=1"
        )


def sector_parity(sector) -> int:
    """Fermionic parity of a sector, summed over the factors of a product sector.

    ``ProductSector`` carries no ``parity`` of its own; a product of a bosonic and
    a fermionic factor is graded by the fermionic one (#52 needs this to give the
    product provider the same oracle as fZ2).
    """
    if hasattr(sector, "parity"):
        return sector.parity
    return sum(sector_parity(c) for c in getattr(sector, "components", ())) % 2


def parity_vector(space: GradedSpace) -> np.ndarray:
    """Parity of each dense index of ``space``, in canonical sector order."""
    return np.concatenate([np.full(m, sector_parity(a)) for a, m in space.sectors])


def supersign(legs, p: tuple[int, ...], *, per_side: bool) -> np.ndarray:
    """Dense-side Koszul sign array, shaped like ``np.transpose(dense, p)``.

    ``per_side=False`` counts every inversion of ``p`` (correct when every leg
    lives on one side); ``per_side=True`` counts only inversions between two axes
    of the same side, which is TeNeT-py's stated convention.
    """
    pars = [parity_vector(legs[ax].space) for ax in p]
    sides = [legs[ax].side for ax in p]
    sign = np.ones(tuple(len(v) for v in pars))
    n = len(p)
    for j in range(n):
        for k in range(j + 1, n):
            if p[j] <= p[k] or (per_side and sides[j] is not sides[k]):
                continue
            shape_j = [1] * n
            shape_j[j] = len(pars[j])
            shape_k = [1] * n
            shape_k[k] = len(pars[k])
            product = pars[j].reshape(shape_j) * pars[k].reshape(shape_k)
            sign = sign * (-1.0) ** product
    return sign
