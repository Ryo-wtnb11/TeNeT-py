"""The two mechanical guards of the coverage policy (issue #146, `tests/COVERAGE.md`).

Everything else the policy asks for is review discipline; these two are greps.
E1 is the failure that has happened three times — `flip` (#142), `full_trace` and
`inner` (#126) each landed in `tenet.__all__` after #95's torch pass and never
joined it. E2 is #145's gap turned into an invariant: the AD suite must keep a
provider whose F/R/B symbols are matrix-valued, or `grad x SU(N)` silently stops
running again.
"""

import pathlib
import re
import sys

import tenet
from tenet.symmetry.base import MultiplicityRecoupling

TESTS = pathlib.Path(__file__).parent

# Public callables that legitimately never touch a torch block, with the reason.
NOT_ON_TORCH = {
    "as_map": "a zero-copy structural view over the tensor's own blocks; no backend kernel",
    "coupled_sectors": "pure sector combinatorics over legs; consumes no array blocks",
    "fusion_trees": "pure sector combinatorics; enumerates trees, consumes no array blocks",
    "map_layout": "static layout metadata computed from the structure alone; no array blocks",
}


def test_e1_every_public_operation_reaches_the_torch_suite():
    src = (TESTS / "backends" / "test_torch.py").read_text()
    public = {n: getattr(tenet, n) for n in tenet.__all__}
    ops = [n for n, o in public.items() if callable(o) and not isinstance(o, type)]
    for name in ops:
        if name in NOT_ON_TORCH:
            assert NOT_ON_TORCH[name].strip(), f"NOT_ON_TORCH[{name!r}] needs a reason"
        else:
            assert re.search(rf"\b{name}\b", src), (
                f"tenet.{name} is public but never appears in tests/backends/test_torch.py; "
                f"add it there or to NOT_ON_TORCH with a reason (tests/COVERAGE.md)"
            )
    stale = set(NOT_ON_TORCH) - set(ops)
    assert not stale, f"NOT_ON_TORCH names nothing public: {sorted(stale)}"


def test_e2_the_ad_suite_carries_a_multiplicity_bearing_provider():
    src = (TESTS / "backends" / "test_ad.py").read_text()
    square = src.split("SQUARE = {", 1)[1].split("}", 1)[0]
    assert '"su3"' in square, "test_ad.py's SQUARE lost its multiplicity-bearing column"
    sys.path.insert(0, str(TESTS / "symmetry"))
    from _su3_fixture import SU3  # noqa: PLC0415  # the provider that column instantiates

    assert isinstance(SU3, MultiplicityRecoupling)
