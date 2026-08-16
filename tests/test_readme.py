"""The README quickstart is executed, not just written (docs-a, #116).

The README is the PyPI long description and the first thing a user runs; the API is
explicitly unstable pre-1.0, so a rotted quickstart is the most expensive stale line in
the repo. Two blocks: the first must run on the *core* install, the second needs JAX.
"""

import pathlib
import re

import pytest

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"

QUICKSTART = README.read_text().split("\n## Quickstart\n", 1)[1].split("\n## ", 1)[0]
BLOCKS = re.findall(r"```python\n(.*?)```", QUICKSTART, re.DOTALL)


def test_quickstart_has_a_core_block_and_a_jax_block():
    assert len(BLOCKS) == 2, "README quickstart should hold exactly two python blocks"


def test_quickstart_runs():
    namespace: dict = {}
    exec(BLOCKS[0], namespace)  # noqa: S102 — executing the README is the point
    pytest.importorskip("jax")
    exec(BLOCKS[1], namespace)  # noqa: S102
