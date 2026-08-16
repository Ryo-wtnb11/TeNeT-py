"""The layer's hygiene invariants, enforced rather than stated (#112).

``src/tenet/network/`` may use **public ``tenet`` API only**. Concretely, and each one is
a test below:

* no ``import jax`` / ``torch`` / ``scipy`` / ``quimb`` / ``opt_einsum``, extending
  ``tests/test_import.py`` and REPOSITORY_RULES:31-33 ("core never imports them") to the
  new package. ``numpy`` and ``autoray`` are core dependencies and are allowed;
* no reach into a ``_``-prefixed name of another ``tenet`` module. The package's own
  private helpers (``mps._as_site``, ``dmrg._schmidt_change``, ``ctmrg._spectrum_change``) are its
  own business; importing someone else's is not;
* no **numerical** use of ``t.blocks``. Reduced blocks are the library's storage, and a
  driver that reads them is doing arithmetic below the public API.

**The one named exception**, spelled out because it is a line and not a loophole:
reading ``t.provider``, ``provider.qdim`` and ``provider.unit`` is allowed. Two reads are
left, and each has one owner: ``common.spectrum`` needs ``sqrt(qdim)`` for its Schmidt
weights, and ``ctmrg.py`` needs ``provider.unit`` for the trivial sector. That is
*symmetry-generic metadata*: ``provider.qdim(c)`` is fine, ``isinstance(provider,
SU2Provider)`` is not, and the second is what the branch test below forbids.

**The duplicated-``scalar`` finding #112 recorded, corrected and closed (#114, #126).** It
was recorded as *three* files writing the same five-line ``sum(qdim(c) * trace(m))``; it
was **two**. ``examples/vmc_mps.py``:158-160 is a *different* function --
``t.blocks[0][0, 0]``, the trivial-leg shortcut for a network with no non-unit sector on
the open bond, not qdim-weighted at all -- and it is deliberately untouched. The two real
copies were ``examples/ctmrg.py`` and ``tenet/network/mps.py::scalar``; #114 moved them
into ``network/common.py`` with their bodies unchanged, and #126 finished the job by
promoting the qdim-weighted trace to ``tenet.full_trace`` (and ``inner`` to
``tenet.inner``) in ``tenet.ops``, next to ``trace``. There is now a public spelling, so
``scalar`` is deleted rather than allowed, and ``spectrum`` alone owns the ``qdim``
allowance above.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).parents[2] / "src" / "tenet" / "network"
MODULES = sorted(PACKAGE.glob("*.py"))
FORBIDDEN = {"jax", "torch", "scipy", "quimb", "opt_einsum"}


def trees():
    for path in MODULES:
        yield path, ast.parse(path.read_text())


def test_the_package_has_the_modules_it_claims():
    assert {p.name for p in MODULES} == {
        "__init__.py",
        "common.py",
        "mps.py",
        "env.py",
        "dmrg.py",
        "ctmrg.py",
    }


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_imports_a_forbidden_dependency(path):
    tree = ast.parse(path.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN, sorted(roots & FORBIDDEN)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_reaches_into_another_modules_private_names(path):
    tree = ast.parse(path.read_text())
    own = f"tenet.network.{path.stem}"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module != own:
            private = [a.name for a in node.names if a.name.startswith("_")]
            assert not private, f"{path.name} imports {private} from {node.module}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_uses_reduced_blocks_numerically(path):
    """``.blocks`` is storage. ``apply_blocks`` is the public spelling and is not this."""
    tree = ast.parse(path.read_text())
    reads = [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "blocks"
    ]
    assert not reads, f"{path.name} reads .blocks at line(s) {[n.lineno for n in reads]}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_branches_on_which_provider_it_has(path):
    """Metadata reads yes, provider dispatch no -- the line the module docstring draws.

    No provider class is so much as *named* in the package, which is the cheapest way to
    say ``isinstance(provider, SU2Provider)`` never happens: the only ``isinstance`` here
    is the write barrier's index check, and it tests ``int``.
    """
    source = path.read_text()
    assert "Provider" not in source, f"{path.name} names a provider class"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "isinstance":
            assert getattr(node.args[1], "id", None) in {"int", "str"}, node.lineno


def test_the_allowed_metadata_reads_are_only_qdim_and_unit():
    """Whatever this package reads off a provider must stay on the named list."""
    allowed = {"qdim", "unit", "provider"}
    seen = set()
    for _path, tree in trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                if node.value.attr == "provider":
                    seen.add(node.attr)
    assert seen <= allowed, sorted(seen - allowed)
