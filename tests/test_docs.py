"""The rendered API page keeps every name ``tenet.__all__`` promises (#173).

``docs/api/tenet.md`` is one ``::: tenet`` directive, so the page is exactly what griffe
resolves the package's members to. Griffe follows a re-export to the module the name was
imported *from* and then looks the name up there -- and in that namespace a submodule of
the same name wins over the function it defines. ``from tenet.ops import repartition``
therefore landed on the ``tenet.ops.repartition`` *module*, which the page filters out
along with every other submodule, and the function was silently absent from the docs.

Building the site here would mean importing mkdocs and griffe, which ``pyproject.toml``
keeps in the docs group precisely because nothing under ``src/`` or ``tests/`` imports
them. The collision is visible without them: read the import statements and check that no
re-export names a module that also has a submodule of that name.
"""

import ast
import importlib
import inspect
import pathlib

import tenet

INIT = pathlib.Path(tenet.__file__)


def _is_submodule_of(package: str, name: str) -> bool:
    """Does ``package`` hold a submodule called ``name``?"""
    module = importlib.import_module(package)
    paths = getattr(module, "__path__", ())
    return any(
        (pathlib.Path(root) / name).is_dir() or (pathlib.Path(root) / f"{name}.py").is_file()
        for root in paths
    )


def test_no_re_export_is_shadowed_by_a_submodule_of_the_same_name():
    shadowed = []
    for node in ast.parse(INIT.read_text()).body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            name = alias.asname or alias.name
            if name not in tenet.__all__:
                continue
            # `tenet.linalg` and `tenet.network` are submodules on purpose and get their
            # own pages; only a non-module losing to a submodule is the bug.
            if inspect.ismodule(getattr(tenet, name)):
                continue
            if _is_submodule_of(node.module, alias.name):
                shadowed.append(f"tenet.{name} (imported from {node.module})")
    assert not shadowed, (
        "griffe resolves these re-exports to the shadowing submodule, so they vanish "
        f"from the ::: tenet API page: {', '.join(shadowed)}. Import each from the "
        "module that defines it instead."
    )


def test_the_names_the_issue_named_are_functions_on_the_package():
    """``repartition``, ``embed`` and the function once spelled ``cast``.

    ``cast`` was renamed ``to_symmetry`` in M31 and so no longer matches its defining
    module ``tenet.ops.cast``; it is listed here because the issue named it, and to
    catch a rename back into the collision.
    """
    for name in ("repartition", "embed", "to_symmetry"):
        assert name in tenet.__all__
        assert inspect.isfunction(getattr(tenet, name))
