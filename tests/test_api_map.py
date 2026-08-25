"""``docs/guide/api-map.md`` names only symbols that exist.

The page's whole value is that a reader can trust it without opening the source, so a
name that quietly stops resolving -- a rename, a removed re-export, a typo -- is worse
than no page. This walks every ``tenet...`` code span on it and resolves it.

Deliberately not a doctest: the page is a table of *names*, and executing them would say
nothing a lookup does not. CI already runs the guide's real doctests
(``pytest --doctest-glob='*.md' docs/guide``); this is the check for the one page that is
prose about the API rather than a use of it.
"""

import importlib
import pathlib
import re

import pytest

PAGE = pathlib.Path(__file__).parents[1] / "docs" / "guide" / "api-map.md"

#: Every inline code span, e.g. ``tenet.linalg.svd``. The trailing ``(...)`` of a call and
#: a trailing ``, …`` are stripped; a span that is not a dotted ``tenet`` path is ignored,
#: which is what lets the page write ``A @ B``, ``bond=`` and ``dual=`` in the same column.
_SPAN = re.compile(r"`([^`]+)`")
_NAME = re.compile(r"^tenet(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


def _names() -> list[str]:
    found = []
    for span in _SPAN.findall(PAGE.read_text()):
        name = span.split("(")[0].strip()
        if _NAME.match(name):
            found.append(name)
    return sorted(set(found))


def _resolve(name: str):
    """``tenet.linalg.svd`` -- import the longest importable prefix, then ``getattr``.

    The two-step walk is not incidental: ``tenet.linalg`` is a re-exported *attribute*
    of the package and not an importable module path, and the page says so. A resolver
    that only tried ``import_module`` would report the page wrong about a name the page
    is right about.
    """
    parts = name.split(".")
    obj, rest = None, parts[1:]
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
            rest = parts[i:]
            break
        except ImportError:
            continue
    assert obj is not None, name
    for attribute in rest:
        obj = getattr(obj, attribute)
    return obj


def test_the_page_names_something():
    """A regex that silently matched nothing would make every assertion below vacuous."""
    assert len(_names()) > 80


@pytest.mark.parametrize("name", _names())
def test_every_name_on_the_api_map_resolves(name):
    _resolve(name)
