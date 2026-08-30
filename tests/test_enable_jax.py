"""``tenet.enable_jax()`` — one test per acceptance criterion of #211.

Both halves it turns on are *process-global* — the pytree node registration is global
to JAX and ``tenet.ad``'s dispatch entries are global to autoray — so the tests that
observe the before/after transition run in a **fresh subprocess**, the pattern
``tests/backends/test_pytree.py`` already uses for its JAX-less environment. Measured
cost: 4 subprocess launches, 0.85 s for the whole module.
"""

import subprocess
import sys
import textwrap

import pytest

import tenet

pytest.importorskip("jax")

# Build one U(1) tensor with more than one coupled sector, so "the leaves are the sector
# matrices" is a statement the un-registered case cannot accidentally satisfy.
FIXTURE = """
    import autoray as ar
    import jax
    import tenet
    from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
    from tenet.symmetry import U1, U1Sector

    V = GradedSpace.new(U1, {U1Sector(0): 1, U1Sector(1): 1})
    T = SymmetricTensor.random((Leg(V, OUT), Leg(V, IN)), seed=0)
    assert T.structure.num_blocks > 1

    def leaves():
        return jax.tree.leaves(T)

    def svd_override():
        return ar.autoray._FUNCS.get(("jax", "linalg.svd"))

    def eigh_override():
        return ar.autoray._FUNCS.get(("jax", "linalg.eigh"))
"""


def run(body: str) -> None:
    """Run ``FIXTURE + body`` in a fresh interpreter; the body must print ``OK``."""
    script = textwrap.dedent(FIXTURE) + textwrap.dedent(body)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("OK"), out.stdout


def test_enable_jax_is_exported_from_the_top_level():
    assert "enable_jax" in tenet.__all__
    assert callable(tenet.enable_jax)


def test_it_does_what_the_three_statements_do():
    """``import tenet.pytree`` + ``import tenet.ad`` + ``tenet.ad.install()``, in one call."""
    run("""
        assert len(leaves()) == 1 and leaves()[0] is T   # not a pytree yet
        assert svd_override() is None and eigh_override() is None

        tenet.enable_jax(ad=True)

        import tenet.ad
        assert len(leaves()) == len(T.data)
        assert all(a is b for a, b in zip(leaves(), T.data))
        assert svd_override() is tenet.ad._svd_dispatch
        assert eigh_override() is tenet.ad._eigh_dispatch
        print("OK")
    """)


def test_the_default_registers_the_pytree_and_leaves_autorays_table_untouched():
    """The invasive half is opted into by name: ``ad`` defaults to ``False`` (#211)."""
    run("""
        tenet.enable_jax()

        assert len(leaves()) == len(T.data)              # effect 1 happened
        assert svd_override() is None                    # effect 2 did not
        assert eigh_override() is None
        print("OK")
    """)


def test_calling_it_twice_is_harmless():
    """Idempotent in both halves, and the second call changes nothing observable."""
    run("""
        tenet.enable_jax(ad=True)
        first = (len(leaves()), svd_override(), eigh_override())

        tenet.enable_jax(ad=True)
        tenet.enable_jax()          # and the narrower call does not uninstall the wider

        assert (len(leaves()), svd_override(), eigh_override()) == first
        print("OK")
    """)


def test_without_jax_it_raises_a_message_naming_the_optional_extra():
    """A blocked ``import jax``, so the guard is exercised with JAX genuinely absent."""
    script = textwrap.dedent("""
        import importlib.abc, sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "jax" or name.startswith("jax."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.meta_path.insert(0, Blocker())
        import tenet                      # core must survive a JAX-less environment
        assert "jax" not in sys.modules
        try:
            tenet.enable_jax()
        except ImportError as exc:
            msg = str(exc)
            assert "tenet.enable_jax()" in msg, msg
            assert "JAX" in msg and "optional dependency" in msg, msg
            assert "symtenet[jax]" in msg, msg
            assert "get_params/set_params" in msg, msg
            # the ImportError is ours, not a traceback surfacing out of tenet.pytree
            assert exc.__traceback__.tb_next.tb_frame.f_globals["__name__"] == "tenet", msg
        else:
            raise AssertionError("enable_jax succeeded without jax")
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "OK"


def test_the_docstring_names_both_effects_separately_and_the_global_reach():
    """The honesty condition #211 states, checked rather than trusted."""
    doc = tenet.enable_jax.__doc__
    assert "tenet.pytree" in doc and "tenet.ad" in doc
    assert "process-global" in doc
    assert "autoray.register_function" in doc
    assert "quimb" in doc  # "affects other libraries in the process", named


def test_the_existing_spellings_are_unchanged():
    """``import tenet.pytree`` and ``tenet.ad.install()`` keep working, same effect."""
    import tenet.ad
    import tenet.pytree  # noqa: F401

    assert tenet.pytree.__all__ == []
    assert callable(tenet.ad.install) and callable(tenet.ad.uninstall)
    assert "enable_jax" not in (tenet.ad.install.__doc__ or "")  # byte-identical (#211)
