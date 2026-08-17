"""The driver layer: finite tensor-network algorithms over the public ``tenet`` API.

The dependency edge is one-way: ``network`` imports ``ops``/``tensor``, never the
reverse. Nothing under ``src/tenet`` outside this package may import it, exactly as
``ops`` imports ``tensor`` and not back (``src/tenet/ops/__init__.py``:1-5).

**The composition rule** (#160), stated once here and referenced from ``mps.py`` and
``env.py``: every two-operand ``tenet.einsum`` in this package is a **composition** --
operand 1 supplies the ``IN`` end of *every* shared wire and operand 2 the ``OUT`` end.
For three-plus operands the same rule applies pairwise in caller order, which
``_contract_path`` guarantees (``ops/contraction.py``:596-603). The rule exists because
the two ends of a wire are not interchangeable for a fermionic provider -- the cap
``V*(x)V -> 1`` is not the cap ``V(x)V* -> 1`` -- so the operand order of every einsum
that can contract an odd wire is load-bearing, and a per-call choice is exactly what
produced #147's ``(-1)^(j-i)`` (the cap-direction Koszul sign, gate 1). A wire that
genuinely turns around in the intended planar diagram is **bent explicitly** with
``tenet.repartition`` before the einsum (``env.py::_composed``), never left to the
einsum's implicit cap. ``tests/network/test_hygiene.py`` pins the rule at every call
site reached by a smoke over the MPS/MPO/DMRG modules, with zero exemptions;
``ctmrg.py`` is outside the smoke -- a 2D network has loops, where operand order is
necessary but not sufficient (``ops/contraction.py``:575-587) and no fermionic PEPS
caller exists.

**Which side of a trace a module lives on is a per-module statement, not a package-level
one.** Until M11b it was package-level ("entirely outside ``jit``/``grad``"); with
``ctmrg.py`` that sentence is no longer true of the package and is still true of the rest,
so it is written where it holds:

* ``mps.py``, ``env.py``, ``dmrg.py`` -- **outside** ``jit``/``grad`` by construction, and
  they make no differentiability claim. ``tenet.linalg.svd_truncated`` re-decides a bond
  :class:`~tenet.GradedSpace` at every bond of every sweep, :func:`lanczos`'s happy
  breakdown tests a norm against ``tol``, and :func:`dmrg_`'s loop exits on a measured
  energy change, and M11c's :meth:`MPS.save` / :meth:`MPS.load`, :meth:`MPS.compress_`
  and :func:`expectation_1site` / :func:`expectation_2site` are outside for the same
  reasons -- filesystem I/O, a re-decided bond space, and a truncating split, and M13's
  :meth:`MPO.from_terms` joins them because its compression sweep is ``svd_truncated``,
  as do M14's :class:`Sweep` and :meth:`MPS.product` with the rest of ``mps.py`` /
  ``dmrg.py`` -- a schedule only re-parameterizes the sweeps above, and a product state's
  bonds are decided sector by sector. One carve-out inside ``env.py`` (M16): the prepared
  two-site matvec is **structure-preserving and traceable** -- :meth:`Env.heff2` returns
  a tensor with its input's structure exactly, so the apply is a pure fixed-structure
  function that an injected ``compile=`` may trace, keyed by structure exactly as the
  rule at the end of this list demands -- while :func:`sweep_` around it stays outside,
  because ``svd_truncated`` re-decides the bond spaces between one matvec and the next;
* ``common.py`` -- **trace-neutral**; :func:`spectrum` is nonetheless only ever called
  outside a trace, its ``sorted`` Python list being driver output rather than a tensor;
* ``ctmrg.py`` -- **both**, stated per function in its own docstrings. :func:`ctmrg`
  reads singular values and a corner spectrum, so it raises under any trace;
  :func:`ctmrg_unrolled` and ``move(bond=B)`` are shape-static and differentiable; and
  :func:`move` is the boundary itself, ``chi=`` outside and ``bond=`` inside. The frozen
  :class:`~tenet.GradedSpace` is the **only** object that crosses between them, and it
  crosses as a jit cache key, never as a jit argument.

That is the complement of docs/design.md invariant 9, not an exception to it: invariant 9 says
structure-changing operations live outside compile boundaries and the library never hides
the distinction; this package is where data-dependent control flow is *allowed to live*,
and ``ctmrg.py`` is where the two sides meet and are named.

Contents. M11a: :class:`MPS`, :class:`MPO`, :class:`Env`, :func:`lanczos`, :func:`sweep_`,
:func:`dmrg_`. M11b: :class:`CTMEnv`, :class:`Absorb`, :func:`single_layer`,
:func:`layers`, :func:`double_layer`, :func:`single_layer_ctm`, :func:`double_layer_ctm`,
:func:`init_env`, :func:`move`, :func:`ctmrg`, :func:`ctmrg_unrolled`, :func:`normalized`
and :func:`ring`. M11c: :meth:`MPS.save` / :meth:`MPS.load` and :meth:`MPS.compress_`,
the two measurements :func:`expectation_1site` and :func:`expectation_2site`, and
:class:`CTMRG_out` -- :func:`ctmrg`'s return, which was a bare ``(CTMEnv, history)``
tuple. M13: :func:`local_op` and :meth:`MPO.from_terms`, the term-list MPO builder whose
bond spaces are derived rather than declared, with :meth:`MPO.to_dense` as its oracle exit.
M13b: no new name -- :func:`local_op` grew an optional ``charge=None`` giving the
symmetry-**invariant** *k*-site form, and :meth:`MPO.from_terms` takes a tuple of sites
for it and splits it with ``svd_truncated``, so it is **no longer Abelian-only**: a
non-Abelian term is one invariant operator whose coupling lives in its own blocks, and the
aux bond it runs through, multiplicities and all, is derived from the SVD.
M14: :class:`Sweep` and :meth:`MPS.product` -- :func:`dmrg_` takes a per-sweep
``schedule`` whose last entry repeats, :func:`sweep_` takes a wavefunction ``noise``, a
per-sweep ``callback`` reports while the run happens, and :class:`DMRG_out` records the
realized schedule; :meth:`MPS.product` builds a product state whose bonds are derived
backwards from the site sectors, putting the total charge on bond 0.
M16: :meth:`MPO.jordan` -- ``from_terms(cutoff=None)`` keeps its finite-state machine as
a per-site block table, :meth:`Env.heff2` and :meth:`Env.update_` fold environments into
those blocks once per bond instead of contracting a dense ``W``, and :class:`Env` takes
an optional ``compile=`` callable applied to the fixed-structure matvec -- injected by
the caller, so this layer still names no accelerator.
Shared: the bond spectrum :func:`spectrum` and the :func:`ones` seed -- the two
scalar exits that sat beside them left for ``tenet.ops`` in #126, as
:func:`tenet.full_trace` and :func:`tenet.inner`. Promoted verbatim from
``examples/toy_codes/dmrg.py`` (#110) and ``examples/toy_codes/ctmrg.py``
(#102/#104/#105/#107) under the rule that no number may move.

Uses **public ``tenet`` API only**: no ``jax``/``torch``/``scipy``/``quimb``/
``opt_einsum``, no ``_``-prefixed reach into other modules, no numerical use of
``t.blocks``. The one named exception is reading ``t.provider`` and the provider's
``qdim``, ``unit``, ``fusion``, ``dual`` and ``permute_tree``: :func:`spectrum`'s
``sqrt(qdim)`` diagonal weight, ``ctmrg.py``'s unit sector, ``mps.py``'s charge
accumulation (:meth:`MPO.from_terms`), :meth:`MPS.product`'s backwards bond derivation
through ``fusion`` and ``dual``, and the braiding probe behind the fermionic refusal.
Every one is symmetry-generic metadata, not a provider branch.
``tests/network/test_hygiene.py`` enforces both halves, including reads through a local
binding such as ``sym = space.provider``.
"""

from tenet.network.common import ones, spectrum
from tenet.network.ctmrg import (
    Absorb,
    CTMEnv,
    CTMRG_out,
    ctmrg,
    ctmrg_unrolled,
    double_layer,
    double_layer_ctm,
    init_env,
    layers,
    move,
    normalized,
    ring,
    single_layer,
    single_layer_ctm,
)
from tenet.network.dmrg import DMRG_out, Sweep, dmrg_, lanczos, sweep_
from tenet.network.env import Env
from tenet.network.mps import MPO, MPS, expectation_1site, expectation_2site, local_op

__all__ = [
    "MPO",
    "MPS",
    "Absorb",
    "CTMEnv",
    "CTMRG_out",
    "DMRG_out",
    "Env",
    "Sweep",
    "ctmrg",
    "ctmrg_unrolled",
    "dmrg_",
    "double_layer",
    "double_layer_ctm",
    "expectation_1site",
    "expectation_2site",
    "init_env",
    "lanczos",
    "layers",
    "local_op",
    "move",
    "normalized",
    "ones",
    "ring",
    "single_layer",
    "single_layer_ctm",
    "spectrum",
    "sweep_",
]
