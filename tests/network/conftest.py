"""``tests/network`` borrows exactly one thing from the example layer: the Hamiltonian.

#112's promotion boundary is "the library takes bond *spaces*; the example computes which
spaces are reachable", so the U(1) Heisenberg ``W``, its grading and ``bond_spaces`` live
in an example on purpose. Copying them here would be a second copy of the physics to keep
in sync with the first, so these unit tests import the example the same way
``tests/integration/test_dmrg.py`` does -- and they stay unit tests because they run at
N <= 8 and chi <= 16, not because they hand-roll a Hamiltonian.

Since #183 the file they import is ``examples/heisenberg_walkthrough.py``, the *usage*
lane's copy: it is the one that hands the hand-graded ``W`` to ``MPO.from_w`` and so
returns the ``MPO`` these tests feed to ``Env``/``dmrg_``. ``examples/toy_codes/dmrg.py``
now writes its own MPO out as a plain list of ``SymmetricTensor`` and imports nothing from
``tenet.network``, which is the teaching lane's rule and the reason it is no longer the
right import here. Both directories go on the path.
"""

import pathlib
import sys

_EXAMPLES = pathlib.Path(__file__).parents[2] / "examples"
sys.path.insert(0, str(_EXAMPLES / "toy_codes"))
sys.path.insert(0, str(_EXAMPLES))
