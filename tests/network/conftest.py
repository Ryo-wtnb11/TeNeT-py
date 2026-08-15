"""``tests/network`` borrows exactly one thing from the example layer: the Hamiltonian.

#112's promotion boundary is "the library takes bond *spaces*; the example computes which
spaces are reachable", so the U(1) Heisenberg ``W``, its grading and ``bond_spaces`` live
in ``examples/dmrg.py`` on purpose. Copying them here would be a second copy of the
physics to keep in sync with the first, so these unit tests import the example the same
way ``tests/integration/test_dmrg.py``:48 does -- and they stay unit tests because they
run at N <= 8 and chi <= 16, not because they hand-roll a Hamiltonian.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples"))
