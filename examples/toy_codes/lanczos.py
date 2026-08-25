"""The Lanczos step: a ground eigenpair over any space with an inner product and a norm.

``dmrg.py``'s inner solver, and a file of its own because nothing in it is about DMRG.
Everything below is written against ``tenet.add``/``tenet.subtract``, scalar multiply and
divide, ``tenet.norm`` and ``tenet.inner`` -- a Krylov method needs a vector space and
nothing else, and a ``SymmetricTensor`` is one. ``matvec`` is a callable, so the same
function serves the two-site effective Hamiltonian, a plain matrix on a rank-2 tensor, or
anything else carrying those five operations.
"""

import numpy as np

import tenet
from tenet import SymmetricTensor


def lanczos(matvec, v: SymmetricTensor, ncv: int = 3, tol: float = 1e-13):
    """Ground eigenpair ``(value, vector)`` of a Hermitian ``matvec`` over SymmetricTensors.

    YASTN's three-term recurrence (``yastn/tensor/_krylov.py``:34-42) and its happy
    breakdown (``H[(j+1,j)] < tol`` -> stop and drop the row, :39-43), then ``eigh`` of
    the ``(m, m)`` tridiagonal and one recombination
    (``yastn/krylov/_krylov.py``:226-239, a single iteration with no restart at :217-219).
    ``hermitian=True, ncv=3, which='SR'`` are YASTN's own DMRG defaults
    (``_dmrg.py``:151-152) and are not knobs this example tunes.

    The only tensor operations are ``tenet.add``/``subtract``, scalar multiply/divide,
    ``tenet.norm`` and ``tenet.inner`` -- a Krylov solver needs a vector space and nothing
    else, and a ``SymmetricTensor`` is one.

    Simplification: **no reorthogonalization**, and neither has YASTN. At ``ncv=3`` the
    recurrence has not had time to lose orthogonality, and the vector is reseeded from the
    current MPS at every bond -- this is an inner solver inside an outer sweep, not a
    standalone eigensolver. Ceiling: raise ``ncv`` past ~10 and full reorthogonalization
    against the stored ``vecs`` becomes the two-line addition.

    Simplification: numpy ``eigh`` on the ``(3, 3)`` tridiagonal, not ``tenet.linalg.eigh``. The
    projected matrix has no symmetry structure to respect -- it is 9 floats.
    """
    # The Krylov basis starts at the normalized seed; every alpha and beta below is an
    # entry of H expressed in this basis, so the seed has to be a unit vector.
    vecs = [v / tenet.norm(v)]
    alphas: list[float] = []
    betas: list[float] = []
    for j in range(ncv):
        # The only place the operator is touched. Everything after this is linear algebra
        # on three floats per step, which is why ncv matvecs is the whole cost.
        w = matvec(vecs[j])
        # alpha_j = <v_j|H|v_j>, the diagonal of the projected matrix.
        alphas.append(float(tenet.inner(vecs[j], w)))
        # Project H v_j orthogonal to the two previous basis vectors. For a Hermitian H
        # those two are enough: <v_k|H|v_j> vanishes for k < j - 1 because H v_k already
        # lies in the span of v_{k-1}, v_k, v_{k+1}. That is the three-term recurrence,
        # and in exact arithmetic it leaves w orthogonal to the whole basis. In floating
        # point the older directions creep back in, which is why a long run needs explicit
        # reorthogonalization against the stored vecs -- at ncv=3 there is no room for it.
        w = tenet.subtract(w, vecs[j] * alphas[j])
        if j:
            w = tenet.subtract(w, vecs[j - 1] * betas[j - 1])
        # beta_j is the length of what is left, i.e. the off-diagonal of the projection.
        beta = float(tenet.norm(w))
        # A vanishing beta means the Krylov space has closed on an invariant subspace:
        # the eigenpair in it is already exact, and dividing by beta would be division by
        # noise. tol is near float64 round-off on a unit vector, so it fires only then.
        if j + 1 == ncv or beta < tol:  # happy breakdown: drop the row, keep the space
            break
        betas.append(beta)
        vecs.append(w / beta)
    # H restricted to the Krylov space: symmetric tridiagonal by the recurrence above,
    # with the alphas on the diagonal and the betas on both off-diagonals.
    tri = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    values, states = np.linalg.eigh(tri)
    # eigh sorts ascending, so column 0 is the lowest Ritz pair -- the ground state of the
    # projection, which is the best approximation the Krylov space can offer.
    ground = states[:, 0]
    # Recombine those coefficients into a tensor: the Ritz vector back in the full space.
    out = vecs[0] * float(ground[0])
    for k in range(1, len(vecs)):
        out = tenet.add(out, vecs[k] * float(ground[k]))
    # Renormalize: the coefficients are unit-norm only if the basis stayed orthonormal,
    # and the caller is handed a state, so the norm is enforced rather than assumed.
    return float(values[0]), out / tenet.norm(out)
