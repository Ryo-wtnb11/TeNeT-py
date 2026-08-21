"""#205 Part 1, the measurement gate: is the shipped double-layer CTMRG corner indefinite?

`network/ctmrg.py`'s `move` projects the enlarged corner with a **single** isometry `u`
taken from `svd_truncated`, which is exact only when that corner is *positive*. Its
docstring says an indefinite corner still gets a self-consistent contraction "but its
corner and edge then differ by a diagonal of signs". #205 refuses to build
`eigh_truncated` before that sentence has a number behind it. This is the instrument that
produces the number.

## The basis pairing, and why it is measured rather than asserted

"Eigenvalue" is a statement about an *endomorphism*, and which row of a coupled-sector
matrix pairs with which column is a basis question the axis order settles. `check_square`
refuses every CTM corner in this file -- the C4v mirror identifies a space with its *dual*,
which is a `flip_dual` and not an equality -- so the pairing is not asserted here. Two
candidates are computed and both are printed: `direct`, `move`'s own `ndim // 2` order, and
`swapped`, which exchanges the domain's last two axes (the ket/bra pair of a double-layer
corner). Eigenvalues are read off whichever one is Hermitian; a within-side transpose is a
*unitary* on the domain, so it leaves `U` and `Sigma` exactly as `move` sees them and only
rotates `V`. Every number below is about the factorization the library actually runs.

## The columns

* `herm direct` / `herm swapped` -- `max_c ||B - B†|| / ||B||` under each pairing. The
  corner is only *meant* to be Hermitian; these say whether it still is, and under which
  identification.
* `neg` / `|w_neg|/w_max` -- eigenvalues of the Hermitian part `(B + B†)/2`, per coupled
  sector, summed: how many are negative and how far above zero the largest of them sits.
* `sigma_cut/sigma_max` -- the SVD's own truncation threshold at this `chi`. A negative
  eigenvalue *below* the smallest kept singular value is discarded by the projector and
  cannot reach the answer; this is the bar the gate compares against.
* `kept neg` -- negatives with `|w| >= sigma_cut`, i.e. the ones the projector keeps. **This
  is the gate's number.**
* `max|u-v|` -- the deviation between the two isometries `svd_truncated` would produce,
  over the kept columns. This is "the diagonal of signs" made into a number, and it needs
  no gauge fixing: the SVD's freedom is a *joint* phase on `(u_j, v_j)`, so `|u_j - v_j|`
  is invariant. It is 0 for a positive corner and `2|v_j|` on every column whose eigenvalue
  is negative -- and `move` uses only `u`.

Fixtures: the c4v iPEPS of `tests/integration/test_ctmrg.py:190` (U(1) and SU(2), at the
example's own `CHI_IPEPS`), the SU(2) ket of `tests/network/test_ctmrg.py:82`, and the
single-layer Ising corner as the **control** -- documented positive, and the row that says
the instrument reads zero when there is nothing to read.

Run: `uv run python benchmarks/bench_ctm_corner_signs.py`. On no CI path.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples" / "toy_codes"))

import ctmrg as example  # noqa: E402

import tenet  # noqa: E402
from tenet import IN, OUT, Leg, SymmetricTensor  # noqa: E402
from tenet.map_view import to_matrices  # noqa: E402
from tenet.network import double_layer_ctm, move, single_layer_ctm  # noqa: E402


def pairings(big_c):
    """`{label: matrices}` for the enlarged corner under each candidate basis pairing.

    "Eigenvalue" is a statement about an endomorphism, and which column of the
    coupled-sector matrix pairs with which row is a *basis* question the axis order
    settles. `check_square` refuses every corner here -- the mirror identifies a space
    with its dual, which is a `flip_dual`, not an equality -- so the pairing is chosen by
    measurement instead of asserted: both candidates are reported, and the eigenvalues are
    read off whichever one is actually Hermitian. `direct` is `move`'s own axis order;
    `swapped` exchanges the domain's last two axes, which is the ket/bra pair of a
    double-layer corner.
    """
    n = big_c.ndim // 2
    axes = (tuple(range(n)), tuple(range(n, 2 * n)))
    out = {}
    for label, perm in (("direct", None), ("swapped", (*range(2 * n - 2), 2 * n - 1, 2 * n - 2))):
        t = big_c if perm is None else tenet.transpose(big_c, perm)
        mats = to_matrices(tenet.repartition(t, *axes))
        if all(np.asarray(b).shape[0] == np.asarray(b).shape[1] for b in mats.values()):
            out[label] = mats
    return out, axes


def defect(mats):
    """`max_c ||B - B†|| / ||B||`."""
    return max(
        float(
            np.linalg.norm(np.asarray(b) - np.asarray(b).conj().T) / np.linalg.norm(np.asarray(b))
        )
        for b in mats.values()
    )


def eigen_report(mats, sigma_cut):
    """`(negatives, |w_neg|/w_max, negatives the projector keeps)`."""
    negatives, kept, w_neg, w_max = 0, 0, 0.0, 0.0
    for b in mats.values():
        a = np.asarray(b)
        w = np.linalg.eigvalsh((a + a.conj().T) / 2)
        negatives += int(np.sum(w < 0))
        kept += int(np.sum((w < 0) & (np.abs(w) >= sigma_cut)))
        if np.any(w < 0):
            w_neg = max(w_neg, float(np.max(np.abs(w[w < 0]))))
        w_max = max(w_max, float(np.max(np.abs(w))))
    return negatives, (w_neg / w_max if w_max else 0.0), kept


def isometry_gap(mats, keep):
    """`max_j |u_j - v_j|` over the kept columns, on the square form.

    Gauge-free without any alignment: the SVD's freedom is a *joint* phase, `u_j -> e^{it}
    u_j` forcing `v_j -> e^{it} v_j`, so `|u_j - v_j|` is invariant while the relative sign
    between them is not free at all. For a positive corner it is 0; for a Hermitian corner
    with a negative eigenvalue `w_j`, `u_j = -v_j` on that column and the entry is `2 |v_j|`.
    NumPy's own SVD, so the number does not depend on `tenet`'s factorization; only *which*
    columns are kept comes from `select_bond`.
    """
    gap = 0.0
    for c, k in keep.items():
        u, _, vh = np.linalg.svd(np.asarray(mats[c]), full_matrices=False)
        v = vh.conj().T
        for j in range(min(k, u.shape[1])):
            gap = max(gap, float(np.max(np.abs(u[:, j] - v[:, j]))))
    return gap


HEAD = (
    "| move | dim | herm direct | herm swapped | neg | `|w_neg|/w_max` | "
    "`sigma_cut/sigma_max` | kept neg | `max|u-v|` |"
)
HERMITIAN = 1e-10  # the corner is Hermitian or it is not; there is no middle at float64


def sweep(name, absorb, c, e, chi, moves=6):
    print(f"\n### {name} (chi={chi})\n")
    print(HEAD)
    print("|---|---|---|---|---|---|---|---|---|")
    verdict, indefinite = 0, 0
    for k in range(moves):
        big_c = absorb.corner(c, e)
        candidates, axes = pairings(big_c)
        defects = {label: defect(mats) for label, mats in candidates.items()}
        best = min(defects, key=defects.get)

        selection = tenet.linalg.select_bond(big_c, axes, max_bond=chi)
        values = [x for x, _, _ in selection.kept]
        sigma_cut, sigma_max = min(values), max(values)

        if defects[best] < HERMITIAN:
            negatives, ratio, kept_neg = eigen_report(candidates[best], sigma_cut)
            gap = isometry_gap(candidates[best], dict(selection.bond.sectors))
            cells = (
                f"{negatives} | {ratio:.3e} | {sigma_cut / sigma_max:.3e} | {kept_neg} | {gap:.3e}"
            )
            verdict += kept_neg
            indefinite += negatives > 0
        else:
            # not Hermitian under either pairing: an eigenvalue is not defined, and the
            # non-Hermiticity is itself the damage -- move 0 is exact, later moves are not
            kept_neg = 0
            cells = f"n/a | n/a | {sigma_cut / sigma_max:.3e} | n/a | n/a"

        dim = sum(np.asarray(b).shape[0] for b in candidates[best].values())
        print(
            f"| {k} | {dim} | {defects.get('direct', float('nan')):.2e} | "
            f"{defects.get('swapped', float('nan')):.2e} | {cells} |"
        )
        c, e, _ = move(c, e, absorb, chi=chi)
    print(
        f"\n**{name}: {indefinite} of {moves} moves had a Hermitian corner with a negative "
        f"eigenvalue; {verdict} of those negatives sit above the projector's own cut.**"
    )
    return verdict


def ipeps(provider, seed=1):
    phys, virt = example.SPACES[provider]
    legs = (Leg(phys, OUT), Leg(virt, OUT), Leg(virt, OUT), Leg(virt, IN), Leg(virt, IN))
    return example.c4v(SymmetricTensor.random(legs, seed=seed))


def main():
    print("# #205 Part 1 — is the double-layer CTMRG corner indefinite?")
    total = 0
    total += sweep(
        "single-layer Ising, beta=0.4 (control)", *single_layer_ctm(example.ising_bulk(0.4)), chi=4
    )
    for provider in ("u1", "su2"):
        chi = example.CHI_IPEPS[provider]
        total += sweep(f"c4v iPEPS, {provider}", *double_layer_ctm(ipeps(provider)), chi=chi)
    total += sweep("c4v iPEPS, su2 (seed 3)", *double_layer_ctm(ipeps("su2", seed=3)), chi=6)
    print(
        "\n## Verdict\n\n"
        f"{total} negative eigenvalues above the projector's own truncation threshold. "
        "The gate fires at anything above zero on a double-layer fixture."
    )


if __name__ == "__main__":
    main()
