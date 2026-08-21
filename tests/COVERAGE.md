# Test coverage: the provider × operation × mode matrix

What the suite *runs*, written down (issue #146). Line coverage measures which
lines ran; the gap #145 found was which *combination* ran — `sun.py` was green
the whole time `grad × SU(N)` had never executed. This file is the combination
ledger: the matrix as measured, the classification of every empty cell, the
standing rule that keeps it honest, and the budget rule that pays for new cells.

Enforced mechanically only where a grep suffices: `tests/test_coverage.py`
carries E1 (every public operation reaches the torch suite or names its reason)
and E2 (the AD suite keeps a multiplicity-bearing provider). Everything else in
this file is review discipline.

## Columns: the providers

The canonical list is the suite's only all-eight parametrization,
`tests/symmetry/test_flip_scalar.py:34`:

`Trivial, U1, Z2, fZ2, SU2, SUNProvider(3), ProductProvider((U1, SU2)), SU3`

`SU3` is the vendored fixture provider (`tests/symmetry/_su3_fixture.py`,
sectors truncated at `27`, `N^8_{88} = 2`, no registry name); `SUNProvider(3)`
is the racah-backed one. Since M24a (#158) there is a ninth, smoke-level
column: the Fibonacci fixture (`tests/symmetry/_fibonacci_fixture.py`, no
registry name, deliberately no `cgc`/`irrep_dim`/`z_matrix` and a chiral
braiding), exercised only in `tests/symmetry/test_fibonacci.py` — build, bend,
`full_trace`, the coherence validators of `tenet.symmetry.coherence`
(pentagon/hexagon/snake/spherical/non-degeneracy, which SU(2) also runs in
`test_su2_recoupling.py`), and the two named refusals (`transpose` on a chiral
braid, `to_dense` without `ClebschGordanData`). It joins no row below by
design; the capability lattice, not this fixture, is what those rows cover. `racah-py` is a core dependency and JAX sits in the default dev group
(`pyproject.toml`), torch has its own CI step — **no cell below is unreachable
in CI**; every empty cell is a decision, not a constraint.

## Rows: operation families × providers (numpy eager)

| # | family | covered on | evidence |
|---|--------|------------|----------|
| 1 | construction / `from_dense` / `to_dense` | all 8 | `tests/ops/test_dense.py:43` (`UU`); `tests/symmetry/test_z2.py:299-317`; `tests/symmetry/test_su3_multiplicity.py:143-155`; `tests/symmetry/test_sun.py:316-328` |
| 1b | keyed construction: `from_blocks` / `with_blocks` (#208) | `{u1, su2, su3}` on numpy; all five torch columns; the rest are (b6) below | `tests/test_tensor_keyed.py` (round trip through `items`, absent-key zero fill, the two refusals, the frozen/pytree contracts); `tests/backends/test_torch.py` (the zero fill follows the supplied block's backend); the two cups it replaced, `tests/symmetry/test_sun.py:349-355` and `tests/symmetry/test_su2_dual.py:189-195` |
| 1c | dtype and backend moves: `astype` / `to_backend(dtype=)` (#207) | `{su2}` × `{numpy, jax, torch}`; provider-independent, (b6) below | `tests/test_tensor_properties.py` (float64↔complex128 round trip, the block-free refusal, and the subprocess that pins what `dtype=` cannot do to a JAX without `jax_enable_x64`); `tests/backends/test_torch.py` (both dtype spellings, since autoray's torch `astype` takes only the name) |
| 2 | `transpose` / braiding | all 8 | `tests/ops/test_permutation.py`; `tests/symmetry/test_z2.py:264-298`; `tests/symmetry/test_fz2.py`; `tests/symmetry/test_product.py`; `tests/symmetry/test_su3_multiplicity.py:157-171`; `tests/symmetry/test_sun.py:330-345` |
| 3 | `repartition` / `bend` | all 8 | `tests/ops/test_repartition.py`; `tests/ops/test_repartition_plan.py:51` (`UF`); `tests/symmetry/test_z2.py:330-357`; `tests/symmetry/test_su3_multiplicity.py:174-207`; `tests/symmetry/test_sun.py:350-382` |
| 4 | `fuse` / `unfuse` | `{su2, u1, trivial, fz2}` | `tests/ops/test_fusion.py:49` (`ALL_LEGS`; fZ2 joined in #146); dense oracles at `:245-262` |
| 5 | `compose` / `tensordot` / `einsum` | `{trivial, u1, su2, fz2, product}`; `einsum_multi` drops product | `tests/ops/test_einsum.py:75`; `tests/ops/test_einsum_multi.py:82`; `tests/ops/test_contraction.py:61` |
| 6 | `full_trace` / `inner` | `{su2, u1}` (+ torch row below) | `tests/ops/test_full_trace.py:20-23` |
| 7 | linalg (`svd`/`qr`/`lq`/`polar`/`eigh`/`eig`/`expm`/`*_null`/`svd_truncated`) | `{trivial, u1, su2, fz2}` | `tests/ops/test_linalg.py:52-62`; `tests/ops/test_svd_truncated.py:15-24` imports that same `PROVIDERS` — the reuse pattern the budget rule mandates |
| 8 | `flip_dual` (#142) | capability on all 8; numerically `{u1, fz2, su2, su3}` (+ Z2/product spaces) | `tests/symmetry/test_flip_scalar.py:34`; `tests/ops/test_flip_dual.py` |
| 9 | `embed` / `restrict` / `direct_sum` / `isometry` | `{trivial, u1, su2, fz2, product}` | `tests/ops/test_embed.py:57`; `tests/ops/test_isometry.py:45-61` |
| 10 | `to_symmetry` | bounded by `BranchingRules`: targets `{SU2, Trivial, fZ2, U1×U1}`; U1-as-source is the refusal | `tests/ops/test_to_symmetry.py:97`, `:386-408` |
| 11 | `save` / `load` | `{trivial, u1, z2, su2, fz2, product, nested}` + SU(N) — all seven registered providers round-trip | `tests/test_serialize.py:36-57`; `tests/symmetry/test_z2.py:390-419`; `tests/symmetry/test_sun.py:451-501` |
| 12 | network MPS/MPO/Env/dmrg/ctmrg | `{u1, su2, z2, su3, fz2}` (fermionic MPO/DMRG since #147; ctmrg stays bosonic) | `tests/network/test_mps.py:24-32`; `tests/network/test_mpo.py`; `tests/network/test_hubbard.py`; `tests/network/test_heff2.py:110-125`; `tests/network/test_deferred.py` (the deferred instantiation boundary, `{u1, fz2}` incl. spinful Hubbard); `tests/network/test_density_matrix.py` (M61 Stage C's second decimation and `Env.heff2_families`, `{u1, fz2, su2}`); `tests/network/test_two_state_env.py` and `tests/integration/test_dmrg_excited.py` (M61 Stage D's `Env(bra=)`, `Env.project2`, `MPO.identity` and `dmrg_(orthogonal_to=)`, `{u1, fz2, su2}`); `tests/network/test_apply.py` (M49's `MPO.apply` and `MPO.variance`, `{u1, su2, fz2}` x both `from_terms` representations -- the dense oracle is what says the virtual-leg turn-around is charged by the leg and not by its `dual` flag); `tests/network/test_measure.py` (M48's `overlap`, `measure_mpo`, `correlation_function` and `expectation_profile`, `{u1, fz2}` -- the distance-`r` fermionic correlator against #147's explicit Jordan-Wigner oracle); `tests/network/test_entanglement.py` (M50's `MPS.schmidt_values`/`schmidt_sectors`/`entanglement_entropy` and `network.spectrum_sectors`/`entropy`, `{u1, su2}` -- the SU(2)-equals-U(1) entropy is what pins the `sqrt(qdim)` multiplet weight); `tests/network/test_ctmrg.py` |
| 13 | map view | `{su2, u1, trivial}` + jax | `tests/ops/test_map.py:143-166`, `:347-364` |
| 14 | `map_diagonal` / `zip_blocks` (#232) | `{u1, fz2, fz2 Hubbard d=4, su2}` for the diagonal; `{trivial, u1, su2, fz2, product}` for `zip_blocks` (+ torch row below) | `tests/ops/test_map_diagonal.py:102-121` (the formed-dense oracle on all four), `:138-166` (the constructed SU(2) two-inner-line case); `tests/ops/test_blocks.py:386-447` |
| 15 | `tenet.models` sites (#198) | `{u1, su2, fz2, trivial}` -- every grading the layer ships | `tests/models/test_sites.py`: the algebra oracles (spin commutators, fermion anticommutators, the hard-core relations) read off the **built** tensors, the SU(2) answer and its two refusals, and the end-to-end `MPO.from_arrays` / `MPO.from_terms` call shapes against dense Jordan-Wigner oracles |
| 16 | `enable_jax` (#211) | provider-independent, JAX only | `tests/test_enable_jax.py` — the pytree half, the `ad=True` half, idempotence and the JAX-absent refusal, each in a fresh subprocess (0.85 s for the module) |

`tenet.PROJECT` (#210) adds no row: it is the *name* of the `atol=math.inf` mode of rows
1, 9 and 10 and is exactly that value, so the cells it runs in are theirs. It is pinned by
the identity assertion plus a both-spellings comparison in `tests/ops/test_dense.py` and
`tests/ops/test_to_symmetry.py`, and by the refusal-message assertions in
`tests/ops/test_dense.py` and `tests/ops/test_embed.py:649`.

## The third axis: backend × mode

- **numpy eager** — every row above.
- **jax eager + jit** — scattered per module, never parametrized over providers:
  `test_blocks.py`, `test_embed.py:384`, `test_einsum.py`,
  `test_contraction_plan.py`, `test_flip_dual.py:325-336`,
  `test_svd_truncated.py:564-586`, `test_map.py:347-364`, `test_to_symmetry.py`,
  `test_dense.py`, `test_basic.py`, `test_adjoint.py`,
  `test_full_trace.py:146-173`. Absent for `fuse`/`unfuse` (jax eager only,
  `test_fusion.py:470-492`), `repartition`/`bend`, `transpose`, core linalg,
  `isometry`, `einsum_multi` — classified (b2) below.
- **jax grad** — `tests/backends/test_ad.py:53-59`: `{u1, su2, fz2, su3}` ×
  `{svd, eigh, polar, qr, lq, svd(bond=)}`. The `su3` column landed with #145
  (M19), so grad now crosses the matrix-valued F/R/B branch — the (a5) hole of
  the #146 survey is **filled**, and E2 pins it. Also
  `test_pytree.py` (`{u1, su2}` × `{norm, transpose, fuse, set_params}`),
  `test_blocks.py`, `test_einsum_multi.py`, `test_linalg_null.py`,
  `test_linalg_expm_eig.py`, `tests/integration/test_vmc.py`, `test_ctmrg.py`.
- **jax vmap** — `test_pytree.py:308-355` (SU(2) legs) and
  `tests/integration/test_vmc.py:63` (`{u1, su2}`). Nothing else.
- **torch eager** — `tests/backends/test_torch.py:76-85`:
  `{trivial, u1, su2, fz2, product}` × a broad op list. `flip_dual` and
  `full_trace`/`inner` joined in #146 (`:249`, `:414`); E1 fails on the next
  public operation that does not.
- **torch autograd** — `test_torch.py` eager-backward section, `{u1, su2}`.
  `torch.compile` / `torch.func` refused in #95.
- **`tenet.models` × torch/jax — out of contract, and E1 does not fire on it.**
  Row 15 has no backend cell to fill: every site's operators are built by
  `local_op` from a NumPy matrix written in `src/tenet/models/sites.py`, so the
  layer produces no kernel of its own and the tensors it returns reach a
  backend only through the same `SymmetricTensor` operations rows 1–14 already
  cover on torch. Nothing in `tenet.models` is exported from `tenet.__all__`
  either (it is an explicitly-imported subpackage, deliberately: see
  `tests/network/test_hygiene.py::test_no_module_imports_tenet_models`), so E1
  neither requires nor would notice a `test_torch.py` row for it.

## Empty cells, classified

Every empty cell carries exactly one of: **(a) genuine gap** (with a
follow-up), **(b) structurally redundant** (with the argument), **(c) out of
contract** (with the refusal test), **(d) blocked** (with what blocks it).

### (a) genuine gaps

- **a4. `fuse`/`unfuse` × SU(3).** `test_fusion.py`'s chunk-tiling pins
  multiplicity-free providers only; `N^c_{ab} = 2` is a distinct tiling.
  Follow-up: "SU(3) joins the fusion and contraction oracles".
- **a6. `tensordot`/`einsum` × SU(3), one dense-oracle test.** Rows 2 and 3
  cover SU(3); what has never run is the composition. Same follow-up as a4.
- *Filled:* a1 (`flip_dual` × torch), a2 (`full_trace`/`inner` × torch) and a3
  (`fuse`/`unfuse` × fZ2 against the dense oracle) by #146; a5 (grad × SU(N))
  by #145.

### (b) structurally redundant

- **b1. Z2 across the ops layer.** `test_z2.py:264-298` proves every
  permutation-plan and bending coefficient is 1, so at the coefficient layer Z2
  is U(1) with a finite charge group — and the ops layer branches on capability,
  not provider, *enforced* by `tests/ops/test_permutation.py:314`,
  `tests/ops/test_repartition.py:520` and `tests/network/test_hygiene.py`
  (no provider name appears in the source). A Z2 column in `test_linalg.py`
  would execute the same lines with the same numbers as the U(1) column.
- **b2. A jit cell per operation.** jit-ability is one property — no
  data-dependent structure decision — and the operations that make one are
  enumerable and pinned: `svd_truncated` (`test_svd_truncated.py:564-586`,
  `tests/network/test_ctmrg.py:203-219`) and the containment-checking
  `embed`/`restrict` refusals (`test_embed.py:384`). Everything else is
  `ar.do` over a static plan, and "no NumPy in the implementation" is asserted
  module by module. A jit × op × provider grid is machinery for a property that
  lives in three places.
- **b3. torch × Z2 / SU(3).** `test_torch.py:1-15`'s docstring is the argument:
  the torch coverage is breadth of *ops*, the numerics bit-identical to NumPy's
  (`np.array_equal`, not `allclose`), so a third fixture family proves nothing
  new. Adopted as the row rule.
- **b4. grad × Trivial / Z2 / ProductProvider.** Coefficient-1 providers add no
  term to a VJP that fZ2 and SU(2) do not already carry;
  the matrix-valued `FMatrixData`/`RMatrixData`/`BMatrixData` are the
  capabilities that do — which is why a5
  was (a) and these are (b).
- **b6. `enable_jax` × every provider, and × torch.** It takes no tensor: it
  registers `SymmetricTensor` with JAX and, on request, `tenet.ad`'s VJPs with
  autoray. No provider, no block and no backend kernel enters it, so a per-provider
  cell would execute the same two registrations. It is in
  `tests/test_coverage.py`'s `NOT_ON_TORCH` for the same reason — eager torch needs
  no registration of its own (`docs/design.md`, "What \"PyTorch backend\" means").
- **b5. `embed`/`restrict`/`direct_sum` × SU(3).** Those act on degeneracy
  multiplicities inside a coupled sector; a fusion-vertex multiplicity is a row
  count inside the same block, orthogonal to the containment check.
- **b6. Rows 1b and 1c across the remaining providers.** `from_blocks`,
  `with_blocks`, `astype` and `to_backend(dtype=)` consume no fusion
  coefficient: the first two are a `block_order` membership test plus a
  positional fill, the last two are one `ar.do` per block. Provider enters only
  through `block_order`, and the three columns run already cover the shapes it
  takes — one channel (`u1`), several channels per coupled sector (`su2`), and a
  vertex multiplicity (`su3`). A Z2 or Trivial column would execute the same
  lines on shorter tuples.

### (c) out of contract — the cell is a refusal test, and it exists

- **c1.** Truncating network entry points under jit: `svd_truncated` decides a
  structure from singular *values* — `tests/network/test_ctmrg.py:203-219`.
- **c2.** Fermionic MPO: the coupling coefficients contain no swap gate —
  `tests/network/test_mpo.py:317-337`.
- **c3.** `to_symmetry` with U(1) as source: `U1Provider` provides no `BranchingRules` —
  `tests/ops/test_to_symmetry.py:386-400`.
- **c4.** `save`/`load` × the SU(3) *fixture*: it has no registry name on
  purpose; `SUNProvider(3)` covers the row — `tests/symmetry/test_sun.py:451-501`.
- **c5.** Non-Abelian MPO terms fusing through three channels —
  `tests/network/test_mpo.py:293-305`.
- **c6.** `inner` × product provider: `inner` bends every leg past the first,
  and a product provider forwards no `BendingCoefficients` (#40) —
  `tests/backends/test_torch.py:414` pins the `CapabilityError`.

### (d) blocked on infrastructure

None — stated because it was the natural suspicion. `racah-py` is a core
dependency and JAX is a default dev-group one, torch has its own CI step; every
cell is reachable in CI today.

## The standing rule

A PR that adds a symmetry provider or a public operation updates
`tests/COVERAGE.md` in the same diff: a provider names its column and the
cells it fills; an operation names its row and the providers and modes it
runs under. Any cell left empty carries one of the four classifications —
genuine gap (with a follow-up), structurally redundant (with the argument),
out of contract (with the refusal test), or blocked (with what blocks it) —
in one line.

(The same bullet lives in `REPOSITORY_RULES.md`'s Tests section and
`AGENTS.md`'s coding rules.)

## The budget rule

The standing suite budget is **290 s** (#145), and the headroom above the
measured baseline is small — see the number in the PR that touched this file
last. So:

- A new cell is a new `parametrize` entry on an existing test body wherever
  the body already exists. `tests/ops/test_svd_truncated.py:15-24` importing
  `PROVIDERS`, `SPLIT`, `tensor` and `dense_matrix` straight out of
  `test_linalg.py` is the pattern to copy.
- A cell that genuinely needs a new test body states its measured cost in the
  PR. If the suite would exceed 290 s, the cell moves onto an existing fixture
  or the budget is renegotiated in the issue that adds it — never silently.
- A cell that would run the same lines with the same coefficients as an
  existing cell is not coverage, it is runtime: classify it (b) instead.
- No coverage-percentage gate, here or in CI: the metric is blind to exactly
  the class of hole this file exists to track.
