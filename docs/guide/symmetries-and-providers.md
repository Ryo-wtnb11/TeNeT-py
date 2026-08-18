# Symmetries and providers

A symmetry in `tenet` is a **provider**: a small frozen object that answers
questions about sectors — how they fuse, what their duals are, what the
recoupling coefficients look like. Spaces, legs and tensors all carry a
reference to one, and every operation asks the provider for exactly the data
it needs. This page covers the sector conventions, what a provider can and
cannot do, and how the library tells you the difference.

## Sectors and their labels

Each provider has its own frozen sector dataclass. The built-in providers and
their labelling:

| provider | sector | label |
| --- | --- | --- |
| `U1` | `U1Sector(charge)` | any integer charge |
| `Z2` | `Z2Sector(parity)` | 0 or 1 |
| `fZ2` | `FZ2Sector(parity)` | 0 or 1, **fermionic** — odd wires braid with Koszul signs |
| `SU2` | `SU2Sector(two_j)` | **2j**, not j |
| `Trivial` | `TrivialSector()` | the one sector |

**SU(2) sectors are labelled by `2j`.** This is the single most common trip
for a new user: `SU2Sector(1)` is the spin-½ doublet, `SU2Sector(2)` is the
spin-1 triplet, and there is no half-integer anywhere in a label. The
convention keeps every label an exact integer:

```python
>>> from tenet.symmetry import SU2, SU2Sector
>>> SU2.irrep_dim(SU2Sector(1))   # spin-1/2: dimension 2j + 1 = 2
2
>>> SU2.fusion(SU2Sector(1), SU2Sector(1))   # 1/2 x 1/2 = 0 + 1
(SU2Sector(two_j=0), SU2Sector(two_j=2))

```

Abelian fusion is single-valued — a U(1) fusion is charge addition:

```python
>>> from tenet.symmetry import U1, U1Sector
>>> U1.fusion(U1Sector(1), U1Sector(2))
(U1Sector(charge=3),)

```

Composite symmetries are built with `ProductProvider`, whose sectors are
tuples of the factors' sectors. SU(N) (SU(3) first) lives in
`tenet.symmetry.sun` and needs the `[sun]` extra.

## Providers are data plus capabilities

Every provider implements the base protocol,
[FusionRules][tenet.symmetry.FusionRules]: a name, a unit sector, `fusion` and
the fusion multiplicity `n_symbol`. Everything beyond that is an optional
**data protocol** — a capability the provider may or may not have. The ones
you will meet first:

- [ClebschGordanData][tenet.symmetry.ClebschGordanData] — irrep dimensions and
  CG tensors; what `to_dense` and physical `shape` need.
- [QuantumDimensionData][tenet.symmetry.QuantumDimensionData] — `qdim`; what
  the weighted [tenet.norm][] and [tenet.full_trace][] need.
- [BraidingData][tenet.symmetry.BraidingData] /
  [AssociatorData][tenet.symmetry.AssociatorData] — R- and F-symbols, the
  recoupling data behind leg permutations.
- [BendingCoefficients][tenet.symmetry.BendingCoefficients] — line bends, what
  [repartition][tenet.SymmetricTensor.repartition] needs.
- [DualityData][tenet.symmetry.DualityData],
  [DualBasis][tenet.symmetry.DualBasis],
  [FSIndicatorData][tenet.symmetry.FSIndicatorData],
  [TwistData][tenet.symmetry.TwistData], ... — the rest of the lattice, each
  documented on its protocol in the [symmetry API page](../api/symmetry.md).

The split matters because operations gate on *exactly* the data they use. An
operation that only routes blocks around runs on any provider; one that needs
CG tensors demands `ClebschGordanData` and nothing more. You can query the
lattice yourself with [supports][tenet.symmetry.supports] and
[requires][tenet.symmetry.requires]:

```python
>>> from tenet.symmetry import Z2, ClebschGordanData, SymmetryCast, supports, requires
>>> supports(U1, ClebschGordanData)
True
>>> supports(Z2, SymmetryCast)
False
>>> requires(Z2, SymmetryCast)
Traceback (most recent call last):
    ...
tenet.symmetry.base.CapabilityError: Z2Provider does not provide capability SymmetryCast

```

## What a `CapabilityError` means

A [CapabilityError][tenet.symmetry.CapabilityError] is a **categorical
refusal, not a missing feature**. When `full_trace` refuses a provider without
quantum dimensions, it is not saying "unimplemented, sorry" — it is saying the
operation *has no meaning* for that symmetry as declared: there is no such
thing as the qdim-weighted trace of a category that has no quantum dimensions.
The error names the provider and the capability, so the fix is never a
workaround inside `tenet`; it is either a different operation (one that does
not need the data) or a different symmetry declaration for your problem. The
`Raises` sections throughout the [API reference](../api/tenet.md) tell you,
per operation, which capabilities are gated and where.

## Validated properties, not declared ones

Whether a provider's braiding is symmetric, whether it is spherical or
modular or unitary, is **computed, never declared**: the
`tenet.symmetry.coherence` module carries validators (pentagon, hexagon,
snake, sphericality, unitarity) that check every instance of an identity over
an explicit sector budget, and `properties()` bundles the derived
classification, cached:

```python
>>> from tenet.symmetry.coherence import properties
>>> p = properties(SU2, (SU2Sector(0), SU2Sector(1)))
>>> (p.braided, p.symmetric, p.spherical, p.unitary)
(True, True, True, True)

```

Nothing here runs on an operation's hot path — validators run in tests and at
two cached property gates. If you implement your own provider, the validators
are how you prove its coefficient tables are coherent before a tensor ever
touches them.

## Gauge fingerprints

Recoupling coefficients are only meaningful against the gauge conventions
that produced them, so the SU(2), SU(N) and fZ2 providers each carry a gauge
fingerprint that [tenet.save][] writes into every file and [tenet.load][]
verifies — a gauge-mismatched file is refused rather than silently misread
(see [load][tenet.load]'s `Raises`).

## Where next

- [Tensors, legs and spaces](tensors-legs-spaces.md) — how spaces and legs put
  sectors to work.
- [Contraction](contraction.md) — where the fermionic Koszul signs actually
  bite, and the rule that tames them.
- The [symmetry API page](../api/symmetry.md) — every protocol, every
  provider, every validator, with examples.
