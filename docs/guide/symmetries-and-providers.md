# Symmetries and providers

A symmetry in `tenet` is a **provider**: a small frozen object that answers questions
about sectors — how they fuse, what their duals are, what the recoupling coefficients
are. Spaces, legs and tensors carry a reference to one, and every operation asks the
provider for exactly the data it needs.

## Sectors and their labels

Each provider has its own frozen sector dataclass:

| provider | sector | label |
| --- | --- | --- |
| `U1` | `U1Sector(charge)` | any integer charge |
| `Z2` | `Z2Sector(parity)` | 0 or 1 |
| `fZ2` | `FZ2Sector(parity)` | 0 or 1, **fermionic** — odd wires braid with Koszul signs |
| `SU2` | `SU2Sector(two_j)` | $2j$, **not** $j$ |
| `Trivial` | `TrivialSector()` | the one sector |

**SU(2) sectors are labelled by $2j$.** `SU2Sector(1)` is the spin-1/2 doublet,
`SU2Sector(2)` the spin-1 triplet. The convention keeps every label an exact integer:

```python
>>> from tenet.symmetry import SU2, SU2Sector
>>> SU2.irrep_dim(SU2Sector(1))              # spin-1/2: dimension 2j + 1 = 2
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

## Products

`ProductProvider` is the Deligne product of two or more providers; its sectors are
tuples of the factors' sectors, and every coefficient factorizes across them:

```python
>>> from tenet.symmetry import ProductProvider, ProductSector, Z2, Z2Sector
>>> P = ProductProvider((U1, Z2))
>>> P.fusion(ProductSector((U1Sector(1), Z2Sector(1))),
...          ProductSector((U1Sector(2), Z2Sector(1))))
(ProductSector(components=(U1Sector(charge=3), Z2Sector(parity=0))),)

```

A capability is forwarded when **every** factor has it, so a product is as capable as
its weakest factor and no more. The spinful-fermion grading — parity, charge and total
spin at once — is the one worth naming:

```python
>>> from tenet.symmetry import SU2, SU2Sector, FZ2Sector, fZ2
>>> from tenet import GradedSpace
>>> F = ProductProvider((fZ2, U1, SU2))
>>> F.name
'fZ2 x U1 x SU2'
>>> def sector(parity, charge, two_j):
...     return ProductSector((FZ2Sector(parity), U1Sector(charge), SU2Sector(two_j)))
>>> site = GradedSpace.new(F, {sector(0, 0, 0): 1, sector(1, 1, 1): 1, sector(0, 2, 0): 1})
>>> site.dim, site.reduced_dim
(4, 3)

```

That is a Hubbard site: empty and doubly-occupied are even spin singlets, the
singly-occupied states an odd $j = 1/2$ doublet. Four physical states, three multiplets
— the SU(2) factor is what makes those two numbers differ, and what a bond of such a
space compresses.

## SU(N)

SU(N) lives in `tenet.symmetry.sun` and is not re-exported from `tenet.symmetry`,
because a provider carries its own `n` and there is no singleton to name:

```python
>>> from tenet.symmetry.sun import SUNProvider, SUNSector
>>> SU3 = SUNProvider(3)
>>> SU3.fusion(SUNSector((1, 0)), SUNSector((0, 1)))   # 3 x 3bar = 1 + 8
(SUNSector(dynkin=(0, 0)), SUNSector(dynkin=(1, 1)))

```

`SUNProvider(n)` works for every `n`: the recoupling coefficients — and SU(2)'s — are
computed by `racah-py`, a core dependency, so `import tenet.symmetry.sun` works on a
plain `pip install tenet-py`. The coefficients carry `racah`'s gauge convention, which
[tenet.save][] records and [tenet.load][] verifies.

The provider set follows `racah`: product groups such as SU(2) × U(1), U(N), and the
exceptional groups (G2 first) are on its roadmap and will appear here as providers when
their coefficients land.

## Providers are data plus capabilities

Every provider implements the base protocol, [FusionRules][tenet.symmetry.FusionRules]:
a name, a unit sector, `fusion`, and the fusion multiplicity `n_symbol`. Everything
beyond that is an optional **data protocol** — a capability the provider may or may not
have:

- [ClebschGordanData][tenet.symmetry.ClebschGordanData] — irrep dimensions and CG
  tensors; what `to_dense` and the physical `shape` read.
- [QuantumDimensionData][tenet.symmetry.QuantumDimensionData] — `qdim`; what the
  weighted [tenet.norm][] and [tenet.full_trace][] read.
- [BraidingData][tenet.symmetry.BraidingData] /
  [AssociatorData][tenet.symmetry.AssociatorData] — R- and F-symbols, the recoupling
  data behind leg permutations.
- [BendingCoefficients][tenet.symmetry.BendingCoefficients] — line bends, what
  [repartition][tenet.SymmetricTensor.repartition] reads.
- [DualityData][tenet.symmetry.DualityData], [DualBasis][tenet.symmetry.DualBasis],
  [FSIndicatorData][tenet.symmetry.FSIndicatorData], [TwistData][tenet.symmetry.TwistData],
  [BranchingRules][tenet.symmetry.BranchingRules] — the rest, each documented on its
  protocol in the [symmetry API page](../api/symmetry.md).

Operations gate on exactly the data they use. One that only routes blocks around runs on
any provider; one that needs CG tensors demands `ClebschGordanData` and nothing more.
Query the lattice with [supports][tenet.symmetry.supports] and
[requires][tenet.symmetry.requires]:

```python
>>> from tenet.symmetry import ClebschGordanData, BranchingRules, supports, requires
>>> supports(U1, ClebschGordanData)
True
>>> supports(Z2, BranchingRules)
False
>>> requires(Z2, BranchingRules)
Traceback (most recent call last):
    ...
tenet.symmetry.base.CapabilityError: Z2Provider does not provide capability BranchingRules

```

## What a `CapabilityError` means

A [CapabilityError][tenet.symmetry.CapabilityError] is a **categorical refusal**. When
`full_trace` refuses a provider without quantum dimensions, it says the operation has no
meaning for that symmetry as declared: there is no qdim-weighted trace of a category
with no quantum dimensions. The error names the provider and the capability. The
response is a different operation — one that does not need the data — or a different
symmetry declaration for your problem. Every operation's `Raises` section in the
[API reference](../api/tenet.md) says which capabilities it gates on.

## Restricting to a smaller symmetry

A provider with [BranchingRules][tenet.symmetry.BranchingRules] can be restricted in the
dense basis. [to_symmetry][tenet.to_symmetry] takes an SU(2) tensor to U(1) by branching
each multiplet into its magnetic quantum numbers:

```python
>>> import tenet
>>> from tenet import IN, OUT, GradedSpace, Leg, SymmetricTensor
>>> S = GradedSpace.new(SU2, {SU2Sector(1): 1})
>>> t = SymmetricTensor.random((Leg(S, OUT), Leg(S, IN)), seed=0)
>>> u = tenet.to_symmetry(t, U1)
>>> u.provider.name, u.shape
('U1', (2, 2))

```

## Validated properties, not declared ones

Whether a provider's braiding is symmetric, whether it is spherical or unitary, is
**computed**: `tenet.symmetry.coherence` carries validators (pentagon, hexagon, snake,
sphericality, unitarity) that check every instance of an identity over an explicit
sector budget, and `properties()` bundles the derived classification, cached:

```python
>>> from tenet.symmetry.coherence import properties
>>> p = properties(SU2, (SU2Sector(0), SU2Sector(1)))
>>> (p.braided, p.symmetric, p.spherical, p.unitary)
(True, True, True, True)

```

This runs in tests only — operations gate through `supports`/`requires`, not through a
validator — with one exception worth knowing about. `transpose` gates on the
symmetric-braiding property, so `symmetric_braiding` is called whenever axes reorder
within a side. It is cached per `(provider, sectors)`: a repeat is a dict lookup, while
the first call over a given sector budget evaluates the provider's R-symbols, which for a
provider whose coefficients come from a runtime generator is milliseconds rather than
microseconds. It is paid once per distinct budget per process.

(`full_trace`'s sphericality refusal is not one of these. It compares exact quantum
dimensions on the sectors actually traced and never calls a validator.)

If you implement your own provider, the validators are how you prove its coefficient
tables are coherent before a tensor touches them.

## Gauge fingerprints

Recoupling coefficients are meaningful only against the gauge conventions that produced
them, so the SU(2), SU(N) and fZ2 providers each carry a gauge fingerprint that
[tenet.save][] writes into every file and [tenet.load][] verifies. A gauge-mismatched
file is refused — see [Saving and loading](saving-and-loading.md).

## Where next

- [Tensors, legs and spaces](tensors-legs-spaces.md) — how spaces and legs put sectors
  to work.
- [Contraction](contraction.md) — where the fermionic Koszul signs bite, and the rule
  that tames them.
- [The symmetry API page](../api/symmetry.md) — every protocol, provider and validator.
