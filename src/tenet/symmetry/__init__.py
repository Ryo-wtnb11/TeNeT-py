"""Symmetry providers and sector labels.

A *provider* answers every categorical question about one symmetry — fusion rules,
quantum dimensions, duality, F/R symbols, braiding — and a *sector* labels one irrep.
[Trivial][tenet.symmetry.Trivial], [Z2][tenet.symmetry.Z2], [fZ2][tenet.symmetry.fZ2],
[U1][tenet.symmetry.U1] and [SU2][tenet.symmetry.SU2] are the shipped instances, each
beside its class and its ``*Sector`` type, and
[ProductProvider][tenet.symmetry.ProductProvider] composes two into one.

The capability protocol is the rest of the page: [supports][tenet.symmetry.supports] and
[requires][tenet.symmetry.requires] ask and demand, the ``*Data`` protocols name what a
provider may implement, and [CapabilityError][tenet.symmetry.CapabilityError] is the
refusal raised when one is missing.
"""

from tenet.symmetry.base import (
    AssociatorData,
    BendingCoefficients,
    BMatrixData,
    BraidingData,
    BranchingRules,
    CapabilityError,
    ClebschGordanData,
    DaggerData,
    DualBasis,
    DualityData,
    FMatrixData,
    FSIndicatorData,
    FusionRules,
    PermutationCoefficients,
    PivotalData,
    QuantumDimensionData,
    RMatrixData,
    Sector,
    StructureChangingError,
    Trivial,
    TrivialProvider,
    TrivialSector,
    TwistData,
    bend_braided,
    bend_unique,
    permute_braided_tree,
    permute_unique_tree,
    requires,
    supports,
)
from tenet.symmetry.fz2 import FZ2Provider, FZ2Sector, fZ2
from tenet.symmetry.product import ProductProvider, ProductSector
from tenet.symmetry.su2 import SU2, SU2Provider, SU2Sector
from tenet.symmetry.u1 import U1, U1Provider, U1Sector
from tenet.symmetry.z2 import Z2, Z2Provider, Z2Sector

__all__ = [
    "AssociatorData",
    "BMatrixData",
    "BendingCoefficients",
    "BraidingData",
    "BranchingRules",
    "CapabilityError",
    "ClebschGordanData",
    "DaggerData",
    "DualBasis",
    "DualityData",
    "FMatrixData",
    "FSIndicatorData",
    "FZ2Provider",
    "FZ2Sector",
    "FusionRules",
    "PermutationCoefficients",
    "PivotalData",
    "ProductProvider",
    "ProductSector",
    "QuantumDimensionData",
    "RMatrixData",
    "SU2",
    "SU2Provider",
    "SU2Sector",
    "Sector",
    "StructureChangingError",
    "Trivial",
    "TrivialProvider",
    "TrivialSector",
    "TwistData",
    "U1",
    "U1Provider",
    "U1Sector",
    "Z2",
    "Z2Provider",
    "Z2Sector",
    "bend_braided",
    "bend_unique",
    "fZ2",
    "permute_braided_tree",
    "permute_unique_tree",
    "requires",
    "supports",
]
