# MPSKit.jl oracle for the open-boundary spinful Hubbard chain (#147, Gate 4).
# Regenerate the fixture with:
#   julia --project=tools/oracles tools/oracles/hubbard_mpskit.jl > tests/fixtures/mpskit_hubbard.json
#
# Convention -- matched to tenet's tests/network/test_hubbard.py verbatim:
#   H = -t sum_{m,sigma} (c+_{m,sigma} c_{m+1,sigma} + h.c.) + U sum_m n_up n_dn
# MPSKitModels' `hubbard_model(...; t, U, mu)` is exactly this operator at mu = 0
# (its docstring: -t sum (e+e- + e-e+) + U sum n_up n_dn - mu sum n), so the mapping
# is the identity: no chemical-potential term, no particle-hole shift, no constant.
# The physical space is Vect[fZ2](0 => 2, 1 => 2), the same parity grading tenet
# uses, and the trivial MPS boundaries fix total parity even -- the same sector
# tenet's DMRG targets and the test's even-block ED diagonalizes.
using MPSKit, MPSKitModels, TensorKit, SHA, Dates
import Pkg

ver(name) = string(only(d for d in values(Pkg.dependencies()) if d.name == name).version)
json(x::Integer) = string(x)
json(x::Real) = string(Float64(x))
json(x::AbstractString) = "\"$x\""
json(x::Vector{<:Pair}) = "{" * join(("\"$k\":$(json(v))" for (k, v) in x), ",") * "}"

function ground(N, U, chi)
    H = hubbard_model(ComplexF64, Trivial, Trivial, FiniteChain(N); t=1.0, U=U, mu=0.0)
    pspace = Vect[FermionParity](0 => 2, 1 => 2)
    vspace = Vect[FermionParity](0 => cld(chi, 2), 1 => cld(chi, 2))
    psi = FiniteMPS(rand, ComplexF64, N, pspace, vspace)
    psi, _, _ = find_groundstate(psi, H, DMRG2(; trscheme=truncrank(chi), tol=1e-12, verbosity=0))
    psi, _, _ = find_groundstate(psi, H, DMRG(; tol=1e-13, verbosity=0))
    real(expectation_value(psi, H))
end

US = (0.0, 2.0, 4.0, 8.0)
n4 = Dict(U => ground(4, U, 16) for U in US)            # chi=16 is exact at N=4
n6 = ground(6, 4.0, 64)                                 # chi=64 is exact at N=6
n8 = Dict((U, chi) => ground(8, U, chi) for U in US, chi in (128, 256))

entry(e; chi, unc) = ["chi" => chi, "energy" => e, "energy_uncertainty" => unc]
print(json([
    "provenance" => ["mpskit" => ver("MPSKit"), "mpskitmodels" => ver("MPSKitModels"),
        "tensorkit" => ver("TensorKit"), "julia" => string(VERSION),
        "generator" => "tools/oracles/hubbard_mpskit.jl",
        "generated_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "convention" => "H = -t sum_(m,s)(c+_ms c_m+1,s + h.c.) + U sum_m n_up n_dn, " *
            "open boundaries, t = 1; hubbard_model(; t=1, U, mu=0) is this operator " *
            "verbatim (identity mapping, no shift); even total parity sector"],
    # N=4 is exact at chi=16; tests/network/test_hubbard.py re-derives the 256-dim
    # even-parity ED in Python and asserts these against it -- the dual oracle.
    "N4" => ["U$(Int(U))" => entry(n4[U]; chi=16, unc=1e-12) for U in US],
    "N6" => ["U4" => entry(n6; chi=64, unc=1e-12)],
    "N8" => ["U$(Int(U))" => entry(n8[(U, 256)]; chi=256,
        unc=max(1e-10, abs(n8[(U, 256)] - n8[(U, 128)]))) for U in US],
]))
