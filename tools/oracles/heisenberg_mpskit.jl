# MPSKit.jl oracle for the open-boundary spin-1/2 Heisenberg chain (#152).
# Regenerate the fixture with:
#   julia --project=tools/oracles tools/oracles/heisenberg_mpskit.jl > tests/fixtures/mpskit_heisenberg.json
# Convention matches tenet's exactly: H = Sum_i S_i . S_{i+1}, J = 1, open boundaries.
using MPSKit, MPSKitModels, TensorKit, SHA, Dates
import Pkg

ver(name) = string(only(d for d in values(Pkg.dependencies()) if d.name == name).version)
json(x::Integer) = string(x)
json(x::Real) = string(Float64(x))
json(x::AbstractString) = "\"$x\""
json(x::Vector) = "[" * join(json.(x), ",") * "]"
json(x::Vector{<:Pair}) = "{" * join(("\"$k\":$(json(v))" for (k, v) in x), ",") * "}"

function ground(N, chi)  # SU(2)-symmetric: DMRG2 grows the bonds, single-site polishes
    H = heisenberg_XXX(ComplexF64, SU2Irrep, FiniteChain(N); spin=1 // 2, J=1.0)
    psi = FiniteMPS(rand, ComplexF64, N, Rep[SU₂](1 // 2 => 1), Rep[SU₂](0 => 4, 1 // 2 => 4, 1 => 2))
    psi, envs, _ = find_groundstate(psi, H, DMRG2(; trscheme=truncrank(chi), tol=1e-12, verbosity=0))
    psi, envs, _ = find_groundstate(psi, H, DMRG(; tol=1e-13, verbosity=0))
    real(expectation_value(psi, H)), psi
end

e12t, _ = let H = heisenberg_XXX(ComplexF64, Trivial, FiniteChain(12); spin=1 // 2, J=1.0)
    psi = FiniteMPS(rand, ComplexF64, 12, ComplexSpace(2), ComplexSpace(64))
    psi, _, _ = find_groundstate(psi, H, DMRG(; tol=1e-12, verbosity=0))
    real(expectation_value(psi, H)), psi
end
e12, _ = ground(12, 128)
e20, _ = ground(20, 128)
runs = Dict((N, chi) => ground(N, chi) for N in (32, 64), chi in (64, 128, 256))
psi32 = runs[(32, 256)][2]
bonds = [real(expectation_value(psi32, (i, i + 1) => S_exchange(ComplexF64, SU2Irrep; spin=1 // 2))) for i in 1:31]
spectrum = ["$(c.j)" => collect(b) for (c, b) in pairs(entanglement_spectrum(psi32, 16))]

entry(N, chi; unc) = ["chi" => chi, "energy" => runs[(N, chi)][1], "energy_uncertainty" => unc]
u(N, chi) = abs(runs[(N, chi)][1] - runs[(N, 256)][1])  # chi-ladder step, under-claiming
print(json([
    "provenance" => ["mpskit" => ver("MPSKit"), "mpskitmodels" => ver("MPSKitModels"),
        "tensorkit" => ver("TensorKit"), "julia" => string(VERSION),
        "generator" => "tools/oracles/heisenberg_mpskit.jl",
        "generated_utc" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "script_sha256" => bytes2hex(sha256(read(@__FILE__)))],
    "validation" => [  # cross-checks against tests/integration/test_dmrg.py's ED literals
        "N12_trivial" => ["chi" => 64, "energy" => e12t, "energy_uncertainty" => 1e-13],
        "N12_su2" => ["chi" => 128, "energy" => e12, "energy_uncertainty" => 1e-12],
        "N20_su2" => ["chi" => 128, "energy" => e20, "energy_uncertainty" => 1e-12]],
    "N32" => ["chi64" => entry(32, 64; unc=u(32, 64)), "chi128" => entry(32, 128; unc=u(32, 128)),
        "chi256" => entry(32, 256; unc=1e-12)],
    # N64 is asserted by no test; its consumer is #148's examples lane (heisenberg.py's report).
    "N64" => ["chi64" => entry(64, 64; unc=u(64, 64)), "chi128" => entry(64, 128; unc=u(64, 128)),
        "chi256" => entry(64, 256; unc=1e-10)],
    "N32_bond_energies" => bonds,  # <S_i . S_{i+1}>, i = 1..31, from the chi=256 state
    "N32_entanglement_spectrum_bond16" => spectrum,  # Schmidt values (not squared) per SU(2) spin s
]))
