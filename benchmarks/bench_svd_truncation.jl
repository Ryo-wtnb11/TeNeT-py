# The TensorKit.jl arm of benchmarks/bench_svd_truncation.py -- see that file for the
# question. Driven from Python: it reads the dense arrays the Python driver dumped,
# rebuilds them as TensorMaps over the same spaces, truncates with `truncrank(D)`, and
# writes back what was kept, at what error, and the dense reconstruction.
#
#   julia --project=benchmarks/su2_libraries_jl benchmarks/bench_svd_truncation.jl <dir>
#
# `<dir>/spec.json` is the case list; results land in `<dir>/tk_results.json` and the
# reconstructions in `<dir>/tk_<case>_D<D>.bin` (raw column-major Float64).

using TensorKit
using LinearAlgebra
using JSON
# MatrixAlgebraKit is TensorKit's dependency, not this project's, so it is reached
# through a type it owns rather than added to the Project.toml.
const MAK = parentmodule(typeof(truncrank(1)))

# The Python driver writes with `ndarray.tofile`, which is always C order.
function readarray(path, shape)
    a = Array{Float64}(undef, reverse(shape)...)
    read!(path, a)
    return permutedims(a, ndims(a):-1:1)
end

writearray(path, a) = write(path, Array(a))

"Degeneracies come over as {\"label\": degeneracy}: 2j for SU(2), the charge for U(1)."
function makespace(sym, degs)
    prs = sort([parse(Int, k) => v for (k, v) in degs]; by = first)
    return sym == "su2" ? SU2Space((k // 2 => v for (k, v) in prs)...) :
        U1Space(prs...)
end

label(c::SU2Irrep) = string(Int(2 * c.j))
label(c::U1Irrep) = string(Int(c.charge))

pow(V, n) = reduce(⊗, fill(V, n))

"{label => degeneracy} of a graded space, plus its dense dimension."
function kept_sectors(V)
    return Dict(label(c) => dim(V, c) for c in sectors(V))
end

function run_case(dir, spec)
    V = makespace(spec["symmetry"], spec["space"])
    n_out, n_in = spec["n_out"], spec["n_in"]
    a = readarray(joinpath(dir, spec["array"]), Int.(spec["shape"]))
    t = TensorMap(a, pow(V, n_out) ← pow(V, n_in))
    # The array round-trip is the premise of the whole comparison: if TensorKit put this
    # array in a different invariant subspace, nothing below would be comparable.
    roundtrip = maximum(abs.(convert(Array, t) .- a))

    rows = []
    for D in spec["Ds"]
        U, S, Vh = svd_trunc(t; trunc = truncrank(D))
        W = space(S, 1)
        vals = Float64[]
        for (_, b) in blocks(S)
            append!(vals, diag(b))
        end
        sort!(vals; rev = true)
        recon = U * S * Vh
        name = "tk_$(spec["name"])_D$(D).bin"
        writearray(joinpath(dir, name), convert(Array, recon))

        # What TensorKit itself calls the truncation error, from the untruncated
        # spectrum and the very index set `truncrank` selects.
        reported = nothing
        try
            _, Sf, _ = svd_compact(t)
            ind = MAK.findtruncated_svd(MAK.diagview(Sf), truncrank(D))
            reported = MAK.truncation_error(MAK.diagview(Sf), ind)
        catch err
            reported = string(err)
        end

        push!(
            rows, Dict(
                "case" => spec["name"], "D" => D,
                "kept" => kept_sectors(W),
                "dense_dim" => dim(W), "reduced_dim" => length(vals),
                "values" => vals,
                "error_recomputed" => norm(t - recon),
                "error_reported" => reported,
                "recon" => name,
            )
        )
    end
    return roundtrip, rows
end

function main()
    dir = ARGS[1]
    spec = JSON.parsefile(joinpath(dir, "spec.json"))
    out = Dict{String, Any}(
        "julia_version" => string(VERSION),
        "tensorkit_version" => string(pkgversion(TensorKit)),
    )
    rows, trips = [], Dict{String, Float64}()
    for s in spec
        rt, rs = run_case(dir, s)
        trips[s["name"]] = rt
        append!(rows, rs)
    end
    out["roundtrip"] = trips
    out["rows"] = rows
    open(joinpath(dir, "tk_results.json"), "w") do io
        JSON.print(io, out)
    end
end

main()
