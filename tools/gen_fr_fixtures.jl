#!/usr/bin/env julia
# Generate SU(2) F-symbol and R-symbol oracle fixtures from TensorKitSectors
# (the reference for the racah crate's F/R conventions), over all admissible
# doubled spins <= 12.
#
# F file line:  dj1 dj2 dj3 dj4 dj5 dj6 fval      (fval = Fsymbol, Float64)
# R file line:  dj1 dj2 dj3 rval                  (rval = Rsymbol, Float64)
# Floats are printed with %.17g so they round-trip exactly to the same f64.
#
# Usage (env must have TensorKit, which re-exports the SU2Irrep sector data):
#   julia --project=@v1.11 tools/gen_fr_fixtures.jl \
#       tests/fixtures/su2_f.txt tests/fixtures/su2_r.txt
#
# TensorKitSectors is the F/R oracle. It is loaded transitively via TensorKit;
# its F/R definitions (src/irreps/su2irrep.jl) are functionally identical from
# 0.3.4 (the 0.3.4 file differs only by a scalartype refactor -- same numeric
# F/R values) and byte-identical across 0.3.6-0.3.8, the version the racah
# conventions cite, so the version the local depot resolves is faithful. The
# provenance header records the resolved versions.

using TensorKit
using Printf
using Dates
using SHA
using Pkg

const SU2 = SU2Irrep
sd(dj) = SU2(dj // 2)
tri(a, b, c) = iseven(a + b + c) && c >= abs(a - b) && c <= a + b
# The four triangles of the 6j actually evaluated by Fsymbol, {j1 j2 j5 / j3 j4 j6}.
f_admissible(d1, d2, d3, d4, d5, d6) =
    tri(d1, d2, d5) && tri(d1, d4, d6) && tri(d3, d2, d6) && tri(d3, d4, d5)

function depversion(name)
    for (_, v) in Pkg.dependencies()
        v.name == name && return string(v.version)
    end
    return "unknown"
end

function header(io, kind, cols)
    script = @__FILE__
    h = bytes2hex(sha256(read(script)))
    println(io, "# racah SU(2) ", kind, " oracle fixtures")
    println(io, "# oracle: TensorKitSectors (via TensorKit)")
    println(io, "# generator: tools/gen_fr_fixtures.jl")
    println(io, "# TensorKitSectors version: ", depversion("TensorKitSectors"))
    println(io, "# TensorKit version: ", depversion("TensorKit"))
    println(io, "# julia version: ", VERSION)
    println(io, "# generated (UTC): ", Dates.now(Dates.UTC))
    println(io, "# script sha256: ", h)
    println(io, "# columns: ", cols)
end

function write_f(path)
    open(path, "w") do io
        header(io, "F-symbol", "dj1 dj2 dj3 dj4 dj5 dj6 fval   (fval = Fsymbol)")
        for d1 in 0:12, d2 in 0:12, d3 in 0:12, d4 in 0:12, d5 in 0:12, d6 in 0:12
            f_admissible(d1, d2, d3, d4, d5, d6) || continue
            fval = Fsymbol(sd(d1), sd(d2), sd(d3), sd(d4), sd(d5), sd(d6))
            @printf(io, "%d %d %d %d %d %d %.17g\n", d1, d2, d3, d4, d5, d6, fval)
        end
    end
end

function write_r(path)
    open(path, "w") do io
        header(io, "R-symbol", "dj1 dj2 dj3 rval   (rval = Rsymbol)")
        for d1 in 0:12, d2 in 0:12, d3 in 0:12
            tri(d1, d2, d3) || continue
            rval = Rsymbol(sd(d1), sd(d2), sd(d3))
            @printf(io, "%d %d %d %.17g\n", d1, d2, d3, rval)
        end
    end
end

function main()
    length(ARGS) == 2 || error("usage: gen_fr_fixtures.jl <f_out> <r_out>")
    write_f(ARGS[1])
    write_r(ARGS[2])
end

main()
