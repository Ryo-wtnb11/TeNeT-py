# The TensorKit.jl arm of benchmarks/bench_su2_libraries.py -- see that file for the
# design. Driven from Python, it emits the same JSON schema onto the same JSONL, so the
# report table does not care which runtime produced a row.
#
#   julia -t 1 --project=benchmarks/su2_libraries_jl benchmarks/bench_su2_libraries.jl \
#       '<spec json>' <out.jsonl> <budget seconds> [dump dir]
#
# `spec` is a list of {"case", "degeneracies"} objects; `dump dir`, when given, also
# writes A/B/C as raw column-major Float64 for the Python cross-library verification.

using TensorKit
using LinearAlgebra
using JSON

BLAS.set_num_threads(1)

# a's legs then b's, as (n_out, n_mid): a is V^n_out <- V^n_mid, b is V^n_mid <- V^n_out.
const CASES = Dict("rank3" => (2, 1), "rank5" => (3, 2))

"Degeneracies come over as {\"2j\": degeneracy}; TensorKit labels an irrep by j itself."
function su2space(degs::Dict)
    pairs = [parse(Int, k) // 2 => v for (k, v) in degs]
    sort!(pairs; by = first)
    return SU2Space(pairs...)
end

pow(V, n) = reduce(⊗, fill(V, n))

# Same ceilings as the Python driver, and the same reason -- see its module docstring.
const MAX_OPERAND_ELEMENTS = 4.0e7
const MAX_DENSE_ELEMENTS = 2.0e7
const MAX_DENSE_FLOPS = 4.0e10
const CALIBRATION_N = 512

"(warmup_ms, steady_ms, reps): the first call alone, then the median of the rest."
function measure(f; budget = 0.5, min_reps = 5, max_reps = 200)
    warmup = @elapsed f()
    walls = Float64[]
    t0 = time()
    while length(walls) < min_reps || (time() - t0 < budget && length(walls) < max_reps)
        push!(walls, @elapsed f())
    end
    return warmup * 1e3, 1e3 * sort(walls)[cld(length(walls), 2)], length(walls)
end

"FLOP/s of one square f64 gemm, this runtime's BLAS, one thread."
const GEMM_RATE = Ref(0.0)
function gemm_rate(budget)
    if GEMM_RATE[] == 0.0
        x = randn(CALIBRATION_N, CALIBRATION_N)
        _, ms, _ = measure(() -> x * x; budget = budget)
        GEMM_RATE[] = 2.0 * CALIBRATION_N^3 / (ms * 1e-3)
    end
    return GEMM_RATE[]
end

function run_point(case, degs, budget, dumpdir)
    n_out, n_mid = CASES[case]
    V = su2space(degs)
    a = randn(Float64, pow(V, n_out) ← pow(V, n_mid))
    b = randn(Float64, pow(V, n_mid) ← pow(V, n_out))

    # FLOPs off the block structure: one (M_c, K_c) * (K_c, N_c) gemm per coupled sector
    # the two operands share. Sectors only the *result* space admits are zero blocks and
    # do no arithmetic -- TensorKit still allocates them, which is a real difference from
    # tenet and frostspin and shows up in the wall, not in this column.
    flops = 0.0
    shapes = []
    for c in intersect(blocksectors(a), blocksectors(b))
        m, k = size(block(a, c))
        _, n = size(block(b, c))
        flops += 2.0 * m * k * n
        push!(shapes, (string(c), m, k, n))
    end

    d = dim(V)
    M, K, N = d^n_out, d^n_mid, d^n_out
    do_dense =
        float(M) * K <= MAX_OPERAND_ELEMENTS &&
        float(M) * N <= MAX_DENSE_ELEMENTS &&
        2.0 * M * K * N <= MAX_DENSE_FLOPS

    dwarm, dms = nothing, nothing
    if do_dense
        A = reshape(convert(Array, a), M, K)
        B = reshape(convert(Array, b), K, N)
        dwarm, dms, _ = measure(() -> A * B; budget = budget)
    end
    wm, ms, reps = measure(() -> a * b; budget = budget)

    # The honest floor: the same per-sector gemm list with no library around it. See the
    # Python driver's `blocks_only` for why this and not the peak-gemm ratio.
    pairs = [(randn(s[2], s[3]), randn(s[3], s[4])) for s in shapes]
    _, bms, _ = measure(() -> [x * y for (x, y) in pairs]; budget = budget)

    if dumpdir !== nothing
        C = convert(Array, a * b)
        tag = "$(case)_" * join(["$(k)-$(v)" for (k, v) in sort(collect(degs); by = first)], "_")
        for (nm, arr) in (("A", convert(Array, a)), ("B", convert(Array, b)), ("C", C))
            write(joinpath(dumpdir, "tk_$(tag)_$(nm).bin"), arr)
        end
        write(joinpath(dumpdir, "tk_$(tag)_shapes.json"),
              JSON.json(Dict("A" => collect(size(convert(Array, a))),
                             "B" => collect(size(convert(Array, b))),
                             "C" => collect(size(C)))))
    end

    return Dict(
        "arm" => "tensorkit",
        "case" => case,
        "degeneracies" => degs,
        "n_irreps" => length(degs),
        "dense_dim" => d,
        "flops" => flops,
        "n_gemm" => length(shapes),
        "dense_flops" => 2.0 * M * K * N,
        "dense_mkn" => [M, K, N],
        "block_shapes" => shapes,
        "gemm_rate_gflops" => gemm_rate(budget) / 1e9,
        "dense_warmup_ms" => dwarm,
        "dense_ms" => dms,
        "warmup_ms" => wm,
        "steady_ms" => ms,
        "reps" => reps,
        "ratio_to_peak" => ms * 1e-3 * gemm_rate(budget) / flops,
        "blocks_ms" => bms,
        "ratio_to_blocks" => ms / bms,
        "speedup_vs_dense" => do_dense ? dms / ms : nothing,
        "julia_version" => string(VERSION),
        "tensorkit_version" => string(pkgversion(TensorKit)),
        "blas" => string(BLAS.get_config().loaded_libs[1].libname),
    )
end

function main()
    spec = JSON.parse(ARGS[1])
    out = ARGS[2]
    budget = parse(Float64, ARGS[3])
    dumpdir = length(ARGS) >= 4 ? ARGS[4] : nothing

    # resumable, the same rule the Python driver uses
    done = Set()
    if isfile(out)
        for line in eachline(out)
            isempty(strip(line)) && continue
            r = JSON.parse(line)
            push!(done, (r["arm"], r["case"], sort(collect(r["degeneracies"]))))
        end
    end

    for pt in spec
        case = pt["case"]
        degs = Dict{String,Int}(k => v for (k, v) in pt["degeneracies"])
        ("tensorkit", case, sort(collect(degs))) in done && continue
        @info "tensorkit $case $degs"
        row = run_point(case, degs, budget, dumpdir)
        open(out, "a") do io
            println(io, JSON.json(row))
        end
    end
end

main()
