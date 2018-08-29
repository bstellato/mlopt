mutable struct OptimizationProblem
    c::Vector{Float64}
    l::Vector{Float64}
    A::SparseMatrixCSC
    u::Vector{Float64}
end


