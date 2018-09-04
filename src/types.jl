
abstract type OptimizationProblem end

"""
Data of the optimization problem in LP form
"""
mutable struct ProblemData
    c::Vector{Float64}
    l::Vector{Float64}
    A::SparseMatrixCSC
    u::Vector{Float64}
end
