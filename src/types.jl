
abstract type OptimizationProblem end

"""
Data of the optimization problem in LP form
"""
mutable struct ProblemData
    c::Vector{Float64}
    l::Vector{Float64}
    A::SparseMatrixCSC
    u::Vector{Float64}
    ProblemData() = new()
end

function ProblemData(c::Vector{Float64}, l::Vector{Float64}, A::SparseMatrixCSC, u::Vector{Float64})
    d = ProblemData()
    d.c = c
    d.l = l
    d.A = A
    d.u = u
    return d
end

function eq_ineq(data::ProblemData)
    eq =  find(abs.(data.u - data.l) .<= 1e-08)
    ineq = setdiff(1:length(data.l), eq)
    return eq, ineq
end

function Base.copy!(data_dest::ProblemData,
                      data::ProblemData)
    copy!(data_dest.c, data.c)
    copy!(data_dest.l, data.l)
    copy!(data_dest.A, data.A)
    copy!(data_dest.u, data.u)
end

function Base.copy(data::ProblemData)
    data_dest = ProblemData()
    data_dest.c = copy(data.c)
    data_dest.l = copy(data.l)
    data_dest.A = copy(data.A)
    data_dest.u = copy(data.u)
    return data_dest
end
