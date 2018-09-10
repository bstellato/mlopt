abstract type OptimizationProblem end

"""
Evaluate cost function at x
"""
function cost(problem::OptimizationProblem,
              x::Vector{Float64})
    return problem.data.c' * x
end

"""
Is the problem mixed-integer?
"""
is_mip(p::OptimizationProblem) = length(p.data.int_idx) > 0

"""
Data of the optimization problem in LP form
"""
mutable struct ProblemData
    c::Vector{Float64}
    l::Vector{Float64}
    A::SparseMatrixCSC
    u::Vector{Float64}
    int_idx::Vector{Int64}  # Indeces of integer variables
    ProblemData() = new()
end

function ProblemData(c::Vector{Float64},
                     l::Vector{Float64}, A::SparseMatrixCSC, u::Vector{Float64};
                     int_idx::Vector{Int64}=Int64[])
    d = ProblemData()
    d.c = c
    d.l = l
    d.A = A
    d.u = u
    d.int_idx = int_idx
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
    copy!(data_dest.int_idx, data.int_idx)
end

function Base.copy(data::ProblemData)
    data_dest = ProblemData()
    data_dest.c = copy(data.c)
    data_dest.l = copy(data.l)
    data_dest.A = copy(data.A)
    data_dest.u = copy(data.u)
    data_dest.int_idx = copy(data.int_idx)
    return data_dest
end


mutable struct Strategy
    active_constraints::Vector{Vector{Int64}}
    int_vars::Vector{Vector{Int64}}
end
