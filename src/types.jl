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
    int_vars::Vector{Int64}
    active_constraints::Vector{Int64}
end
function Strategy(active_constraints::Vector{Int64})
    Strategy(active_constraints, Int64[])
end


function Base.unique(strategy::Vector{Strategy})
    n_int_var = length(strategy[1].int_var)

    # Construct vector of vectors
    strategy_vecs = [[s.int_vars; s.active_constraints] for s in strategy]

    # Get unique vectors
    return [Strategy(s[1:n_int_var], s[n_int_var+1:end]) for s in unique(strategy_vecs)]
end

.==(s1::Strategy, s2::Strategy) = (s1.int_vars == s2.int_vars) & (s1.active_constraints == s2.active_constraints)









