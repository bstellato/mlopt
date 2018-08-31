"""
    @remove_unbounded_constraints A l u

Remove constraints where both l_i and u_i have infinite magnitude.
"""
macro remove_unbounded_constraints(A, l, u)

    return quote
        local _A = $(esc(A))
        local _l = $(esc(l))
        local _u = $(esc(u))
        local idx_constr = setdiff(1:length(_l), intersect(find(isinf.(_u)), find(isinf.(_l))))
        $(esc(A)) = _A[idx_constr, :]
        $(esc(l)) = _l[idx_constr]
        $(esc(u)) = _u[idx_constr]
    end
end

"""
    extract_problem_data(m)

Extract problem data from JuMP model
"""
function extract_problem_data(m::JuMP.Model)
    # Build internal model
    JuMP.build(m)

    # Extract data
    return extract_problem_data(m.internalModel)

end

function extract_problem_data(m::MathProgBase.AbstractLinearQuadraticModel)
    c = MathProgBase.getobj(m)
    A = [MathProgBase.getconstrmatrix(m); eye(length(c))]
    l = [MathProgBase.getconstrLB(m); MathProgBase.getvarLB(m)]
    u = [MathProgBase.getconstrUB(m); MathProgBase.getvarUB(m)]

    # Cap constraints with infinity
    [u[i] = Inf for i in 1:length(u) if u[i] >= 1e20]
    [l[i] = -Inf for i in 1:length(l) if l[i] <= -1e20]

    # Find indices where both bounds are infinity
    MyModule.@remove_unbounded_constraints A l u

    return c, l, A, u
end


function extract_problem_data(file_name::String)
    m = MathProgBase.LinearQuadraticModel(GurobiSolver())
    MathProgBase.loadproblem!(m, file_name)
    return extract_problem_data(m)
end
MyModule.OptimizationProblem(name::String) = MyModule.OptimizationProblem(MyModule.extract_problem_data(name)...)

"""
    infeasibility(x_ml, problem)

Compute infeasibility as ||(Ax - u)_{+} + (l - Ax)_{+}||_2
weighted over the magnitude of l and u.
"""
function infeasibility(x::Vector{Float64},
                       problem::OptimizationProblem)
    l, A, u = problem.l, problem.A, problem.u

    upper = max.(A * x - u, 0.) ./ (abs.(u) + 1e-10)
    lower = max.(l - A * x, 0.) ./ (abs.(l) + 1e-10)
    return norm(upper + lower)

end

"""
    suboptimality(x, x_opt, problem)

Compute suboptimality as || c' * x - c' * x_opt ||
"""
function suboptimality(x::Vector{Float64},
                       x_opt::Vector{Float64},
                       problem::OptimizationProblem)
    c = problem.c
    return (c' * x - c' * x_opt) / abs.(c' * x_opt + 1e-10)
end

