populate!(::OptimizationProblem) = error("custom OptimizationProblem objects must define a `populate!` method")

"""
    @remove_unbounded_constraints A l u

Remove constraints where both l_i and u_i have infinite magnitude.
"""
macro remove_unbounded_constraints(A, l, u)

    return quote
        local _A = $(esc(A))
        local _l = $(esc(l))
        local _u = $(esc(u))
        local idx_constr = setdiff(1:length(_l), intersect(find(Base.isinf.(_u)), find(Base.isinf.(_l))))
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

    # Remove indices where both bounds are infinity
    MyModule.@remove_unbounded_constraints A l u

    return ProblemData(c, l, A, u)
end


function extract_problem_data(file_name::String)
    m = MathProgBase.LinearQuadraticModel(READ_SOLVER)
    MathProgBase.loadproblem!(m, file_name)
    return extract_problem_data(m)
end

