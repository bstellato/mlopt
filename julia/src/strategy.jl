strategies(theta::DataFrame, problem::OptimizationProblem) = solve(theta, problem)[end]

function solve(theta::DataFrame, problem::OptimizationProblem)
    N = size(theta, 1)

    # Get strategy for each point
    x = Vector{Vector{Float64}}(N)
    time = Vector{Float64}(N)
    strategy = Vector{Strategy}(N)
    @showprogress 1 "Solving problem for each theta..." for i = 1:N
        populate!(problem, theta[i, :])
        x[i], time[i], strategy[i] = solve(problem)
    end

    return x, time, strategy

end


"""
    encode(strategy)

Map vector of strategies to numbers
"""
function encode(strategy::Vector{Strategy})

    @printf "Encoding strategies\n"
    N = length(strategy)
    @printf "Getting unique set of strategies\n"
    unique_strategy = unique(strategy)
    n_strategy = length(unique_strategy)  # Number of active constr vectors
    @printf "Found %d unique strategies\n" n_strategy

    # Map strategy to number
    y = Vector{Int64}(N)
    for i = 1:N
        # Get which strategy is the current one
        y[i] = 0
        for j = 1:n_strategy
            if strategy[i] == unique_strategy[j]  # Compare vectors (Expensive?)
                y[i] = j
                break
            end
        end
        (y[i] == 0) && (error("Found no matching strategy"))
    end
    @printf "Encoding done.\n"
    return y, unique_strategy
end

"""
    solve(problem)

Solve linear program and return primal variables `x` and dual variables `y`
"""
function solve(problem::OptimizationProblem)

    c, l, A, u = problem.data.c, problem.data.l, problem.data.A, problem.data.u
    int_idx = problem.data.int_idx

    @assert length(l) == length(u)
    @assert size(A, 1) == length(l)
    @assert size(A, 2) == length(c)
    @assert !any(l .> u)

    n_var = length(c)
    n_constr = length(l)

    var_types = [if i in int_idx
                     :Int
                 else
                     :Cont
                 end for i in 1:n_var]

    # Solve directly with MathProgBase
    m = MathProgBase.LinearQuadraticModel(SOLVER)
    MathProgBase.loadproblem!(m, A, -Inf * ones(n_var), Inf * ones(n_var), c, l, u, :Min)
    MathProgBase.setvartype!(m, var_types)   # Set integer variables
    MathProgBase.optimize!(m)
    status = MathProgBase.status(m)
    ((status != :Optimal) & (status != :Stall)) && error("Problem not solved to optimality. Status $(status)")

    # Get solution, time and integer variables
    x_opt = MathProgBase.getsolution(m)
    time = MathProgBase.getsolvetime(m)
    x_int = round.(Int64, x_opt[int_idx])

    # Solve LP restriction to get basis
    A_lp = [A; speye(n_var)[int_idx, :]]
    u_lp = [u; x_int]
    l_lp = [l; x_int]

    # Solve directly with MathProgBase
    m = MathProgBase.LinearQuadraticModel(SOLVER)
    MathProgBase.loadproblem!(m, A_lp, -Inf * ones(n_var), Inf * ones(n_var),
                              c, l_lp, u_lp, :Min)
    MathProgBase.optimize!(m)
    status = MathProgBase.status(m)
    ((status != :Optimal) & (status != :Stall)) && error("Restricted continuous problem not solved to optimality. Status $(status)")

    # Get active constraints
    _, basis_constr = MathProgBase.getbasis(m)
    basis = zeros(Int, n_constr)
    for i = 1:n_constr
        if basis_constr[i] == :NonBasicAtLower
            basis[i] = -1
        elseif basis_constr[i] == :NonBasicAtUpper
            basis[i] = 1
        end
    end

    return x_opt, time, Strategy(x_int, basis)

end


"""
    solve(problem, strategy)

Solve simplified problem using a predefined strategy describing
which integer variables are fixed and which constraints are active.
"""
function solve(problem::OptimizationProblem, strategy::Strategy)

    # Unpack problem data
    c, l, A, u = problem.data.c, problem.data.l, problem.data.A, problem.data.u
    int_idx = problem.data.int_idx

    # Unpack strategy
    int_vars, active_constr = strategy.int_vars, strategy.active_constraints

    @assert length(l) == length(u)
    @assert size(A, 1) == length(l)
    @assert size(A, 2) == length(c)
    if length(int_idx) > 0
        @assert maximum(int_idx) <= length(l)   # int vector within length
        @assert minimum(int_idx) >= 1
    end
    @assert !any(l .> u)
    n_var = length(c)
    n_constr = length(u)

    # 1) Fix integer variables
    A_red = speye(n_var)[int_idx, :]
    bound_red = int_vars

    # 2) Use active constraints
    active_constr_upper = find(active_constr .== 1)
    n_upper = length(active_constr_upper)
    active_constr_lower = find(active_constr .== -1)
    n_lower = length(active_constr_lower)
    A_upper = A[active_constr_upper, :]
    u_upper = u[active_constr_upper]
    A_lower = A[active_constr_lower, :]
    l_lower = l[active_constr_lower]
    A_red = [A_red; A_lower; A_upper]
    bound_red = [bound_red; l_lower; u_upper]

    # Solve directly with MathProgBase
    m = MathProgBase.LinearQuadraticModel(SOLVER)
    #  m = MathProgBase.LinearQuadraticModel(MosekSolver())
    MathProgBase.loadproblem!(m, A_red, -Inf * ones(n_var), Inf * ones(n_var),
                              c, bound_red, bound_red, :Min)
    MathProgBase.optimize!(m)
    status = MathProgBase.status(m)

    if (status != :Optimal) & (status != :Stall)
        error("Problem not solved to optimality. Status $(status)")
    end

    x = MathProgBase.getsolution(m)
    return x, MathProgBase.getsolvetime(m)

    # Do not return dual variables in general
    #  y = zeros(n_constr)
    #  y_temp = -MathProgBase.getconstrduals(m)
    #  y[active_constr_lower] = y_temp[1:n_lower]
    #  y[active_constr_upper] = y_temp[n_lower+1:end]
    #  return x, y, MathProgBase.getsolvetime(m)

    # Solve ONLY a single linear system (does not always work because it can be non square)
    #  # Find x
    #  x = A_red \ [l_lower; u_upper]
    #
    #  # Find y
    #  y = zeros(n_constr)
    #  y_temp = (A_red') \ (-c)
    #  y[active_constr_lower] = y_temp[1:length(active_constr_lower)]
    #  y[active_constr_upper] = y_temp[length(active_constr_lower) + 1:end]
    #
    #  return x, y

end

