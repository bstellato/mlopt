# Solving and active constraints identification
function Strategy(theta::DataFrame, problem::OptimizationProblem)
    #  _, _, _, active_constr = solve(theta, problem)
    _, _, strategy = solve(theta, problem)
    return strategy
end


function solve(theta::DataFrame, problem::OptimizationProblem)
    N = size(theta, 1)

    # Get active_constr for each point
    x = Vector{Vector{Float64}}(N)
    #  y = Vector{Vector{Float64}}(N)
    time = Vector{Float64}(N)
    strategy = Vector{Strategy}(N)
    @showprogress 1 "Solving problem for each theta..." for i = 1:N
        populate!(problem, theta[i, :])
        #  x[i], y[i], time[i], active_constr[i] = solve(problem)
        x[i], time[i], strategy[i] = solve(problem)
    end

    #  return x, y, time, active_constr
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
    @printf "Found %d unique active constraints\n" n_strategy

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
    setvartype!(m, var_types)
    MathProgBase.optimize!(m)
    status = MathProgBase.status(m)

    if (status != :Optimal) & (status != :Stall)
        error("LP not solved to optimality. Status $(status)")
    end

    # Get strategy
    # Get integer variables
    x_int = MathProgBase.getsolution(m)[idx_int]
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


    #  return MathProgBase.getsolution(m), -MathProgBase.getconstrduals(m), MathProgBase.getsolvetime(m), basis
    return MathProgBase.getsolution(m), MathProgBase.getsolvetime(m), Strategy(int_vars,
                                                                               basis)

end

#  """
#      active_constr(problem)
#
#  Solve optimization problem and get vector of active constraints where each element is:
#
#    - -1: if the lower bound is active
#    - +1: if the upper bound is active
#    -  0: if the constraint is inactive
#  """
#  function active_constraints(problem::OptimizationProblem)
#
#      x, y, time, active_constr = solve(problem)
#
#
#      #  n_constr = length(problem.data.l)
#      #  active_constr = zeros(Int64, n_constr)
#      #  for i = 1:n_constr
#      #      if y[i] >= TOL
#      #          active_constr[i] = 1
#      #      elseif y[i] <= -TOL
#      #          active_constr[i] = -1
#      #      end
#      #  end
#
#      # Active constr
#      return x, y, time, active_constr
#
#  end

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
    int_vars, active_constr = strategy.int_vars, strategy.active_constr

    @assert length(l) == length(u)
    @assert size(A, 1) == length(l)
    @assert size(A, 2) == length(c)
    @assert maximum(int_idx) <= length(l)   # Check if vector of integers is within length
    @assert minimum(int_idx) >= 1
    @assert !any(l .> u)
    n_var = length(c)
    n_constr = length(u)

    # 1) Fix integer variables
    A_red = speye(n_var)[int_idx, :]
    b_red = int_vars

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
        error("LP not solved to optimality. Status $(status)")
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

