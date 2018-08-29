# Solving and active constraints identification

"""
    active_constr_to_number(active_constr::Vector{Vector{Int64}})

Map vector of active constraints vectors to numbers
"""
function active_constr_to_number(active_constr::Vector{Vector{Int64}})

    N = length(active_constr)
    unique_active_constr = unique(active_constr)
    n_active_constr = length(unique_active_constr)  # Number of active constr vectors

    # Map active_constr to number
    y = Vector{Int64}(N)
    for i = 1:N
        # Get which active_constr is the current one
        y[i] = 0
        for j = 1:n_active_constr
            if active_constr[i] == unique_active_constr[j]  # Compare vectors. (Expensive?)
                y[i] = j
                break
            end
        end
        (y[i] == 0) && (error("Found no matching active_constr"))
    end

    return y, unique_active_constr
end

"""
    solve_lp(c, l, A, u)

Solve linear program and return primal variables `x` and dual variables `y`
"""
function solve_lp(c::Vector{Float64},
                  l::Vector{Float64},
                  A::SparseMatrixCSC,
                  u::Vector{Float64})

    @assert length(l) == length(u)
    @assert size(A, 1) == length(l)
    @assert size(A, 2) == length(c)

    n_var = length(c)
    n_constr = length(l)

    # Solve directly with MathProgBase
    m = MathProgBase.LinearQuadraticModel(SOLVER)
    MathProgBase.loadproblem!(m, A, -Inf * ones(n_var), Inf * ones(n_var), c, l, u, :Min)
    MathProgBase.optimize!(m)
    status = MathProgBase.status(m)


    if status != :Optimal
        error("LP not solved to optimality. Status $(status)")
    end

    return MathProgBase.getsolution(m), -MathProgBase.getconstrduals(m)

end

"""
    get_active_constr(c, l, A, u)

Solve linear program and get vector of active constraints where each element is:

  - -1: if the lower bound is active
  - +1: if the upper bound is active
  -  0: if the constraint is inactive
"""
function get_active_constr(c::Vector{Float64},
                           l::Vector{Float64},
                           A::SparseMatrixCSC,
                           u::Vector{Float64})

    @assert length(l) == length(u)
    @assert size(A, 1) == length(l)
    @assert size(A, 2) == length(c)
    n_constr = length(l)

    _, y = solve_lp(c, l, A, u)

    active_constr = zeros(Int64, n_constr)
    for i = 1:n_constr
        if y[i] >= TOL
            active_constr[i] = 1
        elseif y[i] <= -TOL
            active_constr[i] = -1
        end
    end

    # Active constr
    return active_constr

end

"""
    solve_with_active_constr(c, l, A, u, active_constr)

Solve simplified problem using the `active_constr` vector
specifying which constraints are active according to function
`get_active_constr`.
"""
function solve_with_active_constr(c::Vector{Float64},
                                  l::Vector{Float64},
                                  A::SparseMatrixCSC,
                                  u::Vector{Float64},
                                  active_constr::Vector{Int64})
    @assert length(l) == length(u)
    @assert size(A, 1) == length(l)
    @assert size(A, 2) == length(c)
    n_var = length(c)
    n_constr = length(u)

    # Solve using Basis
    n_active = length(active_constr)
    active_constr_upper = find(active_constr .== 1)
    n_upper = length(active_constr_upper)
    active_constr_lower = find(active_constr .== -1)
    n_lower = length(active_constr_lower)
    A_upper = A[active_constr_upper, :]
    u_upper = u[active_constr_upper]
    A_lower = A[active_constr_lower, :]
    l_lower = l[active_constr_lower]
    A_red = [A_lower; A_upper]
    bound_red = [l_lower; u_upper]

    # Solve directly with MathProgBase
    m = MathProgBase.LinearQuadraticModel(SOLVER)
    MathProgBase.loadproblem!(m, A_red, -Inf * ones(n_var), Inf * ones(n_var),
                              c, bound_red, bound_red, :Min)
    MathProgBase.optimize!(m)
    status = MathProgBase.status(m)


    if status != :Optimal
        error("LP not solved to optimality. Status $(status)")
    end

    #  x = getvalue(x)
    x = MathProgBase.getsolution(m)
    y = zeros(n_constr)
    y_temp = -MathProgBase.getconstrduals(m)
    y[active_constr_lower] = y_temp[1:n_lower]
    y[active_constr_upper] = y_temp[n_lower+1:end]
    return x, y

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

