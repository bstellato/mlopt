# Functions


TOL = 1e-05

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

function solve_lp(c::Vector{Float64},
                  l::Vector{Float64},
                  A::SparseMatrixCSC,
                  u::Vector{Float64})

    @assert length(l) == length(u)
    @assert size(A, 1) == length(l)
    @assert size(A, 2) == length(c)

    n_var = length(c)
    n_constr = length(l)

    m = Model(solver = MosekSolver(QUIET=1))
    @variable(m, x[1:n_var])
    @constraint(m, upper_constr[j=1:n_constr], sum(A[j, i] * x[i] for i = 1:n_var) <= u[j])
    @constraint(m, lower_constr[j=1:n_constr], sum(A[j, i] * x[i] for i = 1:n_var) >= l[j])
    @objective(m, Min, sum(c[i] * x[i] for i = 1:n_var))

    status = solve(m)

    if status != :Optimal
        error("LP not solved to optimality. Status $(status)")
    end

    return getvalue(x), -getdual(upper_constr) - getdual(lower_constr)

end

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
    active_constr_upper = find(active_constr .== 1)
    active_constr_lower = find(active_constr .== -1)
    A_upper = @view A[active_constr_upper, :]
    u_upper = @view u[active_constr_upper]
    A_lower = @view A[active_constr_lower, :]
    l_lower = @view l[active_constr_lower]
    A_red = [A_lower; A_upper]
    #  A_red = A_c[active_constr, :]
    #  b_red = b_c[active_constr]

    # Find x
    #  x = A_red \ b_red
    x = A_red \ [l_lower; u_upper]

    # Finx y
    y = zeros(n_constr)
    #  y[active_constr] = A_red' \ (-c)
    y_temp = A_red' \ (-c)
    y[active_constr_lower] = y_temp[1:length(active_constr_lower)]
    y[active_constr_upper] = y_temp[length(active_constr_lower) + 1:end]

    return x, y

end

function gen_supply_chain_model(x0::Array{Float64},
                                w::Array{Float64},
                                T::Int64,
                                bin_vars::Bool=false)
    M = 10  # Maximum ordering capacity
    K = 10
    c = 3
    h = 10      # Storage cost
    p = 30      # Shortage cost

    # Define JuMP model
    m = Model(solver = GurobiSolver())

    # Variables
    @variable(m, x[i=1:T+1])
    (bin_vars) && (@variable(m, u[i=1:T]))
    @variable(m, y[i=1:T])  # Auxiliary: y[t] = max{h * x[t], -p * x[t]}
    @variable(m, v[i=1:T], Bin)

    # Constraints
    @constraint(m, [i=1:length(x0)], x[i] == x0[i])
    @constraint(m, yh[t=1:T], y[t] >= h * x[t])
    @constraint(m, yp[t=1:T], y[t] >= -p * x[t])
    if bin_vars
        @constraint(m, evolution[t=1:T], x[t + 1] == x[t] + u[t] - w[t])
        @constraint(m, [t=1:T], u[t] >= 0)
        @constraint(m, [t=1:T], u[t] <= M * v[t])
    else
        @constraint(m, evolution[t=1:T], x[t + 1] == x[t] - w[t])
    end

    # Cost
    if bin_vars
        @objective(m, Min, sum(y[i] + K * v[i] + c * u[i] for i in 1:T))
    else
        @objective(m, Min, sum(y[i] + K * v[i] for i in 1:T))
    end

    # Solve problem
    JuMP.build(m)

    # Extract problem data
    m_in = m.internalModel

    # Get c, A, b, lb, ub
    c = getobj(m_in)
    A = [getconstrmatrix(m_in); eye(n_var)]
    l = [getconstrLB(m_in); getvarLB(m_in)]
    u = [getconstrUB(m_in); getvarUB(m_in)]

    return c, l, A, u

end




















# Define problem in JuMP
function solve_supply_chain(x0::Array{Float64},
                            w::Array{Float64},
                            T::Int64,
                            bin_vars::Bool=false)
    srand(1)  # reset rng
    M = 10  # Maximum ordering capacity
    K = 10
    c = 3
    h = 10      # Storage cost
    p = 30      # Shortage cost

    # Define JuMP model
    m = Model(solver = GurobiSolver())

    # Variables
    @variable(m, x[i=1:T+1])
    (bin_vars) && (@variable(m, u[i=1:T]))
    @variable(m, y[i=1:T])  # Auxiliary: y[t] = max{h * x[t], -p * x[t]}
    @variable(m, v[i=1:T], Bin)

    # Constraints
    @constraint(m, [i=1:length(x0)], x[i] == x0[i])
    @constraint(m, yh[t=1:T], y[t] >= h * x[t])
    @constraint(m, yp[t=1:T], y[t] >= -p * x[t])
    if bin_vars
        @constraint(m, evolution[t=1:T], x[t + 1] == x[t] + u[t] - w[t])
        @constraint(m, [t=1:T], u[t] >= 0)
        @constraint(m, [t=1:T], u[t] <= M * v[t])
    else
        @constraint(m, evolution[t=1:T], x[t + 1] == x[t] - w[t])
    end

    # Cost
    if bin_vars
        @objective(m, Min, sum(y[i] + K * v[i] + c * u[i] for i in 1:T))
    else
        @objective(m, Min, sum(y[i] + K * v[i] for i in 1:T))
    end

    # Solve problem
    solve(m)

    # Plot behavior x, u, v and w
    #  t_vec = 0:1:T-1
    #  p1 = plot(t_vec, getvalue(x)[1:T], line=:steppost, lab="x")
    #  p2 = plot(t_vec, getvalue(u), line=:steppost, lab="u")
    #  p3 = plot(t_vec, getvalue(v), line=:steppost, lab="v")
    #  p4 = plot(t_vec, w, line=:steppost, lab="w")
    #  plot(p1, p2, p3, p4, layout = (4,1))

    return getobjectivevalue(m)

end


function estimate_cost(w, T)
    # Sample state
    N = 100
    X = randn(N, 1)
    y = Array{Float64}(N)

    # For each state solve optimization problem
    for i = 1:N
        y[i] = solve_supply_chain(X[i, :], w[2:end], T-1)
    end

    # fit tree
    lnr = OptimalTrees.OptimalTreeRegressor()
    lnr = OptimalTrees.OptimalTreeRegressor(max_depth=10)
    OptimalTrees.fit!(lnr, X, y)

    # return tree
    lnr

end


# Solve it and get solution




# Sample States/Parameters

# Solve problem for each one of them

# Learn cost to go

# Solve 1-stage problem
