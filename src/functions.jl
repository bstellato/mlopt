# Functions


TOL = 1e-05

function active_constr_to_number(active_constr::Vector{Vector{Int64}})

    N = length(active_constr)
    unique_active_constr = unique(active_constr)
    n_active_constr = length(unique_active_constr)

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
                  A::SparseMatrixCSC,
                  b::Vector{Float64},
                  lb::Vector{Float64},
                  ub::Vector{Float64})

    n_var = length(c)
    n_constr = length(b)

    # Cumulative constraints taking into account
    # variable bounds
    A_c = [A; eye(n_var); -eye(n_var)]
    b_c = [b; ub; -lb]

    m = Model(solver = MosekSolver(QUIET=1))
    @variable(m, x[1:n_var])
    @constraint(m, lin_constr[j=1:n_constr + 2*n_var], sum(A_c[j, i] * x[i] for i = 1:n_var) <= b_c[j])
    @objective(m, Min, sum(c[i] * x[i] for i = 1:n_var))

    status = solve(m)

    if status != :Optimal
        error("LP not solved to optimality. Status $(status)")
    end

    return getvalue(x), -getdual(lin_constr)

end

function get_active_constr(c::Vector{Float64},
                   A::SparseMatrixCSC,
                   b::Vector{Float64},
                   lb::Vector{Float64},
                   ub::Vector{Float64})

    x, y = solve_lp(c, A, b, lb, ub)

    # Extract active_constr
    return find(abs.(y) .>= TOL)

end

function solve_with_active_constr(c::Vector{Float64},
                          A::SparseMatrixCSC,
                          b::Vector{Float64},
                          lb::Vector{Float64},
                          ub::Vector{Float64},
                          active_constr::Vector{Int64})
    n_var = length(c)
    n_constr = length(b)

    # Create cumulative vecs
    A_c = [A; eye(n_var); -eye(n_var)]
    b_c = [b; ub; -lb]
    n_c = length(b) + 2 * length(c)

    # Solve using Basis
    A_red = A_c[active_constr, :]
    b_red = b_c[active_constr]

    x = A_red \ b_red
    y = zeros(n_c)
    y[active_constr] = A_red' \ (-c)

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
    A = getconstrmatrix(m_in)
    A_ub = getconstrUB(m_in)
    A_lb = getconstrLB(m_in)
    ub = setvarUB(m_in)
    lb = getvarLB(m_in)

    return c, A_lb, A, A_ub, lb, ub

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
