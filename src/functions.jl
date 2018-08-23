# Functions


TOL = 1e-05

function solve_lp(c, A, b)

    n_var = length(c)
    n_constr = length(b)

    m = Model(solver = GurobiSolver(OutputFlag=0))
    @variable(m, x[1:n_var])
    @constraint(m, lin_constr[j=1:n_constr], sum(A[j, i] * x[i] for i = 1:n_var) <= b[j])
    @objective(m, Min, sum(c[i] * x[i] for i = 1:n_var))

    status = solve(m)

    if status != :Optimal
        error("LP not solved to optimality. Status $(status)")
    end

    return getvalue(x), -getdual(lin_constr)

end

function get_basis(c, A, b)

    x, y = solve_lp(c, A, b)


    # Extract basis
    basis = find(y .>= TOL)
    #  n_basis = length(basis)

    return basis
end

function solve_with_basis(c, A, b, basis)
    n_var = length(c)
    n_constr = length(b)
    n_basis = length(basis)

    # Solve using Basis
    A_red = A[basis, :]
    b_red = b[basis]

    x = A_red \ b_red
    y = zeros(n_constr)
    y[basis] = A_red' \ (-c)

    return x, y

end



















# Define problem in JuMP
function solve_supply_chain(x0, w, T)
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
    @variable(m, u[i=1:T])
    @variable(m, y[i=1:T])  # Auxiliary: y[t] = max{h * x[t], -p * x[t]}
    @variable(m, v[i=1:T], Bin)

    # Constraints
    @constraint(m, [i=1:length(x0)], x[i] == x0[i])
    @constraint(m, evolution[t=1:T], x[t + 1] == x[t] + u[t] - w[t])
    @constraint(m, yh[t=1:T], y[t] >= h * x[t])
    @constraint(m, yp[t=1:T], y[t] >= -p * x[t])
    @constraint(m, [t=1:T], u[t] >= 0)
    @constraint(m, [t=1:T], u[t] <= M * v[t])

    # Cost
    @objective(m, Min, sum(y[i] + K * v[i] + c * u[i] for i in 1:T))

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
