# Functions





# Define problem in JuMP
function solve_supply_chain()
    srand(1)  # reset rng
    T = 10
    M = 10  # Maximum ordering capacity
    K = 10
    c = 3
    w = M * rand(T)  # Disturbance
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
    t_vec = 0:1:T-1
    p1 = plot(t_vec, getvalue(x)[1:T], line=:steppre, lab="x")
    p2 = plot(t_vec, getvalue(u), line=:steppre, lab="u")
    p3 = plot(t_vec, getvalue(v), line=:steppre, lab="v")
    p4 = plot(t_vec, w, line=:steppre, lab="w")
    plot(p1, p2, p3, p4, layout = (4,1))


end


# Solve it and get solution




# Sample States/Parameters

# Solve problem for each one of them

# Learn cost to go

# Solve 1-stage problem
