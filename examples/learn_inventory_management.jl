using JuMP
include("../src/MyModule.jl")

function InventoryManagementProblem(x0::Array{Float64};
                                    T = 50,   # Horizon
                                    M = 10.,  # Max ordering capacity
                                    K = 10.,  # Fixed ordering cost
                                    c = 3.,   # Variable ordering cost
                                    h = 10.,  # Storage cost
                                    p = 30.,  # Shortage cost
                                    bin_vars::Bool=false)
    # Constant demand
    w = repmat([1.3], T, 1)

    # Define JuMP model
    m = Model(solver=MyModule.BUILD_SOLVER)

    # Variables
    @variable(m, x[i=1:T+1])
    @variable(m, u[i=1:T])
    @variable(m, y[i=1:T])  # Auxiliary: y[t] = max{h * x[t], -p * x[t]}
    (bin_vars) && (@variable(m, v[i=1:T], Bin))

    # Constraints
    @constraint(m, [i=1:length(x0)], x[i] == x0[i])
    @constraint(m, yh[t=1:T], y[t] >= h * x[t])
    @constraint(m, yp[t=1:T], y[t] >= -p * x[t])
    @constraint(m, evolution[t=1:T], x[t + 1] == x[t] + u[t] - w[t])
    @constraint(m, [t=1:T], u[t] >= 0)
    if bin_vars
        @constraint(m, [t=1:T], u[t] <= M * v[t])
    end

    # Cost
    if bin_vars
        @objective(m, Min, sum(y[i] + c * u[i] + K * v[i]  for i in 1:T))
    else
        @objective(m, Min, sum(y[i] + c * u[i] for i in 1:T))
    end

    # Extract problem data
    return MyModule.OptimizationProblem(MyModule.extract_problem_data(m)...)

end

# Learn Inventory management problem active sets mapping
# ----------------------------------------------------
srand(1)

# Generate training data points
N_train = 500
X_train = [randn(1) for i = 1:N_train]
# Generate testing data points
N_test = 100
X_test = [randn(1) for i = 1:N_test]

# Get active_constr for each point
y_train, enc2active_constr = MyModule.encode(MyModule.active_constraints(X_train, InventoryManagementProblem))

# Learn tree
lnr = MyModule.tree(X_train, y_train, export_tree=false)

# Get active_constr for each point
active_constr_test = MyModule.active_constraints(X_test, InventoryManagementProblem)

# Predict active constraints
active_constr_pred = MyModule.predict(X_test, lnr, enc2active_constr)


# Evaluate performance








# Compare active_constr
n_correct_eval = 0
for i = 1:N_test
    if active_constr_pred[i] == active_constr_test[i]
        n_correct_eval += 1
    else
        println("Bad prediction at index $i")
    end
end


println("Results")
@printf "Number of correct evaluations: %d/%d\n" n_correct_eval N_test

# Solve problems with learned active_constr
x_pred = Vector{Vector{Float64}}(N_test)
x_test = Vector{Vector{Float64}}(N_test)
time_ml = Vector{Float64}(N_test)
time_lp = Vector{Float64}(N_test)
for i = 1:N_test
    problem = InventoryManagementProblem(X_test[i])
    time_ml[i] = @elapsed x_pred[i], _ = MyModule.solve(problem, active_constr_pred[i])
    time_lp[i] = @elapsed x_test[i], _ = MyModule.solve(problem)

    if abs(problem.c'* x_pred[i] - problem.c' * x_test[i]) > 1e-05
        println("Not matching cost at problem $i. Difference $(problem.c'* x_pred[i] - problem.c' * x_test[i])")
    end
end

@printf "Time LP = %.2e\n" mean(time_lp)
@printf "Time ML = %.2e\n" mean(time_ml)





