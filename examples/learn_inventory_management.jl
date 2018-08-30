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

# Learn
# -----
srand(1)

# Generate training data points
num_train = 500
theta_train = [randn(1) for i = 1:num_train]

# Get active_constr for each point
y_train, enc2active_constr = MyModule.encode(MyModule.active_constraints(theta_train, InventoryManagementProblem))

# Learn tree
lnr = MyModule.tree(theta_train, y_train, export_tree=false)


# Test
# ------

# Generate testing data points
num_test = 100
theta_test = [randn(1) for i = 1:num_test]

# Evaluate performance
# TODO: Complete
df_general, df_detail = eval_performance(theta_test, lnr, InventoryManagementProblem, enc2active_constr)

# Write output
output_name = String(Base.function_name(InventoryManagementProblem))
CSV.write("$(output_name)_general.csv", df_general)
CSV.write("$(output_name)_detail.csv", df_detail)


