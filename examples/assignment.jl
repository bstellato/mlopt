using JuMP
include("../src/MyModule.jl")

function assignment(theta::Array{Float64};
                    A = 50,   # Agents
                   )


    T = A  # Same tasks as agents

    c = eye(A) + spdiagm(theta)

    # Define JuMP model
    m = Model(solver=MyModule.BUILD_SOLVER)

    # Variables
    @variable(m, x[i=1:A, j=1:T] >= 0)

    # Constraints
    @constraint(m, [i=1:A], sum(x[i, j] for j = 1:T) == 1)
    @constraint(m, [j=1:T], sum(x[i, j] for i = 1:A) == 1)

    # Objective
    @objective(m, Min, sum(c[i, j] * x[i, j]  for i in 1:A for j = 1:T))

    # Extract problem data
    return MyModule.OptimizationProblem(MyModule.extract_problem_data(m)...)

end


# Generate data
# -------------
theta_dim = 50  # A * T
problem = assignment
# Generate training data points
num_train = 1000
theta_train = [.3 * randn(theta_dim) for i = 1:num_train]

# Generate testing data points
num_test = 100
theta_test = [.3 * randn(theta_dim) for i = 1:num_test]




# Learn
# -----
srand(1)

# Get active_constr for each point
y_train, enc2active_constr = MyModule.encode(MyModule.active_constraints(theta_train, problem))

# Learn tree
lnr = MyModule.tree(theta_train, y_train, export_tree=true, problem=problem)


# Test
# ------
# Evaluate performance
df_general, df_detail = MyModule.eval_performance(theta_test, lnr, problem, enc2active_constr)

# Store results
MyModule.write_output(df_general, df_detail, problem)

