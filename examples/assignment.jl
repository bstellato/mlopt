using JuMP
include("../src/MyModule.jl")

# Generate data
# -------------
problem = MyModule.Assignment()
problem.A = 50
theta_dim = 50  # A

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

