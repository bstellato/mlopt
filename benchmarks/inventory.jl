include("../src/MyModule.jl")
using DataFrames

# Generate data
# -------------
problem = MyModule.Inventory()
problem.T = 3
problem.M = 3.
problem.K = 1.
problem.radius = 1.0

# Operating point
theta_bar = [2.;               # h
             2.;               # p
             5.;               # c
             1.;               # x0
             ones(problem.T);  # d
            ]
radius = 1.0

N_train = 1000
N_test = 100
theta_train = MyModule.sample(problem, theta_bar, N=N_train)
theta_test = MyModule.sample(problem, theta_bar, N=N_test)

# Learn
# -----
srand(1)

# Get active_constr for each point
y_train, enc2active_constr = MyModule.encode(MyModule.active_constraints(theta_train, problem))

# Learn tree
lnr = MyModule.tree(theta_train, y_train, sparse=true, export_tree=true, problem=problem)


# Test
# ------
# Evaluate performance
df, df_detail = MyModule.eval_performance(theta_test, lnr, problem, enc2active_constr)

# Store results
MyModule.write_output(df, df_detail, problem)

