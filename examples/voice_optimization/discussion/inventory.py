# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
import matplotlib.pyplot as plt
import importlib
from mlopt.sampling import uniform_sphere_sample
import mlopt
import pandas as pd
import cvxpy as cp
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

# Inventory example script

# Development
importlib.reload(mlopt)

'''
Define Inventory problem
'''
bin_vars = False

# Generate data
np.random.seed(1)
T = 30
M = 3.
h = 1.
c = 2.
p = 3.
#  x_init = 10.

# Define problem
x = cp.Variable(T+1)
u = cp.Variable(T)

# Define parameters
d = cp.Parameter(T, nonneg=True, name="d")
x_init = cp.Parameter(name="x_init")

# Constaints
constraints = [x[0] == x_init]
for t in range(T):
    constraints += [x[t+1] == x[t] + u[t] - d[t]]
constraints += [u >= 0, u <= M]

# Objective
cost = cp.sum(cp.maximum(h * x, -p * x)) + c * cp.sum(u)

# Define optimizer
m = mlopt.Optimizer(cp.Minimize(cost), constraints)

'''
Sample points
'''
# Average request
theta_bar = np.concatenate(( 2 * np.ones(T),  # d
    [10]               # x_init
))
radius = 1


def sample_inventory(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    X_d = uniform_sphere_sample(theta_bar[:-1], radius, n=n)
    X_x_init = uniform_sphere_sample([theta_bar[-1]], 3 * radius,
                                     n=n)

    df = pd.DataFrame({'d': X_d.tolist(),
                       'x_init': X_x_init.tolist()})

    return df


'''
Train and solve
'''

# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample_inventory(theta_bar, radius, n=n_train)
theta_test = sample_inventory(theta_bar, radius, n=n_test)

# Train solver
m.train(theta_train,
        learner=mlopt.OPTIMAL_TREE,
        max_depth=3,
        parallel_trees=False,
        save_svg=True)
#  m.train(theta_train, learner=mlopt.PYTORCH)

output = "output/optimal_tree_inv_cont"

# Save solver
m.save(output, delete_existing=True)

# Benchmark
results_general, results_detail = m.performance(theta_test)
results_general.to_csv(output + "inv_cont_general.csv")
results_detail.to_csv(output + "inv_cont_detail.csv")

# Solve with single value of theta
theta_plot = sample_inventory(theta_bar, radius, n=1)

# Get optimal solution
result_plot = m.solve(theta_plot)

# Plot behavior

# Write normally
t = np.arange(0, T, 1)
fig, ax = plt.subplots(3, 1)
ax[0].step(t, x.value[:-1], where="post")
ax[0].set_ylabel('x')
ax[1].step(t, u.value, where="post")
ax[1].set_ylabel('u')
ax[2].step(t, theta_plot['d'][0], where="post")
ax[2].set_ylabel('d')
plt.show(block=False)


# Store values for plotting
df_plot = pd.DataFrame({'t': t,
                        'x': x.value[:-1],
                        'u': u.value,
                        'd': theta_plot['d'][0]})
df_plot.to_csv(output + "inventory_plot.csv")
