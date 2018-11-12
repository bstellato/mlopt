# Inventory example script
import numpy as np
import cvxpy as cp
import pandas as pd

# Development
import mlopt
from mlopt.sampling import uniform_sphere_sample
import importlib
importlib.reload(mlopt)

'''
Define Inventory problem
'''
bin_vars = False

# Generate data
np.random.seed(1)
T = 5
K = 2.
M = 1.
h = 1.
c = 2.
p = 3.
x_init = 2.

# Define problem
x = cp.Variable(T+1)
u = cp.Variable(T)
v = cp.Variable(T, boolean=True)


# Define parameters
d = cp.Parameter(T, nonneg=True, name="d")

# Constaints
constraints = [x[0] == x_init]
for t in range(T):
    constraints += [x[t+1] == x[t] + u[t] - d[t]]
constraints += [u >= 0]
constraints += [u <= M * v]


# Objective
cost = cp.sum(cp.maximum(h * x, -p * x)) + c * cp.sum(u) + K * cp.sum(v)

# Define optimizer
m = mlopt.Optimizer(cp.Minimize(cost), constraints)

'''
Sample points
'''
# Average request
theta_bar = 2 * np.ones(T)
radius = 1.5


def sample_inventory(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, radius, n=n)

    df = pd.DataFrame({'d': X.tolist()})

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
m.train(theta_train, learner=mlopt.OPTIMAL_TREE,
        max_depth=3,
        #  cp=0.1,
        #  hyperplanes=True,
        save_svg=True)
#  m.train(theta_train, learner=mlopt.PYTORCH)

# Save solver
m.save("output/optimal_tree_inv_int.tar.gz", delete_existing=True)

# Benchmark
results = m.performance(theta_test)

output_folder = "output/"
for i in range(len(results)):
    results[i].to_csv(output_folder + "inv_int%d.csv" % i)
