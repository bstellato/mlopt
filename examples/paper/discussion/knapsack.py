# Inventory example script
import numpy as np
import cvxpy as cp
import pandas as pd
import os

# Development
import mlopt
from mlopt.sampling import uniform_sphere_sample
import importlib
importlib.reload(mlopt)

'''
Define Knapsack problem
'''

# Generate data
np.random.seed(1)

# Variable
n = 10
x = cp.Variable(n, integer=True)

# Cost
c = np.random.rand(n)

# Weights
a = cp.Parameter(n, nonneg=True, name='a')
x_u = cp.Parameter(n, nonneg=True, name='x_u')
b = 0.5 * n

# Problem
cost = - c * x
constraints = [a * x <= b,
               0 <= x, x <= x_u]


# Define optimizer
m = mlopt.Optimizer(cp.Minimize(cost), constraints)

'''
Sample points
'''
# Average request
theta_bar = 2 * np.ones(2 * n)
radius = 1.0


def sample(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    ndim = int(len(theta_bar)/2)
    X_a = uniform_sphere_sample(theta_bar[:ndim], radius, n=n)
    X_u = uniform_sphere_sample(theta_bar[ndim:], radius, n=n)

    df = pd.DataFrame({
        'a': X_a.tolist(),
        'x_u': X_u.tolist()
        })

    return df


'''
Train and solve
'''

# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample(theta_bar, radius, n=n_train)
theta_test = sample(theta_bar, radius, n=n_test)

# Train solver
m.train(theta_train,
        parallel=False,
        learner=mlopt.OPTIMAL_TREE,
        max_depth=3,
        #  cp=0.1,
        #  hyperplanes=True,
        save_svg=True)
#  m.train(theta_train, learner=mlopt.PYTORCH)

# Save solver
m.save("output/optimal_tree_knapsack", delete_existing=True)

# Benchmark
results = m.performance(theta_test)

output_folder = "output/"
for i in range(len(results)):
    results[i].to_csv(output_folder + "knapsack%d.csv" % i)
