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
bin_vars = False

# Generate data
np.random.seed(1)

# Variable
n = 10
x = cp.Variable(n, integer=True)

# Cost
c = np.random.rand(n)

# Weights
a = cp.Parameter(n, nonneg=True, name='a')
b = 0.5 * n
x_u = 3

# Problem
cost = - c * x
constraints = [a * x <= b,
               0 <= x, x <= 3]


# Define optimizer
m = mlopt.Optimizer(cp.Minimize(cost), constraints)

'''
Sample points
'''
# Average request
theta_bar = 2 * np.ones(n)
radius = 0.5


def sample(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, radius, n=n)

    df = pd.DataFrame({'a': X.tolist()})

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
        max_depth=2,
        #  cp=0.1,
        #  hyperplanes=True,
        save_pdf=True)
#  m.train(theta_train, learner=mlopt.PYTORCH)

# Save solver
output_folder = os.path.join("output", "knapsack")
m.save(os.path.join(output_folder, "optimal_tree_knapsack"),
       delete_existing=True)

# Benchmark
results_general, results_detail = m.performance(theta_test)

results_general.to_csv(os.path.join(output_folder,
                                    "knapsack_general.csv"))
results_detail.to_csv(os.path.join(output_folder,
                                   "knapsack_detail.csv"))
