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
Define choice of vendor problem
'''
problem_name = "vendor"

# Generate data
np.random.seed(1)

# Variable
n = 5   # Number of suppliers
x = cp.Variable(n, integer=True)
u = cp.Variable(n)

# Data
c = np.random.rand(n)  # Supplier cost
m_max = 1. + np.random.rand(n)
gamma = 0.1

# Parameters
d = cp.Parameter(nonneg=True, name='d')
tau = cp.Parameter(n, nonneg=True, name='tau')

# Cost (remove redundancy)
cost = c * u + gamma * cp.max(cp.multiply(tau, x)) + 1e-06 * cp.sum(x)

# Constraints
constraints = [cp.sum(u) >= d]
constraints += [0 <= u,
                u <= cp.multiply(x, m_max)]
constraints += [0 <= x,
                x <= 1]

# Define optimizer
m = mlopt.Optimizer(cp.Minimize(cost), constraints)

'''
Sample points
'''
theta_bar = {}
theta_bar['d'] = [2.]
theta_bar['tau'] = [2., 3., 2.5, 5., 1.]


def sample(theta_bar, n=100):

    # Sample points from multivariate ball
    X_d = uniform_sphere_sample(np.array(theta_bar['d']), 1., n=n)
    X_tau = uniform_sphere_sample(np.array(theta_bar['tau']), 0.5, n=n)

    df = pd.DataFrame({
        'd': X_d.tolist(),
        'tau': X_tau.tolist()
        })

    return df


'''
Train and solve
'''

# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample(theta_bar, n=n_train)
theta_test = sample(theta_bar, n=n_test)

# Train solver
m.train(theta_train,
        parallel=True,
        learner=mlopt.OPTIMAL_TREE,
        max_depth=3,
        hyperplanes=True,
        save_svg=True)
#  m.train(theta_train, learner=mlopt.PYTORCH)

# Save solver
m.save("output/optimal_tree_%s" % problem_name, delete_existing=True)

# Benchmark
results = m.performance(theta_test)

output_folder = "output/"
for i in range(len(results)):
    results[i].to_csv(output_folder + "%s%d.csv" % (problem_name, i))
