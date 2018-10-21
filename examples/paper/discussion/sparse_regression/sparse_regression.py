import numpy as np
import cvxpy as cp
import pandas as pd

# Development
import mlopt
from mlopt.sampling import uniform_sphere_sample

'''
Define Sparse Regression problem
'''
# TODO: This needs to work for different
# n = ... points
# p = ... parameters
# k = ... sparsity

'''
Sample points
'''

# Solve Gurobi


# Solve with learner

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
        save_pdf=True)
#  m.train(theta_train, learner=mlopt.PYTORCH)

# Save solver
m.save("output/optimal_tree_inv_int", delete_existing=True)
