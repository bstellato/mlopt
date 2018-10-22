import numpy as np
import scipy.sparse as spa
import cvxpy as cp
import pandas as pd
import os
import mlopt
from mlopt.sampling import uniform_sphere_sample
np.random.seed(1)


'''
Define Sparse Regression problem
'''
# This needs to work for different
k = 9
n = k * 10
F = spa.random(n, k, density=0.5,
               data_rvs=np.random.randn, format='csc')
D = spa.diags(np.random.rand(n) *
              np.sqrt(k), format='csc')
Sigma = (F.dot(F.T) + D).todense()   # TODO: Add Constant(Sigma)?
gamma = 1.0
mu = cp.Parameter(n, name='mu')
x = cp.Variable(n)
cost = - mu * x + gamma * cp.quad_form(x, Sigma)
constraints = [cp.sum(x) == 1, x >= 0]

# Define optimizer
m = mlopt.Optimizer(cp.Minimize(cost), constraints,
                    name="portfolio")

'''
Sample points
'''
theta_bar = np.random.randn(n)
radius = 0.3


def sample_portfolio(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, radius, n=n)

    df = pd.DataFrame({'mu': X.tolist()})

    return df


'''
Train and solve
'''

output_folder = "output/portfolio"

# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample_portfolio(theta_bar, radius, n=n_train)
theta_test = sample_portfolio(theta_bar, radius, n=n_test)

# Train and test using pytorch
m.train(theta_train, learner=mlopt.PYTORCH)
m.save(os.path.join(output_folder, "pytorch_portfolio"), delete_existing=True)
results_pytorch = m.performance(theta_test)

# DEBUG. DEFINE OPTIMIZER AGAIn
#  mu = cp.Parameter(n, name='mu')
#  x = cp.Variable(n)
#  cost = - mu * x + gamma * cp.quad_form(x, Sigma)
#  constraints = [cp.sum(x) == 1, x >= 0]
#  m = mlopt.Optimizer(cp.Minimize(cost), constraints,
#                      name="portfolio")
#  m.train(theta_train, learner=mlopt.PYTORCH)
results_pytorch = m.performance(theta_test)

# Train and test using optimal trees
#  m.train(theta_train, learner=mlopt.OPTIMAL_TREE,
#          parallel=True,
#          max_depth=10,
#          #  cp=0.1,
#          #  hyperplanes=True,
#          save_pdf=True)
#  m.save(os.path.join(output_folder, "optimaltrees_portfolio"),
#         delete_existing=True)
#  results_optimaltrees = m.performance(theta_test)
#
#
#  # Create cumulative results
#  results_general = pd.concat([results_pytorch[0], results_optimaltrees[0]])
#  results_detail = pd.concat([results_pytorch[1], results_optimaltrees[1]])
#  results_general.to_csv(os.path.join(output_folder,
#                                      "portfolio_cont_general.csv"))
#  results_detail.to_csv(os.path.join(output_folder,
#                                     "portfolio_cont_detail.csv"))
