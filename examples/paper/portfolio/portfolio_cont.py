import numpy as np
import scipy.sparse as spa
import cvxpy as cp
import pandas as pd
import mlopt
from mlopt.sampling import uniform_sphere_sample
np.random.seed(1)


'''
Define Sparse Regression problem
'''
# This needs to work for different
k = 100
n = k * 100
F = spa.random(n, k, density=0.5,
               data_rvs=np.random.randn, format='csc')
D = spa.diags(np.random.rand(n) *
              np.sqrt(k), format='csc')
Sigma = F.dot(F.T) + D
mu = cp.Parameter(n, name='mu')
gamma = 1.0
x = cp.Variable(n)
cost = mu * x - gamma * (cp.quad_form(x, Sigma))
constraints = [cp.sum(x) == 1, x >= 0]

# Define optimizer
m = mlopt.Optimizer(cp.Minimize(cost), constraints)

'''
Sample points
'''
theta_bar = np.random.randn(n)
radius = 1.0


def sample_portfolio(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, radius, n=n)

    df = pd.DataFrame({'mu': X.tolist()})

    return df


'''
Train and solve
'''

# Training and testing data
n_train = 1000
n_test = 100
theta_train = sample_portfolio(theta_bar, radius, n=n_train)
theta_test = sample_portfolio(theta_bar, radius, n=n_test)

# Train solver
m.train(theta_train, learner=mlopt.OPTIMAL_TREE,
        max_depth=10,
        #  cp=0.1,
        #  hyperplanes=True,
        save_pdf=True)
#  m.train(theta_train, learner=mlopt.PYTORCH)

# Save solver
m.save("output/optimal_tree_portfolio", delete_existing=True)

# Benchmark
results = m.performance(theta_test)
output_folder = "output/"
for i in range(len(results)):
    results[i].to_csv(output_folder + "portfolio_cont%d.csv" % i)
