# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())

from mlopt.sampling import uniform_sphere_sample
from mlopt.utils import benchmark
import mlopt
import numpy as np
import scipy.sparse as spa
import cvxpy as cp
import pandas as pd


np.random.seed(1)

# Define loop to train
p_vec = np.array([10, 20, 30, 40, 50])
#  p_vec = np.array([3, 4])

# Output folder
name = "portfolio"
output_folder = "output/%s" % name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Function to sample points
def sample(theta_bar, n=100):

    radius = 0.15

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, radius, n=n)

    df = pd.DataFrame({'mu': X.tolist()})

    return df


results_general = pd.DataFrame()
results_detail = pd.DataFrame()

for p in p_vec:
    '''
    Define portfolio problem
    '''
    # This needs to work for different
    n = p * 10
    F = spa.random(n, p, density=0.5,
                   data_rvs=np.random.randn, format='csc')
    D = spa.diags(np.random.rand(n) *
                  np.sqrt(p), format='csc')
    Sigma = (F.dot(F.T) + D).todense()   # TODO: Add Constant(Sigma)?
    gamma = 1.0
    mu = cp.Parameter(n, name='mu')
    x = cp.Variable(n)
    cost = - mu * x + gamma * cp.quad_form(x, Sigma)
    constraints = [cp.sum(x) == 1, x >= 0]

    # Define optimizer
    m = mlopt.Optimizer(cp.Minimize(cost), constraints,
                        name=name)

    '''
    Define parameters average values
    '''
    theta_bar = np.random.randn(n)

    '''
    Train and solve
    '''

    data_file = os.path.join(output_folder, "%s_p%d" % (name, p))

    # Benchmark and append results
    temp_general, temp_detail = benchmark(m, data_file,
                                          theta_bar,
                                          lambda n: sample(theta_bar, n),
                                          {'p': p})
    results_general = results_general.append(temp_general)
    results_detail = results_detail.append(temp_detail)


# Store cumulative results
results_general.to_csv(os.path.join(output_folder,
                                    "%s_general.csv" % name))
results_detail.to_csv(os.path.join(output_folder,
                                   "%s_detail.csv" % name))
