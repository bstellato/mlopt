# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())

# Standard imports
from mlopt.sampling import uniform_sphere_sample
import mlopt
import numpy as np
import scipy.sparse as spa
import cvxpy as cp
import pandas as pd
from mlopt.utils import benchmark


np.random.seed(1)

# Define data
n_vec = np.array([], dtype=int)  # Facilities
m_vec = np.array([], dtype=int)  # Stores
for i in np.arange(20, 100, 20):
    n_vec = np.append(n_vec, [i] * 2)
    m_vec = np.append(m_vec, [int(i/2), i])
results_general = pd.DataFrame()
results_detail = pd.DataFrame()


name = "facility"

# Output folder
output_folder = "output/" + name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Function to sample points
def sample(theta_bar, n=100):

    radius = 0.25

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, radius, n=n)

    df = pd.DataFrame({'d': X.tolist()})

    return df


# Main script
results_general = pd.DataFrame()
results_detail = pd.DataFrame()

for i in range(len(n_vec)):
    '''
    Define Facility Location problem
    '''
    n_dim = n_vec[i]
    m_dim = m_vec[i]
    print("Solving for n = %d, m = %d" % (n_dim, m_dim))

    # Variables
    x = [cp.Variable(m_dim) for _ in range(n_dim)]
    #  x = cp.Variable((n_dim, m_dim))
    y = cp.Variable(n_dim, integer=True)

    # Define transportation cost
    c = [np.random.rand(m_dim) for _ in range(n_dim)]  # Facilities x stores
    f = 10 * np.random.rand(n_dim)

    # Supply for each warehouse (scalar)
    s = 8 * np.ones(n_dim) + 10 * np.random.rand(n_dim)

    # Parameters
    d = cp.Parameter(m_dim, name='d')

    # Constraints
    constraints = [cp.sum(x) >= d]
    constraints += [cp.sum(x[i]) <= s[i] * y[i] for i in range(n_dim)]
    constraints += [x[i] >= 0 for i in range(n_dim)]
    constraints += [y >= 0, y <= 1]

    # Objective
    cost = f * y
    for i in range(n_dim):
        cost += c[i] * x[i]

    # Define optimizer
    m = mlopt.Optimizer(cp.Minimize(cost), constraints,
                        name=name)

    '''
    Sample points
    '''
    theta_bar = 3 * np.ones(m_dim) + np.random.randn(m_dim)

    '''
    Train and solve
    '''

    # Train and test using pytorch
    data_file = os.path.join(output_folder, "%s_n%d_m%d" %
                             (name, n_dim, m_dim))

    # Benchmark and append results
    temp_general, temp_detail = benchmark(m, data_file,
                                          theta_bar,
                                          lambda n: sample(theta_bar, n),
                                          {'n': n_dim, 'm': m_dim})
    results_general = results_general.append(temp_general)
    results_detail = results_detail.append(temp_detail)


# Store cumulative results
results_general.to_csv(os.path.join(output_folder,
                                    "%s_general.csv" % name))
results_detail.to_csv(os.path.join(output_folder,
                                   "%s_detail.csv" % name))
