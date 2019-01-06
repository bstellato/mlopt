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
    m_vec = np.append(m_vec, [i, int(i/2)])
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

    #  # Training and testing data
    #  theta_test = sample(theta_bar, radius, n=n_test)
    #
    #  # Train and test using pytorch
    #  data_file = os.path.join(output_folder,
    #                           name + "_n%d_m%d_data.pkl" % (n_dim, m_dim))
    #
    #  # Loading data points
    #  if os.path.isfile(data_file):
    #      print("Loading data file %s" % data_file)
    #      m.load_data(data_file)
    #      m.train(parallel=True,
    #              learner=mlopt.PYTORCH)
    #  else:
    #      m.train(sampling_fn=lambda n: sample(theta_bar, radius, n),
    #              parallel=True,
    #              learner=mlopt.PYTORCH)
    #  m.save(os.path.join(output_folder,
    #                      "pytorch_" + name + "_n%d_m%d" % (n_dim, m_dim)),
    #         delete_existing=True)
    #  pytorch_general, pytorch_detail = m.performance(theta_test, parallel=True)
    #
    #  # Fix dataframe by adding elements
    #  add_details(pytorch_general, n=n_dim, m=m_dim)
    #  add_details(pytorch_detail, n=n_dim, m=m_dim)
    #  results_general = results_general.append(pytorch_general)
    #  results_detail = results_detail.append(pytorch_detail)
    #
    #  #  Train and test using optimal trees
    #  m.train(parallel=True,
    #          learner=mlopt.OPTIMAL_TREE,
    #          hyperplanes=False,
    #          max_depth=15,
    #          save_svg=True)
    #  m.save(os.path.join(output_folder,
    #                      "optimaltrees_" + name + "_n%d_m%d" % (n_dim, m_dim)),
    #         delete_existing=True)
    #  optimaltrees_general, optimaltrees_detail = m.performance(theta_test,
    #                                                            parallel=True)
    #  add_details(optimaltrees_general, n=n_dim, m=m_dim)
    #  add_details(optimaltrees_detail, n=n_dim, m=m_dim)
    #  results_general = results_general.append(optimaltrees_general)
    #  results_detail = results_detail.append(optimaltrees_detail)
    #
    #  # Save data to file
    #  if not os.path.isfile(data_file):
    #      print("Saving data file %s" % data_file)
    #      m.save_data(data_file, delete_existing=True)
    #
    #  # Store cumulative results at each iteration
    #  results_general.to_csv(os.path.join(output_folder,
    #                                      name + "_general.csv"))
    #  results_detail.to_csv(os.path.join(output_folder,
    #                                     name + "_detail.csv"))
