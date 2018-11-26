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


np.random.seed(1)

# Define data
n_vec = np.array([], dtype=int)
m_vec = np.array([], dtype=int)
for i in np.arange(20, 40, 20):
    n_vec = np.append(n_vec, [i] * 3)
    m_vec = np.append(m_vec, [i, int(i/2), 2 * i])
n_train = 10000
n_test = 100
results_general = pd.DataFrame()
results_detail = pd.DataFrame()


# DEBUG: only data 3
n_vec = n_vec[2:]
m_vec = m_vec[2:]

name = "transportation"

# Output folder
output_folder = "output/" + name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Function to sample points
def sample(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, radius, n=n)

    df = pd.DataFrame({'d': X.tolist()})

    return df


def add_details(df, n=None, m=None):
    len_df = len(df)

    df['n'] = [n] * len_df
    df['m'] = [m] * len_df


# Main script
for i in range(len(n_vec)):
    '''
    Define Transportation problem
    '''
    n_dim = n_vec[i]
    m_dim = m_vec[i]
    print("Solving for n = %d, m = %d" % (n_dim, m_dim))

    # Define transportation cost
    c = [5 * np.random.rand(m_dim)
         for _ in range(n_dim)]  # c_i for each warehouse
    # Supply for each warehouse (scalar)
    s = 3 * np.ones(n_dim) + 10 * np.random.rand(n_dim)

    # Variables
    x = [cp.Variable(m_dim) for _ in range(n_dim)]  # x_i for each earehouse

    # Parameters
    d = cp.Parameter(m_dim, name='d')

    # Constraints
    constraints = [cp.sum(x[i]) <= s[i] for i in range(n_dim)]
    constraints += [cp.sum(x) >= d]
    constraints += [x[i] >= 0 for i in range(n_dim)]

    # Objective
    cost = 0
    for i in range(n_dim):
        cost += c[i] * x[i]

    # Define optimizer
    m = mlopt.Optimizer(cp.Minimize(cost), constraints,
                        name=name)

    '''
    Sample points
    '''
    theta_bar = 3 * np.ones(m_dim) + np.random.randn(m_dim)
    radius = 0.5

    '''
    Train and solve
    '''

    # Training and testing data
    theta_test = sample(theta_bar, radius, n=n_test)

    # Train and test using pytorch
    data_file = os.path.join(output_folder,
                             name + "_n%d_m%d_data.pkl" % (n_dim, m_dim))

    # Loading data points
    if os.path.isfile(data_file):
        print("Loading data file %s" % data_file)
        m.load_data(data_file)
        m.train(parallel=True,
                learner=mlopt.PYTORCH)
    else:
        m.train(sampling_fn=lambda n: sample(theta_bar, radius, n),
                parallel=True,
                learner=mlopt.PYTORCH)
    m.save(os.path.join(output_folder,
                        "pytorch_" + name + "_n%d_m%d" % (n_dim, m_dim)),
           delete_existing=True)
    pytorch_general, pytorch_detail = m.performance(theta_test, parallel=True)

    # Fix dataframe by adding elements
    add_details(pytorch_general, n=n_dim, m=m_dim)
    add_details(pytorch_detail, n=n_dim, m=m_dim)
    results_general = results_general.append(pytorch_general)
    results_detail = results_detail.append(pytorch_detail)

    #  Train and test using optimal trees
    #  m.train(
    #          #  sampling_fn=lambda n: sample(theta_bar, radius, n),
    #          parallel=True,
    #          parallel_trees=True,
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

    # Save data to file
    if not os.path.isfile(data_file):
        print("Saving data file %s" % data_file)
        m.save_data(data_file, delete_existing=True)

    # Store cumulative results at each iteration
    results_general.to_csv(os.path.join(output_folder,
                                        name + "_general.csv"))
    results_detail.to_csv(os.path.join(output_folder,
                                       name + "_detail.csv"))
