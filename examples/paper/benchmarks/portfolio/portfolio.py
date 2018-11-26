# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())

from mlopt.sampling import uniform_sphere_sample
import mlopt
import numpy as np
import scipy.sparse as spa
import cvxpy as cp
import pandas as pd


np.random.seed(1)

# Define loop to train
p_vec = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#  p_vec = np.array([10, 20])
results_general = pd.DataFrame()
results_detail = pd.DataFrame()

# Output folder
name = "portfolio"
output_folder = "output/%s" % name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Function to sample points
def sample(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, radius, n=n)

    df = pd.DataFrame({'mu': X.tolist()})

    return df


def add_details(df, p=None, n=None):
    len_df = len(df)

    df['n'] = [n] * len_df
    df['p'] = [p] * len_df


for p in p_vec:
    '''
    Define Sparse Regression problem
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
    Sample points
    '''
    theta_bar = np.random.randn(n)
    radius = 0.3

    '''
    Train and solve
    '''

    # Training and testing data
    n_train = 10000
    n_test = 100
    theta_train = sample(theta_bar, radius, n=n_train)
    theta_test = sample(theta_bar, radius, n=n_test)

    # Train and test using pytorch
    data_file = os.path.join(output_folder,
                             "%s_p%d_n%d_data.pkl" % (name, p, n_train))

    # Loading data points
    if os.path.isfile(data_file):
        print("Loading data file %s" % data_file)
        m.load_data(data_file)
        m.train(parallel=True,
                learner=mlopt.PYTORCH)
    else:
        # Train neural network
        m.train(sampling_fn=lambda n: sample(theta_bar, radius, n),
                parallel=True,
                learner=mlopt.PYTORCH)
    m.save(os.path.join(output_folder,
                        "pytorch_%s_p%d_n%d" % (name, p, n_train)),
           delete_existing=True)
    pytorch_general, pytorch_detail = m.performance(theta_test, parallel=True)

    # Fix dataframe by adding elements
    add_details(pytorch_general, n=n, p=p)
    add_details(pytorch_detail, n=n, p=p)
    results_general = results_general.append(pytorch_general)
    results_detail = results_detail.append(pytorch_detail)

    #  Train and test using optimal trees
    m.train(
            #  theta_train,
            parallel=True,
            learner=mlopt.OPTIMAL_TREE,
            hyperplanes=False,
            max_depth=15,
            save_svg=True)
    m.save(os.path.join(output_folder, "optimaltrees_%s_p%d_n%d" % (name, p, n_train)),
           delete_existing=True)
    optimaltrees_general, optimaltrees_detail = m.performance(theta_test,
                                                              parallel=True)
    add_details(optimaltrees_general, n=n, p=p)
    add_details(optimaltrees_detail, n=n, p=p)
    results_general = results_general.append(optimaltrees_general)
    results_detail = results_detail.append(optimaltrees_detail)

    # Save data to file
    if not os.path.isfile(data_file):
        print("Saving data file %s" % data_file)
        m.save_data(data_file, delete_existing=True)

    # Store cumulative results at each iteration
    results_general.to_csv(os.path.join(output_folder,
                                        "%s_general.csv" % name))
    results_detail.to_csv(os.path.join(output_folder,
                                       "%s_detail.csv" % name))
