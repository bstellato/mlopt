import numpy as np
import scipy.sparse as spa
import cvxpy as cp
import pandas as pd
import os
import mlopt
from mlopt.sampling import uniform_sphere_sample
np.random.seed(1)


# Define loop to train
p_vec = np.array([10, 20, 30])
results_general = pd.DataFrame()
results_detail = pd.DataFrame()

# Output folder
output_folder = "output/portfolio_int"


# Function to sample points
def sample_portfolio(theta_bar, radius, n=100):

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, radius, n=n)

    df = pd.DataFrame({'mu': X.tolist()})

    return df


def add_details(df, p=None, n=None, k=None):
    len_df = len(df)

    df['n'] = [n] * len_df
    df['p'] = [p] * len_df
    df['k'] = [k] * len_df


for p in p_vec:
    '''
    Define Sparse Regression problem
    '''
    # This needs to work for different
    n = p * 10
    k = int(p/5)
    F = spa.random(n, p, density=0.5,
                   data_rvs=np.random.randn, format='csc')
    D = spa.diags(np.random.rand(n) *
                  np.sqrt(p), format='csc')
    Sigma = (F.dot(F.T) + D).todense()   # TODO: Add Constant(Sigma)?
    gamma = 1.0
    mu = cp.Parameter(n, name='mu')
    x = cp.Variable(n)
    y = cp.Variable(n, boolean=True)
    cost = - mu * x + gamma * cp.quad_form(x, Sigma)
    constraints = [cp.sum(x) == 1,
                   x >= 0,
                   x <= y,
                   cp.sum(y) <= k]

    # Define optimizer
    m = mlopt.Optimizer(cp.Minimize(cost), constraints,
                        name="portfolio")

    '''
    Sample points
    '''
    theta_bar = np.random.randn(n)
    radius = 0.3

    '''
    Train and solve
    '''

    # Training and testing data
    n_train = 2000
    n_test = 100
    theta_train = sample_portfolio(theta_bar, radius, n=n_train)
    theta_test = sample_portfolio(theta_bar, radius, n=n_test)

    # Train and test using pytorch
    m.train(theta_train,
            parallel=False,
            learner=mlopt.PYTORCH)
    m.save(os.path.join(output_folder, "pytorch_portfolio_%d" % p),
           delete_existing=True)
    pytorch_general, pytorch_detail = m.performance(theta_test, parallel=False)

    # Fix dataframe by adding elements
    add_details(pytorch_general, n=n, p=p, k=k)
    add_details(pytorch_detail, n=n, p=p, k=k)
    results_general = results_general.append(pytorch_general)
    results_detail = results_detail.append(pytorch_detail)

    #  Train and test using optimal trees
    m.train(theta_train,
            parallel=False,
            learner=mlopt.OPTIMAL_TREE,
            max_depth=10,
            #  cp=0.1,
            #  hyperplanes=True,
            save_svg=True)
    m.save(os.path.join(output_folder, "optimaltrees_portfolio_%d" % p),
           delete_existing=True)
    optimaltrees_general, optimaltrees_detail = m.performance(theta_test,
                                                              parallel=False)
    add_details(optimaltrees_general, n=n, p=p, k=k)
    add_details(optimaltrees_detail, n=n, p=p, k=k)
    results_general = results_general.append(optimaltrees_general)
    results_detail = results_detail.append(optimaltrees_detail)


# Create cumulative results
results_general.to_csv(os.path.join(output_folder,
                                    "portfolio_int_general.csv"))
results_detail.to_csv(os.path.join(output_folder,
                                   "portfolio_int_detail.csv"))
