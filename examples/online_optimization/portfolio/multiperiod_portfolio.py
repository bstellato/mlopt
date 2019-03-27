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
from cvxpy.atoms.affine.wraps import psd_wrap
import pandas as pd
import datetime as dt


np.random.seed(1)

# Get real data for learning
with pd.HDFStore("./data/learn_data.h5") as sim_data:
    df_real = sim_data['data']


# Reduce number of assets
n_assets = 30
k_factors = 5

# F
F_red = []
for F in df_real['F']:
    F_red.append(F[:n_assets, :k_factors])

# hat_r_red
hat_r_red = {}
for hat_r_col in [col for col in df_real.columns if 'hat_r' in col]:
    hat_r = df_real[hat_r_col]
    hat_r_temp = []
    for r in hat_r:
        hat_r_temp.append(r[:n_assets])
    hat_r_red[hat_r_col] = hat_r_temp

#  w_init
w_init_red = []
for w in df_real['w_init']:
    w_init_red.append(w[:n_assets])

#  Sigma_F
Sigma_F_red = []
for S in df_real['Sigma_F']:
    Sigma_F_red.append(S[:k_factors, :k_factors])

#  sqrt_D
sqrt_D_red = []
for sqrt_D in df_real['sqrt_D']:
    sqrt_D_red.append(sqrt_D[:n_assets])


df_red = pd.DataFrame()
df_red['F'] = F_red
df_red['w_init'] = w_init_red
df_red['Sigma_F'] = Sigma_F_red
df_red['sqrt_D'] = sqrt_D_red
for k, v in hat_r_red.items():
    df_red[k] = v


def create_mlopt_problem(df):

    # Get number of periods from data
    n_periods = len([col for col in df.columns if 'hat_r' in col])

    lam = {'risk': 50,
           'borrow': 0.0001,
           #  'norm1_trade': 0.02,
           'norm2_trade': 0.02,
           'norm0_trade': 1.}

    borrow_cost = 0.0001
    p = 3

    # Initialize problem
    n, k = df.iloc[0]['F'].shape

    # Parameters
    hat_r = [cp.Parameter(n, name="hat_r_%s" % (t + 1))
             for t in range(n_periods)]
    w_init = cp.Parameter(n, name="w_init")
    F = cp.Parameter((n, k), name="F")
    Sigma_F = cp.Parameter((k, k), PSD=True, name="Sigma_F")
    sqrt_D = cp.Parameter(n, name="sqrt_D")

    Sigma = psd_wrap(F * (Sigma_F * F.T) + cp.diag(cp.power(sqrt_D, 2)))

    # Formulate problem
    w = [cp.Variable(n) for t in range(n_periods + 1)]

    # Sparsity constraints
    s = [cp.Variable(n, boolean=True) for t in range(n_periods)]

    # Define cost components
    cost = 0
    constraints = [w[0] == w_init]
    for t in range(1, n_periods + 1):

        #  risk_cost = lam['risk'] * (
        #      cp.quad_form(F.T * w[t], Sigma_F) +
        #      cp.sum_squares(cp.multiply(sqrt_D, w[t])))
        risk_cost = lam['risk'] * cp.quad_form(w[t], Sigma)

        holding_cost = lam['borrow'] * \
            cp.sum(borrow_cost * cp.neg(w[t]))

        #  transaction_cost = lam['norm1_trade'] * cp.norm(w[t] - w[t-1], 1)
        transaction_cost = lam['norm2_trade'] * cp.sum_squares(w[t] - w[t-1])

        cost += \
            hat_r[t-1] * w[t] \
            - risk_cost \
            - holding_cost \
            - transaction_cost

        constraints += [cp.sum(w[t]) == 1.]

        # Cardinality constraint (big-M)
        constraints += [-s[t-1] <= w[t] - w[t-1], w[t] - w[t-1] <= s[t-1],
                        cp.sum(s[t-1]) <= p]

    return mlopt.Optimizer(cp.Maximize(cost), constraints,
                           FeasibiliyTol=1e-09, OptimalityTol=1e-09)


m = create_mlopt_problem(df_red)

params = {
    'learning_rate': [0.01],
    'batch_size': [32],
    'n_epochs': [200]
}

m._get_samples(df_red, parallel=True)
#  m.save_training_data("./data/train_data.pkl",
#                       delete_existing=True)
#  m.load_training_data("./data/train_data.pkl")
















# Test repeat first element of df_real
#  df_train = pd.DataFrame()
#
#  for i in range(1000):
#      df_train = df_train.append(df_real.iloc[0])


#  m.train(df_real,
#          parallel=True,
#          learner=mlopt.PYTORCH,
#          params=params)

# Function to sample points
# Use portfolio data
#  def sample(theta_bar, n=100):
#
#      radius = 0.15
#
#      # Sample points from multivariate ball
#      X = uniform_sphere_sample(theta_bar, radius, n=n)
#
#      df = pd.DataFrame({'mu': X.tolist()})
#
#      return df

#  results_general = pd.DataFrame()
#  results_detail = pd.DataFrame()
#
#  for p in p_vec:
#      '''
#      Define portfolio problem
#      '''
#      # This needs to work for different
#      n = p * 10
#      F = spa.random(n, p, density=0.5,
#                     data_rvs=np.random.randn, format='csc')
#      D = spa.diags(np.random.rand(n) *
#                    np.sqrt(p), format='csc')
#      Sigma = (F.dot(F.T) + D).todense()   # TODO: Add Constant(Sigma)?
#      gamma = 1.0
#      mu = cp.Parameter(n, name='mu')
#      x = cp.Variable(n)
#      cost = - mu * x + gamma * cp.quad_form(x, Sigma)
#      constraints = [cp.sum(x) == 1, x >= 0]
#
#      # Define optimizer
#      m = mlopt.Optimizer(cp.Minimize(cost), constraints,
#                          name=name)
#
#      '''
#      Define parameters average values
#      '''
#      theta_bar = np.random.randn(n)
#
#      '''
#      Train and solve
#      '''
#
#      data_file = os.path.join(output_folder, "%s_p%d" % (name, p))
#
#      # Benchmark and append results
#      temp_general, temp_detail = benchmark(m, data_file,
#                                            theta_bar,
#                                            lambda n: sample(theta_bar, n),
#                                            {'p': p})
#      results_general = results_general.append(temp_general)
#      results_detail = results_detail.append(temp_detail)
#
#
#  # Store cumulative results
#  results_general.to_csv(os.path.join(output_folder,
#                                      "%s_general.csv" % name))
#  results_detail.to_csv(os.path.join(output_folder,
#                                     "%s_detail.csv" % name))
#
