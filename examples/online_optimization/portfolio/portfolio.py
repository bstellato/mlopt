# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())

from mlopt.utils import benchmark
import online_optimization.portfolio.simulation.settings as stg
from online_optimization.portfolio.learning_data import learning_data, sample_around_points
import mlopt
import numpy as np
import scipy.sparse as spa
import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
import pandas as pd
import datetime as dt
import logging
import argparse
import datetime as dt


np.random.seed(1)

STORAGE_DIR = "/pool001/stellato/online/portfolio"


def create_mlopt_problem(df, k=None, lambda_cost=None,
                         tight_constraints=True):

    # Get number of periods from data
    n_periods = len([col for col in df.columns if 'hat_r' in col])

    if lambda_cost is None:
        lambda_cost = {'risk': stg.RISK_COST,
                       'borrow': stg.BORROW_COST,
                       #  'norm0_trade': stg.NORM0_TRADE_COST,
                       'norm1_trade': stg.NORM1_TRADE_COST}
    else:
        lambda_cost = lambda_cost

    # Initialize problem
    n, m = df.iloc[0]['F'].shape

    # Parameters
    hat_r = [cp.Parameter(n, name="hat_r_%s" % (t + 1))
             for t in range(n_periods)]
    w_init = cp.Parameter(n, name="w_init")
    F = cp.Parameter((n, m), name="F")
    sqrt_Sigma_F = cp.Parameter(m, name="sqrt_Sigma_F")
    sqrt_D = cp.Parameter(n, name="sqrt_D")

    #  Sigma = psd_wrap(F * (Sigma_F * F.T) + cp.diag(cp.power(sqrt_D, 2)))

    # Formulate problem
    w = [cp.Variable(n) for t in range(n_periods + 1)]

    # Sparsity constraints
    if k is not None:
        s = [cp.Variable(n, boolean=True) for t in range(n_periods)]

    # Define cost components
    cost = 0
    constraints = [w[0] == w_init]
    for t in range(1, n_periods + 1):

        risk_cost = lambda_cost['risk'] * (
            #  cp.quad_form(F.T * w[t], Sigma_F) +
            cp.sum_squares(cp.multiply(sqrt_Sigma_F, F.T * w[t])) +
            cp.sum_squares(cp.multiply(sqrt_D, w[t])))
        #  risk_cost = lambda_cost['risk'] * cp.quad_form(w[t], Sigma)

        holding_cost = lambda_cost['borrow'] * \
            cp.sum(stg.BORROW_COST * cp.neg(w[t]))

        transaction_cost = \
            lambda_cost['norm1_trade'] * cp.norm(w[t] - w[t-1], 1)

        cost += \
            hat_r[t-1] * w[t] \
            - risk_cost \
            - holding_cost \
            - transaction_cost

        constraints += [cp.sum(w[t]) == 1.]

        if k is not None:
            # Cardinality constraint (big-M)
            constraints += [-s[t-1] <= w[t] - w[t-1], w[t] - w[t-1] <= s[t-1],
                            cp.sum(s[t-1]) <= k]

    return mlopt.Optimizer(cp.Maximize(cost), constraints,
                           log_level=logging.INFO,
                           #  verbose=True,
                           tight_constraints=tight_constraints,
                           parallel=True,
                           )


'''
Main script
'''


def main():

    desc = 'Online Portfolio Example'

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--sparsity', type=int, default=None, metavar='N',
                        help='sparsity level (default: Full)')
    arguments = parser.parse_args()
    k = arguments.sparsity

    EXAMPLE_NAME = STORAGE_DIR + '/portfolio_%d_' % k

    n_train = 100000

    # Define cost weights
    lambda_cost = {'risk': stg.RISK_COST,
                   'borrow': stg.BORROW_WEIGHT_COST,
                   #  'norm0_trade': stg.NORM0_TRADE_COST,
                   #  'norm1_trade': stg.NORM1_TRADE_COST,
                   'norm1_trade': 0.01}

    nn_params = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [32, 64],
        'n_epochs': [200, 300],
        'n_layers': [7, 10]
        #  {'learning_rate': 0.0001, 'batch_size': 64, 'n_epochs': 300, 'n_layers': 10}
        #  'learning_rate': [0.0001],
        #  'batch_size': [64],
        #  'n_epochs': [300],
        #  'n_layers': [10]
    }

    # Define initial value
    t_start = dt.date(2008, 1, 1)
    t_end = dt.date(2013, 1, 1)
    T_periods = 1

    # TODO: Pass initial value

    # Get data for learning
    print("Get learning data by simulating with no integer variables (faster)")
    df_history = learning_data(t_start=t_start,
                               t_end=t_end,
                               T_periods=T_periods,
                               lambda_cost=lambda_cost)
    n_history_train = int(len(df_history) * 0.8)
    df_history_train = df_history[:n_history_train]
    df_history_test = df_history[n_history_train:]

    # Sample around points
    df_train = sample_around_points(df_history_train,
                                    n_total=n_train)

    # Define mlopt problem
    m_mlopt = create_mlopt_problem(df_train, k=k,
                                   lambda_cost=lambda_cost)
    with m_mlopt:

        # Get samples
        print("Get samples in parallel")
        m_mlopt.get_samples(df_train,
                            parallel=True,
                            filter_strategies=False)
        m_mlopt.save_training_data(EXAMPLE_NAME + 'data.pkl',
                                   delete_existing=True)

        # m_mlopt.load_training_data(EXAMPLE_NAME + 'data.pkl')

        # m_mlopt.filter_strategies()

        # m_mlopt.save_training_data(EXAMPLE_NAME + 'data_filtered.pkl',
        #                            delete_existing=True)
        
        # Learn
        # m_mlopt.train(learner=mlopt.PYTORCH,
        #               n_best=10,
        #               filter_strategies=False,
        #               parallel=True,
        #               params=nn_params)

        # df_test = sample_around_points(df_history_test,
        #                                n_total=n_test)
        # res_general, res_detail = m_mlopt.performance(df_test,
        #                                               parallel=True,
        #                                               use_cache=True)

        # res_general.to_csv(EXAMPLE_NAME + "test_general.csv",
        #                    header=True)
        # res_detail.to_csv(EXAMPLE_NAME + "test_detail.csv")

        # Evaluate loop performance
        # TODO: Add!

if __name__ == '__main__':
    main()
