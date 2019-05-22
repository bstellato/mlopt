# Script to plot behavior of strategies number as parameters get tuned
from online_optimization.portfolio.portfolio import create_mlopt_problem

# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())

from mlopt.sampling import uniform_sphere_sample
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


np.random.seed(1)

STORAGE_DIR = "/pool001/stellato/online/portfolio"


def compute_strategies(sparsity=5,
                       lambda_norm1=stg.NORM1_TRADE_COST,
                       tight_constraints=False):

    k = sparsity

    print("Calculating number of strategies for " +
          "k=%d, lambda_norm1=%.2e, tight_constraints=%s" %
          (k, lambda_norm1, tight_constraints))
    print("-------------------------------------------")

    # Define cost weights
    lambda_cost = {'risk': stg.RISK_COST,
                   'borrow': stg.BORROW_WEIGHT_COST,
                   'norm1_trade': lambda_norm1}

    # Define initial value
    t_start = dt.date(2008, 1, 1)
    t_end = dt.date(2013, 1, 1)
    T_periods = 1

    # Get data for learning
    print("Get learning data by simulating with no integer variables (faster)")
    df_history = learning_data(t_start=t_start,
                               t_end=t_end,
                               T_periods=T_periods,
                               lambda_cost=lambda_cost)

    # Sample around points
    df = df_history

    # Define mlopt problem
    m_mlopt = create_mlopt_problem(df, k=k,
                                   lambda_cost=lambda_cost,
                                   tight_constraints=tight_constraints)
    with m_mlopt:

        # Get samples
        print("Get samples in parallel")
        m_mlopt.get_samples(df, parallel=True, filter_strategies=False)
        return m_mlopt.n_strategies


def main():
    n_lambda = 10
    k_vec = [2, 3, 4]
    lambda_norm1_vec = np.logspace(-2., 1., num=n_lambda)[::-1]

    n_strategies = {}
    results = {'k': [],
               'lambda_norm1': [],
               'n_strategies': [],
               'tight_constraints': []}
    for k in k_vec:
        for tight_constraints in [False, True]:
            for lambda_norm1 in lambda_norm1_vec:
                n_strategies = \
                    compute_strategies(sparsity=k,
                                       lambda_norm1=lambda_norm1,
                                       tight_constraints=tight_constraints)
                results['k'].append(k)
                results['lambda_norm1'].append(lambda_norm1)
                results['n_strategies'].append(n_strategies)
                results['tight_constraints'].append(tight_constraints)

                pd.DataFrame(results).to_csv('strategies_behavior_small.csv')


if __name__ == '__main__':
    main()
