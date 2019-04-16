import unittest
import numpy as np
import mlopt
from mlopt.sampling import uniform_sphere_sample
import pandas as pd
import cvxpy as cp
import logging


class TestCondensing(unittest.TestCase):

    def setUp(self):
        """Setup simple problem"""
        np.random.seed(1)
        # This needs to work for different
        p = 50
        n = 200
        F = np.random.randn(n, p)
        D = np.diag(np.random.rand(n)*np.sqrt(p))
        Sigma = F.dot(F.T) + D
        gamma = 1.0
        mu = cp.Parameter(n, name='mu')
        x = cp.Variable(n)
        cost = - mu * x + gamma * cp.quad_form(x, Sigma) + .5 * cp.norm(x, 1)
        constraints = [cp.sum(x) == 1, x >= 0]

        # Define optimizer
        # Force mosek to be single threaded
        m = mlopt.Optimizer(cp.Minimize(cost),
                            constraints,
                            log_level=logging.DEBUG)

        '''
        Sample points
        '''
        theta_bar = 10 * np.random.rand(n)
        radius = 1.0

        '''
        Train and solve
        '''

        # Training and testing data
        n_train = 100
        n_test = 10

        # Sample points from multivariate ball
        X_d = uniform_sphere_sample(theta_bar, radius, n=n_train)
        self.df_train = pd.DataFrame({'mu': list(X_d)})

        #  # Train and test using pytorch
        #  params = {
        #      'learning_rate': [0.01],
        #      'batch_size': [100],
        #      'n_epochs': [200]
        #  }
        #
        #  m.train(df, parallel=False, learner=mlopt.PYTORCH, params=params)

        # Testing data
        X_d_test = uniform_sphere_sample(theta_bar, radius, n=n_test)
        df_test = pd.DataFrame({'mu': list(X_d_test)})

        # Store stuff
        self.m = m
        self.df_test = df_test

    def test_condensing_simple(self):

        # TODO: Fix this test with parallel vs serial
        self.m._get_samples(self.df_train, parallel=True)
        logging.info("Number of original strategies %d" %
                     len(self.m.encoding))

        self.m.condense_strategies(k_max_strategies=5, parallel=False)
        logging.info("Number of condensed strategies (serial): %d" %
                     len(self.m.encoding))
        n_condensed_serial = len(self.m.encoding)

        self.m._get_samples(self.df_train, parallel=True)
        self.m.condense_strategies(k_max_strategies=5, parallel=True)
        logging.info("Number of condensed strategies (parallel): %d" %
                     len(self.m.encoding))
        n_condensed_parallel = len(self.m.encoding)

        assert len(self.m.encoding_full) >= n_condensed_parallel
        assert n_condensed_serial == n_condensed_parallel
