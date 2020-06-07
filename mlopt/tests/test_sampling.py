import unittest
import numpy as np
import numpy.testing as npt
from mlopt import Optimizer, PYTORCH
from mlopt.tests.settings import TEST_TOL as TOL
from mlopt.sampling import uniform_sphere_sample
import mlopt.settings as s
import tempfile
import os
import pandas as pd
import cvxpy as cp


def sampling_function(n):
    """
    Sample data points.
    """
    theta_bar = 3. * np.ones(5)

    # Sample points from multivariate ball
    X = uniform_sphere_sample(theta_bar, 1, n=n)

    df = pd.DataFrame({'d': list(X)})

    return df


class TestSampling(unittest.TestCase):

    def setUp(self):
        # Generate data
        np.random.seed(1)
        T = 5
        M = 2.
        h = 1.
        c = 1.
        p = 1.
        x_init = 2.

        # Number of test points
        n_test = 10

        # Define problem
        x = cp.Variable(T+1)
        u = cp.Variable(T)

        # Define parameter and sampling points
        d = cp.Parameter(T, nonneg=True, name="d")

        # Constaints
        constraints = [x[0] == x_init]
        for t in range(T):
            constraints += [x[t+1] == x[t] + u[t] - d[t]]
        constraints += [u >= 0, u <= M]
        self.constraints = constraints

        # Objective
        self.cost = cp.sum(cp.maximum(h * x, -p * x)) + c * cp.sum(u)

        # Define problem
        problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        self.optimizer = Optimizer(problem)

        # Test set
        self.df_test = sampling_function(n_test)

    def test_sample(self):
        """Test sampling scheme"""

        # Train optimizer
        self.optimizer.train(sampling_fn=sampling_function,
                             learner=PYTORCH,
                             n_train_trials=10)

        # Check tolerance
        self.assertTrue(self.optimizer._sampler.good_turing_smooth
                        < s.SAMPLING_TOL)

        # Check that smoothed is larger than unsmoothed
        self.assertTrue(self.optimizer._sampler.good_turing
                        < self.optimizer._sampler.good_turing_smooth)


if __name__ == '__main__':
    unittest.main()

