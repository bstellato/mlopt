import unittest
import numpy as np
import numpy.testing as npt
from mlopt.optimizer import Optimizer
from mlopt.settings import PYTORCH
from mlopt.problem import Problem
from mlopt.tests.settings import TEST_TOL as TOL
from mlopt.sampling import uniform_sphere_sample
import pandas as pd
import cvxpy as cp


class TestParallel(unittest.TestCase):

    def test_parallel_vs_serial_learning(self):
        """Test parallel VS serial learning"""

        # Generate data
        np.random.seed(1)
        T = 5
        M = 2.
        h = 1.
        c = 1.
        p = 1.
        x_init = 2.
        radius = 3.
        N = 100   # Number of points

        # Define problem
        x = cp.Variable(T+1)
        u = cp.Variable(T)

        # Define parameter and sampling points
        d = cp.Parameter(T, nonneg=True, name="d")
        d_bar = 3. * np.ones(T)
        X_d = uniform_sphere_sample(d_bar, radius, n=N)
        df = pd.DataFrame({'d': X_d.tolist()})

        # Constaints
        constraints = [x[0] == x_init]
        for t in range(T):
            constraints += [x[t+1] == x[t] + u[t] - d[t]]
        constraints += [u >= 0, u <= M]

        # Objective
        cost = cp.sum(cp.maximum(h * x, -p * x)) + c * cp.sum(u)

        # Define problem
        problem = Problem(cp.Minimize(cost), constraints)

        # Solve for all theta in serial
        results_serial = problem.solve_parametric(df,
                                                  parallel=False)

        # Solve for all theta in parallel
        results_parallel = problem.solve_parametric(df,
                                                    parallel=True)

        # Assert all results match
        for i in range(N):
            serial = results_serial[i]
            parallel = results_parallel[i]

            # Compare x
            npt.assert_array_almost_equal(serial['x'],
                                          parallel['x'],
                                          decimal=TOL)

            # Compare cost
            npt.assert_array_almost_equal(serial['cost'],
                                          parallel['cost'],
                                          decimal=TOL)

            # Compare strategy
            self.assertTrue(serial['strategy'] == parallel['strategy'])

    def test_parallel_resolve(self):
        """Test parallel resolve (to avoid hanging)"""

        np.random.seed(1)
        # This needs to work for different
        p = 10
        n = p * 10
        F = np.random.randn(n, p)
        D = np.diag(np.random.rand(n)*np.sqrt(p))
        Sigma = F.dot(F.T) + D
        gamma = 1.0
        mu = cp.Parameter(n, name='mu')
        x = cp.Variable(n)
        cost = - mu * x + gamma * cp.quad_form(x, Sigma)
        constraints = [cp.sum(x) == 1, x >= 0]

        # Define optimizer
        # Force mosek to be single threaded
        m = Optimizer(cp.Minimize(cost), constraints, name="portfolio")

        '''
        Sample points
        '''
        theta_bar = np.random.randn(n)
        radius = 0.3

        '''
        Train and solve
        '''

        # Training and testing data
        n_train = 1000
        n_test = 100
        # Sample points from multivariate ball
        X_d = uniform_sphere_sample(theta_bar, radius, n=n_train)
        X_d_test = uniform_sphere_sample(theta_bar, radius, n=n_test)
        df = pd.DataFrame({'mu': X_d.tolist()})
        df_test = pd.DataFrame({'mu': X_d_test.tolist()})

        # Train and test using pytorch
        m.train(df,
                parallel=True,
                learner=PYTORCH)
        m.performance(df_test, parallel=True)

        # Run parallel loop again to enforce instability
        # in multiprocessing
        m.performance(df_test, parallel=True)

        return

    def test_parallel_strategy_selection(self):
        """Choose best strategy in parallel"""
        np.random.seed(1)
        # This needs to work for different
        p = 10
        n = p * 10
        F = np.random.randn(n, p)
        D = np.diag(np.random.rand(n)*np.sqrt(p))
        Sigma = F.dot(F.T) + D
        gamma = 1.0
        mu = cp.Parameter(n, name='mu')
        x = cp.Variable(n)
        cost = - mu * x + gamma * cp.quad_form(x, Sigma)
        constraints = [cp.sum(x) == 1, x >= 0]

        # Define optimizer
        # Force mosek to be single threaded
        m = Optimizer(cp.Minimize(cost), constraints)

        '''
        Sample points
        '''
        theta_bar = np.random.randn(n)
        radius = 0.3

        '''
        Train and solve
        '''

        # Training and testing data
        n_train = 1000
        n_test = 1  # Choose only one point to check parallel strategy evaluation

        # Sample points from multivariate ball
        X_d = uniform_sphere_sample(theta_bar, radius, n=n_train)
        df = pd.DataFrame({'mu': X_d.tolist()})
        X_d_test = uniform_sphere_sample(theta_bar, radius, n=n_test)
        df_test = pd.DataFrame({'mu': X_d_test.tolist()})

        # Train and test using pytorch
        m.train(df, parallel=True, learner=PYTORCH)

        # Test
        serial = m.solve(df_test, parallel=False)
        parallel = m.solve(df_test)

        # Compare x
        npt.assert_array_almost_equal(serial['x'],
                                      parallel['x'],
                                      decimal=TOL)

        # Compare cost
        npt.assert_array_almost_equal(serial['cost'],
                                      parallel['cost'],
                                      decimal=TOL)


if __name__ == '__main__':
    unittest.main()
