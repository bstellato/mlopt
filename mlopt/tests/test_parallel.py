import unittest
import numpy as np
import numpy.testing as npt
from mlopt.problem import Problem
from mlopt.tests.settings import TEST_TOL as TOL
from mlopt.sampling import uniform_sphere_sample
import pandas as pd
import cvxpy as cp


class TestParallel(unittest.TestCase):

    def test_parallel_vs_serial(self):
        """Test parallel VS serial solution"""

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


if __name__ == '__main__':
    unittest.main()
