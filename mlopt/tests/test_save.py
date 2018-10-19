import unittest
import numpy as np
import numpy.testing as npt
from mlopt import Optimizer
from mlopt.tests.settings import TEST_TOL as TOL
from mlopt.sampling import uniform_sphere_sample
import mlopt.settings as s
import tempfile
import pandas as pd
import cvxpy as cp


class TestSave(unittest.TestCase):

    def setUp(self):
        # Generate data
        np.random.seed(1)
        T = 5
        M = 2.
        h = 1.
        c = 1.
        p = 1.
        x_init = 2.
        radius = 3.
        n = 100   # Number of points
        n_test = 10

        # Define problem
        x = cp.Variable(T+1)
        u = cp.Variable(T)

        # Define parameter and sampling points
        d = cp.Parameter(T, nonneg=True, name="d")
        d_bar = 3. * np.ones(T)
        X_d = uniform_sphere_sample(d_bar, radius, n=n)
        X_d_test = uniform_sphere_sample(d_bar, radius, n=n_test)
        self.df = pd.DataFrame({'d': X_d.tolist()})
        self.df_test = pd.DataFrame({'d': X_d_test.tolist()})

        # Constaints
        constraints = [x[0] == x_init]
        for t in range(T):
            constraints += [x[t+1] == x[t] + u[t] - d[t]]
        constraints += [u >= 0, u <= M]
        self.constraints = constraints

        # Objective
        self.cost = cp.sum(cp.maximum(h * x, -p * x)) + c * cp.sum(u)

        # Define problem
        self.optimizer = Optimizer(cp.Minimize(self.cost),
                                   self.constraints)

    def test_save_load(self):
        """Test save load"""

        learners = [
            #  s.PYTORCH,
            s.OPTIMAL_TREE
        ]

        for learner in learners:

            # Train optimizer
            self.optimizer.train(self.df, learner=learner)

            with tempfile.TemporaryDirectory() as tmpdir:

                print("Using temporary directory ", tmpdir)

                # Save optimizer
                self.optimizer.save(tmpdir, delete_existing=True)

                # Create new optimizer and load
                new_optimizer = Optimizer.from_file(tmpdir)

                # Predict with optimizer
                res = self.optimizer.solve(self.df_test)

                # Predict with new_optimizer
                res_new = new_optimizer.solve(self.df_test)

                # Make sure predictions match
                for i in range(len(self.df_test)):
                    npt.assert_almost_equal(res[i]['x'],
                                            res_new[i]['x'],
                                            decimal=TOL)
                    npt.assert_almost_equal(res[i]['cost'],
                                            res_new[i]['cost'],
                                            decimal=TOL)
                    self.assertTrue(res[i]['strategy'] ==
                                    res_new[i]['strategy'])


if __name__ == '__main__':
    unittest.main()
