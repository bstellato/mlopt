import unittest
import numpy as np
import numpy.testing as npt
from mlopt import Optimizer, installed_learners
from mlopt.tests.settings import TEST_TOL as TOL
from mlopt.sampling import uniform_sphere_sample
import mlopt.settings as s
import tempfile
import os
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
        self.radius = 2.
        n = 1000   # Number of points
        n_test = 100

        # Define problem
        x = cp.Variable(T+1)
        u = cp.Variable(T)

        # Define parameter and sampling points
        d = cp.Parameter(T, nonneg=True, name="d")
        self.d_bar = 2. * np.ones(T)
        X_d = uniform_sphere_sample(self.d_bar, self.radius, n=n)
        X_d_test = uniform_sphere_sample(self.d_bar, self.radius, n=n_test)
        self.df = pd.DataFrame({'d': list(X_d)})
        self.df_test = pd.DataFrame({'d': list(X_d_test)})

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

        #  # Define learners
        #  self.learners = [
        #      #  s.OPTIMAL_TREE,  # Disable. Too slow
        #      s.PYTORCH
        #  ]

    def test_save_load_data(self):
        """Test save load data"""
        m = self.optimizer

        for learner in installed_learners():
            with tempfile.TemporaryDirectory() as tmpdir:
                data_file = os.path.join(tmpdir, "data.pkl")

                # Sample and store
                m.train(self.df,
                        #  sampling_fn=lambda n: sample(self.d_bar,
                        #                               self.radius,
                        #                               n),
                        filter_strategies=True,
                        parallel=True,
                        learner=learner,
                        n_train_trials=10,
                        #  params=nn_params
                        )
                store_general, store_detail = m.performance(self.df_test)

                # Save datafile
                m.save_training_data(data_file, delete_existing=True)

                # Create new optimizer, load data, train and
                # evaluate performance
                problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
                self.optimizer = Optimizer(problem)

                m = self.optimizer
                m.load_training_data(data_file)
                m.train(parallel=True,
                        learner=learner,
                        n_train_trials=10)
                load_general, load_detail = m.performance(self.df_test)

                # test same things
                npt.assert_almost_equal(store_general['max_infeas'],
                                        load_general['max_infeas'],
                                        decimal=1e-8)
                npt.assert_almost_equal(store_general['avg_infeas'],
                                        load_general['avg_infeas'],
                                        decimal=1e-8)
                npt.assert_almost_equal(store_general['max_subopt'],
                                        load_general['max_subopt'],
                                        decimal=1e-8)
                npt.assert_almost_equal(store_general['avg_subopt'],
                                        load_general['avg_subopt'],
                                        decimal=1e-8)
#
    #  def test_save_load(self):
    #      """Test save load"""
    #
    #      for learner in self.learners:
    #
    #          # Train optimizer
    #          self.optimizer.train(self.df, learner=learner)
    #
    #          # Create temporary directory where
    #          # to do stuff
    #          with tempfile.TemporaryDirectory() as tmpdir:
    #
    #              # Archive name
    #              file_name = os.path.join(tmpdir, learner + ".tar.gz")
    #
    #              # Save optimizer
    #              self.optimizer.save(file_name)
    #
    #              # Create new optimizer and load
    #              new_optimizer = Optimizer.from_file(file_name)
    #
    #              # Predict with optimizer
    #              res = self.optimizer.solve(self.df_test)
    #
    #              # Predict with new_optimizer
    #              res_new = new_optimizer.solve(self.df_test)
    #
    #              # Make sure predictions match
    #              for i in range(len(self.df_test)):
    #                  npt.assert_almost_equal(res[i]['x'],
    #                                          res_new[i]['x'],
    #                                          decimal=TOL)
    #                  npt.assert_almost_equal(res[i]['cost'],
    #                                          res_new[i]['cost'],
    #                                          decimal=TOL)
    #                  self.assertTrue(res[i]['strategy'] ==
    #                                  res_new[i]['strategy'])
    #
    #
if __name__ == '__main__':
    unittest.main()
