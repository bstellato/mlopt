import unittest
import scipy.sparse as spa
import numpy as np
import numpy.testing as npt
from mlo.problem import OptimizationProblem
from mlo.utils import problem_data


class TestStrategy(unittest.TestCase):
    def test_compute(self):

        # Define problem
        c = np.ones(2)
        l = np.array([10., -100., -2.])
        u = np.array([10., 5., 100.])
        A = spa.csc_matrix([[2., 1.],
                            [0.5, -1.],
                            [0., 1.]])
        int_idx = np.array([0])
        problem = OptimizationProblem()
        problem.data = problem_data(c, l, A, u, int_idx)

        # Solve and compute strategy
        x_opt, time, strategy = problem.solve()

        # Solve just with strategy
        x_new, time_new = problem.solve_with_strategy(strategy)

        # Verify both solutions are equal
        npt.assert_almost_equal(x_opt, x_new)
