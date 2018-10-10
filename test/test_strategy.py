import unittest
import scipy.sparse as spa
import numpy as np
import numpy.testing as npt
from mlopt.problem import OptimizationProblem
from mlopt.utils import problem_data
import cvxpy as cp


class TestStrategy(unittest.TestCase):
    def test_compute(self):

        # Define problem
        c = -1 * np.ones(2)
        x = cp.Variable(2, boolean=True)
        cost = c * x
        constraints = [x[1] <= 0.5 * x[0] + 1.5,
                       x[1] <= -0.5 * x[0] + 3.5,
                       x[1] <= -5.0 * x[0] + 10]
        cp_problem = cp.Problem(cp.Minimize(cost), constraints)
        problem = OptimizationProblem(cp_problem)

        # Solve and compute strategy
        results = problem.solve(solver='MOSEK')

        # Solve just with strategy
        results_new = problem.solve_with_strategy(results['strategy'],
                                                  solver='MOSEK')

        # Verify both solutions are equal
        npt.assert_almost_equal(results['x'], results_new['x'])
        npt.assert_almost_equal(results['cost'], results_new['cost'])
