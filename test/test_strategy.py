import unittest
import numpy as np
import scipy.sparse as spa
import numpy.testing as npt
from mlopt.problem import OptimizationProblem
import cvxpy as cp


class TestStrategy(unittest.TestCase):
    def test_small(self):
        """Test small continuous LP"""

        # Define problem
        c = np.array([-1, -2])
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


    def test_random_cont(self):
        """Test random continuous LP test"""

        # Seed for reproducibility
        np.random.seed(1)

        # Define problem
        n = 100
        m = 250

        # Define constraints
        v = np.random.rand(n)   # Solution
        A = spa.random(m, n, density=0.8,
                       data_rvs=np.random.randn,
                       format='csc')
        b = A.dot(v) + np.random.rand(m)

        # Split in 2 parts
        A1 = A[:int(m/2), :]
        b1 = b[:int(m/2)]
        A2 = A[int(m/2):, :]
        b2 = b[int(m/2):]

        # Cost
        c = np.random.rand(n)
        x = cp.Variable(n)  # Variable
        cost = c * x

        # Define constraints
        constraints = [A1 * x <= b1,
                       A2 * x <= b2]

        # Problem
        cp_problem = cp.Problem(cp.Minimize(cost), constraints)
        problem = OptimizationProblem(cp_problem)

        # Solve and compute strategy
        results = problem.solve(solver='MOSEK',
                                #  verbose=True
                                )


        # Solve just with strategy
        results_new = problem.solve_with_strategy(results['strategy'],
                                                  solver='MOSEK',
                                                  #  verbose=True
                                                  )

        # Verify both solutions are equal
        npt.assert_almost_equal(results['x'], results_new['x'])
        npt.assert_almost_equal(results['cost'], results_new['cost'])

    #  def test_random_integer(self):
    #      """Mixed-integer random LP test"""
