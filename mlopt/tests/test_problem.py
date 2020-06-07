import unittest
import numpy as np
import numpy.testing as npt
import cvxpy as cp
from mlopt.problem import Problem
from mlopt.settings import DEFAULT_SOLVER
from mlopt.tests.settings import TEST_TOL as TOL
from copy import deepcopy


class TestProblem(unittest.TestCase):
    def setUp(self):
        pass

    def test_violation(self):
        """Test problem violation"""

        np.random.seed(1)

        # Define problem
        n = 5
        m = 5
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        prob_cvxpy = cp.Problem(cp.Minimize(c @ x), [A @ x <= b])
        mlprob = Problem(prob_cvxpy)
        data, _, _ = prob_cvxpy.get_problem_data(solver=DEFAULT_SOLVER)

        # Set variable value
        x_val = 10 * np.random.randn(n)
        x.value = x_val

        # Check violation
        viol_cvxpy = mlprob.infeasibility(x_val, data)
        viol_manual = np.linalg.norm(np.maximum(A.dot(x_val) - b, 0), np.inf)

        self.assertTrue(abs(viol_cvxpy - viol_manual) <= TOL)

    def test_solve_cvxpy(self):
        """Solve cvxpy problem vs optimizer problem.
           Expect similar solutions."""
        np.random.seed(1)
        n = 5
        m = 15
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        cost = c @ x
        constraints = [A @ x <= b]

        cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
        cvxpy_problem.solve(solver=DEFAULT_SOLVER)
        x_cvxpy = deepcopy(x.value)
        cost_cvxpy = cost.value
        problem = Problem(cvxpy_problem)
        problem.solve()
        x_problem = x.value
        cost_problem = cost.value

        npt.assert_almost_equal(x_problem, x_cvxpy, decimal=TOL)
        npt.assert_almost_equal(cost_problem, cost_cvxpy, decimal=TOL)


if __name__ == "__main__":
    unittest.main()
