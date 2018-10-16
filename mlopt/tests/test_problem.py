import unittest
import numpy as np
import numpy.testing as npt
import cvxpy as cp
import mlopt as mo
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
        mlprob = mo.Problem(cp.Minimize(c * x), [A * x <= b])

        # Set variable value
        x_val = 10 * np.random.randn(n)
        x.value = x_val

        # Check violation
        viol_cvxpy = mlprob.infeasibility()
        viol_manual = np.linalg.norm(np.maximum(A.dot(x_val) - b,
                                                0))
        self.assertTrue(abs(viol_cvxpy - viol_manual) <= TOL)

    def test_solve_perturbation(self):
        """Solve cvxpy problem vs perturbed problem.
           Expect similar solutions."""
        np.random.seed(1)
        n = 5
        m = 15
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        cost = c * x
        constraints = [A * x <= b]

        cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
        cvxpy_problem.solve(solver=DEFAULT_SOLVER, verbose=True)
        x_cvxpy = deepcopy(x.value)
        cost_cvxpy = cost.value
        problem = mo.Problem(cp.Minimize(cost), constraints)
        problem.solve(verbose=True)
        x_problem = deepcopy(x.value)
        cost_problem = cost.value

        npt.assert_almost_equal(x_problem,
                                x_cvxpy,
                                decimal=TOL)
        npt.assert_almost_equal(cost_problem,
                                cost_cvxpy,
                                decimal=TOL)


if __name__ == '__main__':
    unittest.main()
