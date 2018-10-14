import unittest
import numpy as np
import cvxpy as cp
import mlopt as mo
from mlopt.settings import TOL


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
        prob = cp.Problem(cp.Minimize(c * x), [A * x <= b])
        mlprob = mo.OptimizationProblem(prob)

        # Set variable value
        x_val = 10 * np.random.randn(n)
        x.value = x_val

        # Check violation
        viol_cvxpy = mlprob.infeasibility()
        viol_manual = np.linalg.norm(np.maximum(A.dot(x_val) - b,
                                                0))
        assert abs(viol_cvxpy - viol_manual) <= TOL
