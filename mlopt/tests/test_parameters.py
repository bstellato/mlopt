import unittest
import cvxpy as cp
import numpy as np
import mlopt
from mlopt.sampling import uniform_sphere_sample
import pandas as pd
import numpy.testing as npt
import logging
from mlopt.tests.settings import TEST_TOL as TOL


class TestParameters(unittest.TestCase):

    def test_parameters_in_matrices(self):
        """Check if parameters in matrices are recognized
        """
        # Problem data.
        m = 2
        n = 1
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        gamma = cp.Parameter(nonneg=True)
        gamma.value = 0.8
        theta = cp.Parameter(nonneg=True)
        theta.value = 8.9
        x = cp.Variable(n)
        objective = cp.Minimize(gamma * cp.sum_squares(A @ x - b) +
                                cp.norm(x, 1))
        constraints = [0 <= x, x <= theta]
        problem = cp.Problem(objective, constraints)
        m = mlopt.Optimizer(problem)

        self.assertTrue(m._problem.parameters_in_matrices)

    def test_parameters_in_matrices2(self):
        """Check if parameters in matrices are recognized
        """
        # Problem data.
        m = 2
        n = 1
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        gamma = cp.Parameter(nonneg=True)
        gamma.value = 0.8
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A @ x - b) +
                                cp.norm(x, 1))
        constraints = [0 <= x, gamma * x <= 1]
        problem = cp.Problem(objective, constraints)
        m = mlopt.Optimizer(problem)

        self.assertTrue(m._problem.parameters_in_matrices)

    def test_parameters_in_vectors(self):
        """Check if parameters not in matrices are recognized
        """
        # Problem data.
        m = 2
        n = 1
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        theta = cp.Parameter(nonneg=True)
        theta.value = 8.9
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A @ x - b) + cp.norm(x, 1))
        constraints = [0 <= x, x <= theta]
        problem = cp.Problem(objective, constraints)
        m = mlopt.Optimizer(problem)

        self.assertFalse(m._problem.parameters_in_matrices)
