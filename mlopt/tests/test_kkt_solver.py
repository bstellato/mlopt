import unittest
import cvxpy as cp
from cvxpy.error import SolverError
from mlopt.kkt import KKT
from mlopt.tests.settings import TEST_TOL as TOL
import numpy as np
import numpy.testing as npt
import scipy.sparse as spa


class TestKKT(unittest.TestCase):

    def setUp(self):
        n = 10
        m = 10
        P = spa.random(n, n, density=0.5,
                       data_rvs=np.random.randn,
                       format='csc')
        self.P = P.dot(P.T).tocsc()
        self.q = np.random.randn(n)
        self.A = spa.random(m, n, density=0.5,
                            data_rvs=np.random.randn,
                            format='csc')
        self.b = np.random.randn(m)
        self.n = n
        self.m = m

    def test_solution(self):
        """Test solution of KKT solver
           compared to another solver
        """

        # Define equality constrained QP
        x = cp.Variable(self.n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, self.P) + self.q.T * x),
                          [self.A * x == self.b])
        obj_solver = prob.solve(solver=cp.ECOS)
        x_solver = x.value
        obj_kkt = prob.solve(solver=KKT)
        x_kkt = x.value

        npt.assert_almost_equal(x_solver,
                                x_kkt,
                                decimal=TOL)
        npt.assert_almost_equal(obj_solver,
                                obj_kkt,
                                decimal=TOL)

    def test_not_applicable(self):
        """
        Test that it complains if problem is not an equality
        constrained QP.
        """
        x = cp.Variable(self.n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, self.P) + self.q.T * x),
                          [self.A * x <= self.b])
        with npt.assert_raises(SolverError):
            prob.solve(solver=KKT)
