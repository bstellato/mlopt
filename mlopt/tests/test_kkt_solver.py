import unittest
import cvxpy as cp
from cvxpy.error import SolverError
from mlopt.kkt import KKTSolver
from mlopt.tests.settings import TEST_TOL as TOL
from mlopt.strategy import Strategy
import mlopt.settings as stg
import numpy as np
import numpy.testing as npt
import scipy.sparse as spa


class TestKKT(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
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
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, self.P) + self.q.T @ x),
                          [self.A @ x == self.b])
        obj_solver = prob.solve(solver=stg.DEFAULT_SOLVER)
        x_solver = np.copy(x.value)
        y_solver = np.copy(prob.constraints[0].dual_value)

        # Solve using KKT solver
        data, chain, inverse_data = \
            prob.get_problem_data(solver=stg.DEFAULT_SOLVER)
        # Get Strategy
        strategy = Strategy(x_solver, data)

        # Apply strategy and solve with KKT solver
        strategy.apply(data, inverse_data[-1])
        solver = KKTSolver()
        raw_solution = solver.solve_via_data(data, warm_start=True,
                                             verbose=True, solver_opts={})
        inv_solution = solver.invert(raw_solution, inverse_data[-1])
        x_kkt = inv_solution.primal_vars[KKTSolver.VAR_ID]
        y_kkt = inv_solution.dual_vars[KKTSolver.DUAL_VAR_ID]
        obj_kkt = raw_solution['cost']

        # Assert matching
        npt.assert_almost_equal(x_solver,
                                x_kkt,
                                decimal=TOL)
        npt.assert_almost_equal(y_solver,
                                y_kkt,
                                decimal=TOL)
        npt.assert_almost_equal(obj_solver,
                                obj_kkt,
                                decimal=TOL)

    #  def test_not_applicable(self):
    #      """
    #      Test that it complains if problem is not an equality
    #      constrained QP.
    #      """
    #      x = cp.Variable(self.n)
    #      prob = cp.Problem(cp.Minimize(cp.quad_form(x, self.P) +
    #                                    self.q.T * x),
    #                        [self.A * x <= self.b])
    #      with npt.assert_raises(SolverError):
    #          prob.solve(solver=KKT)
