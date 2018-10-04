import unittest
import scipy.sparse as spa
import numpy as np
import numpy.testing as npt
from mlopt.problem import OptimizationProblem
from mlopt.utils import problem_data


class TestSolver(unittest.TestCase):
    def test_solve(self):

        # Define problem
        np.random.seed(1)
        n = 100
        m = 200
        c = np.random.randn(n)
        A = spa.random(m, n, density=0.8, format='csc')
        l = -1 * np.random.rand(m)
        u = 1 * np.random.rand(m)
        int_idx = np.array([0, 1, 50])
        problem = OptimizationProblem()
        problem.data = problem_data(c, l, A, u, int_idx)

        # Solve
        #  x_opt_cplex, time_cplex, strategy_cplex = \
        #      problem.solve(solver='CPLEX',
        #                    settings={'verbose': True}
        #                    )
        x_opt_mosek, time_mosek, strategy_mosek = \
            problem.solve(solver='MOSEK',
                          settings={'verbose': True}
                          )
        x_opt_gurobi, time_gurobi, strategy_gurobi = \
            problem.solve(solver='GUROBI',
                          settings={'verbose': True}
                          )

        # Verify both solutions are equal
        #  npt.assert_almost_equal(x_opt_cplex, x_opt_mosek)
        #  assert strategy_cplex == strategy_mosek
        npt.assert_almost_equal(x_opt_gurobi, x_opt_mosek)
        assert strategy_gurobi == strategy_mosek
