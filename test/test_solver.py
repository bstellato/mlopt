import unittest
import scipy.sparse as spa
import numpy as np
import numpy.testing as npt
from mlo.problem import OptimizationProblem
from mlo.utils import problem_data


class TestSolver(unittest.TestCase):
    def test_solve(self):

        # Define problem
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
        x_opt_cplex, time_cplex, strategy_cplex = problem.solve(solver='CPLEX',
                                                                #  settings={'verbose': True}
                                                                )
        x_opt_mosek, time_mosek, strategy_mosek = problem.solve(solver='MOSEK',
                                                                #  settings={'verbose': True}
                                                                )

        # Verify both solutions are equal
        npt.assert_almost_equal(x_opt_cplex, x_opt_mosek)
        assert strategy_cplex == strategy_mosek
