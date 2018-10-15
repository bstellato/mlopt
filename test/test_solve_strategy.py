import unittest
import numpy as np
import scipy.sparse as spa
import numpy.testing as npt
from .settings import TEST_TOL as TOL
from mlopt.problem import OptimizationProblem
from mlopt.strategy import Strategy
from mlopt.settings import DEFAULT_SOLVER
import cvxpy as cp


class TestSolveStrategy(unittest.TestCase):
    def test_small(self):
        """Test small continuous LP"""

        # Define problem
        c = np.array([-1, -2])
        x = cp.Variable(2, boolean=True)
        cost = c * x
        constraints = [x[1] <= 0.5 * x[0] + 1.5,
                       x[1] <= -0.5 * x[0] + 3.5,
                       x[1] <= -5.0 * x[0] + 10]
        problem = OptimizationProblem(cp.Minimize(cost), constraints)

        # Solve and compute strategy
        results = problem.solve(solver='MOSEK')
        violation1 = problem.infeasibility()

        # Solve just with strategy
        results_new = problem.solve_with_strategy(results['strategy'],
                                                  solver='MOSEK')
        violation2 = problem.infeasibility()

        # Verify both solutions are equal
        npt.assert_almost_equal(results['x'],
                                results_new['x'],
                                decimal=TOL)
        npt.assert_almost_equal(results['cost'],
                                results_new['cost'],
                                decimal=TOL)
        self.assertTrue(abs(violation1 - violation2) <= TOL)

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
        problem = OptimizationProblem(cp.Minimize(cost), constraints)

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
        npt.assert_almost_equal(results['x'],
                                results_new['x'],
                                decimal=TOL)
        npt.assert_almost_equal(results['cost'],
                                results_new['cost'],
                                decimal=TOL)

    def test_random_boolean(self):
        """Mixed-boolean random LP test"""

        # Seed for reproducibility
        np.random.seed(1)

        # Define problem
        n = 20
        m = 70

        # Define constraints
        v = np.random.rand(n)   # Solution
        A = spa.random(m, n, density=0.8,
                       data_rvs=np.random.randn,
                       format='csc')
        b = A.dot(v) + 10 * np.random.rand(m)

        # Split in 2 parts
        A1 = A[:int(m/2), :]
        b1 = b[:int(m/2)]
        A2 = A[int(m/2):, :]
        b2 = b[int(m/2):]

        # Cost
        c = np.random.rand(n)
        x = cp.Variable(n)  # Variable
        y = cp.Variable(m, boolean=True)  # Variable
        cost = c * x - cp.sum(y) + 3 * y[7]

        # Define constraints
        constraints = [A1 * x + y[:int(m/2)] <= b1,
                       A2 * x + y[int(m/2):] <= b2]

        # Problem
        problem = OptimizationProblem(cp.Minimize(cost), constraints)

        # Solve and compute strategy
        results = problem.solve(solver='MOSEK',
                                verbose=True
                                )

        # Solve just with strategy
        results_new = problem.solve_with_strategy(results['strategy'],
                                                  solver='MOSEK',
                                                  verbose=True
                                                  )

        # Verify both solutions are equal
        npt.assert_almost_equal(results['x'],
                                results_new['x'],
                                decimal=TOL)
        npt.assert_almost_equal(results['cost'],
                                results_new['cost'],
                                decimal=TOL)

    def test_small_inventory(self):
        # Generate data
        np.random.seed(1)
        T = 5
        M = 2.
        h = 1.
        c = 1.
        p = 1.
        x_init = 2.

        # Define problem
        x = cp.Variable(T+1)
        u = cp.Variable(T)

        # Explicitly define parameter
        d = np.array([3.94218985, 2.98861724,
                      2.48309709, 1.91226946,
                      2.33123841])

        # Constaints
        constraints = [x[0] == x_init]
        for t in range(T):
            constraints += [x[t+1] == x[t] + u[t] - d[t]]
        constraints += [u >= 0, u <= M]

        # Objective
        # TODO: If you remove that part it reports a crappy solution
        cost = cp.sum(cp.maximum(h * x, -p * x)) + c * cp.sum(u)

        # Define problem
        problem = OptimizationProblem(cp.Minimize(cost), constraints)
        results = problem.solve(solver=DEFAULT_SOLVER)

        # NB. This is the strategy that you would get if
        #     you do not perturb the cost.
        #  int_vars = {}
        #  binding_constraints = {constraints[0].id: np.array([1]),
        #                         constraints[1].id: np.array([1]),
        #                         constraints[2].id: np.array([1]),
        #                         constraints[3].id: np.array([1]),
        #                         constraints[4].id: np.array([1]),
        #                         constraints[5].id: np.array([1]),
        #                         constraints[6].id: np.array([0, 0, 0, 0, 0]),
        #                         constraints[7].id: np.array([1, 1, 1, 1, 0])
        #                         }
        #  strategy = Strategy(binding_constraints, int_vars)

        # Solve with strategy!
        results_strategy = problem.solve_with_strategy(results['strategy'])

        # TODO: Solve issue!
        # Correct strategy but variable is infeasible for original problem.
        # Need to rethink how we choose the strategy!
        self.assertTrue(problem.infeasibility() >= 0)
        self.assertTrue(problem.infeasibility() <= TOL)
        self.assertTrue(abs(results['cost'] - results_strategy['cost']) <= TOL)
