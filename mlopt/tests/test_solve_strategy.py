import unittest
import mlopt
import numpy as np
import scipy.sparse as spa
import numpy.testing as npt
from mlopt.tests.settings import TEST_TOL as TOL
from mlopt.problem import Problem
import cvxpy as cp


class TestSolveStrategy(unittest.TestCase):
    def test_small(self):
        """Test small continuous LP"""

        # Define problem
        c = np.array([-1, -2])
        x = cp.Variable(2, integer=True)
        #  x = cp.Variable(2)
        cost = c @ x
        constraints = [
            x[1] <= 0.5 * x[0] + 1.5,
            x[1] <= -0.5 * x[0] + 3.5,
            x[1] <= -5.0 * x[0] + 10,
            x >= 0, x <= 1,
        ]
        cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
        problem = Problem(cvxpy_problem)

        # Solve and compute strategy
        results = problem.solve()
        #  violation1 = problem.infeasibility()

        # Solve just with strategy
        results_new = problem.solve(strategy=results["strategy"])
        #  violation2 = problem.infeasibility()

        # Verify both solutions are equal
        npt.assert_almost_equal(results["x"], results_new["x"], decimal=TOL)
        npt.assert_almost_equal(
            results["cost"], results_new["cost"], decimal=TOL
        )
        #  self.assertTrue(abs(violation1 - violation2) <= TOL)

    def test_random_cont(self):
        """Test random continuous LP test"""

        # Seed for reproducibility
        np.random.seed(1)

        # Define problem
        n = 100
        m = 250

        # Define constraints
        v = np.random.rand(n)  # Solution
        A = spa.random(
            m, n, density=0.8, data_rvs=np.random.randn, format="csc"
        )
        b = A.dot(v) + np.random.rand(m)

        # Split in 2 parts
        A1 = A[: int(m / 2), :]
        b1 = b[: int(m / 2)]
        A2 = A[int(m / 2):, :]
        b2 = b[int(m / 2):]

        # Cost
        c = np.random.rand(n)
        x = cp.Variable(n)  # Variable
        cost = c @ x

        # Define constraints
        constraints = [A1 @ x <= b1, A2 @ x <= b2]

        # Problem
        cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
        problem = Problem(cvxpy_problem)

        # Solve and compute strategy
        results = problem.solve()

        # Solve just with strategy
        results_new = problem.solve(strategy=results["strategy"])

        # Verify both solutions are equal
        npt.assert_almost_equal(results["x"], results_new["x"], decimal=TOL)
        npt.assert_almost_equal(
            results["cost"], results_new["cost"], decimal=TOL
        )

    def test_small_inventory(self):
        # Generate data
        np.random.seed(1)
        T = 5
        M = 2.0
        h = 1.0
        c = 1.0
        p = 1.0
        x_init = 2.0

        # Define problem
        x = cp.Variable(T + 1)
        u = cp.Variable(T)
        t = cp.Variable(T + 1)

        # Explicitly define parameter
        d = np.array(
            [3.94218985, 2.98861724, 2.48309709, 1.91226946, 2.33123841]
        )

        # Constaints
        constraints = [x[0] == x_init]
        for t in range(T):
            constraints += [x[t + 1] == x[t] + u[t] - d[t]]
        constraints += [u >= 0, u <= M]

        # Maximum
        constraints += [t >= h * x, t >= -p * x]

        # Objective
        cost = cp.sum(t) + c * cp.sum(u)

        # Define problem
        cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
        problem = Problem(cvxpy_problem)
        results = problem.solve()

        # Solve with strategy!
        results_strategy = problem.solve(strategy=results["strategy"])

        # Verify both solutions are equal
        npt.assert_almost_equal(
            results["x"], results_strategy["x"], decimal=TOL
        )
        npt.assert_almost_equal(
            results["cost"], results_strategy["cost"], decimal=TOL
        )

    def test_random_integer(self):
        """Mixed-integer random LP test"""

        # Seed for reproducibility
        np.random.seed(1)

        # Define problem
        n = 20
        m = 70

        # Define constraints
        v = np.random.rand(n)  # Solution
        A = spa.random(
            m, n, density=0.8, data_rvs=np.random.randn, format="csc"
        )
        b = A.dot(v) + 10 * np.random.rand(m)

        # Split in 2 parts
        A1 = A[: int(m / 2), :]
        b1 = b[: int(m / 2)]
        A2 = A[int(m / 2):, :]
        b2 = b[int(m / 2):]

        # Cost
        c = np.random.rand(n)
        x = cp.Variable(n)  # Variable
        y = cp.Variable(integer=True)  # Variable
        cost = c @ x - cp.sum(y) + y

        # Define constraints
        constraints = [A1 @ x - y <= b1, A2 @ x + y <= b2, y >= 2]

        # Problem
        cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
        problem = Problem(cvxpy_problem)

        # Solve and compute strategy
        results = problem.solve()

        # Solve just with strategy
        results_new = problem.solve(strategy=results["strategy"])

        # Verify both solutions are equal
        npt.assert_almost_equal(results["x"], results_new["x"], decimal=TOL)
        npt.assert_almost_equal(
            results["cost"], results_new["cost"], decimal=TOL
        )

    def test_random_reform_integer(self):
        """Mixed-integer random reformulated LP test"""

        # Seed for reproducibility
        np.random.seed(1)

        # Define problem
        n = 20
        m = 70

        # Define constraints
        v = np.random.rand(n)  # Solution
        A = spa.random(
            m, n, density=0.8, data_rvs=np.random.randn, format="csc"
        )
        b = A.dot(v) + 10 * np.random.rand(m)

        # Split in 2 parts
        A1 = A[: int(m / 2), :]
        b1 = b[: int(m / 2)]
        A2 = A[int(m / 2):, :]
        b2 = b[int(m / 2):]

        # Cost
        c = np.random.rand(n)
        x = cp.Variable(n)  # Variable
        y = cp.Variable(integer=True)  # Variable
        cost = c @ x - cp.sum(y) + y + 0.1 * cp.sum(cp.pos(x))

        # Define constraints
        constraints = [A1 @ x - y <= b1, A2 @ x + y <= b2, y >= 2]

        # Problem
        cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
        problem = Problem(cvxpy_problem)
        results = problem.solve()

        # Solve just with strategy
        results_new = problem.solve(strategy=results["strategy"])

        # Verify both solutions are equal
        npt.assert_almost_equal(results["x"], results_new["x"], decimal=TOL)
        npt.assert_almost_equal(
            results["cost"], results_new["cost"], decimal=TOL
        )

    def test_random_cont_qp_reform(self):
        """Test random continuous QP reform test"""

        # Seed for reproducibility
        np.random.seed(1)

        # Define problem
        n = 100
        m = 250

        # Define constraints
        v = np.random.rand(n)  # Solution
        A = spa.random(
            m, n, density=0.8, data_rvs=np.random.randn, format="csc"
        )
        b = A.dot(v) + np.random.rand(m)

        # Split in 2 parts
        A1 = A[: int(m / 2), :]
        b1 = b[: int(m / 2)]
        A2 = A[int(m / 2):, :]
        b2 = b[int(m / 2):]

        # Cost
        c = np.random.rand(n)
        x = cp.Variable(n)  # Variable
        cost = cp.sum_squares(c @ x) + cp.norm(x, 1)

        # Define constraints
        constraints = [A1 @ x <= b1, A2 @ x <= b2]

        # Problem
        cvxpy_problem = cp.Problem(cp.Minimize(cost), constraints)
        problem = Problem(cvxpy_problem)

        # Solve and compute strategy
        results = problem.solve()

        # Solve just with strategy
        results_new = problem.solve(strategy=results["strategy"])

        # Verify both solutions are equal
        npt.assert_almost_equal(results["x"], results_new["x"], decimal=TOL)
        npt.assert_almost_equal(
            results["cost"], results_new["cost"], decimal=TOL
        )


if __name__ == "__main__":
    unittest.main()
