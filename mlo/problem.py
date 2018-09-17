# Define optimization problem
import numpy as np
import scipy.sparse as spa
from . import constants as con
from solvers.solvers import SOLVER_MAP
from .strategy import Strategy
import tqdm


class ProblemData(object):
    def __init__(self, c, l, A, u, int_idx=np.array([])):
        self.c = c
        self.A = A
        self.l = l
        self.u = u
        self.int_idx = int_idx

    def eq_ineq(self):
        """
        Return equality and inequality constraints
        """
        n_con = len(self.l)
        eq = np.where(self.u - self.l) <= con.TOL
        ineq = np.array(set(range(n_con)) - set(eq))
        return eq, ineq


class OptimizationProblem(object):
    def is_mip(self):
        return len(self.data) > 0

    def cost(self, x):
        return np.dot(self.data.c, x)

    def solve(self, solver, settings={}):
        """
        Solve optimization problem

        Returns
        -------
        numpy array
            Solution.
        float
            Time.
        strategy
            Strategy.
        """
        s = SOLVER_MAP[solver](settings)
        results = s.solve(self)
        x_opt = results.x
        time = results.run_time
        x_int = results.x[self.data.int_idx]

        # Get strategy
        strategy = Strategy(x_int, results.active_constraints)

        return x_opt, time, strategy

    def solve_parametric(self, theta):
        """
        Solve parametric problems

        Args:
            theta (DataFrame): parameter values
            problem (Optimizationproblem): optimization problem to solve

        Returns:
            x (numpy array list): solutions
            time (float list): computation times
            strategy (Strategy list): strategies
        """
        n = len(theta)  # Number of points

        # Preallocate solutions
        x = [None for i in range(n)]
        time = [None for i in range(n)]
        strategy = [None for i in range(n)]

        for i in tqdm(range(n)):
            self.populate(theta[i, :])
            x[i], time[i], strategy[i] = self.solve()

        return x, time, strategy

    def solve_with_strategy(self, strategy, solver, settings={}):
        """
        Solve problem using strategy

        Parameters
        ----------
        strategy : Strategy
            Strategy to be used.
        solver : string
            Solver to use as defined in solvers.py

        Returns
        -------
        numpy array
            Solution.
        float
            Time.
        """

        c, l, A, u = self.data.c, self.data.l, self.data.A, self.data.u
        int_idx, active_constr = strategy.int_vars, strategy.active_constraints

        if self.is_mip():
            assert np.max(strategy.int_vars) <= len(self.data.l)
            assert np.min(strategy.int_vars) >= 0

        n_var = len(c)

        # Create equivalent problem
        # 1) Fix integer variables
        A_red = spa.eye(n_var)[int_idx, :]
        bound_red = int_idx

        # 2) Use active constraints
        active_constraints_upper = np.where(active_constr == 1)
        #  n_upper = len(active_constraints_upper)
        active_constraints_lower = np.where(active_constr == -1)
        #  n_lower = len(active_constraints_lower)
        A_upper = A[active_constraints_upper, :]
        u_upper = u[active_constraints_upper]
        A_lower = A[active_constraints_lower, :]
        l_lower = l[active_constraints_lower]
        A_red = spa.vstack([A_red, A_lower, A_upper])
        bound_red = np.vstack([bound_red, l_lower, u_upper])

        # 3) Create problem
        problem = OptimizationProblem()
        problem.data = ProblemData(c, bound_red, A_red, bound_red)

        # Solve problem
        x, time, _ = problem.solve(solver=solver, settings=settings)

        return x, time
