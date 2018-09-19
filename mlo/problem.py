# Define optimization problem
import numpy as np
import scipy.sparse as spa
from . import constants as con
from .solvers.solvers import SOLVER_MAP, DEFAULT_SOLVER
from .solvers.statuses import SOLUTION_PRESENT
from .strategy import Strategy
from .constants import TOL
from tqdm import tqdm


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

    def cost(self, x):
        """Return cost function value"""
        return self.c.dot(x)


class OptimizationProblem(object):
    def is_mip(self):
        """Check if problem has integer variables."""
        return len(self.data.int_idx) > 0

    def infeasibility(self, x):
        """
        Compute infeasibility for vector x
        """
        l, A, u = self.data.l, self.data.A, self.data.u

        norm_A = [spa.linalg.norm(A[i, :]) for i in range(A.shape[0])]

        upper = np.maximum(A.dot(x) - u, 0.)
        lower = np.maximum(l - A.dot(x), 0.)

        # Normalize nonzero ones
        for i in range(len(upper)):
            if upper[i] >= TOL:
                upper[i] /= np.maximum(norm_A[i], np.abs(u[i]))
        for i in range(len(lower)):
            if lower[i] >= TOL:
                lower[i] /= np.maximum(norm_A[i], np.abs(l[i]))

        return np.linalg.norm(upper + lower)

    def suboptimality(self, x, x_opt):
        """
        Compute suboptimality for vector x
        """
        c = self.data.c
        return (self.data.cost(x) - self.data.cost(x.opt)) / np.linalg.norm(c, np.inf)

    def cost(self, x):
        return self.data.cost(x)

    def solve(self, solver=DEFAULT_SOLVER, settings={}):
        """
        Solve optimization problem

        Parameters
        ----------
        solver : string, optional
            Solver to be used. Default Mosek.
        settings : dict, optional
            Solver settings. Default empty.

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

        # DEBUG CHECK OTHER SOLVER
        #  s = SOLVER_MAP['GUROBI'](settings)
        #  res_gurobi = s.solve(self)
        #  assert np.linalg.norm(self.cost(results.x) - self.cost(res_gurobi.x)) <= 1e-05

        if results.status not in SOLUTION_PRESENT:
            import cvxpy as cvx
            x = cvx.Variable(len(self.data.c))
            problem = cvx.Problem(cvx.Minimize(self.data.c * x),
                                  [self.data.A * x <= self.data.u,
                                   self.data.A * x >= self.data.l])
            problem.solve(cvx.GUROBI, verbose=True)
            import ipdb; ipdb.set_trace()
            raise ValueError('Problem not solved. Status %s' % results.status)

        x_opt = results.x
        time = results.run_time
        x_int = np.array([], dtype=int)
        if self.is_mip():
            x_int = results.x[self.data.int_idx]

        # Get strategy
        strategy = Strategy(x_int, results.active_constraints)

        return x_opt, time, strategy

    def solve_parametric(self, theta, solver=DEFAULT_SOLVER, settings={}):
        """
        Solve parametric problems

        Parameters
        ----------
        theta : DataFrame
            parameter values
        problem : Optimizationproblem
            optimization problem to solve
        solver : string, optional
            Solver to be used. Default Mosek.
        settings : dict, optional
            Solver settings. Default empty.

        Returns
        -------
        x : numpy array list
            solutions
        time : float list
            computation times
        strategy : Strategy list
            strategies
        """
        n = len(theta)  # Number of points

        # Preallocate solutions
        x = [None for i in range(n)]
        time = [None for i in range(n)]
        strategy = [None for i in range(n)]

        for i in tqdm(range(n), desc="Solving for all theta..."):
            self.populate(theta.iloc[i, :])
            x[i], time[i], strategy[i] = self.solve(solver, settings)

        return x, time, strategy

    def solve_with_strategy(self, strategy, solver=DEFAULT_SOLVER,
                            settings={}):
        """
        Solve problem using strategy

        Parameters
        ----------
        strategy : Strategy
            Strategy to be used.
        solver : string, optional
            Solver to be used. Default Mosek.
        settings : dict, optional
            Solver settings. Default empty.

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
        active_constraints_lower = np.where(active_constr == -1)
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
