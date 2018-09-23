from multiprocessing import Pool, cpu_count
from itertools import repeat
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
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

    def is_mip(self):
        """Check if problem has integer variables."""
        return len(self.int_idx) > 0


class OptimizationProblem(object):

    def cost(self, x):
        """Compute cost function value"""
        return self.data.cost(x)

    def is_mip(self):
        """Check if problem has integer variables."""
        return self.data.is_mip()

    def infeasibility(self, x):
        """
        Compute infeasibility for vector x
        """
        l, A, u = self.data.l, self.data.A, self.data.u

        norm_A = [spla.norm(A[i, :], np.inf) for i in range(A.shape[0])]

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
        return (self.cost(x) - self.cost(x_opt)) / \
            np.linalg.norm(c, np.inf)

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
        results = SOLVER_MAP[solver](settings).solve(self.data)

        # DEBUG CHECK OTHER SOLVER
        #  res_gurobi = SOLVER_MAP['GUROBI'](settings).solve(self.data)
        #  assert np.linalg.norm(self.cost(results.x) -
        #                        self.cost(res_gurobi.x)) <= 1e-05

        # DEBUG Solve with CVXPY
        #  self.problem.solve()
        #  if not self.is_mip():
        #      x_cvxpy = np.concatenate((self.vars['y'].value,
        #                                self.vars['u'].value,
        #                                self.vars['x'].value))
        #  else:
        #      x_cvxpy = np.concatenate((self.vars['y'].value,
        #                                self.vars['u'].value,
        #                                self.vars['v'].value,
        #                                self.vars['x'].value))
        #
        #  #  import ipdb; ipdb.set_trace()
        #  if np.linalg.norm(x_cvxpy - results.x) > TOL:
        #      print("Solutions are different of %.2e" % np.linalg.norm(x_cvxpy -
        #          results.x))
        #      print(x_cvxpy)
        #      print(results.x)
        #

        n_constr = len(self.data.u)  # Number of constraints
        if results.status not in SOLUTION_PRESENT:
            #  import cvxpy as cvx
            #  x = cvx.Variable(len(self.data.c))
            #  problem = cvx.Problem(cvx.Minimize(self.data.c * x),
            #                        [self.data.A * x <= self.data.u,
            #                         self.data.A * x >= self.data.l])
            #  problem.solve(cvx.GUROBI, verbose=True)
            raise ValueError('Problem not solved. Status %s' % results.status)

        x_opt = results.x
        time = results.run_time
        x_int = np.array([], dtype=int)
        if self.is_mip():
            x_int = np.round(results.x[self.data.int_idx])

            # If mixed-integer, solve continuous restriction and get basis
            # Create continuous restriction
            c_cont = self.data.c
            n_var = len(c_cont)
            A_cont = spa.vstack([self.data.A,
                                 spa.eye(n_var,
                                         format='csc')[self.data.int_idx, :]])
            u_cont = np.concatenate((self.data.u, x_int))
            l_cont = np.concatenate((self.data.l, x_int))
            data_cont = ProblemData(c_cont, l_cont, A_cont, u_cont)

            # Solve and get active constraints
            results_cont = SOLVER_MAP[solver](settings).solve(data_cont)
            if results_cont.status not in SOLUTION_PRESENT:
                raise ValueError('Continuous restriction problem not ' +
                                 'solved. Status %s' % results_cont.status)
            # Get only active constraints of the original problem (ignore x_int
            # fixing)
            active_constraints = results_cont.active_constraints[:n_constr]
        else:
            # Continuous problem. Get active constraints directly
            active_constraints = results.active_constraints

        # Get strategy
        strategy = Strategy(x_int, active_constraints)

        return x_opt, time, strategy

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

        c = self.data.c
        l, A, u = self.data.l, self.data.A, self.data.u
        int_idx = self.data.int_idx
        int_vars = strategy.int_vars
        active_constr = strategy.active_constraints

        if self.is_mip():
            assert np.max(int_vars) <= len(self.data.l)
            assert np.min(int_vars) >= 0

        n_var = c.size

        # Create equivalent problem
        A_red = spa.csc_matrix((0, n_var))
        bound_red = np.array([])

        # 1) Use active constraints
        active_constraints_upper = np.where(active_constr == 1)[0]
        active_constraints_lower = np.where(active_constr == -1)[0]
        if len(active_constraints_upper) > 0:
            A_upper = A[active_constraints_upper, :]
            u_upper = u[active_constraints_upper]
            A_red = spa.vstack([A_red, A_upper])
            bound_red = np.concatenate((bound_red, u_upper))

        if len(active_constraints_lower) > 0:
            A_lower = A[active_constraints_lower, :]
            l_lower = l[active_constraints_lower]
            A_red = spa.vstack([A_red, A_lower])
            bound_red = np.concatenate((bound_red, l_lower))

        # 2) If integer program, fix integer variable
        if self.is_mip():
            A_red = spa.vstack([A_red,
                                spa.eye(n_var, format='csc')[int_idx, :]])
            bound_red = np.concatenate((bound_red, int_vars))

        # 3) Create problem
        problem = OptimizationProblem()
        problem.data = ProblemData(c, bound_red, A_red, bound_red)

        # Solve problem
        x, time, _ = problem.solve(solver=solver, settings=settings)

        return x, time

    def _populate_and_solve(self, args):
        """Single function to populate the problem with
           theta and solve it with the solver. Useful for
           multiprocessing."""
        theta, solver, settings = args
        self.populate(theta)
        results = self.solve(solver, settings)
        #  self.pbar.update()
        return results

    def solve_parametric(self, theta,
                         solver=DEFAULT_SOLVER, settings={},
                         message="Solving for all theta",
                         parallel=False  # Solve problems in parallel
                         ):
        """
        Solve parametric problems for each value of theta.

        Parameters
        ----------
        theta : DataFrame
            Parameter values.
        problem : Optimizationproblem
            Optimization problem to solve.
        solver : string, optional
            Solver to be used. Default Mosek.
        settings : dict, optional
            Solver settings. Default empty.
        parallel : bool, optional
            Solve problems in parallel. Default True.

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

        if parallel:
            print("Solving for all theta (parallel %i processors)..." %
                  cpu_count())
            #  self.pbar = tqdm(total=n, desc=message + " (parallel)")
            #  with tqdm(total=n, desc=message + " (parallel)") as self.pbar:
            with Pool(processes=min(n, cpu_count())) as pool:
                # Solve in parallel
                results = \
                    pool.map(self._populate_and_solve,
                             zip([theta.iloc[i, :] for i in range(n)],
                                 repeat(solver),
                                 repeat(settings)))
                # Solve in parallel and print tqdm progress bar
                #  results = list(tqdm(pool.imap(self._populate_and_solve,
                #                                zip([theta.iloc[i, :]
                #                                     for i in range(n)],
                #                                    repeat(solver),
                #                                    repeat(settings))),
                #                      total=n))
        else:
            # Preallocate solutions
            results = []
            for i in tqdm(range(n), desc=message + " (serial)"):
                results.append(self._populate_and_solve((theta.iloc[i, :],
                                                         solver, settings)))

        # Reorganize results
        x = [results[i][0] for i in range(n)]
        time = [results[i][1] for i in range(n)]
        strategy = [results[i][2] for i in range(n)]

        return x, time, strategy
