from multiprocessing import Pool, cpu_count
from copy import deepcopy  # For copying problem
from itertools import repeat
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
from . import settings as con
#  from .solvers.solvers import SOLVER_MAP, DEFAULT_SOLVER
#  from .solvers.statuses import SOLUTION_PRESENT
from cvxpy.settings import SOLUTION_PRESENT
from .strategy import Strategy
from .settings import TOL, DEFAULT_SOLVER
from .utils import cvxpy2data, problem_data
from tqdm import tqdm


class OptimizationProblem(object):

    def __init__(self, cvxpy_problem, name="problem"):
        """Initialize Optimization problem

        Parameters
        ----------
        cvxpy_problem: cvxpy Problem
            Problem generated in CVXPY.
        name : string
            Problem name for outputting learner.
        """
        self.name = name
        self.cvxpy_problem = cvxpy_problem

        #  # Get objective and constraints
        #  self.objective =
        #  self.constraints = cvxpy_problem.constraints
        #
        #  # Convert parameters to dict
        #  params = cvxpy_problem.parameters()
        #  self.params = {}
        #  for p in params:
        #      self.params[p.name()] = p
        #
        import ipdb; ipdb.set_trace()


    def populate(self, theta):
        """
        Populate problem using parameter theta
        """
        for p in self.cvxpy_problem.parameters():
            p.value = theta[p.name()]

        # OLD
        #  for c in theta.index.values:
        #      self.params[c].value = theta[c]
        #  for p in params:
        #      p.value = theta[p.name()]
        #  for c in theta.index.values:
        #      self.params[c].value = theta[c]

    def cost(self):
        """Compute cost function value"""
        #  return self.data['c'].dot(x)
        return self.cvxpy_problem.objective.value

    def is_mip(self):
        """Check if problem has integer variables."""
        return self.cvxpy_problem.is_mixed_integer()

    def infeasibility(self):
        """
        Compute infeasibility for vector x
        """
        violations = [c.violation() for c in self.cvxpy_problem.constraints]

        return np.linalg.norm(violations)

        #  l, A, u = self.data['l'], self.data['A'], self.data['u']
        #
        #  norm_A = [spla.norm(A[i, :], np.inf) for i in range(A.shape[0])]
        #
        #  upper = np.maximum(A.dot(x) - u, 0.)
        #  lower = np.maximum(l - A.dot(x), 0.)
        #
        #  # Normalize nonzero ones
        #  for i in range(len(upper)):
        #      if upper[i] >= TOL:
        #          upper[i] /= np.maximum(norm_A[i], np.abs(u[i]))
        #  for i in range(len(lower)):
        #      if lower[i] >= TOL:
        #          lower[i] /= np.maximum(norm_A[i], np.abs(l[i]))
        #
        #  return np.linalg.norm(upper + lower)

    #  def suboptimality(self, x, x_opt):
    #      """
    #      Compute suboptimality for vector x
    #      """
    #      c = self.data['c']
    #      return (self.cost(x) - self.cost(x_opt)) / \
    #          np.linalg.norm(c, np.inf)

    def _solve(self, problem, solver, settings):
        """
        Solve problem with CVXPY and return results dictionary
        """
        problem.solve(solver=solver)

        results = {}
        results['x'] = np.array([v.value for v in problem.variables()])
        results['time'] = problem.solver_stats.setup_time + \
            problem.solver_stats.solve_time

        if not problem.is_mixed_integer():
            active_constraints = dict()
            for c in problem.constraints:
                active_constraints[c.id] = \
                    [1 if y >= TOL else 0 for y in c.dual_value]
            results['active_constraints'] = active_constraints

        return results

    def _is_var_mip(self, var):
        """
        Is the cvxpy variable mixed-integer?
        """
        return var.attributes['boolean'] or var.attributes['integer']

    def _set_bool_var(self, var):
        """Set variable to be boolean"""
        var.attributes['boolean'] = True
        var.boolean_idx = list(np.ndindex(max(var.shape, (1,))))

    def _set_cont_var(self, var):
        """Set variable to be continuous"""
        var.attributes['boolean'] = False
        var.boolean_idx = []

    #  def _get_vars(self, objective, constraints):
    #      vars_ = objective.variables()
    #      for constr in contraints:
    #          vars_ += constr.variables()
    #      seen = set()
    #      # never use list as a variable name
    #      return [seen.add(obj.id) or obj for obj in vars_ if obj.id not in seen]

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
        # Solve complete problem with CVXPY
        results = self._solve(self.cvxpy_problem, solver, settings)

        # Get solution and integer variables
        x_opt = results['x']
        time = results['time']
        x_int = dict()

        if self.is_mip():
            # Get integer variables
            int_vars = [v if self._is_var_mip(v)
                        for v in self.cvxpy_problem.variables()]

            # Get value of integer variables
            x_int = dict()
            for x in int_vars:
                x_int[x.id] = deepcopy(x.value)

            # Change attributes to continuous
            int_vars_fix = []
            for x in int_vars:
                self._set_cont_var(x)  # Set continuous variable
                int_vars_fix += [x == x_int[x.id]]

            # Define new constraints to fix integer variables
            prob_fix_vars = cp.Problem(cp.Minimize(0), int_vars_fix)
            prob_cont = self.cvxpy_problem + prob_fix_vars

            # Solve
            results_cont = self._solve(prob_cont, solver, settings)
            active_constraints = results['active_constraints']

            # Restore integer variables
            for x in int_vars:
                self._set_bool_var(x)

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

        c = self.data['c']
        l, A, u = self.data['l'], self.data['A'], self.data['u']
        int_idx = self.data['int_idx']
        int_vars = strategy.int_vars
        active_constr = strategy.active_constraints

        if self.is_mip():
            assert np.max(int_idx) <= len(self.data['l'])
            assert np.min(int_idx) >= 0

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
            A_red = spa.vstack([A_red, A_upper]).tocsc()
            bound_red = np.concatenate((bound_red, u_upper))

        if len(active_constraints_lower) > 0:
            A_lower = A[active_constraints_lower, :]
            l_lower = l[active_constraints_lower]
            A_red = spa.vstack([A_red, A_lower]).tocsc()
            bound_red = np.concatenate((bound_red, l_lower))

        # 2) If integer program, fix integer variable
        if self.is_mip():
            A_red = spa.vstack([A_red,
                                spa.eye(n_var, format='csc')[int_idx, :]])
            bound_red = np.concatenate((bound_red, int_vars))

        # 3) Create problem data
        problem_red = OptimizationProblem()
        problem_red.data = problem_data(c, bound_red, A_red, bound_red)

        # Solve reduced problem
        x, time, _ = problem_red.solve(solver=solver, settings=settings)

        return x, time

    def populate_and_solve(self, args):
        """Single function to populate the problem with
           theta and solve it with the solver. Useful for
           multiprocessing."""
        theta, solver, settings = args
        self.populate(theta)
        results = self.solve(solver, settings)

        # DEBUG. Solve with other solvers and check
        #  res_gurobi = self.solve('GUROBI', settings)
        #  if res_gurobi[2] != results[2]:
        #      print("Wrong strategy")
        #      import ipdb; ipdb.set_trace()

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
                    pool.map(self.populate_and_solve,
                             zip([theta.iloc[i, :] for i in range(n)],
                                 repeat(solver),
                                 repeat(settings)))
                # Solve in parallel
                #  results = pool.starmap(self._populate_and_solve,)
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
                results.append(self.populate_and_solve((theta.iloc[i, :],
                                                        solver, settings)))

        # Reorganize results
        x = [results[i][0] for i in range(n)]
        time = [results[i][1] for i in range(n)]
        strategy = [results[i][2] for i in range(n)]

        return x, time, strategy
