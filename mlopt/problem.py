import os
from multiprocessing import Pool, cpu_count
#  import multiprocessing, logging
#  logger = multiprocessing.log_to_stderr()
#  logger.setLevel(multiprocessing.SUBDEBUG)
#  from itertools import repeat
from warnings import warn
import numpy as np
from mlopt.strategy import Strategy
from mlopt.settings import TIGHT_CONSTRAINTS_TOL, \
    DEFAULT_SOLVER
# Import cvxpy and constraint types
import cvxpy as cp
from cvxpy.constraints.nonpos import NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
# Progress bars
from tqdm import tqdm

#  def populate_and_solve(args):
#      """Single function to populate the problem with
#         theta and solve it with the solver. Useful for
#         multiprocessing."""
#      problem, theta = args
#      problem.populate(theta)
#      results = problem.solve()
#
#      return results
#


class Problem(object):

    def __init__(self,
                 objective, constraints,
                 solver=DEFAULT_SOLVER,
                 **solver_options):
        """
        Initialize optimization problem.


        Parameters
        ----------
        objective : cvxpy objective
            Objective defined in CVXPY.
        constraints : cvxpy constraints
            Constraints defined in CVXPY.
        solver : str, optional
            Solver to solve internal problem. Defaults to DEFAULT_SOLVER.
        solver_options : dict, optional
            A dict of options for the internal solver.
        """
        # Assign solver
        self.solver = solver

        # Get problem cost
        cost = objective.args[0]

        # Define problem
        self.cvxpy_problem = cp.Problem(cp.Minimize(cost),
                                        constraints)
        # Set options
        self.solver_options = solver_options

    @property
    def solver(self):
        """Internal optimization solver"""
        return self._solver

    @solver.setter
    def solver(self, s):
        """Set internal solver."""
        if s not in INSTALLED_SOLVERS:
            raise ValueError('Solver %s not installed.' % s)
        self._solver = s

    @property
    def n_var(self):
        """Number of variables"""
        return sum([x.size for x in self.cvxpy_problem.variables()])

    @property
    def n_constraints(self):
        """Number of constraints"""
        return sum([x.size for x in self.constraints])

    @property
    def n_parameters(self):
        """Number of parameters."""
        return sum([x.size for x in self.cvxpy_problem.parameters()])

    def populate(self, theta):
        """
        Populate problem using parameter theta
        """
        for p in self.cvxpy_problem.parameters():
            theta_val = theta[p.name()]
            if len(theta_val) == 1:
                theta_val = theta_val[0]  # Make it a scaler in case
            p.value = theta_val

    @property
    def objective(self):
        """Inner problem objective"""
        return self.cvxpy_problem.objective

    @property
    def constraints(self):
        """Inner problem constraints"""
        return self.cvxpy_problem.constraints

    def cost(self):
        """Compute cost function value"""
        return self.objective.value

    def is_mip(self):
        """Check if problem has integer variables."""
        return self.cvxpy_problem.is_mixed_integer()

    #  def infeasibility(self, variables):
    def infeasibility(self):
        """
        Compute infeasibility for variables.
        """
        # Compute violations
        violations = np.concatenate([np.atleast_1d(c.violation())
                                     for c in self.constraints])

        return np.linalg.norm(violations, np.inf)

    def _solve(self, problem):
        """
        Solve problem with CVXPY and return results dictionary
        """
        problem.solve(solver=self.solver,
                      #  verbose=True,
                      **self.solver_options)

        results = {}
        results['time'] = problem.solver_stats.solve_time
        results['x'] = np.concatenate([np.atleast_1d(v.value)
                                       for v in problem.variables()])
        results['cost'] = np.inf
        results['status'] = problem.status

        if results['status'] in cp.settings.SOLUTION_PRESENT:
            results['cost'] = self.cost()
            results['infeasibility'] = self.infeasibility()
            tight_constraints = dict()
            for c in problem.constraints:
                val = c.args[0].value
                tight_constraints[c.id] = np.abs(val) <= TIGHT_CONSTRAINTS_TOL
            results['tight_constraints'] = tight_constraints
        else:
            results['cost'] = np.inf
            results['infeasibility'] = np.inf
            results['tight_constraints'] = dict()

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

    def _set_int_var(self, var):
        """Set variable to be integer"""
        var.attributes['integer'] = True
        var.integer_idx = list(np.ndindex(max(var.shape, (1,))))

    def _set_cont_var(self, var):
        """Set variable to be continuous"""
        var.attributes['boolean'] = False
        var.attributes['integer'] = False
        var.boolean_idx = []
        var.integer_idx = []

    def solve(self):
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
        dict
            Results dictionary.
        """
        # Solve complete problem with CVXPY
        results = self._solve(self.cvxpy_problem)

        # Get tight constraints
        tight_constraints = results['tight_constraints']

        # Get solution and integer variables
        x_int = dict()

        if self.is_mip():
            # Get integer variables
            int_vars = [v for v in self.cvxpy_problem.variables()
                        if self._is_var_mip(v)]

            # Get value of integer variables
            # by rounding them to the nearest integer
            for x in int_vars:
                x_int[x.id] = np.rint(x.value).astype(int)

        # Get strategy
        strategy = Strategy(tight_constraints, x_int)

        # Define return dictionary
        return_dict = {}
        return_dict['x'] = results['x']
        return_dict['time'] = results['time']
        return_dict['cost'] = results['cost']
        return_dict['infeasibility'] = results['infeasibility']
        return_dict['strategy'] = strategy
        return_dict['status'] = results['status']

        return return_dict

    def _verify_strategy(self, strategy):
        """Verify that strategy is compatible with current problem."""
        # Compare keys for tight constraints and integer variables
        variables = {v.id: v
                     for v in self.cvxpy_problem.variables()}
        con_keys = [c.id for c in self.constraints]

        for key in strategy.tight_constraints.keys():
            if key not in con_keys:
                raise ValueError("Tight constraints not compatible " +
                                 "with problem. Constaint IDs not matching.")

        int_var_error = ValueError("Integer variables not compatible " +
                                   "with problem. Constaint IDs not " +
                                   "matching an integer variable.")
        for key in strategy.int_vars.keys():
            try:
                v = variables[key]
            except KeyError:
                raise int_var_error
            if not self._is_var_mip(v):
                raise int_var_error

    def solve_with_strategy(self, strategy):
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
        dict
            Results.
        """
        self._verify_strategy(strategy)

        # Unpack strategy
        tight_constraints = strategy.tight_constraints
        int_vars = strategy.int_vars

        # Unpack original problem
        orig_objective = self.objective
        orig_variables = self.cvxpy_problem.variables()
        orig_constraints = self.constraints
        orig_int_vars = [v for v in orig_variables if v.attributes['integer']]
        orig_bool_vars = [v for v in orig_variables if v.attributes['boolean']]

        # Get same objective
        objective = orig_objective

        # Get only constraints in strategy
        constraints = []
        for con in orig_constraints:
            try:
                idx_tight = np.where(tight_constraints[con.id])[0]
            except KeyError:
                import ipdb; ipdb.set_trace()
            if len(idx_tight) > 0:
                # Tight constraints in expression
                con_expr = con.args[0]
                if con_expr.shape == ():
                    # Scalar case no slicing
                    tight_expr = con_expr
                else:
                    # Get tight constraints
                    tight_expr = con.args[0][idx_tight]

                # Set linear inequalities as equalities
                new_type = type(con)
                if type(con) == NonPos:
                    new_type = Zero

                # Add constraints
                constraints += [new_type(tight_expr)]

        # Fix discrete variables and
        # set them to continuous.
        discrete_fix = []
        for var in orig_int_vars + orig_bool_vars:
            self._set_cont_var(var)
            discrete_fix += [var == int_vars[var.id]]

        # Solve problem
        prob_red = cp.Problem(objective, constraints + discrete_fix)
        results = self._solve(prob_red)

        # Make variables discrete again
        for var in orig_int_vars:  # Integer
            self._set_int_var(var)
        for var in orig_bool_vars:  # Boolean
            self._set_bool_var(var)

        return results

    def populate_and_solve(self, theta):
        """Single function to populate the problem with
           theta and solve it with the solver. Useful for
           multiprocessing."""
        self.populate(theta)
        results = self.solve()

        return results

    def solve_parametric(self, theta,
                         parallel=True,  # Solve problems in parallel
                         message="Solving for all theta",
                         ):
        """
        Solve parametric problems for each value of theta.

        Parameters
        ----------
        theta : DataFrame
            Parameter values.
        parallel : bool, optional
            Solve problems in parallel. Default True.
        message : str, optional
            Message to be printed on progress bar.

        Returns
        -------
        dict
            Results dictionary.
        """
        n = len(theta)  # Number of points
        #  with open('txt.txt', 'a') as f:
        #      f.write(str(os.environ))
        try:
            n_cpus = int(os.environ["SLURM_CPUS_PER_NODE"])
        except KeyError:
            n_cpus = cpu_count()

        # DEBUG: TODO: Remove!
        #  n_cpus = 28
        #  parallel = False
        n_proc = min(n, n_cpus)

        if parallel:
            print("Solving for all theta (parallel %i processors)..." %
                  n_proc)
            #  self.pbar = tqdm(total=n, desc=message + " (parallel)")
            #  with tqdm(total=n, desc=message + " (parallel)") as self.pbar:
            pool = Pool(processes=n_proc)
            # Solve in parallel
            #  results = \
            #      pool.map(populate_and_solve,
            #               zip(repeat(self), [theta.iloc[i, :]
            #                                  for i in range(n)]))
            # SEPARATE FUNCTION
            # Solve in parallel and print tqdm progress bar
            #  results = list(tqdm(pool.imap(populate_and_solve,
            #                                zip(repeat(self), [theta.iloc[i, :]
            #                                                   for i in range(n)])),
            #                      total=n))
            results = list(tqdm(pool.imap(self.populate_and_solve,
                                          [theta.iloc[i, :]
                                           for i in range(n)]),
                                total=n))
            pool.close()
            pool.join()

        else:
            # Preallocate solutions
            results = []
            for i in tqdm(range(n), desc=message + " (serial)"):
                #  results.append(populate_and_solve((self, theta.iloc[i, :])))
                results.append(self.populate_and_solve(theta.iloc[i, :]))

        return results
