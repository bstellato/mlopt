from multiprocessing import Pool
#  from pathos.multiprocessing import ProcessingPool as Pool
#  import multiprocessing, logging
#  logger = multiprocessing.log_to_stderr()
#  logger.setLevel(multiprocessing.SUBDEBUG)
#  from itertools import repeat
#  from warnings import warn
import numpy as np
#  import scipy as sp
#  import scipy.sparse as spa
#  import scipy.sparse.linalg as sla
from mlopt.strategy import Strategy
from mlopt.settings import DEFAULT_SOLVER, DIVISION_TOL
from mlopt.kkt import KKT
# Import cvxpy and constraint types
import cvxpy as cp
from cvxpy.constraints.nonpos import NonPos, Inequality
from cvxpy.constraints.zero import Zero, Equality
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
# Progress bars
from tqdm import tqdm
from mlopt.utils import get_n_processes, args_norms, tight_components
import logging

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
#  from mlopt.kkt import KKT


#  def _solve_with_strategy_multiprocess(problem, strategy, cache):
#      """Wrapper function to be called with multiprocessing"""
#      return problem.solve_with_strategy(strategy, cache)


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

        # Reformulate and extract constraints
        # if problem is QP compatible
        #  if self.cvxpy_problem.is_qp():  #
        # TODO!

        # Store discrete variables to restore later
        self.int_vars = [v for v in self.cvxpy_problem.variables()
                         if v.attributes['integer']]
        self.bool_vars = [v for v in self.cvxpy_problem.variables()
                          if v.attributes['boolean']]

        # Set options
        self.solver_options = solver_options

        # Set solver cache
        self._solver_cache = None

    @property
    def solver(self):
        """Internal optimization solver"""
        return self._solver

    @solver.setter
    def solver(self, s):
        """Set internal solver."""
        if s not in INSTALLED_SOLVERS:
            err = 'Solver %s not installed.' % s
            logging.error(err)
            raise ValueError(err)
        self._solver = s

    @property
    def n_var(self):
        """Number of variables"""
        return sum([x.size for x in self.cvxpy_problem.variables()])

    @property
    def n_disc_var(self):
        """Number of discrete variables"""
        return sum([x.size for x in self.cvxpy_problem.variables()
                    if (x.attributes['boolean'] or
                        x.attributes['integer'])
                    ])

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

    def is_qp(self):
        """Is problem QP representable (LP/QP/MILP/MIQP)"""
        return self.cvxpy_problem.is_qp()

    #  def params_in_vectors(self):
    #      """Do the parameters affect only problem vectors?"""
    #      # Use constr parameters()
    #      #  for c in self.constraints:
    #          # extract rhs
    #
    #          # check if parameters are there

    #  def infeasibility(self, variables):
    def infeasibility(self):
        """
        Compute infeasibility for variables.
        """

        # Compute relative constraint violation
        violations = []
        for c in self.constraints:
            # Get constraint arguments norms
            arg_norms = args_norms(c)

            try:
                relative_viol = np.amax(arg_norms)
            except:
                import ipdb; ipdb.set_trace()
            # Get relative value for all of the expression arguments
            relative_viol = np.amax(arg_norms)

            # Normalize relative tolerance if too small
            relative_viol = relative_viol \
                if relative_viol > DIVISION_TOL else 1.

            # Append violation
            violations.append(np.atleast_1d(c.violation().flatten() /
                                            relative_viol))

        # Create numpy array
        violations = np.concatenate(violations)

        # Compute violations
        #  violations = np.concatenate([np.atleast_1d(c.violation())
        #                               for c in self.constraints])

        return np.linalg.norm(violations, np.inf)

    # Constraint error
    #  def get_constr_error(constr):
    #      if isinstance(constr, cvx.constraints.EqConstraint):
    #          error = cvx.abs(constr.args[0] - constr.args[1])
    #      elif isinstance(constr, cvx.constraints.LeqConstraint):
    #          error = cvx.pos(constr.args[0] - constr.args[1])
    #      elif isinstance(constr, cvx.constraints.PSDConstraint):
    #          mat = constr.args[0] - constr.args[1]
    #          error = cvx.neg(cvx.lambda_min(mat + mat.T)/2)
    #      return cvx.sum_entries(error)

    def _solve(self, problem, use_KKT=False, KKT_cache=None):
        """
        Solve problem with CVXPY and return results dictionary
        """
        # DEBUG: Remove KKT
        if use_KKT:
            # Solve problem with KKT system
            problem.solve(solver=KKT, KKT_cache=KKT_cache,
                          **self.solver_options)
        else:
            problem.solve(solver=self.solver,
                          **self.solver_options)

        results = {}
        results['time'] = problem.solver_stats.solve_time
        results['cost'] = np.inf
        results['status'] = problem.status

        if results['status'] in cp.settings.SOLUTION_PRESENT:
            results['x'] = np.concatenate([np.atleast_1d(v.value.flatten())
                                           for v in problem.variables()])
            results['cost'] = self.cost()
            results['infeasibility'] = self.infeasibility()
            tight_constraints = dict()
            for c in problem.constraints:
                tight_constraints[c.id] = tight_components(c)
            results['tight_constraints'] = tight_constraints
        else:
            # DEBUG
            #  problem.solve(solver=self.solver,
            #                verbose=True,
            #                **self.solver_options)
            results['x'] = np.nan * np.ones(self.n_var)
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

    def _restore_disc_var(self):
        """
        Restore relaxed original variables
        to be discrete
        """
        for var in self.int_vars:  # Integer
            self._set_int_var(var)
        for var in self.bool_vars:  # Boolean
            self._set_bool_var(var)

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
                err = "Tight constraints not compatible " + \
                    "with problem. Constaint IDs not matching."
                logging.error(err)
                raise ValueError(err)

        int_var_err = "Integer variables not compatible " + \
            "with problem. Constaint IDs not " + \
            "matching an integer variable."
        for key in strategy.int_vars.keys():
            try:
                v = variables[key]
            except KeyError:
                logging.error(err)
                raise ValueError(int_var_err)
            if not self._is_var_mip(v):
                logging.error(int_var_err)
                raise ValueError(int_var_err)

    def _construct_reduced_problem(self, strategy):
        """Construct reduced problem using strategy.
           NB. This function relaxed integrality of the original variables."""
        # Unpack strategy
        tight_constraints = strategy.tight_constraints
        int_vars = strategy.int_vars

        # Unpack original problem
        objective = self.objective
        constraints = self.constraints

        # Get only tight constraints in strategy
        reduced_constraints = []
        for con in constraints:
            idx_tight = np.where(tight_constraints[con.id])[0]
            if len(idx_tight) > 0:

                # Get tight constraints in expression
                if con.expr.shape == ():
                    # Scalar case no slicing
                    tight_expr = con.expr
                else:
                    # Get tight constraints
                    tight_expr = con.expr[idx_tight]

                # Set affine inequalities as equalities
                new_type = type(con)

                # Remove inequality and equality constraints
                if type(con) == Inequality:
                    new_type = NonPos
                elif type(con) == Equality:
                    new_type = Zero

                if new_type == NonPos and tight_expr.is_affine():
                    new_type = Zero

                reduced_constraints += [new_type(tight_expr)]

        # Fix discrete variables and
        # set them to continuous.
        discrete_fix = []
        for var in self.int_vars + self.bool_vars:
            self._set_cont_var(var)
            discrete_fix += [var == int_vars[var.id]]

        return cp.Problem(objective, reduced_constraints + discrete_fix)

    def solve_with_strategy(self,
                            strategy,
                            cache=None):
        """
        Solve problem using strategy

        By default the we solve a linear system if
        the problem is an (MI)LP/(MI)QP.

        Parameters
        ----------
        strategy : Strategy
            Strategy to be used.
        cache : dict, optional
            KKT Solver cache.

        Returns
        -------
        dict
            Results.
        """
        self._verify_strategy(strategy)

        prob_red = self._construct_reduced_problem(strategy)

        # Solve problem
        if prob_red.is_qp():
            # If QP, if must be an equality constrained QP because
            # of the loop above fixing linear inequality constraints
            # to be equalities
            results = self._solve(prob_red, use_KKT=True,
                                  KKT_cache=cache)
        else:
            results = self._solve(prob_red)

        self._restore_disc_var()

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
        n_proc = get_n_processes(n)

        if parallel:
            logging.info("Solving for all theta (parallel %i processors)..." %
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
