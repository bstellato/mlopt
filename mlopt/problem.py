from joblib import Parallel, delayed
import numpy as np
# Mlopt stuff
from mlopt.strategy import Strategy
import mlopt.settings as stg
from mlopt.kkt import KKT
import mlopt.utils as u
# Import cvxpy and constraint types
import cvxpy as cp
from cvxpy.constraints.nonpos import NonPos, Inequality
from cvxpy.constraints.zero import Zero, Equality
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
# Progress bars
from tqdm.autonotebook import tqdm


class Problem(object):

    def __init__(self,
                 objective, constraints,
                 solver=stg.DEFAULT_SOLVER,
                 tight_constraints=True,
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


        # Define problem
        self.cvxpy_problem = cp.Problem(objective,
                                        constraints)

        # Canonicalize problem
        self._canonicalize()

        # Store discrete variables to restore later
        self.int_vars = [v for v in
                         self.cvxpy_problem._intermediate_problem.variables()
                         if v.attributes['integer']]
        self.bool_vars = [v for v in
                          self.cvxpy_problem._intermediate_problem.variables()
                          if v.attributes['boolean']]

        # Add default solver options to solver options
        if solver == stg.DEFAULT_SOLVER:
            solver_options.update(stg.DEFAULT_SOLVER_OPTIONS)

        # Set options
        self.tight_constraints = tight_constraints
        self.solver_options = solver_options

        # Set solver cache
        self._solver_cache = None

    def sense(self):
        return type(self.cvxpy_problem.objective)

    def _canonicalize(self):
        self.cvxpy_problem._construct_chains(solver=self.solver)

    @property
    def solver(self):
        """Internal optimization solver"""
        return self._solver

    @solver.setter
    def solver(self, s):
        """Set internal solver."""
        if s not in INSTALLED_SOLVERS:
            err = 'Solver %s not installed.' % s
            stg.logger.error(err)
            raise ValueError(err)
        self._solver = s

    @property
    def n_var(self):
        """Number of variables"""
        return sum([x.size for x in self.variables()])

    @property
    def n_disc_var(self):
        """Number of discrete variables"""
        return sum([x.size for x in self.variables()
                    if (x.attributes['boolean'] or
                        x.attributes['integer'])
                    ])

    @property
    def n_constraints(self):
        """Number of constraints"""
        return sum([x.size for x in self.constraints])

    def parameters(self):
        """Problem parameters."""
        return self.cvxpy_problem.parameters()

    def variables(self):
        """Problem variables."""
        return self.cvxpy_problem.variables()

    @property
    def n_parameters(self):
        """Number of parameters."""
        return sum([x.size for x in self.parameters()])

    def populate(self, theta):
        """
        Populate problem using parameter theta
        """
        for p in self.parameters():
            p.value = theta[p.name()]

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
            arg_norms = u.args_norms(c.expr)

            # Get relative value for all of the expression arguments
            relative_viol = np.amax(arg_norms)

            # Normalize relative tolerance if too small
            relative_viol = relative_viol \
                if relative_viol > stg.DIVISION_TOL else 1.

            # Append violation
            violations.append(np.atleast_1d(c.violation().flatten() /
                                            relative_viol))

        # Create numpy array
        violations = np.concatenate(violations)

        return np.linalg.norm(violations, np.inf)

    def _parse_solution(self, problem):
        """
        Parse solution of problem.

        NB. It can be a reformulation of the original problem.
        """
        results = {}

        # Specific to problem
        results['time'] = problem.solver_stats.solve_time
        results['status'] = problem.status

        # From original problem (and below)
        orig_problem = self.cvxpy_problem
        intermediate_problem = orig_problem._intermediate_problem
        results['cost'] = np.inf

        if results['status'] in cp.settings.SOLUTION_PRESENT:
            results['x'] = np.concatenate([np.atleast_1d(v.value.flatten())
                                           for v in orig_problem.variables()])
            results['cost'] = self.cost()
            results['infeasibility'] = self.infeasibility()
            tight_constraints = {}

            if self.tight_constraints:
                for c in intermediate_problem.constraints:
                    tight_constraints[c.id] = u.tight_components(c)

            # Get integer variables
            x_int = {}
            if self.is_mip():
                # Get integer variables
                int_vars = [v for v in orig_problem.variables()
                            if self._is_var_mip(v)]

                # Get value of integer variables
                # by rounding them to the nearest integer
                # NB. Some solvers do not return exactly integers
                for x in int_vars:
                    x_int[x.id] = np.rint(x.value).astype(int)

            results['strategy'] = Strategy(tight_constraints, x_int)
        else:
            results['x'] = np.nan * np.ones(self.n_var)
            results['cost'] = np.inf
            results['infeasibility'] = np.inf
            results['strategy'] = Strategy({}, {})

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

    def _relax_disc_var(self):
        """Relax variables"""
        for var in self.int_vars:  # Integer
            self._set_cont_var(var)
        for var in self.bool_vars:  # Boolean
            self._set_cont_var(var)

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
        problem = self.cvxpy_problem
        self._solve(problem,
                    solver=self.solver,
                    **self.solver_options)

        return self._parse_solution(problem)

    def _verify_strategy(self, strategy):
        """Verify that strategy is compatible with current problem."""
        # Compare keys for tight constraints and integer variables
        intermediate_problem = self.cvxpy_problem._intermediate_problem
        variables = {v.id: v
                     for v in intermediate_problem.variables()
                     if self._is_var_mip(v)}
        con_keys = [c.id for c in
                    intermediate_problem.constraints]

        for key in strategy.tight_constraints.keys():
            if key not in con_keys:
                err = "Tight constraints not compatible " + \
                    "with problem. Constaint IDs not matching."
                stg.logger.error(err)
                raise ValueError(err)

        int_var_err = "Integer variables not compatible " + \
            "with problem. IDs not " + \
            "matching an integer variable."
        for key in strategy.int_vars.keys():
            try:
                v = variables[key]
            except KeyError:
                stg.logger.error(err)
                raise ValueError(int_var_err)
            if not self._is_var_mip(v):
                stg.logger.error(int_var_err)
                raise ValueError(int_var_err)

    def _construct_reduced_problem(self, strategy):
        """Construct reduced problem from intermediate cvxpy problem
           using strategy information.

           NB. This function assumes all the variables to be continuous."""
        # Unpack strategy
        tight_constraints = strategy.tight_constraints
        int_vars = strategy.int_vars

        # Unpack original intermediate problem
        objective = self.cvxpy_problem._intermediate_problem.objective
        constraints = self.cvxpy_problem._intermediate_problem.constraints

        # Get only tight constraints in strategy
        if not self.tight_constraints:
            # Pass all problem constraints
            reduced_constraints = constraints
        else:
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
            discrete_fix += [var == int_vars[var.id]]

        return cp.Problem(objective, reduced_constraints + discrete_fix)

    def _solve(self, problem,
               solver=None,
               verbose=False, warm_start=True, **kwargs):
        """
        Custom solve method replacing cvxpy one to store intermediate
        problem results.
        """

        problem._construct_chains(solver=solver)
        intermediate_problem = problem._intermediate_problem

        data, solving_inverse_data = \
            problem._solving_chain.apply(intermediate_problem)

        solution = problem._solving_chain.solve_via_data(problem, data,
                                                         warm_start,
                                                         verbose, kwargs)

        # Unpack intermediate results
        intermediate_problem.unpack_results(solution, problem._solving_chain,
                                            solving_inverse_data)

        # Unpack final results
        full_chain = \
            problem._solving_chain.prepend(problem._intermediate_chain)
        inverse_data = problem._intermediate_inverse_data + \
            solving_inverse_data

        problem.unpack_results(solution, full_chain, inverse_data)

        return problem.value

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

        n_jobs = u.get_n_processes() if parallel else 1

        stg.logger.info(message + " (n_jobs = %d)" % n_jobs)

        results = Parallel(n_jobs=n_jobs)(
            delayed(populate_and_solve)(self, theta.iloc[i])
            for i in tqdm(range(n))
        )

        return results


def populate_and_solve(problem, theta):
    """Single function to populate the problem with
       theta and solve it with the solver. Useful for
       multiprocessing."""
    problem.populate(theta)
    results = problem.solve()

    return results


def solve_with_strategy(problem,
                        strategy,
                        cache=None):
    """
    Solve problem using strategy

    By default the we solve a linear system if
    the problem is an (MI)LP/(MI)QP.

    Parameters
    ----------
    problem : Problem
        Problem to solve.
    strategy : Strategy
        Strategy to be used.
    cache : dict, optional
        KKT Solver cache.

    Returns
    -------
    dict
        Results.
    """
    problem._verify_strategy(strategy)

    # Relax discrete variables
    problem._relax_disc_var()

    reduced_problem = problem._construct_reduced_problem(strategy)

    if problem.tight_constraints and reduced_problem.is_qp():
        # If tight constraints enabled and QP representable problem
        problem._solve(reduced_problem,
                       solver=KKT,
                       KKT_cache=cache,
                       **problem.solver_options)
    else:
        problem._solve(reduced_problem,
                       solver=problem.solver,
                       **problem.solver_options)

    problem._restore_disc_var()

    results = problem._parse_solution(reduced_problem)

    return results

