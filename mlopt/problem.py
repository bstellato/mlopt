from joblib import Parallel, delayed
import numpy as np
# Mlopt stuff
from mlopt.strategy import Strategy
import mlopt.settings as stg
from mlopt.kkt import KKT
import mlopt.utils as u
import mlopt.error as e
# Import cvxpy and constraint types
import cvxpy as cp
import cvxpy.settings as cps
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
# Progress bars
from tqdm import tqdm


class Problem(object):

    def __init__(self,
                 cvxpy_problem,
                 solver=stg.DEFAULT_SOLVER,
                 verbose=False,
                 **solver_options):
        """
        Initialize optimization problem.


        Parameters
        ----------
        problem : cvxpy.Problem
            CVXPY problem.
        solver : str, optional
            Solver to solve internal problem. Defaults to DEFAULT_SOLVER.
        solver_options : dict, optional
            A dict of options for the internal solver.
        """
        # Assign solver
        self.solver = solver
        self.verbose = verbose

        # Define problem
        if not cvxpy_problem.is_qp():
            e.value_error("MLOPT supports only MIQP-based problems " +
                          "LP/QP/MILP/MIQP")

        self.cvxpy_problem = cvxpy_problem

        # Canonicalize problem
        self._canonicalize()
        self._x = None   # Raw solution

        # Add default solver options to solver options
        if solver == stg.DEFAULT_SOLVER:
            solver_options.update(stg.DEFAULT_SOLVER_OPTIONS)

        # Set options
        self.solver_options = solver_options

        #  # Set solver cache
        #  self._solver_cache = None

    def _canonicalize(self):
        """Canonicalize optimizaton problem.
        It constructs CVXPY solving chains.
        """
        data, solving_chain, inverse_data = \
            self.cvxpy_problem.get_problem_data(self.solver, enforce_dpp=True)

        # Cache contains
        # - solving_chain
        # - inverse_data (not solver_inverse_data)
        # - param_prog (parametric_program)
        self._cache = self.cvxpy_problem._cache
        self._data = data

    def sense(self):
        return type(self.cvxpy_problem.objective)

    @property
    def solver(self):
        """Internal optimization solver"""
        return self._solver

    @solver.setter
    def solver(self, s):
        """Set internal solver."""
        if s not in INSTALLED_SOLVERS:
            e.error('Solver %s not installed.' % s)
        self._solver = s

    @property
    def n_var(self):
        """Number of variables"""
        return self._cache.param_prog.x.size

    @property
    def n_constraints(self):
        """Number of constraints"""
        return self._cache.param_prog.constr_size

    @property
    def parameters(self):
        """Problem parameters."""
        return self._cache.param_prog.parameters

    def variables(self):
        """Problem variables."""
        return self._cache.param_prog.variables

    @property
    def n_parameters(self):
        """Number of parameters."""
        return sum([x.size for x in self.parameters])

    def populate(self, theta):
        """
        Populate problem using parameter theta
        """
        for p in self.parameters:
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

    def infeasibility(self, x, data):
        """Compute infeasibility for variables given internally stored solution.
        TODO: Make it independent from the problem.

        Args:
            x (TODO): TODO
            data (TODO): TODO

        Returns: TODO

        """

        A, b = data['A'], data['b']
        F, g = data['F'], data['g']

        eq_viol = np.linalg.norm(A.dot(x) - b, np.inf)
        ineq_viol = np.linalg.norm(np.max(F.dot(x) - g, 0), np.inf)

        return np.maximum(eq_viol, ineq_viol)

    def _get_problem_data(self):
        """TODO: Docstring for _get_problem_data.
        Returns: TODO

        """
        cache = self._cache
        solving_chain = cache.solving_chain
        inverse_data = cache.inverse_data
        param_prog = cache.param_prog

        # Compute raw solution using parametric program
        data, solver_inverse_data = solving_chain.solver.apply(param_prog)
        inverse_data = inverse_data + [solver_inverse_data]

        return data, inverse_data, solving_chain

    def solve(self, strategy=None, cache=None):
        """Solve optimization problem.

        Kwargs:
            strategy (Strategy): Strategy to apply. Default none.
            cache (dict): KKT solver cache

        Returns: Dictionary of results

        """

        if strategy is not None:
            self._verify_strategy(strategy)

        data, inverse_data, solving_chain = self._get_problem_data()

        if strategy is not None:
            strategy.parse_data(data)

        raw_solution = solving_chain.solver.solve_via_data(
            data, warm_start=True, verbose=self.verbose,
            solver_opts=self.solver_options
        )

        return self._parse_solution(raw_solution, data, self.cvxpy_problem,
                                    solving_chain, inverse_data)

    def _parse_solution(self, raw_solution, data, problem,
                        solving_chain, inverse_data):
        """TODO: Docstring for _parse_solution.

        Args:
            raw_solution (TODO): TODO
            data (TODO): TODO
            problem (TODO): TODO
            solving_chain (TODO): TODO
            inverse_data (TODO): TODO

        Returns: TODO

        """

        # Unpack raw solution
        self.cvxpy_problem.unpack_results(raw_solution, solving_chain,
                                          inverse_data)

        results = {}

        # Get time and status
        results['time'] = problem.solver_stats.solve_time
        results['status'] = problem.status
        results['cost'] = np.inf  # Initialize infinite cost

        if results['status'] in cp.settings.SOLUTION_PRESENT:
            # Get raw solution
            # Invert solver solution and get raw x!
            solver = solving_chain.solver
            solver_solution = solver.invert(raw_solution, inverse_data[-1])
            x = solver_solution.primal_vars[solver.VAR_ID]
            results['x'] = x
            results['cost'] = solver_solution.opt_val
            results['infeasibility'] = self.infeasibility(x, data)

            # Get strategy
            # TODO: Construct in object-oriented way
            tight_constraints = self.tight_constraints(x, data)
            int_vars = x[data[cps.INT_IDX]]
            results['strategy'] = Strategy(tight_constraints, int_vars)
        else:
            results['x'] = np.nan * np.ones(self.n_var)
            results['cost'] = np.inf
            results['infeasibility'] = np.inf
            results['strategy'] = Strategy()

        return results

    def _verify_strategy(self, strategy):
        """Verify that strategy is compatible with current problem.
        If not, it raises an error.

        TODO: Make it object-oriented.
        Check that strategy type accepts problem type.

        Args:
            strategy (Strategy): Strategy to compare.

        """
        if len(strategy.tight_constrants) != self._data['n_ineq']:
            e.error("Tight constraints not compatible with problem. " +
                    "Different than the number of inequality constraints.")

        if len(strategy.int_vars) != self._data['int_vars_idx']:
            e.error("Integer variables not compatible " +
                    "with problem. IDs not " +
                    "matching an integer variable.")

    def populate_and_solve(self, theta):
        """Single function to populate the problem with
           theta and solve it with the solver.
           Useful for multiprocessing."""
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

        n_jobs = u.get_n_processes() if parallel else 1

        stg.logger.info(message + " (n_jobs = %d)" % n_jobs)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.populate_and_solve)(theta.iloc[i])
            for i in tqdm(range(n))
        )

        return results
