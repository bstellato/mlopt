from multiprocessing import Pool, cpu_count
import numpy as np
from mlopt.strategy import Strategy
from mlopt.settings import BINDING_CONSTRAINTS_TOL, \
    DEFAULT_SOLVER, PERTURB_TOL
# Import cvxpy and constraint types
import cvxpy as cp
from cvxpy.constraints.nonpos import NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
# Progress bars
from tqdm import tqdm


class Problem(object):

    def __init__(self,
                 objective, constraints,
                 solver=DEFAULT_SOLVER,
                 perturb=True,
                 **solver_options):
        """
        Initialize optimization problem.

        The problem cost is perturbed to avoid degeneracy when the
        solution lies on a hyperplane.

        Parameters
        ----------
        objective : cvxpy objective
            Objective defined in CVXPY.
        constraints : cvxpy constraints
            Constraints defined in CVXPY.
        perturb : bool, optional
            Do you want to slightly perturb the cost to avoid degeneracy?
            Defaults to True.
        solver : str, optional
            Solver to solve internal problem. Defaults to DEFAULT_SOLVER.
        solver_options : dict, optional
            A dict of options for the internal solver.
        """
        # Assign solver
        self.solver = solver

        # Perturb objective to avoid degeneracy
        cost = objective.args[0]
        if perturb:
            perturbed_cost = cost
            for v in objective.variables():
                perturbation = PERTURB_TOL * np.random.randn(*v.shape)
                perturbed_cost += perturbation * v
            cost = perturbed_cost

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

    #  def infeasibility(self, variables):
    def infeasibility(self):
        """
        Compute infeasibility for variables.
        """
        # Compute violations
        violations = np.concatenate([np.atleast_1d(c.violation())
                                     for c in self.constraints])

        return np.linalg.norm(violations)

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

        if problem.status in cp.settings.SOLUTION_PRESENT:
            results['cost'] = self.cost()
            results['infeasibility'] = self.infeasibility()

            if not problem.is_mixed_integer():
                binding_constraints = dict()
                for c in problem.constraints:
                    binding_constraints[c.id] = \
                        np.array([1 if abs(y) >= BINDING_CONSTRAINTS_TOL else 0
                                  for y in np.atleast_1d(c.dual_value)])
                results['binding_constraints'] = binding_constraints
        else:
            results['cost'] = np.inf
            results['infeasibility'] = np.inf
            results['binding_constraints'] = dict()

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

        # Get solution and integer variables
        x_int = dict()

        if self.is_mip():
            # Get integer variables
            int_vars = [v for v in self.cvxpy_problem.variables()
                        if self._is_var_mip(v)]

            # Get value of integer variables
            x_int = dict()
            for x in int_vars:
                x_int[x.id] = np.array(x.value)

            # Change attributes to continuous
            int_vars_fix = []
            for x in int_vars:
                self._set_cont_var(x)  # Set continuous variable
                int_vars_fix += [x == x_int[x.id]]

            prob_cont = cp.Problem(self.objective,
                                   self.constraints +
                                   int_vars_fix)

            # Solve
            results_cont = self._solve(prob_cont)

            # Get binding constraints from original problem
            binding_constraints = dict()
            for c in self.constraints:
                binding_constraints[c.id] = \
                    results_cont['binding_constraints'][c.id]

            # Restore integer variables
            for x in int_vars:
                self._set_bool_var(x)
        else:
            binding_constraints = results['binding_constraints']

        # Get strategy
        strategy = Strategy(binding_constraints, x_int)

        # Define return dictionary
        return_dict = {}
        return_dict['x'] = results['x']
        return_dict['time'] = results['time']
        return_dict['cost'] = results['cost']
        return_dict['infeasibility'] = results['infeasibility']
        return_dict['strategy'] = strategy

        return return_dict

    def _verify_strategy(self, strategy):
        """Verify that strategy is compatible with current problem."""
        # Compare keys for binding constraints and integer variables
        variables = {v.id: v
                     for v in self.cvxpy_problem.variables()}
        con_keys = [c.id for c in self.constraints]

        for key in strategy.binding_constraints.keys():
            if key not in con_keys:
                raise ValueError("Binding constraints not compatible " +
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
        binding_constraints = strategy.binding_constraints
        int_vars = strategy.int_vars

        # Unpack original problem
        orig_objective = self.objective
        orig_variables = self.cvxpy_problem.variables()
        orig_constraints = self.constraints

        # Get same objective
        objective = orig_objective

        # Get only constraints in strategy
        constraints = []
        for con in orig_constraints:
            idx_binding = np.where(binding_constraints[con.id])[0]
            if len(idx_binding) > 0:
                # Binding constraints in expression
                con_expr = con.args[0]
                if con_expr.shape == ():
                    # Scalar case no slicing
                    binding_expr = con_expr
                else:
                    # Get binding constraints
                    binding_expr = con.args[0][idx_binding]

                # Set linear inequalities as equalities
                new_type = type(con)
                if type(con) == NonPos:
                    new_type = Zero

                constraints += [new_type(binding_expr)]

        # Fix integer variables
        int_fix = []
        for v in orig_variables:
            if self._is_var_mip(v):
                self._set_cont_var(v)
                int_fix += [v == int_vars[v.id]]

        # Solve problem
        prob_red = cp.Problem(objective, constraints + int_fix)
        results = self._solve(prob_red)

        # Make variables discrete again
        for v in self.cvxpy_problem.variables():
            if v.id in int_vars.keys():
                self._set_bool_var(v)

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
        n_proc = cpu_count()

        if parallel:
            print("Solving for all theta (parallel %i processors)..." %
                  cpu_count())
            #  self.pbar = tqdm(total=n, desc=message + " (parallel)")
            #  with tqdm(total=n, desc=message + " (parallel)") as self.pbar:
            with Pool(processes=min(n, n_proc)) as pool:
                # Solve in parallel
                #  results = \
                    #  pool.map(self.populate_and_solve,
                    #           zip([theta.iloc[i, :] for i in range(n)]))
                # Solve in parallel and print tqdm progress bar
                results = list(tqdm(pool.imap(self.populate_and_solve,
                                              [theta.iloc[i, :]
                                               for i in range(n)]),
                                    total=n))
        else:
            # Preallocate solutions
            results = []
            for i in tqdm(range(n), desc=message + " (serial)"):
                results.append(self.populate_and_solve(theta.iloc[i, :]))

        return results
