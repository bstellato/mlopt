import cplex as cpx
import numpy as np
from . import statuses as s
from .results import Results
from ..constants import TOL
from .solver import Solver


class CPLEXSolver(Solver):
    """
    An interface for the CPLEX QP solver.
    """

    # Map of CPLEX status to CVXPY status.
    STATUS_MAP = {1: s.OPTIMAL,
                  3: s.PRIMAL_INFEASIBLE,
                  2: s.DUAL_INFEASIBLE,
                  21: s.DUAL_INFEASIBLE,
                  22: s.PRIMAL_INFEASIBLE,
                  4: s.PRIMAL_OR_DUAL_INFEASIBLE,
                  10: s.MAX_ITER_REACHED,
                  101: s.OPTIMAL,
                  102: s.OPTIMAL,
                  103: s.PRIMAL_INFEASIBLE,
                  107: s.TIME_LIMIT,
                  118: s.DUAL_INFEASIBLE}

    def __init__(self, settings={}):
        '''
        Initialize solver object by setting require settings
        '''
        self._settings = settings

    @property
    def settings(self):
        """Solver settings"""
        return self._settings

    def solve(self, p):
        '''
        Solve problem

        Parameters
        ----------
        problem : dict
            Data of the problem to be solved.

        Returns
        -------
        Results structure
            Optimization results
        '''

        #  if p['P'] is not None:
        #      p['P'] = p['P'].tocsr()

        if p['A'] is not None:
            # Convert Matrices in CSR format
            p['A'] = p['A'].tocsr()

        # Get problem dimensions
        m, n = p['A'].shape

        # Convert infinity values to Cplex Infinity
        u = np.minimum(p['u'], cpx.infinity)
        l = np.maximum(p['l'], -cpx.infinity)

        # Define CPLEX problem
        model = cpx.Cplex()

        # Minimize problem
        model.objective.set_sense(model.objective.sense.minimize)

        # Add variables
        var_idx = model.variables.add(obj=p['c'],
                                      lb=-cpx.infinity*np.ones(n),
                                      ub=cpx.infinity*np.ones(n))

        # Constrain integer variables if present
        if len(p['int_idx']) != 0:
            for i in p['int_idx']:
                model.variables.set_types(var_idx[i],
                                          model.variables.type.integer)

        # Add constraints
        eq_idx = []
        for i in range(m):  # Add inequalities
            start = p['A'].indptr[i]
            end = p['A'].indptr[i+1]
            row = [[p['A'].indices[start:end].tolist(),
                    p['A'].data[start:end].tolist()]]
            #  if (l[i] != -cpx.infinity) & (u[i] == cpx.infinity):
            #      model.linear_constraints.add(lin_expr=row,
            #                                   senses=["G"],
            #                                   rhs=[l[i]])
            #  elif (l[i] == -cpx.infinity) & (u[i] != cpx.infinity):
            #          model.linear_constraints.add(lin_expr=row,
            #                                       senses=["L"],
            #                                       rhs=[u[i]])
            #  else:
            model.linear_constraints.add(lin_expr=row,
                                         senses=["R"],
                                         range_values=[l[i] - u[i]],
                                         rhs=[u[i]])
            if np.abs(u[i] - l[i]) <= TOL:
                eq_idx.append(i)

        # Set quadratic Cost
        #  if p['P'].count_nonzero():  # Only if quadratic form is not null
        #      qmat = []
        #      for i in range(n):
        #          start = p['P'].indptr[i]
        #          end = p['P'].indptr[i+1]
        #          qmat.append([p['P'].indices[start:end].tolist(),
        #                      p['P'].data[start:end].tolist()])
        #      model.objective.set_quadratic(qmat)

        # Set parameters
        if "verbose" not in self._settings:
            model.set_results_stream(None)
            model.set_log_stream(None)
            model.set_error_stream(None)
            model.set_warning_stream(None)
        for param, value in self._settings.items():
            if param == "verbose":
                if value == 0:
                    model.set_results_stream(None)
                    model.set_log_stream(None)
                    model.set_error_stream(None)
                    model.set_warning_stream(None)
            else:
                exec("model.parameters.%s.set(%d)" % (param, value))

        # Solve problem
        # -------------
        try:
            start = model.get_time()
            model.solve()
            end = model.get_time()
        except:  # Error in the solution
            if self._settings['verbose']:
                print("Error in CPLEX solution\n")
            return Results(s.SOLVER_ERROR, None, None, None, np.inf, None, None)

        # Return results
        # ---------------
        # Get status
        status = self.STATUS_MAP.get(model.solution.get_status(),
                                     s.SOLVER_ERROR)

        if status == s.SOLVER_ERROR:
            print(model.solution.get_status())
            import ipdb; ipdb.set_trace()


        # Get computation time
        cputime = end-start

        # Get total number of iterations
        total_iter = \
            int(model.solution.progress.get_num_barrier_iterations())

        if status in s.SOLUTION_PRESENT:
            # Get objective value
            objval = model.solution.get_objective_value()

            # Get solution
            sol = np.array(model.solution.get_values())

            # Get dual values
            if len(p['int_idx']) == 0:
                dual = -np.array(model.solution.get_dual_values())

                # Get active constraints
                active_cons = self.active_constraints(dual, eq_idx)

            else:
                dual = None
                active_cons = np.array([])

            return Results(status, objval, sol, dual,
                                   cputime, total_iter, active_cons)
        else:
            return Results(status, None, None, None,
                                   cputime, total_iter, None)

    #  def active_constraints(self, model, eq_idx):
    #      var, ineq = model.solution.basis.get_basis()
    #      n_constr = len(ineq)
    #      active_constr = np.zeros(n_constr, dtype=int)
    #      for i in eq_idx:
    #          active_constr[i] = 1
    #
    #      for i in range(len(ineq)):
    #          if ineq[i] == model.solution.basis.status.at_upper_bound:
    #              active_constr[i] = 1
    #          elif ineq[i] == model.solution.basis.status.at_lower_bound:
    #              if i not in eq_idx:
    #                  active_constr[i] = -1
    #
    #      import ipdb; ipdb.set_trace()
    #
    #      # Check active constraints
    #      #  for i in range(n_constr):
    #      #      if active_constr[i] == 1 and u[i] == cpx.infinity:
    #      #          print("wrong active upper bounds")
    #      #          import ipdb; ipdb.set_trace()
    #      #      elif active_constr[i] == -1 and l[i] == -cpx.infinity:
    #      #          print("wrong active lower bound")
    #      #          import ipdb; ipdb.set_trace()
    #      return active_constr
