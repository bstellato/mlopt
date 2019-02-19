# Define and solve equality constrained QP
# Add this function as a solve method in CVXPY
#  from cvxpy.reductions import (EvalParams, FlipObjective,
#                                Qp2SymbolicQp, QpMatrixStuffing,
#                                CvxAttr2Constr)
#  from cvxpy.problems.objective import Maximize
#  from cvxpy.reductions.solvers.solving_chain import SolvingChain
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.constraints import Zero
from cvxpy.reductions.utilities import are_args_affine
from cvxpy.error import SolverError
from cvxpy.reductions.solvers import utilities
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.solvers.qp_solvers.qp_solver \
    import is_stuffed_qp_objective
from cvxpy.reductions.solvers.defines import SOLVER_MAP_QP, QP_SOLVERS, INSTALLED_SOLVERS
import cvxpy.interface as intf
import cvxpy.settings as s
import scipy.sparse as spa
from cvxpy.reductions import Solution
import numpy as np
import pypardiso as pardiso
import time
import logging

KKT = "KKT"


def create_kkt_matrix(data):
    """Create KKT matrix from data."""
    A_con = data['A']
    n_con = A_con.shape[0]
    O_con = spa.csc_matrix((n_con, n_con))

    # Create KKT linear system
    KKT = spa.vstack([spa.hstack([data['P'], A_con.T]),
                      spa.hstack([A_con, O_con])], format='csc')
    return KKT


def create_kkt_rhs(data):
    """Create KKT rhs from data."""
    return np.concatenate((-data['q'], data['b']))


def create_kkt_system(data):
    """Create KKT linear system from data."""

    KKT = create_kkt_matrix(data)
    rhs = create_kkt_rhs(data)

    return KKT, rhs


class KKTSolver(QpSolver):
    """KKT solver for equality constrained QPs"""

    def name(self):
        return KKT

    def import_solver(self):
        pass

    def invert(self, solution, inverse_data):
        attr = {s.SOLVE_TIME: solution['time']}

        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['cost']
            primal_vars = {
                list(inverse_data.id_map.keys())[0]:
                intf.DEFAULT_INTF.const_to_matrix(np.array(solution['x']))
            }
            dual_vars = utilities.get_dual_values(
                intf.DEFAULT_INTF.const_to_matrix(solution['y']),
                utilities.extract_dual_value,
                inverse_data.sorted_constraints)
        else:
            primal_vars = None
            dual_vars = None
            opt_val = np.inf
            if status == s.UNBOUNDED:
                opt_val = -np.inf
        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def solve_via_data(self, data, warm_start, verbose,
                       solver_opts,
                       solver_cache=None):

        KKT_cache = solver_opts.get('KKT_cache', None)

        n_var = data['P'].shape[0]
        n_con = len(data['b'])
        if data['F'].shape[0] > 0:
            err = 'KKT supports only equality constrained QPs.'
            logging.error(err)
            raise SolverError(err)

        #  if verbose:
        logging.info("Solving %d x %d linear system A x = b " %
                     (n_var + n_con, n_var + n_con) + "using pardiso")

        if KKT_cache is None:
            logging.info("Not using KKT solver cache")

            KKT, rhs = create_kkt_system(data)

            t_start = time.time()
            try:
                x = pardiso.spsolve(KKT, rhs)
            except ValueError:
                x = np.full(n_var + n_con, np.nan)
            t_end = time.time()

        else:
            logging.info("Using KKT solver cache")

            rhs = create_kkt_rhs(data)

            t_start = time.time()
            try:
                x = KKT_cache['factors'](rhs)
            except ValueError:
                x = np.full(n_var + n_con, np.nan)
            t_end = time.time()

        # Get results
        results = {}
        results['x'] = x[:n_var]
        results['y'] = x[n_var:]

        if np.any(np.isnan(results['x'])):
            results['status'] = s.INFEASIBLE
        else:
            results['status'] = s.OPTIMAL
            results['cost'] = \
                .5 * results['x'].T.dot(data['P'].dot(results['x'])) \
                + data['q'].dot(results['x'])
        results['time'] = t_end - t_start

        return results


# Add solver to CVXPY solvers
QP_SOLVERS.insert(0, KKT)
SOLVER_MAP_QP[KKT] = KKTSolver()
INSTALLED_SOLVERS.append(KKT)


# OLD: REGISTER A NEW SOLVE METHOD
#  def construct_kkt_solving_chain(problem):
#      """
#      Construct solving chaing using the same QP steps as
#      cvxpy.reductions.solvers/solving_chain.py
#
#      However, in the end we add LsSolver() to solve the
#      equality constrained QP using least squares.
#      """
#      reductions = []
#      #  if problem.parameters():
#      #      reductions += [EvalParams()]
#      if type(problem.objective) == Maximize:  # Force minimization
#          reductions.append(FlipObjective())
#
#      # Conclude the chain with one of the following:
#      #  reductions += [CvxAttr2Constr(),
#      #                 Qp2SymbolicQp(),
#      #                 QpMatrixStuffing(),
#      #                 KKTSolver()]
#
#      # Canonicalization
#      reductions += [CvxAttr2Constr(),
#                     Qp2SymbolicQp()]
#
#      # Parameters evaluation
#      if problem.parameters():
#          reductions += [EvalParams()]
#
#      # Lower level matrix stuffing and solution
#      reductions += [QpMatrixStuffing(),
#                     KKTSolver()]
#
#      return SolvingChain(reductions=reductions)


#  def solve_kkt(self,
#                verbose=False,
#                warm_start=True,  # Unused
#                *args, **kwargs):
#
#      #  chain_key = (KKT)
#      #  if chain_key != self._cached_chain_key:
#      try:
#          self._solving_chain = construct_kkt_solving_chain(self)
#      except Exception as e:
#          raise e
#      #  self._cached_chain_key = chain_key
#
#      # Get data from chain
#      data, inverse_data = self._solving_chain.apply(self)
#
#      # Solve problem
#      solver_output = self._solving_chain.solve_via_data(
#              self, data, warm_start, verbose, kwargs)
#
#      # Unpack results
#      self.unpack_results(solver_output,
#                          self._solving_chain,
#                          inverse_data)
#
#      return self.value
#
#
#  # Register solve method
#  cp.Problem.register_solve(KKT, solve_kkt)
#


# OLD function to solve equality constrained QP
    #  # Get problem constraints and objective
    #  #  orig_objective = self.objective
    #  orig_variables = self.cvxpy_problem.variables()
    #  orig_constraints = self.constraints
    #  #  orig_int_vars = [v for v in orig_variables
    #  #                   if v.attributes['integer']]
    #  #  orig_bool_vars = [v for v in orig_variables
    #  #                    if v.attributes['boolean']]
    #  orig_disc_vars = [v for v in orig_variables
    #                    if (v.attributes['boolean'] or
    #                        v.attributes['integer'])]
    #
    #  # Unpack strategy
    #  tight_constraints = strategy.tight_constraints
    #  int_vars = strategy.int_vars
    #
    #  # Extract problem data
    #  qp = self.cvxpy_problem.get_problem_data(solver=DEFAULT_SOLVER)[0]
    #
    #  # Construct constraints matrix
    #  A_con = spa.vstack([qp['A'], qp['F']]).tocsc()
    #  b_con = np.concatenate((qp['b'], qp['G']))
    #
    #  # Extract tight constraints
    #  tight_con_vec = np.array([])
    #  for con in orig_constraints:
    #      tight_con_vec = \
    #              np.concatenate((tight_con_vec,
    #                              np.atleast_1d(tight_constraints[con.id])))
    #  idx_tight = np.where(tight_con_vec)[0]
    #  n_tight = len(idx_tight)
    #  A_con_tight = A_con[idx_tight, :]
    #  b_con_tight = b_con[idx_tight]
    #  O_con_tight = spa.csc_matrix((n_tight, n_tight))
    #  I_con_disc = spa.eye(self.n_var).tocsc()
    #  O_con_disc = spa.csc_matrix((self.n_disc_var, n_tight))
    #
    #
    #  # Discrete variables vector (values)
    #  disc_vars_vec = np.array([])
    #  for var in orig_disc_vars:
    #      disc_vars_vec = np.concatenate((disc_vars_vec,
    #                                      np.atleast_1d(int_vars[var.id])))
    #
    #  # Discrete variables index
    #  disc_var_idx = np.array([])
    #  for v in orig_variables:
    #      if v.attributes['boolean'] or v.attributes['integer']:
    #          disc_var_idx = np.concatenate((disc_var_idx,
    #                                         [True] * v.size))
    #      else:
    #          disc_var_idx = np.concatenate((disc_var_idx,
    #                                         [False] * v.size))
    #  disc_var_idx = np.where(disc_var_idx)[0]
    #  I_con_disc = I_con_disc[disc_var_idx, :]
    #
    #
    #  # Create linear system
    #  KKT = spa.vstack([spa.hstack([qp['P'], A_con_tight.T]),
    #                    spa.hstack([A_con_tight, O_con_tight]),
    #                    spa.hstack([I_con_disc, O_con_disc])])
    #
    #  # Concatenate rhs
    #  #  rhs = np.concatenate((-qp['q'], b_con_tight))
    #  rhs = np.concatenate((-qp['q'], b_con_tight, disc_vars_vec))
    #
    #  # Solve linear system
    #  t_start = time.time()
    #  sol = spa.linalg.lsqr(KKT, rhs)[0]
    #  t_end = time.time()
    #  x_sol = sol[:self.n_var]
    #
    #  # Get results
    #  results = {}
    #  results['x'] = x_sol
    #  results['time'] = t_end - t_start
    #  results['cost'] = .5 * x_sol.T.dot(qp['P'].dot(x_sol)) + \
    #      qp['q'].dot(x_sol)
    #  violation = np.maximum(A_con.dot(x_sol) - b_con, 0.)
    #  relative_violation = np.amax(sla.norm(A_con, axis=1))
    #  results['infeasibility'] = np.linalg.norm(violation / relative_violation,
    #          np.inf)
    #
    #  return results
    #
