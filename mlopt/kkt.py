# Define and solve equality constrained QP
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
import cvxpy.settings as cps
import cvxpy.interface as intf
import cvxpy.settings as s
import scipy.sparse as spa
from cvxpy.reductions import Solution
from cvxpy.constraints import Zero
import numpy as np

#  from pypardiso import spsolve
#  from pypardiso.pardiso_wrapper import PyPardisoError
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from scikits.umfpack import UmfpackWarning
import time
import warnings
import mlopt.settings as stg

KKT = "KKT"


class CatchSingularMatrixWarnings(object):

    def __init__(self):
        self.catcher = warnings.catch_warnings()

    def __enter__(self):
        self.catcher.__enter__()
        warnings.simplefilter("ignore", UmfpackWarning)

        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in double_scalars"
        )

    def __exit__(self, *args):
        self.catcher.__exit__()


def create_kkt_matrix(data):
    """Create KKT matrix from data."""
    A_con = data[cps.A + "_red"]
    n_con = A_con.shape[0]
    O_con = spa.csc_matrix((n_con, n_con))

    # Create KKT linear system
    KKT = spa.vstack([spa.hstack([data[cps.P], A_con.T]),
                      spa.hstack([A_con, O_con])], format='csc')
    return KKT


def create_kkt_rhs(data):
    """Create KKT rhs from data."""
    return np.concatenate((-data[cps.Q], data[cps.B + "_red"]))


def create_kkt_system(data):
    """Create KKT linear system from data."""

    KKT = create_kkt_matrix(data)
    rhs = create_kkt_rhs(data)

    return KKT, rhs


def factorize_kkt_matrix(KKT):

    with CatchSingularMatrixWarnings():
        return factorized(KKT)


class KKTSolver(QpSolver):
    """KKT solver for equality constrained QPs"""

    SUPPORTED_CONSTRAINTS = [Zero]  # Support only equality constraints

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
                KKTSolver.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(solution['x']))
            }

            # Build dual variables
            n_eq, n_ineq = inverse_data['n_eq'], inverse_data['n_ineq']
            # equalities
            y_eq = solution['y'][:n_eq]
            # only dual variables for inequalities (not integer variables)
            y_ineq = np.zeros(n_ineq)

            n_tight = np.sum(inverse_data['tight_constraints'])
            y_ineq[inverse_data['tight_constraints']] = \
                solution['y'][n_eq:n_eq + n_tight]
            y = np.concatenate([y_eq, y_ineq])

            dual_vars = {KKTSolver.DUAL_VAR_ID: y}
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

        n_var = data[cps.P].shape[0]
        n_con = len(data[cps.B + "_red"])  # Only equality constraints

        stg.logger.debug("Solving %d x %d linear system A x = b " %
                         (n_var + n_con, n_var + n_con))

        if solver_cache is None:
            stg.logger.debug("Not using KKT solver cache")

            KKT, rhs = create_kkt_system(data)

            t_start = time.time()
            with CatchSingularMatrixWarnings():
                x = spsolve(KKT, rhs, use_umfpack=True)
            t_end = time.time()

        else:
            stg.logger.debug("Using KKT solver cache")

            rhs = create_kkt_rhs(data)

            t_start = time.time()

            with CatchSingularMatrixWarnings():
                x = solver_cache['factors'](rhs)
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


#  # Add solver to CVXPY solvers
#  QP_SOLVERS.insert(0, KKT)
#  SOLVER_MAP_QP[KKT] = KKTSolver()
#  INSTALLED_SOLVERS.append(KKT)
