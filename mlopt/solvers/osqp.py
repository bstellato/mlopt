import osqp
from . import statuses as s
from .results import Results
from .solver import Solver
from ..constants import ACTIVE_CONSTRAINTS_TOL as TOL


class OSQPSolver(Solver):

    m = osqp.OSQP()
    STATUS_MAP = {m.constant('OSQP_SOLVED'): s.OPTIMAL,
                  m.constant('OSQP_MAX_ITER_REACHED'): s.MAX_ITER_REACHED,
                  m.constant('OSQP_PRIMAL_INFEASIBLE'): s.PRIMAL_INFEASIBLE,
                  m.constant('OSQP_DUAL_INFEASIBLE'): s.DUAL_INFEASIBLE}

    def __init__(self, settings={}):
        '''
        Initialize solver object by setting require settings
        '''
        self._settings = settings

    @property
    def settings(self):
        """Solver settings"""
        return self._settings

    def solve(self, problem):
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

        assert (len(problem['int_idx']) == 0), "OSQP does not support integer variables."

        p = problem
        settings = self._settings.copy()

        # Setup OSQP
        m = osqp.OSQP()
        m.setup(q=p['c'], A=p['A'], l=p['l'],
                u=p['u'], **settings)

        # Solve
        results = m.solve()
        status = self.STATUS_MAP.get(results.info.status_val, s.SOLVER_ERROR)

        # Get equality constraints
        eq_idx = np.where(p['u'] - p['l'] <= TOL)[0]

        # get active constraints
        active_cons = self.active_constraints(results.y, eq_idx)

        return_results = Results(status,
                                 results.info.obj_val,
                                 results.x,
                                 results.y,
                                 results.info.run_time,
                                 results.info.iter,
                                 None)  # No info on active constraints

        return return_results
