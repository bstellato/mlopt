import osqp
from . import statuses as s
from .results import Results


class OSQPSolver(object):

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

        Args:
            problem (OptimizationProblem): problem to be solved

        Returns:
            Results structure
        '''

        assert (not problem.is_mip()), "OSQP does not support integer variables."

        p = problem
        settings = self._settings.copy()

        # Setup OSQP
        m = osqp.OSQP()
        m.setup(q=p.c, A=p.A, l=p.l,
                u=p.u, **settings)

        # Solve
        results = m.solve()
        status = self.STATUS_MAP.get(results.info.status_val, s.SOLVER_ERROR)

        return_results = Results(status,
                                 results.info.obj_val,
                                 results.x,
                                 results.y,
                                 results.info.run_time,
                                 results.info.iter,
                                 None)  # No info on active constraints

        return return_results
