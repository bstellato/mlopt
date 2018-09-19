from .gurobi import GUROBISolver
from .mosek import MOSEKSolver
from .osqp import OSQPSolver

GUROBI = 'GUROBI'
OSQP = 'OSQP'
MOSEK = 'MOSEK'

SOLVER_MAP = {OSQP: OSQPSolver,
              GUROBI: GUROBISolver,
              MOSEK: MOSEKSolver}

DEFAULT_SOLVER = MOSEK
