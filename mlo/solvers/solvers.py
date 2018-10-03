from .gurobi import GUROBISolver
from .mosek import MOSEKSolver
from .cplex import CPLEXSolver
from .osqp import OSQPSolver

GUROBI = 'GUROBI'
OSQP = 'OSQP'
MOSEK = 'MOSEK'
CPLEX = 'CPLEX'

SOLVER_MAP = {OSQP: OSQPSolver,
              CPLEX: CPLEXSolver,
              GUROBI: GUROBISolver,
              MOSEK: MOSEKSolver}

#  DEFAULT_SOLVER = GUROBI
DEFAULT_SOLVER = MOSEK
#  DEFAULT_SOLVER = CPLEX
