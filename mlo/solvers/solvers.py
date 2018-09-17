from solvers.gurobi import GUROBISolver
from solvers.mosek import MOSEKSolver
from solvers.osqp import OSQPSolver

GUROBI = 'GUROBI'
OSQP = 'OSQP'
MOSEK = 'MOSEK'

SOLVER_MAP = {OSQP: OSQPSolver,
              GUROBI: GUROBISolver,
              MOSEK: MOSEKSolver}
