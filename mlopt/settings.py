import cvxpy as cp

# Define constants
TOL = 1e-06
ACTIVE_CONSTRAINTS_TOL = 1e-05
INFINITY = 1e15

# Define default solver
DEFAULT_SOLVER=cp.MOSEK
#  DEFAULT_SOLVER=cp.GUROBI
