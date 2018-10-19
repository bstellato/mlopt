import cvxpy as cp

# Define constants
TOL = 1e-06
TIGHT_CONSTRAINTS_TOL = 1e-06
INFINITY = 1e15

# Define default solver
DEFAULT_SOLVER = cp.GUROBI

# Define learners
PYTORCH = "pytorch"
TENSORFLOW = "tensorflow"
OPTIMAL_TREE = "optimaltree"
DEFAULT_LEARNER = PYTORCH

# Learners settings
N_BEST = 3
